# SPDX-License-Identifier: MIT
"""Fused Triton prepare for the INT4-QK SageAttention kernel (gfx12).

From sage_int4 0.1.4 (sage_int4/quant.py), unchanged; kernel in csrc/qattn/gfx12_int4/attn_int4_qk_opt.cuh.

Layout in / out (all contiguous):
  q, k, v      bf16 [B, S, H, 128] (NHD)
  q_i4, k_i4   int8 [B, S_pad, H, 64]   two int4 per byte: dim 2j -> low nibble, 2j+1 -> high nibble
  q_scale      f32  [B, H, S_pad]       one scale per row; dequant q = code * scale (log2 domain, sm_scale folded in)
  k_scale      f32  [B, H, S_pad / 32]  one scale per 32-key tile
  v_fp8        e4m3 [B, H, 128, S_pad]  v / v_scale (same as the int8 SageAttention prepare)
  v_scale      f32  [B, H, 128]         max_s |v| / 448
Q and K are both rotated by the same orthonormal 128x128 Hadamard matrix (QK^T unchanged); K additionally has its
per-channel sequence mean removed (softmax-invariant). The rotation is a tl.dot with a +-1 matrix (exact products,
fp32 accumulation); 1/sqrt(128) is folded into the scales.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

D = 128
INT4_MAX = 7.0
K_TILE = 32
S_ALIGN = 32
FP8_MAX = 448.0
LOG2E = 1.4426950408889634

_HADAMARD = {}


def hadamard_pm1(device, dtype=torch.bfloat16):
    """Sylvester Hadamard matrix with +-1 entries (H @ H == 128 * I)."""
    key = (str(device), dtype)
    if key not in _HADAMARD:
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < D:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HADAMARD[key] = h.to(device=device, dtype=dtype).contiguous()
    return _HADAMARD[key]


@triton.jit
def _pack_int4(codes, BLOCK: tl.constexpr):
    pairs = tl.reshape(codes, (BLOCK, 64, 2))
    lo, hi = tl.split(pairs)
    return ((lo & 15) | ((hi & 15) << 4)).to(tl.int8)


@triton.jit
def _quant_q_kernel(X, HM, QO, QS, S, S_PAD, HEADS, sxb, sxs, sxh, mul, BLOCK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HEADS
    h = pid_bh % HEADS
    rows = pid_s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    x = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxs + d[None, :], mask=rows[:, None] < S, other=0.0)
    hm = tl.load(HM + d[:, None] * 128 + d[None, :])
    y = tl.dot(x, hm) * mul
    scale = tl.maximum(tl.math.div_rn(tl.max(tl.abs(y), axis=1), 7.0), 1e-12)
    codes = libdevice.rint(tl.math.div_rn(y, scale[:, None]))
    codes = tl.minimum(tl.maximum(codes, -7.0), 7.0).to(tl.int32)
    dp = tl.arange(0, 64)
    tl.store(QO + ((b * S_PAD + rows[:, None]) * HEADS + h) * 64 + dp[None, :], _pack_int4(codes, BLOCK),
             mask=rows[:, None] < S_PAD)
    tl.store(QS + (b * HEADS + h) * S_PAD + rows, scale, mask=rows < S_PAD)


@triton.jit
def _quant_k_kernel(X, HM, MEAN_H, KO, KS, S, S_PAD, HEADS, sxb, sxs, sxh, mul, BLOCK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HEADS
    h = pid_bh % HEADS
    rows = pid_s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    x = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxs + d[None, :], mask=rows[:, None] < S, other=0.0)
    hm = tl.load(HM + d[:, None] * 128 + d[None, :])
    mean_h = tl.load(MEAN_H + pid_bh * 128 + d)
    # (k - mean) @ H == k @ H - mean @ H; padded keys stay exactly zero.
    y = tl.where(rows[:, None] < S, (tl.dot(x, hm) - mean_h[None, :]) * mul, 0.0)
    tiles: tl.constexpr = BLOCK // 32
    tmax = tl.max(tl.reshape(tl.abs(y), (tiles, 32 * 128)), axis=1)
    tscale = tl.maximum(tl.math.div_rn(tmax, 7.0), 1e-12)
    scale = tl.reshape(tl.broadcast_to(tscale[:, None], (tiles, 32)), (BLOCK,))
    codes = libdevice.rint(tl.math.div_rn(y, scale[:, None]))
    codes = tl.minimum(tl.maximum(codes, -7.0), 7.0).to(tl.int32)
    dp = tl.arange(0, 64)
    tl.store(KO + ((b * S_PAD + rows[:, None]) * HEADS + h) * 64 + dp[None, :], _pack_int4(codes, BLOCK),
             mask=rows[:, None] < S_PAD)
    t = pid_s * tiles + tl.arange(0, tiles)
    tl.store(KS + (b * HEADS + h) * (S_PAD // 32) + t, tscale, mask=t < S_PAD // 32)


@triton.jit
def _kv_stats_kernel(K, V, KSUM, VMAX, S, HEADS, NBLK, skb, sks, skh, svb, svs, svh, BLOCK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HEADS
    h = pid_bh % HEADS
    rows = pid_s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    m = rows[:, None] < S
    k = tl.load(K + b * skb + h * skh + rows[:, None] * sks + d[None, :], mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + b * svb + h * svh + rows[:, None] * svs + d[None, :], mask=m, other=0.0).to(tl.float32)
    out = (pid_bh * NBLK + pid_s) * 128 + d
    tl.store(KSUM + out, tl.sum(k, axis=0))
    tl.store(VMAX + out, tl.max(tl.abs(v), axis=0))


@triton.jit
def _quant_v_kernel(X, VS, VO, S, S_PAD, HEADS, sxb, sxs, sxh, BLOCK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HEADS
    h = pid_bh % HEADS
    rows = pid_s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    v = tl.load(X + b * sxb + h * sxh + rows[:, None] * sxs + d[None, :], mask=rows[:, None] < S, other=0.0)
    vq = (v.to(tl.float32) / tl.load(VS + pid_bh * 128 + d)[None, :]).to(VO.dtype.element_ty)
    vq = tl.where(rows[:, None] < S, vq, 0.0).to(VO.dtype.element_ty)
    tl.store(VO + (pid_bh * 128 + d[None, :]) * S_PAD + rows[:, None], vq, mask=rows[:, None] < S_PAD)


def _check(q, k, v):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[-1] != D:
        raise ValueError(f"expected equal [B, S, H, 128] q/k/v, got {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    for t in (q, k, v):
        if t.dtype != torch.bfloat16 or not t.is_cuda or t.stride(-1) != 1:
            raise ValueError("expected bf16 CUDA tensors with a contiguous head dim")


@torch.no_grad()
def prepare_int4(q, k, v, sm_scale=None, block=64):
    """bf16 NHD q/k/v -> (q_i4, q_scale, k_i4, k_scale, v_fp8, v_scale) for the INT4-QK kernel."""
    _check(q, k, v)
    batch, seq, heads, _ = q.shape
    s_pad = (seq + S_ALIGN - 1) // S_ALIGN * S_ALIGN
    dev = q.device
    sm = (D**-0.5 if sm_scale is None else sm_scale) * LOG2E
    hm = hadamard_pm1(dev)
    grid = (triton.cdiv(s_pad, block), batch * heads)

    nblk = triton.cdiv(seq, block)
    ksum = torch.empty(batch * heads, nblk, D, device=dev, dtype=torch.float32)
    vmax = torch.empty_like(ksum)
    _kv_stats_kernel[(nblk, batch * heads)](k, v, ksum, vmax, seq, heads, nblk, k.stride(0), k.stride(1), k.stride(2),
                                           v.stride(0), v.stride(1), v.stride(2), BLOCK=block, num_warps=4)
    mean_h = (ksum.sum(1) / seq) @ hm.float()                 # [B*H, 128], deterministic two-stage reduction
    v_scale = (vmax.amax(1) / FP8_MAX).view(batch, heads, D)

    q_i4 = torch.empty(batch, s_pad, heads, D // 2, device=dev, dtype=torch.int8)
    k_i4 = torch.empty_like(q_i4)
    q_scale = torch.empty(batch, heads, s_pad, device=dev, dtype=torch.float32)
    k_scale = torch.empty(batch, heads, s_pad // K_TILE, device=dev, dtype=torch.float32)
    v_fp8 = torch.empty(batch, heads, D, s_pad, device=dev, dtype=torch.float8_e4m3fn)
    rs = D**-0.5
    _quant_q_kernel[grid](q, hm, q_i4, q_scale, seq, s_pad, heads, q.stride(0), q.stride(1), q.stride(2), sm * rs,
                          BLOCK=block, num_warps=4)
    _quant_k_kernel[grid](k, hm, mean_h, k_i4, k_scale, seq, s_pad, heads, k.stride(0), k.stride(1), k.stride(2), rs,
                          BLOCK=block, num_warps=4)
    _quant_v_kernel[grid](v, v_scale, v_fp8, seq, s_pad, heads, v.stride(0), v.stride(1), v.stride(2),
                          BLOCK=block, num_warps=4)
    return q_i4, q_scale, k_i4, k_scale, v_fp8, v_scale


# ---------------------------------------------------------------------------------------------------------------
# prepare_int4_norm_rope (0.1.3): QK RMSNorm + split-half RoPE fused into the prepare, straight from the [S, H*128]
# Q/K/V projections, bit-exact with 0.1.2 (LightX2V split_qkv_norm_rope -> prepare_int4). Launches:
#   1. Q: norm + RoPE + Hadamard + int4          | K: norm + RoPE -> per-64-row channel sums; V: per-64-row |max|
#   2. K mean @ Hadamard, V scale: the same torch ops as prepare_int4
#   3. K: norm + RoPE + Hadamard - mean + int4   | V: fp8, transposed
# Every order-dependent fp32 sum reproduces the original kernel's order with explicit pairwise trees (a + b is
# order-free), so the result does not depend on num_warps; the WMMA dots keep prepare_int4's 64x128x128 tile.
# Grids are (heads, row blocks, part): the heads of one row block run together and share the cos/sin rows in L2.
# ---------------------------------------------------------------------------------------------------------------
NR_BLOCK = 64
NR_WARPS = 8


@triton.jit
def _sumsq_rows(x, BLOCK: tl.constexpr):
    # Row sums of x^2 in the order of split_qkv_norm_rope's 1-D tl.sum over 128 lanes (4 wave32, DPP) on gfx12:
    # inside 16-lane rows over column bits 3, 2, 1, 0, then bit 4 (the two rows), then warps (bit 6, then bit 5).
    s = x * x
    s = tl.sum(tl.reshape(s, (BLOCK, 8, 2, 8)), axis=2)
    s = tl.sum(tl.reshape(s, (BLOCK, 8, 2, 4)), axis=2)
    s = tl.sum(tl.reshape(s, (BLOCK, 8, 2, 2)), axis=2)
    s = tl.sum(s, axis=2)
    s = tl.sum(tl.reshape(s, (BLOCK, 4, 2)), axis=2)
    s = tl.sum(tl.reshape(s, (BLOCK, 2, 2)), axis=1)
    return tl.sum(s, axis=1)


@triton.jit
def _colsum_kv_stats(k):
    # Column sums of k [64, 128] in the order of prepare_int4's _kv_stats_kernel tl.sum(axis=0) (num_warps 4, gfx12):
    # row = 8 * rep + 2 * warp + lane; per thread over rep (row bits 3, 4, 5), then lane (bit 0), then warps (bit 2, 1).
    s = tl.sum(tl.reshape(k, (4, 2, 8, 128)), axis=1)
    s = tl.sum(tl.reshape(s, (2, 2, 8, 128)), axis=1)
    s = tl.sum(s, axis=0)
    s = tl.sum(tl.reshape(s, (4, 2, 128)), axis=1)
    s = tl.sum(tl.reshape(s, (2, 2, 128)), axis=0)
    return tl.sum(s, axis=0)


@triton.jit
def _norm_rope_tile(X, sxs, col, W, COS, SIN, scs, rows, S, eps, ROTARY: tl.constexpr, BLOCK: tl.constexpr):
    # Per element the same arithmetic as LightX2V minimax_h3 split_qkv_norm_rope (launch with enable_fp_fusion=False).
    d = tl.arange(0, 128)
    m = rows[:, None] < S
    pair = tl.where(d < ROTARY, tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2), d)
    base = X + rows[:, None] * sxs + col
    x = tl.load(base + d[None, :], mask=m, other=0.0).to(tl.float32)
    xp = tl.load(base + pair[None, :], mask=m, other=0.0).to(tl.float32)
    r = tl.rsqrt(_sumsq_rows(x, BLOCK) / 128 + eps)
    y = (x * r[:, None] * tl.load(W + d).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
    yp = (xp * r[:, None] * tl.load(W + pair).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
    cm = m & (d[None, :] < ROTARY)
    c = tl.load(COS + rows[:, None] * scs + d[None, :], mask=cm, other=0.0)
    s = tl.load(SIN + rows[:, None] * scs + d[None, :], mask=cm, other=0.0)
    a = y * c
    b = yp * s
    out = tl.where(d[None, :] < ROTARY, tl.where(d[None, :] < ROTARY // 2, a - b, a + b), y)
    return out.to(tl.bfloat16)


@triton.jit
def _nr_q_stats_kernel(Q, K, V, QW, KW, COS, SIN, HM, QO, QS, KSUM, VMAX, S, S_PAD, HEADS, NBLK,
                       sqs, sks, svs, scs, q_eps, k_eps, mul, ROTARY: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    if tl.program_id(2) == 0:
        x = _norm_rope_tile(Q, sqs, h * 128, QW, COS, SIN, scs, rows, S, q_eps, ROTARY, BLOCK)
        hm = tl.load(HM + d[:, None] * 128 + d[None, :])
        y = tl.dot(x, hm) * mul
        scale = tl.maximum(tl.math.div_rn(tl.max(tl.abs(y), axis=1), 7.0), 1e-12)
        codes = libdevice.rint(tl.math.div_rn(y, scale[:, None]))
        codes = tl.minimum(tl.maximum(codes, -7.0), 7.0).to(tl.int32)
        dp = tl.arange(0, 64)
        tl.store(QO + (rows[:, None] * HEADS + h) * 64 + dp[None, :], _pack_int4(codes, BLOCK),
                 mask=rows[:, None] < S_PAD)
        tl.store(QS + h * S_PAD + rows, scale, mask=rows < S_PAD)
    else:
        k = _norm_rope_tile(K, sks, h * 128, KW, COS, SIN, scs, rows, S, k_eps, ROTARY, BLOCK).to(tl.float32)
        v = tl.load(V + rows[:, None] * svs + h * 128 + d[None, :], mask=rows[:, None] < S, other=0.0).to(tl.float32)
        out = (h * NBLK + pid_s) * 128 + d
        tl.store(KSUM + out, _colsum_kv_stats(k))
        tl.store(VMAX + out, tl.max(tl.abs(v), axis=0))


@triton.jit
def _nr_kv_quant_kernel(K, V, KW, COS, SIN, HM, MEAN_H, VS, KO, KS, VO, S, S_PAD, HEADS, sks, svs, scs, k_eps, mul,
                        ROTARY: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    tiles: tl.constexpr = BLOCK // 32
    if tl.program_id(2) == 0:
        x = _norm_rope_tile(K, sks, h * 128, KW, COS, SIN, scs, rows, S, k_eps, ROTARY, BLOCK)
        hm = tl.load(HM + d[:, None] * 128 + d[None, :])
        mean_h = tl.load(MEAN_H + h * 128 + d)
        y = tl.where(rows[:, None] < S, (tl.dot(x, hm) - mean_h[None, :]) * mul, 0.0)
        tmax = tl.max(tl.reshape(tl.abs(y), (tiles, 32 * 128)), axis=1)
        tscale = tl.maximum(tl.math.div_rn(tmax, 7.0), 1e-12)
        scale = tl.reshape(tl.broadcast_to(tscale[:, None], (tiles, 32)), (BLOCK,))
        codes = libdevice.rint(tl.math.div_rn(y, scale[:, None]))
        codes = tl.minimum(tl.maximum(codes, -7.0), 7.0).to(tl.int32)
        dp = tl.arange(0, 64)
        tl.store(KO + (rows[:, None] * HEADS + h) * 64 + dp[None, :], _pack_int4(codes, BLOCK),
                 mask=rows[:, None] < S_PAD)
        t = pid_s * tiles + tl.arange(0, tiles)
        tl.store(KS + h * (S_PAD // 32) + t, tscale, mask=t < S_PAD // 32)
    else:
        v = tl.load(V + rows[:, None] * svs + h * 128 + d[None, :], mask=rows[:, None] < S, other=0.0)
        vq = (v.to(tl.float32) / tl.load(VS + h * 128 + d)[None, :]).to(VO.dtype.element_ty)
        vq = tl.where(rows[:, None] < S, vq, 0.0).to(VO.dtype.element_ty)
        tl.store(VO + (h * 128 + d[None, :]) * S_PAD + rows[:, None], vq, mask=rows[:, None] < S_PAD)


def _check_norm_rope(q, k, v, q_weight, k_weight, cos, sin):
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape or q.shape[0] == 0 or q.shape[1] % D:
        raise ValueError(f"expected equal non-empty [S, H*128] q/k/v, got {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    for t in (q, k, v):
        if t.dtype != torch.bfloat16 or not t.is_cuda or t.stride(1) != 1:
            raise ValueError("expected bf16 CUDA q/k/v with contiguous channels")
    for w in (q_weight, k_weight):
        if w.shape != (D,) or w.dtype != torch.bfloat16 or not w.is_contiguous():
            raise ValueError("expected contiguous bf16 [128] norm weights")
    rot = cos.shape[-1]
    if (cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != q.shape[0] or not 0 < rot <= D or rot % 2
            or cos.dtype != torch.float32 or sin.dtype != torch.float32):
        raise ValueError("expected fp32 cos/sin [S, rot] with an even rot <= 128")


@torch.no_grad()
def prepare_int4_norm_rope(q, k, v, q_weight, k_weight, cos, sin, q_eps, k_eps, sm_scale=None):
    """bf16 [S, H*128] Q/K/V projections -> prepare_int4 outputs (B = 1) of norm_rope(Q), norm_rope(K), V.

    norm_rope(x), per head: y = bf16(x * rsqrt(mean(x^2) + eps) * weight), then split-half RoPE on the first
    rot = cos.shape[-1] dims with full-width fp32 cos/sin [S, rot] (rot even <= 128), rounded to bf16: LightX2V
    minimax_h3 split_qkv_norm_rope. Bit-identical to prepare_int4(split_qkv_norm_rope(...)) on gfx12.
    """
    _check_norm_rope(q, k, v, q_weight, k_weight, cos, sin)
    if sin.stride() != cos.stride():
        cos, sin = cos.contiguous(), sin.contiguous()
    seq, heads = q.shape[0], q.shape[1] // D
    s_pad = (seq + S_ALIGN - 1) // S_ALIGN * S_ALIGN
    nblk = triton.cdiv(s_pad, NR_BLOCK)                       # == cdiv(seq, 64): the ksum shape of prepare_int4
    dev = q.device
    sm = (D**-0.5 if sm_scale is None else sm_scale) * LOG2E
    rs = D**-0.5
    hm = hadamard_pm1(dev)

    q_i4 = torch.empty(1, s_pad, heads, D // 2, device=dev, dtype=torch.int8)
    k_i4 = torch.empty_like(q_i4)
    q_scale = torch.empty(1, heads, s_pad, device=dev, dtype=torch.float32)
    k_scale = torch.empty(1, heads, s_pad // K_TILE, device=dev, dtype=torch.float32)
    v_fp8 = torch.empty(1, heads, D, s_pad, device=dev, dtype=torch.float8_e4m3fn)
    ksum = torch.empty(heads, nblk, D, device=dev, dtype=torch.float32)
    vmax = torch.empty_like(ksum)
    common = dict(ROTARY=cos.shape[1], BLOCK=NR_BLOCK, num_warps=NR_WARPS, enable_fp_fusion=False)

    _nr_q_stats_kernel[(heads, nblk, 2)](q, k, v, q_weight, k_weight, cos, sin, hm, q_i4, q_scale, ksum, vmax,
                                         seq, s_pad, heads, nblk, q.stride(0), k.stride(0), v.stride(0),
                                         cos.stride(0), q_eps, k_eps, sm * rs, **common)
    mean_h = (ksum.sum(1) / seq) @ hm.float()
    v_scale = (vmax.amax(1) / FP8_MAX).view(1, heads, D)
    _nr_kv_quant_kernel[(heads, nblk, 2)](k, v, k_weight, cos, sin, hm, mean_h, v_scale, k_i4, k_scale, v_fp8,
                                          seq, s_pad, heads, k.stride(0), v.stride(0), cos.stride(0), k_eps, rs,
                                          **common)
    return q_i4, q_scale, k_i4, k_scale, v_fp8, v_scale

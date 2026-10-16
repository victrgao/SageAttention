"""INT4-QK / FP8-PV SageAttention for gfx12 (gfx1200 / gfx1201), head_dim 128, non-causal.

Q and K are rotated by the same orthonormal 128x128 Hadamard matrix and quantized to int4 (Q per row, K per 32-key
tile after removing its per-channel mean); V is FP8 per channel. From sage_int4 0.1.4 (MiniMax-H3 mixed int4/int8).
"""
import importlib

import torch

from .triton.quant_gfx12_int4 import prepare_int4, prepare_int4_norm_rope

SUPPORTED_ARCHS = ("gfx1200", "gfx1201")


def is_gfx12_int4_supported(device=None):
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(device or torch.cuda.current_device()).gcnArchName.split(":")[0]
    return arch in SUPPORTED_ARCHS


def attention_int4_from_quantized(q_i4, q_scale, k_i4, k_scale, v_fp8, v_scale, seq_len):
    """Attention on prepare_int4 / prepare_int4_norm_rope outputs; returns bf16 [B, seq_len, H, 128] (a view)."""
    batch, s_pad, heads, _ = q_i4.shape
    out = torch.empty(batch, s_pad, heads, 128, device=q_i4.device, dtype=torch.bfloat16)
    importlib.import_module("sageattention._qattn_gfx12_int4").attention(
        q_i4, q_scale, k_i4, k_scale, v_fp8, v_scale, out, seq_len)
    return out[:, :seq_len]


@torch.no_grad()
def sageattn_qk_int4_pv_fp8_gfx12(q, k, v, tensor_layout="NHD", sm_scale=None):
    """q, k, v: bf16 with equal shapes and head_dim 128, "NHD" [B, S, H, 128] or "HND" [B, H, S, 128].
    Returns bf16 in the input layout."""
    if tensor_layout not in ("NHD", "HND"):
        raise ValueError(f"tensor_layout must be 'NHD' or 'HND', got {tensor_layout!r}")
    if tensor_layout == "HND":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = attention_int4_from_quantized(*prepare_int4(q, k, v, sm_scale=sm_scale), q.shape[1])
    return out.transpose(1, 2) if tensor_layout == "HND" else out


@torch.no_grad()
def sageattn_qk_int4_pv_fp8_gfx12_norm_rope(q, k, v, q_weight, k_weight, cos, sin, q_eps, k_eps, sm_scale=None):
    """Attention of (per-head QK RMSNorm + split-half RoPE) applied to bf16 [S, H*128] Q/K/V projections, with the norm
    and RoPE fused into the int4 prepare (prepare_int4_norm_rope). Returns bf16 [S, H, 128]."""
    inputs = prepare_int4_norm_rope(q, k, v, q_weight, k_weight, cos, sin, q_eps, k_eps, sm_scale=sm_scale)
    return attention_int4_from_quantized(*inputs, q.shape[0])[0]

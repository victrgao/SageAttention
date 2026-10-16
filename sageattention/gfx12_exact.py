import importlib
import math

import torch


@torch.no_grad()
def sageattn_qk_int8_pv_fp8_gfx1201_exact(
    q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=None,
):
    """Exact-exp2 INT8/FP8 attention; quantization error versus FP32 remains."""
    if torch.version.hip is None:
        raise RuntimeError("This operator requires ROCm")
    if tensor_layout not in ("NHD", "HND") or is_causal:
        raise ValueError("Expected noncausal NHD or HND attention")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError("Expected equal four-dimensional Q/K/V shapes with D128")
    length = q.shape[1 if tensor_layout == "NHD" else 2]
    if min(q.shape) < 1 or length > 131072:
        raise ValueError("Expected positive dimensions and sequence length <= 131072")
    for tensor in (q, k, v):
        if (tensor.device != q.device or tensor.device.type != "cuda"
                or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()
                or tensor.requires_grad):
            raise ValueError("Expected contiguous inference-only BF16 Q/K/V on one GPU")
    scale = 128 ** -0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0 or scale > torch.finfo(torch.float32).max:
        raise ValueError("Expected a finite positive FP32 softmax scale")
    with torch.cuda.device(q.device):
        if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx1201":
            raise RuntimeError("This operator requires gfx1201")
        attention = importlib.import_module("sageattention._qattn_gfx12_exact")
        native = importlib.import_module("sageattention._qattn_gfx12_native")
        from .triton.quant_gfx12_exact import prepare_key, prepare_value

        if tensor_layout == "HND":
            q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
        query_int8, query_scale = native.quant_q_nhd_per_warp(q)
        key_int8, key_scale = prepare_key(k)
        value_fp8, value_scale = prepare_value(v)
        output = torch.empty(
            (q.shape[0], ((length + 511) // 512) * 512, q.shape[2], 128),
            device=q.device, dtype=torch.bfloat16,
        )
        output = attention.attention(
            query_int8, key_int8, value_fp8, output, query_scale, key_scale, value_scale, scale, length,
        )[:, :length].contiguous()
        return output if tensor_layout == "NHD" else output.transpose(1, 2).contiguous()


@torch.no_grad()
def sageattn_qk_int8_pv_fp8_gfx1201_exact_prepacked(
    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, seq_len, sm_scale=None,
):
    """The exact-exp2 core on inputs already quantized in its layout, e.g. by the LightX2V MiniMax-H3 fused
    QK-norm/RoPE/int8 prepare with lane-major K/V.

    q_int8 int8 [B, S, H, 128], q_scale fp32 [B, H, ceil(S/32)] (log2(e) folded in, sm_scale applied by the kernel);
    k_int8 int8 [B, H, S_pad, 128] lane-major, k_scale fp32 [B, H, S_pad/64];
    v_fp8 fp8 e4m3 bits [B, H, 128, S_pad] lane-major, v_scale fp32 [B, H, 128]; seq_len = valid K/V length.
    Returns bf16 [B, S, H, 128] (a view of a padded buffer).
    """
    attention = importlib.import_module("sageattention._qattn_gfx12_exact")
    if v_fp8.dtype == torch.float8_e4m3fn:
        v_fp8 = v_fp8.view(torch.uint8)
    batch, length, heads, _ = q_int8.shape
    scale = 128 ** -0.5 if sm_scale is None else float(sm_scale)
    output = torch.empty((batch, ((length + 511) // 512) * 512, heads, 128), device=q_int8.device,
                         dtype=torch.bfloat16)
    return attention.attention(
        q_int8, k_int8, v_fp8, output, q_scale, k_scale, v_scale, scale, seq_len,
    )[:, :length]
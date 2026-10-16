import importlib
import math

import torch


@torch.no_grad()
def sageattn_qk_int8_pv_fp8_gfx1201_compact(
    q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=None,
):
    """Compact INT8/FP8 attention; approximate probabilities add error versus exact P."""
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
        attention = importlib.import_module("sageattention._qattn_gfx12_compact")
        native = importlib.import_module("sageattention._qattn_gfx12_native")
        from .triton.quant_gfx12_compact import prepare_key, prepare_value

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
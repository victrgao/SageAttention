import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _key_quant_nhd_hnd(Input, Mean, Output, Scale, Length: tl.constexpr,
                       Heads: tl.constexpr, Padded: tl.constexpr):
    head = tl.program_id(0)
    block = tl.program_id(1)
    batch = tl.program_id(2)
    rows = block * 64 + tl.arange(0, 64)
    dims = tl.arange(0, 128)
    values = tl.load(
        Input + ((batch * Length + rows[:, None]) * Heads + head) * 128 + dims[None, :],
        rows[:, None] < Length, 0,
    ).to(tl.float32)
    mean = tl.load(Mean + (batch * Heads + head) * 128 + dims).to(tl.float32)
    centered = tl.where(rows[:, None] < Length, values - mean[None, :], 0.0)
    maximum = tl.maximum(tl.max(tl.max(tl.abs(centered), axis=1), axis=0), 1e-7)
    quantized = libdevice.nearbyint(centered * (127.0 / maximum)).to(tl.int8)
    offsets = (batch * Heads + head) * Padded * 128 + (rows[:, None] // 32) * 4096 + (((rows[:, None] % 32 // 16) * 8 + dims[None, :] // 16) * 32 + rows[:, None] % 16 + 16 * (dims[None, :] % 16 // 8)) * 8 + dims[None, :] % 8
    tl.store(Output + offsets, quantized)
    tl.store(Scale + (batch * Heads + head) * (Padded // 64) + block, maximum / 127.0)


@triton.jit
def _value_amax_nhd(Input, Partial, Length: tl.constexpr, Heads: tl.constexpr,
                    Chunks: tl.constexpr, Rows: tl.constexpr):
    head = tl.program_id(0)
    chunk = tl.program_id(1)
    batch = tl.program_id(2)
    rows = chunk * Rows + tl.arange(0, Rows)
    dims = tl.arange(0, 128)
    offsets = ((batch * Length + rows[:, None]) * Heads + head) * 128 + dims[None, :]
    values = tl.load(Input + offsets, rows[:, None] < Length, 0).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=0)
    tl.store(Partial + ((batch * Chunks + chunk) * Heads + head) * 128 + dims, maximum)


@triton.jit
def _value_scale_nhd(Partial, Scale, Heads: tl.constexpr, Chunks: tl.constexpr,
                     ChunkBlock: tl.constexpr, ScaleMax: tl.constexpr):
    group = tl.program_id(0)
    batch = tl.program_id(1)
    dims = group * 32 + tl.arange(0, 32)
    chunks = tl.arange(0, ChunkBlock)
    offsets = (batch * Chunks + chunks[:, None]) * Heads * 128 + dims[None, :]
    values = tl.load(Partial + offsets, chunks[:, None] < Chunks, 0)
    maximum = tl.max(values, axis=0)
    tl.store(Scale + batch * Heads * 128 + dims, maximum / ScaleMax)


@triton.jit
def _value_quant_transpose_nhd(Input, Scale, Output, Length: tl.constexpr,
                              Padded: tl.constexpr, Heads: tl.constexpr,
                              Rows: tl.constexpr):
    head = tl.program_id(0)
    block = tl.program_id(1)
    batch = tl.program_id(2)
    rows = block * Rows + tl.arange(0, Rows)
    dims = tl.arange(0, 128)
    values = tl.load(
        Input + ((batch * Length + rows[:, None]) * Heads + head) * 128 + dims[None, :],
        rows[:, None] < Length, 0,
    ).to(tl.float32)
    scales = tl.load(Scale + (batch * Heads + head) * 128 + dims)
    values = values / tl.where(scales[None, :] == 0, 1.0, scales[None, :])
    packed = values.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    offsets = (batch * Heads + head) * 128 * Padded + (rows[:, None] // 32) * 4096 + (((rows[:, None] % 32 // 16) * 8 + dims[None, :] // 16) * 32 + dims[None, :] % 16 + 16 * (rows[:, None] % 16 // 8)) * 8 + rows[:, None] % 8
    tl.store(Output + offsets, packed, rows[:, None] < Padded)


def prepare_key(key):
    batch, length, heads, _ = key.shape
    padded = triton.cdiv(length, 64) * 64
    mean = key.mean(dim=1)
    output = torch.empty((batch, heads, padded, 128), device=key.device, dtype=torch.int8)
    scale = torch.empty((batch, heads, padded // 64), device=key.device, dtype=torch.float32)
    _key_quant_nhd_hnd[(heads, padded // 64, batch)](
        key, mean, output, scale, length, heads, padded, num_warps=4,
    )
    return output, scale


def prepare_value(value):
    batch, length, heads, _ = value.shape
    chunks = triton.cdiv(length, 256)
    padded = triton.cdiv(length, 64) * 64
    partial = torch.empty((batch, chunks, heads, 128), device=value.device, dtype=torch.float32)
    scale = torch.empty((batch, heads, 128), device=value.device, dtype=torch.float32)
    output = torch.empty((batch, heads, 128, padded), device=value.device, dtype=torch.uint8)
    _value_amax_nhd[(heads, chunks, batch)](
        value, partial, length, heads, chunks, 256, num_warps=8,
    )
    _value_scale_nhd[(heads * 4, batch)](
        partial, scale, heads, chunks, triton.next_power_of_2(chunks), 2.25, num_warps=4,
    )
    _value_quant_transpose_nhd[(heads, triton.cdiv(padded, 64), batch)](
        value, scale, output, length, padded, heads, 64, num_warps=8,
    )
    return output, scale
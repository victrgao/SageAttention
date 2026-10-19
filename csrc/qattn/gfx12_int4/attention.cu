// SPDX-License-Identifier: MIT
// INT4 QK^T + FP8 PV attention for gfx12 (gfx1200 / gfx1201), D128, non-causal: torch binding.
// The kernel is based on sage_int4 0.1.4, with optional lossy FP8 probabilities; inputs come from
// sageattention/triton/quant_gfx12_int4.py (prepare_int4 / prepare_int4_norm_rope).
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>

#include <string>

#include "attn_int4_qk_opt.cuh"

torch::Tensor attention(torch::Tensor q, torch::Tensor q_scale, torch::Tensor k, torch::Tensor k_scale,
                        torch::Tensor v, torch::Tensor v_scale, torch::Tensor out, int64_t valid_seq_len,
                        bool approximate) {
  TORCH_CHECK(q.is_cuda(), "Expected query on a ROCm GPU");
  const c10::DeviceGuard device_guard(q.device());
  const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  TORCH_CHECK(arch.rfind("gfx1200", 0) == 0 || arch.rfind("gfx1201", 0) == 0, "INT4 attention requires gfx1200/gfx1201");
  for (const auto& t : {q_scale, k, k_scale, v, v_scale, out})
    TORCH_CHECK(t.device() == q.device(), "All tensors must share the query device");
  TORCH_CHECK(q.dim() == 4 && q.scalar_type() == torch::kInt8 && q.is_contiguous() && q.size(3) == HEAD_DIM / 2,
              "Expected contiguous packed int4 [B, S_pad, H, 64] query");
  const int64_t batch = q.size(0), s_pad = q.size(1), heads = q.size(2);
  TORCH_CHECK(batch > 0 && heads > 0 && s_pad > 0 && s_pad % 32 == 0 && s_pad <= (int64_t{1} << 24),
              "Invalid padded sequence length");
  TORCH_CHECK(valid_seq_len > 0 && valid_seq_len <= s_pad, "Invalid valid sequence length");
  TORCH_CHECK(k.sizes() == q.sizes() && k.scalar_type() == torch::kInt8 && k.is_contiguous(),
              "Expected key shaped like the query");
  TORCH_CHECK((v.scalar_type() == torch::kFloat8_e4m3fn || v.scalar_type() == torch::kByte) && v.is_contiguous() &&
              v.dim() == 4 && v.size(0) == batch && v.size(1) == heads && v.size(2) == HEAD_DIM && v.size(3) == s_pad,
              "Expected contiguous fp8 [B, H, 128, S_pad] value");
  TORCH_CHECK(q_scale.scalar_type() == torch::kFloat32 && q_scale.is_contiguous() &&
              q_scale.numel() == batch * heads * s_pad, "Expected fp32 [B, H, S_pad] query scale");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat32 && k_scale.is_contiguous() &&
              k_scale.numel() == batch * heads * (s_pad / 32), "Expected fp32 [B, H, S_pad / 32] key scale");
  TORCH_CHECK(v_scale.scalar_type() == torch::kFloat32 && v_scale.is_contiguous() &&
              v_scale.numel() == batch * heads * HEAD_DIM, "Expected fp32 [B, H, 128] value scale");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous() && out.dim() == 4 && out.size(0) == batch &&
              out.size(1) == s_pad && out.size(2) == heads && out.size(3) == HEAD_DIM,
              "Expected contiguous bf16 [B, S_pad, H, 128] output");
  const unsigned grid = static_cast<unsigned>(batch * ((valid_seq_len + BLOCK_M - 1) / BLOCK_M) * heads);
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto mode) {
    sage_hip_attn_i4_bn32<decltype(mode)::value><<<grid, BLOCK_SIZE, 0, stream>>>(
        q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), reinterpret_cast<const int8_t*>(v.data_ptr()),
        q_scale.data_ptr<float>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
        static_cast<int>(batch), static_cast<int>(s_pad), static_cast<int>(valid_seq_len), static_cast<int>(heads));
  };
  if (approximate) launch(std::true_type{});
  else launch(std::false_type{});
  TORCH_CHECK(hipGetLastError() == hipSuccess, "sage_hip_attn_i4_bn32 launch failed");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("attention", &attention, pybind11::arg("q"), pybind11::arg("q_scale"),
             pybind11::arg("k"), pybind11::arg("k_scale"), pybind11::arg("v"), pybind11::arg("v_scale"),
             pybind11::arg("out"), pybind11::arg("valid_seq_len"), pybind11::arg("approximate") = false);
}

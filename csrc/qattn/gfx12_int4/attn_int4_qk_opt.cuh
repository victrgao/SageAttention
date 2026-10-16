// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SageAttention core for gfx12 (RDNA4, tested on gfx1201), INT4 QK^T + FP8 PV, head_dim 128, non-causal.
//   QK^T: V_WMMA_I32_16X16X32_IU4 (same ~8 cycles as the int8 16x16x16 WMMA, twice the K per instruction)
//   PV:   V_WMMA_F32_16X16X16_FP8_FP8
// Inputs (produced by sage_int4.quant.prepare_int4):
//   Q      int4 packed [B, S_pad, H, 64 B], QScale [B, H, S_pad]: one scale per row, sm_scale*log2(e) folded in
//   K      int4 packed [B, S_pad, H, 64 B], KScale [B, H, S_pad/32]: one scale per 32-key tile
//   V      fp8 e4m3 [B, H, 128, S_pad], VScale [B, H, 128]
//   O      bf16 [B, S_pad, H, 128] (rows >= valid_seq_len are not written)
// Launch: grid = B * ceil(S / BLOCK_M) * H, block = BLOCK_M threads (BLOCK_M / 32 waves, 32 query rows each).
// LDS:    two 8 KB tile buffers (K then V); per-lane read bases flip by one XOR per tile, so reads use immediates.
// Build:  hipcc --offload-arch=gfx1201 -O3 --cuda-device-only --no-gpu-bundle-output -c
//         -Xclang -target-feature -Xclang -real-true16   (true16 off: the inline fp8 pack writes full dwords)
//         -mllvm -amdgpu-sched-strategy=max-ilp
// Full tiles run the softmax core (scale, exp2, row sum, fp8 pack) as hand-scheduled asm with pinned VGPRs
// and s_delay_alu hints (tools/gen_softmax_asm.py --auto-delay); the arithmetic is the C path's, op for op,
// so the output is bit-identical.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

#include <type_traits>

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x8_t = int __attribute__((ext_vector_type(8)));
using float32x8_t = float __attribute__((ext_vector_type(8)));

#ifndef SAGE_BLOCK_M
#define SAGE_BLOCK_M 384
#endif
#ifndef SAGE_WAVES_PER_EU
#define SAGE_WAVES_PER_EU 6
#endif

static constexpr int WARP = 32;
static constexpr int WMMA_N = 16;
static constexpr int PV_WMMA_K = 16;
static constexpr int PV_LANE_K = 8;
static constexpr int QK_WMMA_K = 32;                          // int4 elements per WMMA k-step
static constexpr int BLOCK_M = SAGE_BLOCK_M;
static constexpr int BLOCK_N = 32;
static constexpr int HEAD_DIM = 128;
static constexpr int HEAD_BYTES = HEAD_DIM / 2;               // packed int4 row
static constexpr int ROWS_PER_WAVE = 32;
static constexpr int ROW_GROUPS = ROWS_PER_WAVE / 16;         // 2
static constexpr int NUM_WAVES = BLOCK_M / ROWS_PER_WAVE;
static constexpr int BLOCK_SIZE = NUM_WAVES * WARP;
static constexpr int K_STEPS_QK = HEAD_DIM / QK_WMMA_K;       // 4
static constexpr int QK_LANE_BYTES = 8;                       // 16 int4 per lane per k-step
static constexpr int VEC_WIDTH = 16;
static constexpr int K_THREADS_PER_ROW = HEAD_BYTES / VEC_WIDTH;
static constexpr int K_LOAD_ITEMS = BLOCK_N * K_THREADS_PER_ROW;
static constexpr int K_LOAD_BATCHES = (K_LOAD_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
static constexpr int V_CHUNKS_PER_ROW = BLOCK_N / 16;
static constexpr int V_LOAD_ITEMS = HEAD_DIM * V_CHUNKS_PER_ROW;
static constexpr int V_LOAD_BATCHES = (V_LOAD_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
static constexpr int K_GROUP = BLOCK_N;                       // one K scale per tile
static constexpr int NUM_S_ACCS = 2;                          // two 16-key halves per tile
static constexpr int NUM_S_VALS = NUM_S_ACCS * 8;
static constexpr int D_CHUNKS = HEAD_DIM / WMMA_N;
static constexpr int PV_K_STEPS = BLOCK_N / PV_WMMA_K;
static constexpr int LDS_PAD = 8;
static constexpr int K_STRIDE = HEAD_BYTES + LDS_PAD;
static constexpr int V_STRIDE = BLOCK_N + LDS_PAD;
static constexpr int LDS_K_TILE = BLOCK_N * K_STRIDE;
static constexpr int LDS_V_TILE = HEAD_DIM * V_STRIDE;
static constexpr int LDS_BUF_BYTES = 8192;                    // buffer b at b * 8 KB: bit 13 selects it
static_assert(LDS_K_TILE + LDS_V_TILE <= LDS_BUF_BYTES, "a K+V tile must fit in one buffer");
static constexpr float FP8_P_OFFSET = 8.807f;                 // log2(448): P lands in (0, 448]
static constexpr int MAGIC_BITS = 0x4b400000;                 // 1.5 * 2^23: acc bits read as MAGIC + dot
static constexpr float MAGIC = 12582912.0f;

__device__ __forceinline__ int32x2_t load8(const int8_t* p) {
  const int64_t v = *reinterpret_cast<const int64_t*>(p);
  return (int32x2_t){(int)(uint32_t)v, (int)(uint32_t)(v >> 32)};
}

__device__ __forceinline__ int32x8_t wmma_qk_i4(int32x2_t a, int32x2_t b, int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c, false);
}

__device__ __forceinline__ float32x8_t wmma_pv_fp8(int32x2_t a, int32x2_t b, float32x8_t c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

__device__ __forceinline__ int32x2_t pack_fp8_prob(const float p[8]) {
  // Each pair fully overwrites its dword, so no zero seed is needed.
  int32_t lo, hi;
  asm("v_cvt_pk_fp8_f32 %0, %1, %2\n\tv_cvt_pk_fp8_f32 %0, %3, %4 op_sel:[0,0,1]"
      : "=&v"(lo) : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]));
  asm("v_cvt_pk_fp8_f32 %0, %1, %2\n\tv_cvt_pk_fp8_f32 %0, %3, %4 op_sel:[0,0,1]"
      : "=&v"(hi) : "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7]));
  return (int32x2_t){lo, hi};
}

__device__ __forceinline__ float shfl_xor16(float v) { return __shfl_xor(v, 16, WARP); }
// Generated by tools/gen_softmax_asm.py; do not edit.
__device__ __forceinline__ void softmax_asm(int32x8_t (&s)[2][2], const float (&sc)[2], const float (&m)[2],
                                            int32x2_t (&p_frags)[2][2], float (&sum)[2]) {
  int pf[2][2][2];
  asm volatile("v_dual_add_f32 v100, 0xcb400000, v100 :: v_dual_add_f32 v117, 0xcb400000, v117\n\tv_dual_add_f32 v101, 0xcb400000, v101 :: v_dual_add_f32 v118, 0xcb400000, v118\n\ts_delay_alu instid0(VALU_DEP_2)\n\tv_fma_f32 v100, v100, %[sc0], -%[m0]\n\ts_delay_alu instid0(VALU_DEP_3)\n\tv_fma_f32 v117, v117, %[sc1], -%[m1]\n\ts_delay_alu instid0(VALU_DEP_2)\n\tv_exp_f32 v100, v100\n\tv_dual_add_f32 v102, 0xcb400000, v102 :: v_dual_add_f32 v119, 0xcb400000, v119\n\tv_fma_f32 v101, v101, %[sc0], -%[m0]\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v117, v117\n\tv_fma_f32 v118, v118, %[sc1], -%[m1]\n\ts_delay_alu instid0(VALU_DEP_3)\n\tv_exp_f32 v101, v101\n\tv_dual_add_f32 v103, 0xcb400000, v103 :: v_dual_add_f32 v120, 0xcb400000, v120\n\tv_fma_f32 v102, v102, %[sc0], -%[m0]\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v118, v118\n\tv_fma_f32 v119, v119, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_mov_b32 v140, v100 :: v_dual_mov_b32 v141, v117\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v102, v102\n\tv_dual_add_f32 v104, 0xcb400000, v104 :: v_dual_add_f32 v121, 0xcb400000, v121\n\tv_fma_f32 v103, v103, %[sc0], -%[m0]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p000], v100, v101\n\tv_exp_f32 v119, v119\n\tv_fma_f32 v120, v120, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v101, v140 :: v_dual_add_f32 v141, v118, v141\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p100], v117, v118\n\tv_exp_f32 v103, v103\n\tv_dual_add_f32 v105, 0xcb400000, v105 :: v_dual_add_f32 v122, 0xcb400000, v122\n\tv_fma_f32 v104, v104, %[sc0], -%[m0]\n\tv_exp_f32 v120, v120\n\tv_fma_f32 v121, v121, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v102, v140 :: v_dual_add_f32 v141, v119, v141\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v104, v104\n\tv_dual_add_f32 v106, 0xcb400000, v106 :: v_dual_add_f32 v123, 0xcb400000, v123\n\tv_fma_f32 v105, v105, %[sc0], -%[m0]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p000], v102, v103 op_sel:[0,0,1]\n\tv_exp_f32 v121, v121\n\tv_fma_f32 v122, v122, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v103, v140 :: v_dual_add_f32 v141, v120, v141\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p100], v119, v120 op_sel:[0,0,1]\n\tv_exp_f32 v105, v105\n\tv_dual_add_f32 v107, 0xcb400000, v107 :: v_dual_add_f32 v124, 0xcb400000, v124\n\tv_fma_f32 v106, v106, %[sc0], -%[m0]\n\tv_exp_f32 v122, v122\n\tv_fma_f32 v123, v123, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v104, v140 :: v_dual_add_f32 v141, v121, v141\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v106, v106\n\tv_dual_add_f32 v108, 0xcb400000, v108 :: v_dual_add_f32 v125, 0xcb400000, v125\n\tv_fma_f32 v107, v107, %[sc0], -%[m0]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p001], v104, v105\n\tv_exp_f32 v123, v123\n\tv_fma_f32 v124, v124, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v105, v140 :: v_dual_add_f32 v141, v122, v141\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p101], v121, v122\n\tv_exp_f32 v107, v107\n\tv_dual_add_f32 v109, 0xcb400000, v109 :: v_dual_add_f32 v126, 0xcb400000, v126\n\tv_fma_f32 v108, v108, %[sc0], -%[m0]\n\tv_exp_f32 v124, v124\n\tv_fma_f32 v125, v125, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v106, v140 :: v_dual_add_f32 v141, v123, v141\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v108, v108\n\tv_dual_add_f32 v110, 0xcb400000, v110 :: v_dual_add_f32 v127, 0xcb400000, v127\n\tv_fma_f32 v109, v109, %[sc0], -%[m0]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p001], v106, v107 op_sel:[0,0,1]\n\tv_exp_f32 v125, v125\n\tv_fma_f32 v126, v126, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v107, v140 :: v_dual_add_f32 v141, v124, v141\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p101], v123, v124 op_sel:[0,0,1]\n\tv_exp_f32 v109, v109\n\tv_dual_add_f32 v111, 0xcb400000, v111 :: v_dual_add_f32 v128, 0xcb400000, v128\n\tv_fma_f32 v110, v110, %[sc0], -%[m0]\n\tv_exp_f32 v126, v126\n\tv_fma_f32 v127, v127, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v108, v140 :: v_dual_add_f32 v141, v125, v141\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v110, v110\n\tv_dual_add_f32 v112, 0xcb400000, v112 :: v_dual_add_f32 v129, 0xcb400000, v129\n\tv_fma_f32 v111, v111, %[sc0], -%[m0]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p010], v108, v109\n\tv_exp_f32 v127, v127\n\tv_fma_f32 v128, v128, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v109, v140 :: v_dual_add_f32 v141, v126, v141\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p110], v125, v126\n\tv_exp_f32 v111, v111\n\tv_dual_add_f32 v113, 0xcb400000, v113 :: v_dual_add_f32 v130, 0xcb400000, v130\n\tv_fma_f32 v112, v112, %[sc0], -%[m0]\n\tv_exp_f32 v128, v128\n\tv_fma_f32 v129, v129, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v110, v140 :: v_dual_add_f32 v141, v127, v141\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v112, v112\n\tv_dual_add_f32 v114, 0xcb400000, v114 :: v_dual_add_f32 v131, 0xcb400000, v131\n\tv_fma_f32 v113, v113, %[sc0], -%[m0]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p010], v110, v111 op_sel:[0,0,1]\n\tv_exp_f32 v129, v129\n\tv_fma_f32 v130, v130, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v111, v140 :: v_dual_add_f32 v141, v128, v141\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p110], v127, v128 op_sel:[0,0,1]\n\tv_exp_f32 v113, v113\n\tv_dual_add_f32 v115, 0xcb400000, v115 :: v_dual_add_f32 v132, 0xcb400000, v132\n\tv_fma_f32 v114, v114, %[sc0], -%[m0]\n\tv_exp_f32 v130, v130\n\tv_fma_f32 v131, v131, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v112, v140 :: v_dual_add_f32 v141, v129, v141\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v114, v114\n\tv_fma_f32 v115, v115, %[sc0], -%[m0]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p011], v112, v113\n\tv_exp_f32 v131, v131\n\tv_fma_f32 v132, v132, %[sc1], -%[m1]\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_dual_add_f32 v140, v113, v140 :: v_dual_add_f32 v141, v130, v141\n\ts_delay_alu instid0(TRANS32_DEP_3)\n\tv_cvt_pk_fp8_f32 %[p111], v129, v130\n\tv_exp_f32 v115, v115\n\ts_delay_alu instid0(VALU_DEP_4)\n\tv_exp_f32 v132, v132\n\ts_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_4)\n\tv_dual_add_f32 v140, v114, v140 :: v_dual_add_f32 v141, v131, v141\n\ts_delay_alu instid0(TRANS32_DEP_2)\n\tv_cvt_pk_fp8_f32 %[p011], v114, v115 op_sel:[0,0,1]\n\ts_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)\n\tv_dual_add_f32 v140, v115, v140 :: v_dual_add_f32 v141, v132, v141\n\ts_delay_alu instid0(TRANS32_DEP_1)\n\tv_cvt_pk_fp8_f32 %[p111], v131, v132 op_sel:[0,0,1]"
      : "+{v[100:107]}"(s[0][0]), "+{v[108:115]}"(s[0][1]), "+{v[117:124]}"(s[1][0]), "+{v[125:132]}"(s[1][1]),
        "=&{v140}"(sum[0]), "=&{v141}"(sum[1]), [p000] "=&v"(pf[0][0][0]), [p001] "=&v"(pf[0][0][1]), [p010] "=&v"(pf[0][1][0]), [p011] "=&v"(pf[0][1][1]), [p100] "=&v"(pf[1][0][0]), [p101] "=&v"(pf[1][0][1]), [p110] "=&v"(pf[1][1][0]), [p111] "=&v"(pf[1][1][1])
      : [sc0] "v"(sc[0]), [sc1] "v"(sc[1]), [m0] "v"(m[0]), [m1] "v"(m[1]));
  #pragma unroll
  for (int rg = 0; rg < 2; ++rg)
    #pragma unroll
    for (int pks = 0; pks < 2; ++pks) p_frags[rg][pks] = (int32x2_t){pf[rg][pks][0], pf[rg][pks][1]};
}

// Only LDS is shared inside the workgroup, so the barrier needs no global cache invalidate.
__device__ __forceinline__ void lds_release() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); }
__device__ __forceinline__ void lds_acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); }
__device__ __forceinline__ void barrier_arrive() { lds_release(); __builtin_amdgcn_s_barrier_signal(-1); }
__device__ __forceinline__ void barrier_wait() { __builtin_amdgcn_s_barrier_wait(-1); lds_acquire(); }

extern "C" __global__
__launch_bounds__(BLOCK_SIZE, SAGE_WAVES_PER_EU)
__attribute__((amdgpu_waves_per_eu(SAGE_WAVES_PER_EU)))
void sage_hip_attn_i4_bn32(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    const float* __restrict__ QScale,
    const float* __restrict__ KScale,
    const float* __restrict__ VScale,
    __hip_bfloat16* __restrict__ O,
    int batch_size,
    int padded_seq_len,
    int valid_seq_len,
    int num_heads
) {
  __shared__ int8_t lds[LDS_BUF_BYTES + LDS_K_TILE + LDS_V_TILE];
  (void)batch_size;

  const int tid = (int)threadIdx.x;
  const int wave_id = tid / WARP;
  const int lane = tid % WARP;
  const int lane16 = lane % 16;
  const int klane = lane / 16;

  // Head-major workgroup order: adjacent workgroups share a head (and its K/V in L2).
  const int q_tiles = (valid_seq_len + BLOCK_M - 1) / BLOCK_M;
  const unsigned per_batch = (unsigned)(q_tiles * num_heads);
  const unsigned rem = blockIdx.x % per_batch;
  const int block_id = (int)((blockIdx.x / per_batch) * per_batch + (rem % q_tiles) * num_heads + rem / q_tiles);
  const int head_idx = block_id % num_heads;
  const int batch_q_tile = block_id / num_heads;
  const int q_tile_idx = batch_q_tile % q_tiles;
  const int batch_idx = batch_q_tile / q_tiles;
  const int q_start = q_tile_idx * BLOCK_M;
  const int seq = padded_seq_len;
  const long long bh = (long long)batch_idx * num_heads + head_idx;
  const long long tok_stride = (long long)num_heads * HEAD_BYTES;

  int q_row[ROW_GROUPS];
  bool q_ok[ROW_GROUPS];
  int32x2_t q_frags[ROW_GROUPS][K_STEPS_QK];
  float q_scale[ROW_GROUPS];
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    q_row[rg] = q_start + wave_id * ROWS_PER_WAVE + rg * 16 + lane16;
    q_ok[rg] = q_row[rg] < valid_seq_len;
    const int q_safe = q_ok[rg] ? q_row[rg] : 0;
    const int8_t* q_ptr = Q + ((long long)batch_idx * seq + q_safe) * tok_stride + head_idx * HEAD_BYTES;
    #pragma unroll
    for (int ks = 0; ks < K_STEPS_QK; ++ks)
      q_frags[rg][ks] = q_ok[rg] ? load8(q_ptr + ks * (QK_WMMA_K / 2) + klane * QK_LANE_BYTES) : (int32x2_t){0, 0};
    q_scale[rg] = QScale[bh * seq + q_safe];
  }

  float m_running[ROW_GROUPS], l_running[ROW_GROUPS];
  float32x8_t o_acc[ROW_GROUPS][D_CHUNKS];
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    m_running[rg] = -INFINITY;
    l_running[rg] = 0.f;
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) o_acc[rg][dc] = (float32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
  }

  // Tile-dependent address parts stay uniform (SALU); per-thread parts are fixed 32-bit offsets.
  const int8_t* k_head = K + (long long)batch_idx * seq * tok_stride + head_idx * HEAD_BYTES;
  const int8_t* v_head = V + bh * HEAD_DIM * seq;
  const float* ks_head = KScale + bh * (seq / K_GROUP);
  auto uni = [](int x) -> int { return __builtin_amdgcn_readfirstlane(x); };
  // Read bases of the current tile's buffer; every K/V read below is base + immediate (<= 1936 B).
  uint32_t k_rd = lane16 * K_STRIDE + klane * QK_LANE_BYTES;
  uint32_t v_rd_lo = LDS_K_TILE + lane16 * V_STRIDE + klane * PV_LANE_K;
  uint32_t v_rd_hi = v_rd_lo + (D_CHUNKS / 2) * WMMA_N * V_STRIDE;

  static_assert(K_LOAD_ITEMS + V_LOAD_ITEMS == BLOCK_SIZE && K_LOAD_ITEMS % WARP == 0, "one 16 B item per thread");
  // One 16 B item per thread: waves [0, 4) move K rows, the others V^T rows; the K/V choice is wave-uniform.
  const bool mv_k = uni(wave_id) < K_LOAD_ITEMS / WARP;
  const int vi = tid - K_LOAD_ITEMS;
  const uint32_t mv_gl = tid < K_LOAD_ITEMS
      ? (uint32_t)((tid / K_THREADS_PER_ROW) * num_heads * HEAD_BYTES + (tid % K_THREADS_PER_ROW) * VEC_WIDTH)
      : (uint32_t)((vi / V_CHUNKS_PER_ROW) * seq + (vi % V_CHUNKS_PER_ROW) * VEC_WIDTH);
  const uint32_t mv_st = tid < K_LOAD_ITEMS
      ? (uint32_t)((tid / K_THREADS_PER_ROW) * K_STRIDE + (tid % K_THREADS_PER_ROW) * VEC_WIDTH)
      : (uint32_t)(LDS_K_TILE + (vi / V_CHUNKS_PER_ROW) * V_STRIDE + (vi % V_CHUNKS_PER_ROW) * VEC_WIDTH);
  int4 mv_pref;
  auto prefetch = [&](int tile) {
    const int8_t* src = mv_k ? k_head + tile * tok_stride : v_head + tile;
    mv_pref = *reinterpret_cast<const int4*>(src + mv_gl);
  };
  auto commit = [&](int buf) {
    // K / V rows are only 8 B aligned in LDS: two 8 B stores.
    int8_t* dst = lds + uni(buf * LDS_BUF_BYTES) + mv_st;
    reinterpret_cast<int2*>(dst)[0] = make_int2(mv_pref.x, mv_pref.y);
    reinterpret_cast<int2*>(dst)[1] = make_int2(mv_pref.z, mv_pref.w);
  };
  // K/V are padded to a multiple of BLOCK_N, so any tile < seq is in bounds.
  auto safe_tile = [&](int tile) -> int { return tile < seq ? tile : 0; };

  prefetch(0);
  commit(0);
  prefetch(safe_tile(BLOCK_N));
  barrier_arrive();

  auto process_tile = [&](int kv_start, auto tail_tag) {
    constexpr bool MASK_TAIL = decltype(tail_tag)::value;
    const int cur = (kv_start / BLOCK_N) & 1;
    // The tail's wait is issued after the loop so every signal is paired.
    if constexpr (!MASK_TAIL) barrier_wait();

    int32x8_t s_acc[ROW_GROUPS][NUM_S_ACCS];
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg)
      #pragma unroll
      for (int h = 0; h < NUM_S_ACCS; ++h)
        s_acc[rg][h] = (int32x8_t){MAGIC_BITS, MAGIC_BITS, MAGIC_BITS, MAGIC_BITS,
                                   MAGIC_BITS, MAGIC_BITS, MAGIC_BITS, MAGIC_BITS};
    // A = K rows (keys), B = Q rows: lane holds query lane16, keys h*16 + klane*8 + [0, 8).
    // Q and K use the same per-lane packing, so the dot product is independent of the nibble order.
    // All K reads go out before the first WMMA, ahead of this tile's LDS stores.
    int32x2_t kav[K_STEPS_QK], kbv[K_STEPS_QK];
    #pragma unroll
    for (int ks = 0; ks < K_STEPS_QK; ++ks) {
      kav[ks] = load8(lds + k_rd + ks * (QK_WMMA_K / 2));
      kbv[ks] = load8(lds + k_rd + 16 * K_STRIDE + ks * (QK_WMMA_K / 2));
    }
    #pragma unroll
    for (int ks = 0; ks < K_STEPS_QK; ++ks) {
      const int32x2_t ka = kav[ks], kb = kbv[ks];
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) {
        s_acc[rg][0] = wmma_qk_i4(ka, q_frags[rg][ks], s_acc[rg][0]);
        s_acc[rg][1] = wmma_qk_i4(kb, q_frags[rg][ks], s_acc[rg][1]);
      }
    }
    // Publish the prefetched tile into the idle buffer (free since the barrier), then fetch two tiles ahead.
    commit(cur ^ 1);
    prefetch(safe_tile(kv_start + 2 * BLOCK_N));

    const float k_scale = ks_head[kv_start / K_GROUP];
    float sc[ROW_GROUPS];
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) sc[rg] = q_scale[rg] * k_scale;

    if constexpr (MASK_TAIL) {
      // Zero bits sort below every MAGIC + dot, and p is forced to 0 below.
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg)
        #pragma unroll
        for (int h = 0; h < NUM_S_ACCS; ++h)
          #pragma unroll
          for (int i = 0; i < 8; ++i)
            if (kv_start + h * 16 + klane * 8 + i >= valid_seq_len) s_acc[rg][h][i] = 0;
    }

    int32x2_t p_frags[ROW_GROUPS][PV_K_STEPS];
    // Row max: integer max on the raw bits (bit order == value order for MAGIC + dot), then scale.
    float row_max[ROW_GROUPS];
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) {
      int im = s_acc[rg][0][0];
      #pragma unroll
      for (int i = 0; i < NUM_S_VALS; ++i) im = max(im, s_acc[rg][i / 8][i % 8]);
      row_max[rg] = (__int_as_float(im) - MAGIC) * sc[rg];
    }
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) row_max[rg] = fmaxf(row_max[rg], shfl_xor16(row_max[rg]));

    float m_new[ROW_GROUPS], corr[ROW_GROUPS], local_sum[ROW_GROUPS];
    bool m_moved = false;
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) {
      m_new[rg] = fmaxf(m_running[rg], row_max[rg] - FP8_P_OFFSET);
      // exp2(m_running - m_new) == 1 unless the max moved; the rescale branch computes the rest.
      corr[rg] = 1.f;
      m_moved |= m_new[rg] != m_running[rg];
      local_sum[rg] = 0.f;
    }
    // m unchanged => corr == 1 exactly, so rescaling a superset of rows is harmless.
    const bool any_rescale = __builtin_amdgcn_ballot_w32(m_moved) != 0;

    float p_vals[ROW_GROUPS][NUM_S_VALS];
    if constexpr (!MASK_TAIL) {
      softmax_asm(s_acc, sc, m_new, p_frags, local_sum);
    } else
    {
    #pragma unroll
    for (int i = 0; i < NUM_S_VALS; ++i) {
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) {
        // Same binade as MAGIC, so the subtraction is exact (folding MAGIC*scale into the bias is not).
        const float x = __int_as_float(s_acc[rg][i / 8][i % 8]) - MAGIC;
        float p = __builtin_amdgcn_exp2f(fmaf(x, sc[rg], -m_new[rg]));
        if constexpr (MASK_TAIL) {
          if (kv_start + (i / 8) * 16 + klane * 8 + i % 8 >= valid_seq_len) p = 0.f;
        }
        p_vals[rg][i] = p;
        local_sum[rg] += p;
      }
    }
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg)
      #pragma unroll
      for (int pks = 0; pks < PV_K_STEPS; ++pks) p_frags[rg][pks] = pack_fp8_prob(&p_vals[rg][pks * 8]);
    }
    if (any_rescale) {
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) corr[rg] = __builtin_amdgcn_exp2f(m_running[rg] - m_new[rg]);
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg)
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; ++dc)
          #pragma unroll
          for (int i = 0; i < 8; ++i) o_acc[rg][dc][i] *= corr[rg];
    }
    // Explicit fma: written as corr * l + s, the add would sink past the branch and lose the fusion M3 has.
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) {
      l_running[rg] = fmaf(corr[rg], l_running[rg], local_sum[rg] + shfl_xor16(local_sum[rg]));
      m_running[rg] = m_new[rg];
    }

    auto load_v = [&](int pks, int dc) -> int32x2_t {
      constexpr int HALF = D_CHUNKS / 2;
      return load8(lds + (dc < HALF ? v_rd_lo : v_rd_hi) + (dc % HALF) * WMMA_N * V_STRIDE + pks * PV_WMMA_K);
    };
    // Issue every V read up front so their latency hides behind the first PV half.
    int32x2_t pv_first[D_CHUNKS], pv_last[D_CHUNKS];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) pv_first[dc] = load_v(0, dc);
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) pv_last[dc] = load_v(PV_K_STEPS - 1, dc);
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg)
        o_acc[rg][dc] = wmma_pv_fp8(pv_first[dc], p_frags[rg][0], o_acc[rg][dc]);
    // All LDS reads of this tile are issued: arrive now, wait at the next tile's start.
    if constexpr (!MASK_TAIL) barrier_arrive();
    // Keep the last PV WMMAs after the signal; otherwise the scheduler sinks the signal past them.
    __builtin_amdgcn_sched_barrier(0);
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg)
        o_acc[rg][dc] = wmma_pv_fp8(pv_last[dc], p_frags[rg][PV_K_STEPS - 1], o_acc[rg][dc]);
    k_rd ^= LDS_BUF_BYTES;
    v_rd_lo ^= LDS_BUF_BYTES;
    v_rd_hi ^= LDS_BUF_BYTES;
  };

  const int full_end = (valid_seq_len / BLOCK_N) * BLOCK_N;
  for (int tile_start = 0; tile_start < full_end; tile_start += BLOCK_N)
    process_tile(tile_start, std::false_type{});
  barrier_wait();
  if (full_end < valid_seq_len) process_tile(full_end, std::true_type{});

  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    if (!q_ok[rg]) continue;
    const float inv_l = 1.f / l_running[rg];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) {
      const int d_col = dc * WMMA_N + klane * 8;
      float vs[8];
      *reinterpret_cast<float4*>(vs) = *reinterpret_cast<const float4*>(VScale + bh * HEAD_DIM + d_col);
      *reinterpret_cast<float4*>(vs + 4) = *reinterpret_cast<const float4*>(VScale + bh * HEAD_DIM + d_col + 4);
      __hip_bfloat16 outv[8];
      #pragma unroll
      for (int i = 0; i < 8; ++i) outv[i] = __float2bfloat16(o_acc[rg][dc][i] * inv_l * vs[i]);
      *reinterpret_cast<int4*>(O + (((long long)batch_idx * seq + q_row[rg]) * num_heads + head_idx) * HEAD_DIM + d_col) =
          *reinterpret_cast<int4*>(outv);
    }
  }
}

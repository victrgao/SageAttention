// SPDX-License-Identifier: MIT
// SageAttention core for gfx12 (RDNA4): INT8 QK^T (V_WMMA_I32_16X16X16_IU8) + FP8 PV, D128, non-causal,
// exact exp2 probabilities.
// Data layout and arithmetic = the approx_lsum_latescale snapshot kernel built with APPROX_FP8_P=0 (bitwise target):
//   Q   int8 [B, S, H, 128], QScale [B, H, ceil(S/32)]   (sm_scale * log2(e) applied here)
//   K   int8 [B, H, S_pad64, 128], lane-major 32-key tiles, KScale [B, H, S_pad64/64]
//   V   fp8 e4m3 [B, H, 128, >= S_pad64], lane-major 32-key tiles, VScale [B, H, 128]
//   O   bf16 [B, >= S, H, 128] (rows >= S are not written)
// Pipeline = sage_int4 0.1.2 (M5): XOR-flipped LDS read base, branchless K/V mover, LDS commit + prefetch after QK,
// K reads ahead of the first QK WMMA, hand-scheduled asm softmax with s_delay_alu on full tiles, max-ilp scheduling.
// int8-specific: cross-half row max on integer bits, K scale loaded one tile ahead, one merged max-growth branch.
// From dualprob: wave-uniform (SGPR) K/V tile bases with unsigned 32-bit lane offsets for the global prefetch.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

#ifndef SAGE_BLOCK_M
#define SAGE_BLOCK_M 256
#endif
#ifndef SAGE_WAVES_PER_EU
#define SAGE_WAVES_PER_EU 6
#endif
// K-steps whose K fragments are read before the first QK WMMA; deeper preloads spill at 6 waves/SIMD.
#ifndef SAGE_K_PRELOAD
#define SAGE_K_PRELOAD 4
#endif
#ifndef SAGE_ASM_SOFTMAX
#define SAGE_ASM_SOFTMAX 1
#endif
#ifndef SAGE_V_PRELOAD_ALL
#define SAGE_V_PRELOAD_ALL 1
#endif
// Cross-half row max on the MAGIC bits (no float canonicalization); exact since the score scale is > 0.
#ifndef SAGE_INT_XMAX
#define SAGE_INT_XMAX 1
#endif
// Load the next tile's K scale one tile ahead instead of waiting on it mid-tile.
#ifndef SAGE_KSCALE_AHEAD
#define SAGE_KSCALE_AHEAD 1
#endif
// Keep the MAGIC seed tuple live across the loop instead of rematerializing it every tile.
#ifndef SAGE_MAGIC_PERSIST
#define SAGE_MAGIC_PERSIST 0
#endif
// Row max as hand-scheduled asm: four interleaved max3 chains instead of one serial chain per row group.
#ifndef SAGE_ASM_ROWMAX
#define SAGE_ASM_ROWMAX 0
#endif
// Timing ablation only (wrong results): skip the row max and rescale after the first tile.
#ifndef SAGE_ABL_FROZEN_M
#define SAGE_ABL_FROZEN_M 0
#endif
// One uniform branch for both row groups, tested on half-wave maxima (cross-half max only when a row grows).
#ifndef SAGE_GUARD_MERGE
#define SAGE_GUARD_MERGE 1
#endif
// dualprob SCALAR_KV_BASE + UNSIGNED_KV_ADDRESS: K/V prefetch as global_load v_offset, s[base] (no per-lane 64-bit math).
#ifndef SAGE_SCALAR_KV_BASE
#define SAGE_SCALAR_KV_BASE 1
#endif

// Every product / sum below is written out explicitly to match the baseline's rounding sequence.
#pragma clang fp contract(off)

namespace sage_port {

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x4_t = int __attribute__((ext_vector_type(4)));
using int32x8_t = int __attribute__((ext_vector_type(8)));
using float32x8_t = float __attribute__((ext_vector_type(8)));

constexpr int WARP = 32;
constexpr int BLOCK_M = SAGE_BLOCK_M;
constexpr int BLOCK_N = 32;
constexpr int HEAD_DIM = 128;
constexpr int QUERY_GRID_GROUP = 64;
constexpr int ROWS_PER_WAVE = 32;
constexpr int ROW_GROUPS = 2;
constexpr int NUM_WAVES = BLOCK_M / ROWS_PER_WAVE;
constexpr int BLOCK_SIZE = NUM_WAVES * WARP;
constexpr int K_STEPS = HEAD_DIM / 16;
constexpr int D_CHUNKS = HEAD_DIM / 16;
constexpr int PV_K_STEPS = BLOCK_N / 16;
constexpr int FRAG_BYTES = WARP * 8;                  // one lane-major WMMA operand fragment
constexpr int TILE_BYTES = BLOCK_N * HEAD_DIM;        // one K or V tile
constexpr int LDS_BUF_BYTES = 2 * TILE_BYTES;         // K then V; buffer b at b * 8 KB, bit 13 selects it
constexpr int TILE_ITEMS = TILE_BYTES / 16;
constexpr int MOVES_PER_THREAD = 2 * TILE_ITEMS / BLOCK_SIZE;
constexpr int K_SCALE_GROUP = 64;
constexpr int Q_SCALE_GROUP = 32;
constexpr int K_PRELOAD = SAGE_K_PRELOAD;
constexpr float FP8_P_OFFSET = 8.807f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int MAGIC_BITS = 0x4b400000;                // 1.5 * 2^23: acc bits read as MAGIC + dot (|dot| <= 2^21)
constexpr float MAGIC = 12582912.0f;
static_assert(LDS_BUF_BYTES == 8192, "XOR flip assumes 8 KB buffers");
static_assert(MOVES_PER_THREAD == 1 || MOVES_PER_THREAD == 2, "BLOCK_M must be 256 or 512");
static_assert(K_PRELOAD >= 0 && K_PRELOAD <= K_STEPS, "invalid K preload depth");

__device__ __forceinline__ int32x2_t load8(const int8_t* p) {
  const int64_t v = *reinterpret_cast<const int64_t*>(p);
  return (int32x2_t){(int)(uint32_t)v, (int)(uint32_t)(v >> 32)};
}

__device__ __forceinline__ int32x4_t load16_uniform_base(const int8_t* base, uint32_t byte_offset) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(base);
  const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address));
  const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address >> 32));
  using GlobalVec = const int32x4_t __attribute__((address_space(1)))*;
  return reinterpret_cast<GlobalVec>((static_cast<uintptr_t>(hi) << 32) | lo)[byte_offset / 16];
}

__device__ __forceinline__ int32x8_t wmma_qk_i8(int32x2_t a, int32x2_t b, int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, true);
}

__device__ __forceinline__ float32x8_t wmma_pv_fp8(int32x2_t a, int32x2_t b, float32x8_t c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

__device__ __forceinline__ int32x2_t pack_fp8_prob(const float p[8]) {
  int32_t lo, hi;
  asm("v_cvt_pk_fp8_f32 %0, %1, %2\n\tv_cvt_pk_fp8_f32 %0, %3, %4 op_sel:[0,0,1]"
      : "=&v"(lo) : "v"(p[0]), "v"(p[1]), "v"(p[2]), "v"(p[3]));
  asm("v_cvt_pk_fp8_f32 %0, %1, %2\n\tv_cvt_pk_fp8_f32 %0, %3, %4 op_sel:[0,0,1]"
      : "=&v"(hi) : "v"(p[4]), "v"(p[5]), "v"(p[6]), "v"(p[7]));
  return (int32x2_t){lo, hi};
}

__device__ __forceinline__ float swap_half(float v) {
  const int bits = __builtin_bit_cast(int, v);
  return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(bits, bits, 0x76543210u, 0xfedcba98u, true, true));
}

__device__ __forceinline__ int swap_half_i(int v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, true, true);
}

// fmaxf without the IEEE-mode input canonicalization; operands here are never NaN.
__device__ __forceinline__ float max_num(float a, float b) {
  float r;
  asm("v_max_num_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}

#include "softmax_scalar_scale.cuh"
#include "rowmax_asm.cuh"

__device__ __forceinline__ void lds_release() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); }
__device__ __forceinline__ void lds_acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); }
__device__ __forceinline__ void barrier_arrive() { lds_release(); __builtin_amdgcn_s_barrier_signal(-1); }
__device__ __forceinline__ void barrier_wait() { __builtin_amdgcn_s_barrier_wait(-1); lds_acquire(); }

__global__ __launch_bounds__(BLOCK_SIZE, SAGE_WAVES_PER_EU)
__attribute__((amdgpu_waves_per_eu(SAGE_WAVES_PER_EU)))
void attn_int8_port_kernel(
    const int8_t* __restrict__ Q, const int8_t* __restrict__ K, const int8_t* __restrict__ V,
    const float* __restrict__ QScale, const float* __restrict__ KScale, const float* __restrict__ VScale,
    __hip_bfloat16* __restrict__ O,
    int qo_len, int kv_len, int kv_pad, int num_heads,
    int64_t q_sb, int64_t q_sn, int64_t q_sh, int64_t k_sb, int64_t k_sh, int64_t v_sb, int64_t v_sh,
    int64_t qs_sb, int64_t qs_sh, int64_t ks_sb, int64_t ks_sh, int64_t o_sb, int64_t o_sn, int64_t o_sh,
    float sm_scale) {
  __shared__ __attribute__((aligned(16))) int8_t lds[2 * LDS_BUF_BYTES];

  const int tid = (int)threadIdx.x;
  const int wave = tid / WARP;
  const int lane = tid % WARP;
  const int lane16 = lane % 16;
  const int klane = lane / 16;
  const int head = (int)blockIdx.x / QUERY_GRID_GROUP;
  const int64_t batch = blockIdx.z;
  const int query_block = (int)blockIdx.y * QUERY_GRID_GROUP + (int)blockIdx.x % QUERY_GRID_GROUP;
  if (query_block * BLOCK_M >= qo_len) return;
  const int wave_q = query_block * BLOCK_M + wave * ROWS_PER_WAVE;
  auto uni = [](int x) -> int { return __builtin_amdgcn_readfirstlane(x); };

  // B operand of QK: lane holds query lane16, dims ks*16 + klane*8 + [0, 8).
  int q_row[ROW_GROUPS];
  bool q_ok[ROW_GROUPS];
  int32x2_t q_frags[ROW_GROUPS][K_STEPS];
  float qs[ROW_GROUPS];
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    const int q_start = wave_q + rg * 16;
    q_row[rg] = q_start + lane16;
    q_ok[rg] = q_row[rg] < qo_len;
    const float query_scale = q_start < qo_len ? QScale[batch * qs_sb + head * qs_sh + q_start / Q_SCALE_GROUP] : 1.0f;
    qs[rg] = query_scale * sm_scale * LOG2E;
    const int8_t* q_ptr = Q + batch * q_sb + (int64_t)(q_ok[rg] ? q_row[rg] : 0) * q_sn + head * q_sh + klane * 8;
    #pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks)
      q_frags[rg][ks] = q_ok[rg] ? load8(q_ptr + ks * 16) : (int32x2_t){0, 0};
  }

  float m_run[ROW_GROUPS], l_run[ROW_GROUPS];
  float32x8_t o_acc[ROW_GROUPS][D_CHUNKS];
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    m_run[rg] = -FLT_MAX * 0.5f;
    l_run[rg] = 0.f;
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) o_acc[rg][dc] = (float32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
  }

  const int8_t* k_head = K + batch * k_sb + head * k_sh;
  const int8_t* v_head = V + batch * v_sb + head * v_sh;
  const float* ks_head = KScale + batch * ks_sb + head * ks_sh;
  // Lane-major read base of the current tile's buffer; every K/V read is base + immediate.
  uint32_t rd = lane * 8;

  // Each K / V tile is 256 contiguous 16 B items in global memory and in LDS.
  const bool mv_k = MOVES_PER_THREAD == 2 || uni(wave) < NUM_WAVES / 2;
  const uint32_t mv_off = (uint32_t)(tid % TILE_ITEMS) * 16;
  int32x4_t mv_a, mv_b;
  auto prefetch = [&](int tile) {
    if constexpr (SAGE_SCALAR_KV_BASE) {
      // The host checks kv_pad <= 2^24, so tile * HEAD_DIM fits the unsigned 32-bit offset.
      const uint64_t tile_off = (uint64_t)((uint32_t)tile * HEAD_DIM);
      if constexpr (MOVES_PER_THREAD == 2) {
        mv_a = load16_uniform_base(k_head + tile_off, mv_off);
        mv_b = load16_uniform_base(v_head + tile_off, mv_off);
      } else {
        mv_a = load16_uniform_base((mv_k ? k_head : v_head) + tile_off, mv_off);
      }
      return;
    }
    const int64_t tile_off = (int64_t)tile * HEAD_DIM;
    if constexpr (MOVES_PER_THREAD == 2) {
      mv_a = *reinterpret_cast<const int32x4_t*>(k_head + tile_off + mv_off);
      mv_b = *reinterpret_cast<const int32x4_t*>(v_head + tile_off + mv_off);
    } else {
      const int8_t* src = mv_k ? k_head + tile_off : v_head + tile_off;
      mv_a = *reinterpret_cast<const int32x4_t*>(src + mv_off);
    }
  };
  auto commit = [&](int buf) {
    int8_t* dst = lds + uni(buf * LDS_BUF_BYTES) + mv_off;
    if constexpr (MOVES_PER_THREAD == 2) {
      *reinterpret_cast<int32x4_t*>(dst) = mv_a;
      *reinterpret_cast<int32x4_t*>(dst + TILE_BYTES) = mv_b;
    } else {
      *reinterpret_cast<int32x4_t*>(dst + (mv_k ? 0 : TILE_BYTES)) = mv_a;
    }
  };
  // K/V are padded to a multiple of 64 keys, so any tile < kv_pad is in bounds.
  auto safe_tile = [&](int tile) -> int { return tile < kv_pad ? tile : 0; };

  prefetch(0);
  commit(0);
  prefetch(safe_tile(BLOCK_N));
  barrier_arrive();

  float k_scale_next = ks_head[0];
  int32x8_t magic = (int32x8_t){MAGIC_BITS, MAGIC_BITS, MAGIC_BITS, MAGIC_BITS,
                                MAGIC_BITS, MAGIC_BITS, MAGIC_BITS, MAGIC_BITS};
  if constexpr (SAGE_MAGIC_PERSIST) asm volatile("" : "+v"(magic));

  auto process_tile = [&](int kv_start, auto tail_tag) {
    constexpr bool MASK_TAIL = decltype(tail_tag)::value;
    const int cur = (kv_start / BLOCK_N) & 1;
    // The tail's wait is issued after the loop so every signal is paired.
    if constexpr (!MASK_TAIL) barrier_wait();
    asm volatile("s_setprio 2");

    float k_scale;
    if constexpr (SAGE_KSCALE_AHEAD) {
      k_scale = k_scale_next;
      const int next = kv_start + BLOCK_N;
      k_scale_next = ks_head[(next < kv_pad ? next : 0) / K_SCALE_GROUP];
    }

    int32x8_t s_acc[ROW_GROUPS][2];
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg)
      #pragma unroll
      for (int h = 0; h < 2; ++h) s_acc[rg][h] = magic;
    // A = K: fragment (h, ks) holds keys h*16 + lane16, dims ks*16 + klane*8 + [0, 8).
    int32x2_t ka[K_STEPS], kb[K_STEPS];
    #pragma unroll
    for (int ks = 0; ks < K_PRELOAD; ++ks) {
      ka[ks] = load8(lds + rd + ks * FRAG_BYTES);
      kb[ks] = load8(lds + rd + (K_STEPS + ks) * FRAG_BYTES);
    }
    #pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks) {
      if (ks >= K_PRELOAD) {
        ka[ks] = load8(lds + rd + ks * FRAG_BYTES);
        kb[ks] = load8(lds + rd + (K_STEPS + ks) * FRAG_BYTES);
      }
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) {
        s_acc[rg][0] = wmma_qk_i8(ka[ks], q_frags[rg][ks], s_acc[rg][0]);
        s_acc[rg][1] = wmma_qk_i8(kb[ks], q_frags[rg][ks], s_acc[rg][1]);
      }
    }
    // The idle buffer has been free since the barrier; publishing after QK keeps the stores behind the K reads.
    asm volatile("s_setprio 1");
    commit(cur ^ 1);
    prefetch(safe_tile(kv_start + 2 * BLOCK_N));
    asm volatile("s_setprio 0");

    if constexpr (!SAGE_KSCALE_AHEAD) k_scale = ks_head[kv_start / K_SCALE_GROUP];
    float sc[ROW_GROUPS];
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) sc[rg] = qs[rg] * k_scale;

    if constexpr (MASK_TAIL) {
      // Zero bits sort below every MAGIC + dot, and p is forced to 0 below.
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
          #pragma unroll
          for (int i = 0; i < 8; ++i)
            if (kv_start + h * 16 + klane * 8 + i >= kv_len) s_acc[rg][h][i] = 0;
    }

    // Row max on the raw bits (bit order == value order for MAGIC + dot); bits - MAGIC == float(dot) exactly.
    float tile_max[ROW_GROUPS];
    int im[ROW_GROUPS];  // per-lane max over this lane's 16 keys
    const bool skip_max = SAGE_ABL_FROZEN_M && kv_start != 0;
    const bool has_prev = kv_start != 0;
    if (skip_max) {
    } else if constexpr (SAGE_ASM_ROWMAX) {
      rowmax_asm(s_acc, im);
    } else {
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) {
        im[rg] = s_acc[rg][0][0];
        #pragma unroll
        for (int i = 1; i < 16; ++i) im[rg] = max(im[rg], s_acc[rg][i / 8][i % 8]);
      }
    }

    if constexpr (SAGE_GUARD_MERGE) {
      // A row's tile max is the larger of its two half-wave maxima and m_run is equal in both halves, so
      // "any lane's half max > m" == "any row's tile max > m". Rows that did not grow get new_m == m and alpha == 1.
      bool grow = false;
      if (!skip_max) {
        #pragma unroll
        for (int rg = 0; rg < ROW_GROUPS; ++rg)
          grow |= __builtin_amdgcn_ballot_w32((__int_as_float(im[rg]) - MAGIC) * sc[rg] > m_run[rg]) != 0;
      }
      if (grow) {
        float alpha[ROW_GROUPS];
        #pragma unroll
        for (int rg = 0; rg < ROW_GROUPS; ++rg) {
          tile_max[rg] = (__int_as_float(max(im[rg], swap_half_i(im[rg]))) - MAGIC) * sc[rg];
          const float new_m = max_num(m_run[rg], tile_max[rg]);
          alpha[rg] = has_prev ? __builtin_amdgcn_exp2f(m_run[rg] - new_m) : 0.f;
          m_run[rg] = new_m;
          l_run[rg] *= alpha[rg];
        }
        #pragma unroll
        for (int rg = 0; rg < ROW_GROUPS; ++rg) {
          if (has_prev && __builtin_amdgcn_ballot_w32(alpha[rg] != 1.f) != 0) {
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS; ++dc)
              #pragma unroll
              for (int i = 0; i < 8; ++i) o_acc[rg][dc][i] *= alpha[rg];
          }
        }
      }
    } else {
    if (!skip_max) {
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) {
        if constexpr (SAGE_INT_XMAX) {
          tile_max[rg] = (__int_as_float(max(im[rg], swap_half_i(im[rg]))) - MAGIC) * sc[rg];
        } else {
          const float local_max = (__int_as_float(im[rg]) - MAGIC) * sc[rg];
          tile_max[rg] = fmaxf(local_max, swap_half(local_max));
        }
      }
    }

    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) {
      if (skip_max) continue;
      if (__builtin_amdgcn_ballot_w32(tile_max[rg] > m_run[rg]) != 0) {
        const float new_m = SAGE_INT_XMAX ? max_num(m_run[rg], tile_max[rg]) : fmaxf(m_run[rg], tile_max[rg]);
        const float alpha = has_prev ? __builtin_amdgcn_exp2f(m_run[rg] - new_m) : 0.f;
        m_run[rg] = new_m;
        l_run[rg] *= alpha;
        if (has_prev && __builtin_amdgcn_ballot_w32(alpha != 1.f) != 0) {
          #pragma unroll
          for (int dc = 0; dc < D_CHUNKS; ++dc)
            #pragma unroll
            for (int i = 0; i < 8; ++i) o_acc[rg][dc][i] *= alpha;
        }
      }
    }
    }

    int32x2_t p_frags[ROW_GROUPS][PV_K_STEPS];
    float local_sum[ROW_GROUPS];
    if constexpr (!MASK_TAIL && SAGE_ASM_SOFTMAX) {
      // The asm computes fma(x, sc, -m_off); -(m - 8.807) rounds exactly like the baseline's 8.807 - m.
      const float m_off[ROW_GROUPS] = {m_run[0] - FP8_P_OFFSET, m_run[1] - FP8_P_OFFSET};
      softmax_scalar_scale(s_acc, sc, m_off, p_frags, local_sum);
    } else {
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) {
        const float shift = FP8_P_OFFSET - m_run[rg];
        float p_vals[16];
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float x = __int_as_float(s_acc[rg][i / 8][i % 8]) - MAGIC;
          float p = __builtin_amdgcn_exp2f(__builtin_fmaf(x, sc[rg], shift));
          if (MASK_TAIL && kv_start + (i / 8) * 16 + klane * 8 + i % 8 >= kv_len) p = 0.f;
          p_vals[i] = p;
          sum += p;
        }
        local_sum[rg] = sum;
        #pragma unroll
        for (int pks = 0; pks < PV_K_STEPS; ++pks) p_frags[rg][pks] = pack_fp8_prob(&p_vals[pks * 8]);
      }
    }
    // Deferred denominator: each half-wave keeps its own partial sum until the epilogue.
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) l_run[rg] += local_sum[rg];

    // A = V^T: fragment (pks, dc) holds dims dc*16 + lane16, keys pks*16 + klane*8 + [0, 8).
    int32x2_t pv_first[D_CHUNKS], pv_last[D_CHUNKS];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) pv_first[dc] = load8(lds + rd + TILE_BYTES + dc * FRAG_BYTES);
    if constexpr (SAGE_V_PRELOAD_ALL) {
      #pragma unroll
      for (int dc = 0; dc < D_CHUNKS; ++dc)
        pv_last[dc] = load8(lds + rd + TILE_BYTES + (D_CHUNKS + dc) * FRAG_BYTES);
    }
    asm volatile("s_setprio 3");
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg)
        o_acc[rg][dc] = wmma_pv_fp8(pv_first[dc], p_frags[rg][0], o_acc[rg][dc]);
    if constexpr (!SAGE_V_PRELOAD_ALL) {
      #pragma unroll
      for (int dc = 0; dc < D_CHUNKS; ++dc)
        pv_last[dc] = load8(lds + rd + TILE_BYTES + (D_CHUNKS + dc) * FRAG_BYTES);
    }
    // All LDS reads of this tile are issued: arrive now, wait at the next tile's start.
    if constexpr (!MASK_TAIL) barrier_arrive();
    // Keep the last PV WMMAs after the signal; otherwise the scheduler sinks the signal past them.
    __builtin_amdgcn_sched_barrier(0);
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg)
        o_acc[rg][dc] = wmma_pv_fp8(pv_last[dc], p_frags[rg][PV_K_STEPS - 1], o_acc[rg][dc]);
    rd ^= LDS_BUF_BYTES;
  };

  const int full_end = (kv_len / BLOCK_N) * BLOCK_N;
  for (int tile_start = 0; tile_start < full_end; tile_start += BLOCK_N)
    process_tile(tile_start, std::false_type{});
  barrier_wait();
  if (full_end < kv_len) process_tile(full_end, std::true_type{});
  asm volatile("s_setprio 0");

  const float* vs_head = VScale + ((int64_t)batch * num_heads + head) * HEAD_DIM;
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    const float l_sum = l_run[rg] + swap_half(l_run[rg]);
    const float inv_l = l_sum != 0.f ? __builtin_amdgcn_rcpf(l_sum) : 0.f;
    if (!q_ok[rg]) continue;
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) {
      const int d0 = dc * 16 + klane * 8;
      float vs[8];
      *reinterpret_cast<float4*>(vs) = *reinterpret_cast<const float4*>(vs_head + d0);
      *reinterpret_cast<float4*>(vs + 4) = *reinterpret_cast<const float4*>(vs_head + d0 + 4);
      __hip_bfloat16 outv[8];
      #pragma unroll
      for (int i = 0; i < 8; ++i) outv[i] = __float2bfloat16((inv_l * vs[i]) * o_acc[rg][dc][i]);
      *reinterpret_cast<int4*>(O + batch * o_sb + (int64_t)q_row[rg] * o_sn + head * o_sh + d0) =
          *reinterpret_cast<int4*>(outv);
    }
  }
}

}  // namespace sage_port

torch::Tensor attention(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output,
                        torch::Tensor query_scale, torch::Tensor key_scale, torch::Tensor value_scale,
                        double sm_scale, int64_t valid_kv_len) {
  using namespace sage_port;
  TORCH_CHECK(query.is_cuda(), "Expected query on a ROCm GPU");
  const c10::DeviceGuard device_guard(query.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(std::string(properties->gcnArchName).rfind("gfx1201", 0) == 0,
              "Exact-P attention requires gfx1201");
  for (const auto& tensor : {key, value, output, query_scale, key_scale, value_scale})
    TORCH_CHECK(tensor.device() == query.device(), "All tensors must share the query device");
  TORCH_CHECK(query.is_cuda() && query.dim() == 4 && query.scalar_type() == torch::kInt8 && query.size(3) == HEAD_DIM &&
              query.stride(3) == 1, "Expected int8 [B, S, H, 128] query");
  TORCH_CHECK(key.dim() == 4 && key.scalar_type() == torch::kInt8 && key.is_contiguous() && key.size(3) == HEAD_DIM,
              "Expected contiguous int8 [B, H, S_pad, 128] key");
  TORCH_CHECK(value.dim() == 4 && value.scalar_type() == torch::kByte && value.is_contiguous() && value.size(2) == HEAD_DIM,
              "Expected contiguous fp8 [B, H, 128, S_pad] value");
  TORCH_CHECK(output.dim() == 4 && output.scalar_type() == torch::kBFloat16 && output.stride(3) == 1 &&
              output.size(3) == HEAD_DIM, "Expected bf16 [B, >= S, H, 128] output");
  const int64_t batch = query.size(0), qo_len = query.size(1), heads = query.size(2), kv_pad = key.size(2);
  TORCH_CHECK(batch > 0 && qo_len > 0 && qo_len <= 131072 && heads > 0, "Invalid query dimensions");
  TORCH_CHECK(key.size(0) == batch && key.size(1) == heads && value.size(0) == batch && value.size(1) == heads &&
              output.size(0) == batch && output.size(1) >= qo_len && output.size(2) == heads, "Shape mismatch");
  TORCH_CHECK(kv_pad % 64 == 0 && valid_kv_len > 0 && valid_kv_len <= kv_pad && value.size(3) >= kv_pad,
              "Invalid padded KV length");
  TORCH_CHECK(kv_pad <= (int64_t{1} << 24), "K/V tile offsets are formed in 32 bits");
  TORCH_CHECK(query_scale.scalar_type() == torch::kFloat32 && query_scale.dim() == 3 && query_scale.stride(2) == 1 &&
              query_scale.size(0) == batch && query_scale.size(1) == heads && query_scale.size(2) >= (qo_len + 31) / 32,
              "Invalid query scale");
  TORCH_CHECK(key_scale.scalar_type() == torch::kFloat32 && key_scale.dim() == 3 && key_scale.stride(2) == 1 &&
              key_scale.size(0) == batch && key_scale.size(1) == heads && key_scale.size(2) >= kv_pad / 64,
              "Invalid key scale");
  TORCH_CHECK(value_scale.scalar_type() == torch::kFloat32 && value_scale.is_contiguous() &&
              value_scale.numel() == batch * heads * HEAD_DIM, "Invalid value scale");
  const unsigned query_blocks = (qo_len + BLOCK_M - 1) / BLOCK_M;
  const dim3 grid(static_cast<unsigned>(heads * QUERY_GRID_GROUP),
                  (query_blocks + QUERY_GRID_GROUP - 1) / QUERY_GRID_GROUP,
                  static_cast<unsigned>(batch));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  attn_int8_port_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), reinterpret_cast<const int8_t*>(value.data_ptr<uint8_t>()),
      query_scale.data_ptr<float>(), key_scale.data_ptr<float>(), value_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
      (int)qo_len, (int)valid_kv_len, (int)kv_pad, (int)heads,
      query.stride(0), query.stride(1), query.stride(2), key.stride(0), key.stride(1), value.stride(0), value.stride(1),
      query_scale.stride(0), query_scale.stride(1), key_scale.stride(0), key_scale.stride(1),
      output.stride(0), output.stride(1), output.stride(2), (float)sm_scale);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "attn_int8_port_kernel launch failed");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("attention", &attention);
}

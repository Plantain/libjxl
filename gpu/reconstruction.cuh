// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Batched reconstruction matching the pinned e8ff097 decoder.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "batch.cuh"
namespace satellite_reconstruction_gpu {
constexpr int width = 512, pixels = 512 * 512, blocks = 64 * 64;
// Coefficients retain libjxl's canonical min(H,W) x max(H,W) order, reshaped
// row-major into the physical W x H footprint. This avoids overlap for tall
// rectangles, whose canonical row width exceeds their physical width.
__host__ __device__ inline int physical_width(int t) {
  const int w[] = {8,  8, 8, 8, 16, 32, 8, 16, 8,  32, 16,
                   32, 8, 8, 8, 8,  8,  8, 64, 32, 64};
  return w[t];
}
__host__ __device__ inline int physical_height(int t) {
  const int h[] = {8,  8, 8, 8, 16, 32, 16, 8,  32, 8, 32,
                   16, 8, 8, 8, 8,  8,  8,  64, 64, 32};
  return h[t];
}
__host__ __device__ inline bool supported_kind(int t) {
  return t == 0 || (t >= 4 && t <= 11) || (t >= 18 && t <= 20);
}
__device__ inline int coefficient_at(int anchor, int raw, int w) {
  return ((anchor / 64) * 8 + raw / w) * 512 + (anchor % 64) * 8 + raw % w;
}

using Params = jxl_wp3::Params;
__device__ inline int mirror(int x, int n) {
  return x < 0 ? -x - 1 : (x >= n ? 2 * n - x - 1 : x);
}
__device__ inline float sample(const float* a, int c, int x, int y) {
  return a[c * pixels + mirror(y, 512) * 512 + mirror(x, 512)];
}
__global__ void gaborish(const float* a, float* out, const Params* params) {
  const Params& p = *jxl_wp3::batch_ptr(params);
  a = jxl_wp3::batch_ptr(a);
  out = jxl_wp3::batch_ptr(out);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixels) return;
  int x = i % 512, y = i / 512;
  for (int c = 0; c < 3; ++c) {
    float w1 = p.gab[2 * c], w2 = p.gab[2 * c + 1],
          w0 = 1.0f / (1.0f + 4.0f * (w1 + w2));
    w1 *= w0;
    w2 *= w0;
    float side = (sample(a, c, x - 1, y) + sample(a, c, x + 1, y)) +
                 (sample(a, c, x, y - 1) + sample(a, c, x, y + 1));
    float corner = (sample(a, c, x - 1, y - 1) + sample(a, c, x + 1, y - 1)) +
                   (sample(a, c, x - 1, y + 1) + sample(a, c, x + 1, y + 1));
    out[c * pixels + i] =
        fmaf(corner, w2, fmaf(side, w1, a[c * pixels + i] * w0));
  }
}
__global__ void epf1(const float* a, const float* sigma, float* out,
                     const Params* params) {
  const Params& p = *jxl_wp3::batch_ptr(params);
  a = jxl_wp3::batch_ptr(a);
  sigma = jxl_wp3::batch_ptr(sigma);
  out = jxl_wp3::batch_ptr(out);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixels) return;
  int x = i % 512, y = i / 512;
  float sig = sigma[(y / 8) * 64 + x / 8];
  if (sig < -3.90524291751269967465540850526868f) {
    for (int c = 0; c < 3; ++c) out[c * pixels + i] = a[c * pixels + i];
    return;
  }
  float mul = 1.65f;
  if (x % 8 == 0 || x % 8 == 7 || y % 8 == 0 || y % 8 == 7) mul *= p.border;
  const float inv = sig * mul;
  float sad[4] = {};
  for (int c = 0; c < 3; ++c) {
    float p20 = sample(a, c, x, y - 2), p21 = sample(a, c, x, y - 1),
          p11 = sample(a, c, x - 1, y - 1), p31 = sample(a, c, x + 1, y - 1);
    float p02 = sample(a, c, x - 2, y), p12 = sample(a, c, x - 1, y),
          p22 = a[c * pixels + i], p32 = sample(a, c, x + 1, y),
          p42 = sample(a, c, x + 2, y);
    float p13 = sample(a, c, x - 1, y + 1), p23 = sample(a, c, x, y + 1),
          p33 = sample(a, c, x + 1, y + 1), p24 = sample(a, c, x, y + 2);
    float s0 = fabsf(p20 - p21), s1 = fabsf(p11 - p21), s2 = fabsf(p31 - p21),
          t;
    s1 += fabsf(p02 - p12);
    s0 += fabsf(p11 - p12);
    t = fabsf(p12 - p22);
    s1 += t;
    s2 += t;
    t = fabsf(p22 - p21);
    float s3 = t;
    s0 += t;
    s0 += fabsf(p31 - p32);
    t = fabsf(p22 - p32);
    s1 += t;
    s2 += t;
    s2 += fabsf(p42 - p32);
    s3 += fabsf(p13 - p12);
    t = fabsf(p22 - p23);
    s0 += t;
    s3 += t;
    s1 += fabsf(p13 - p23);
    s2 += fabsf(p33 - p23);
    s3 += fabsf(p33 - p32);
    s3 += fabsf(p24 - p23);
    sad[0] = fmaf(s0, p.channel[c], sad[0]);
    sad[1] = fmaf(s1, p.channel[c], sad[1]);
    sad[2] = fmaf(s2, p.channel[c], sad[2]);
    sad[3] = fmaf(s3, p.channel[c], sad[3]);
  }
  float wt[4];
  float sum = 1.0f;
  for (int k = 0; k < 4; ++k) {
    wt[k] = fmaxf(0.0f, fmaf(sad[k], inv, 1.0f));
    sum += wt[k];
  }
  const float recip = 1.0f / sum;
  const int dx[4] = {0, -1, 1, 0}, dy[4] = {-1, 0, 0, 1};
  for (int c = 0; c < 3; ++c) {
    float v = a[c * pixels + i];
    for (int k = 0; k < 4; ++k)
      v = fmaf(wt[k], sample(a, c, x + dx[k], y + dy[k]), v);
    out[c * pixels + i] = v * recip;
  }
}
__global__ void xyb_linear(const float* a, float* out, const Params* params) {
  const Params& p = *jxl_wp3::batch_ptr(params);
  a = jxl_wp3::batch_ptr(a);
  out = jxl_wp3::batch_ptr(out);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixels) return;
  float r = (a[pixels + i] + a[i]) - p.opsin_cbrt[0],
        g = (a[pixels + i] - a[i]) - p.opsin_cbrt[1],
        b = a[2 * pixels + i] - p.opsin_cbrt[2];
  r = fmaf(r * r, r, p.opsin_bias[0]);
  g = fmaf(g * g, g, p.opsin_bias[1]);
  b = fmaf(b * b, b, p.opsin_bias[2]);
  for (int c = 0; c < 3; ++c)
    out[c * pixels + i] =
        fmaf(p.matrix[c * 3 + 2], b,
             fmaf(p.matrix[c * 3 + 1], g, p.matrix[c * 3] * r));
}
__global__ void dc_dequant(const int32_t* a, float* out, const Params* params) {
  const Params& p = *jxl_wp3::batch_ptr(params);
  a = jxl_wp3::batch_ptr(a);
  out = jxl_wp3::batch_ptr(out);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= blocks) return;
  const float y = float(a[blocks + i]) * (p.dc_mul[1] * p.dc_precision);
  out[blocks + i] = y;
  out[i] = fmaf(y, p.dc_cfl[0], float(a[i]) * (p.dc_mul[0] * p.dc_precision));
  out[2 * blocks + i] =
      fmaf(y, p.dc_cfl[2],
           float(a[2 * blocks + i]) * (p.dc_mul[2] * p.dc_precision));
}
__global__ void dc_smooth(const float* a, float* out, const Params* params) {
  const Params& p = *jxl_wp3::batch_ptr(params);
  a = jxl_wp3::batch_ptr(a);
  out = jxl_wp3::batch_ptr(out);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= blocks) return;
  int x = i % 64, y = i / 64;
  if (p.skip_dc_smoothing || x == 0 || x == 63 || y == 0 || y == 63) {
    for (int c = 0; c < 3; ++c) out[c * blocks + i] = a[c * blocks + i];
    return;
  }
  const float w1 = .20345139757231578f, w2 = .0334829185968739f,
              w0 = 1.0f - 4.0f * (w1 + w2);
  float smooth[3], gap = .5f;
  for (int c = 0; c < 3; ++c) {
    const float* b = a + c * blocks;
    float side = (b[i - 1] + b[i + 1]) + (b[i - 64] + b[i + 64]);
    float corner = (b[i - 65] + b[i - 63]) + (b[i + 63] + b[i + 65]);
    smooth[c] = fmaf(corner, w2, fmaf(side, w1, b[i] * w0));
    gap = fmaxf(gap, fabsf((b[i] - smooth[c]) / p.dc_mul[c]));
  }
  float factor = fmaxf(0.0f, fmaf(-4.0f, gap, 3.0f));
  for (int c = 0; c < 3; ++c)
    out[c * blocks + i] =
        fmaf(smooth[c] - a[c * blocks + i], factor, a[c * blocks + i]);
}
// Pinned DCT III recursive factorization. Explicit FMA matches the selected
// reference SIMD path; shared-memory staging preserves transposition order.
__device__ inline float wc(int n, int k) {
  const float v4[2] = {.541196100146197f, 1.3065629648763764f};
  const float v8[4] = {.5097955791041592f, .6013448869350453f,
                       .8999762231364156f, 2.5629154477415055f};
  const float v16[8] = {.5024192861881557f,  .5224986149396889f,
                        .5669440348163577f,  .6468217833599901f,
                        .7881546234512502f,  1.060677685990347f,
                        1.7224470982383342f, 5.101148618689155f};
  const float v32[16] = {
      .5006029982351963f,  .5054709598975436f, .5154473099226246f,
      .5310425910897841f,  .5531038960344445f, .5829349682061339f,
      .6225041230356648f,  .6748083414550057f, .7445362710022986f,
      .8393496454155268f,  .9725682378619608f, 1.1694399334328847f,
      1.4841646163141662f, 2.057781009953411f, 3.407608418468719f,
      10.190008123548033f};
  const float v64[32] = {
      0.500150636020651,  0.5013584524464084, 0.5037887256810443,
      0.5074711720725553, 0.5124514794082247, 0.5187927131053328,
      0.52657731515427,   0.535909816907992,  0.5469204379855088,
      0.5597698129470802, 0.57465518403266,   0.5918185358574165,
      0.6115573478825099, 0.6342389366884031, 0.6603198078137061,
      0.6903721282002123, 0.7251205223771985, 0.7654941649730891,
      0.8127020908144905, 0.8683447152233481, 0.9345835970364075,
      1.0144082649970547, 1.1120716205797176, 1.233832737976571,
      1.3892939586328277, 1.5939722833856311, 1.8746759800084078,
      2.282050068005162,  2.924628428158216,  4.084611078129248,
      6.796750711673633,  20.373878167231453,
  };
  return n == 64   ? v64[k]
         : n == 4  ? v4[k]
         : n == 8  ? v8[k]
         : n == 16 ? v16[k]
                   : v32[k];
}
__device__ inline float idct_madd(float a, float b, float c) {
#ifdef SATELLITE_RECONSTRUCTION_IDCT_NOFMA
  return __fadd_rn(__fmul_rn(a, b), c);
#else
  return fmaf(a, b, c);
#endif
}
template <int N>
__device__ inline void idct(float* a) {
  if constexpr (N == 1)
    return;
  else if constexpr (N == 2) {
    float x = a[0], y = a[1];
    a[0] = x + y;
    a[1] = x - y;
  } else {
    float t[N];
    for (int k = 0; k < N / 2; ++k) {
      t[k] = a[2 * k];
      t[N / 2 + k] = a[2 * k + 1];
    }
    idct<N / 2>(t);
    for (int k = N / 2 - 1; k > 0; --k) t[N / 2 + k] += t[N / 2 + k - 1];
    t[N / 2] *= 1.41421356237f;
    idct<N / 2>(t + N / 2);
    for (int k = 0; k < N / 2; ++k) {
      a[k] = idct_madd(wc(N, k), t[N / 2 + k], t[k]);
      a[N - k - 1] = idct_madd(-wc(N, k), t[N / 2 + k], t[k]);
    }
  }
}
__device__ inline void idct_size(float* a, int n) {
  if (n == 4)
    idct<4>(a);
  else if (n == 8)
    idct<8>(a);
  else if (n == 16)
    idct<16>(a);
  else if (n == 32)
    idct<32>(a);
  else
    idct<64>(a);
}
// Both axes stay within one CTA. Padding prevents shared-memory bank
// conflicts; the final pixel stores are coalesced across output columns.
__global__ void inverse_dct(const float* a, const uint8_t* kind,
                            const uint8_t* first, float* out) {
  a = jxl_wp3::batch_ptr(a);
  kind = jxl_wp3::batch_ptr(kind);
  first = jxl_wp3::batch_ptr(first);
  out = jxl_wp3::batch_ptr(out);
  const int anchor = blockIdx.x, c = blockIdx.y, k = threadIdx.x;
  if (!first[anchor] || !supported_kind(kind[anchor])) return;
  const int w = physical_width(kind[anchor]), h = physical_height(kind[anchor]);
  const int x = anchor % 64 * 8, y = anchor / 64 * 8;
  __shared__ float transpose[64 * 65];
  if (k < h) {
    float row[64];
    for (int j = 0; j < w; ++j)
      row[j] = a[c * pixels +
                 coefficient_at(anchor, w <= h ? j * h + k : k * w + j, w)];
    idct_size(row, w);
    for (int j = 0; j < w; ++j) transpose[k * 65 + j] = row[j];
  }
  __syncthreads();
  if (k < w) {
    float column[64];
    for (int j = 0; j < h; ++j) column[j] = transpose[j * 65 + k];
    idct_size(column, h);
    for (int j = 0; j < h; ++j)
      out[c * pixels + (y + j) * 512 + x + k] = column[j];
  }
}
__global__ void inverse_dct2(const float* a, const uint8_t* kind,
                             const uint8_t* first, float* out) {
  a = jxl_wp3::batch_ptr(a);
  kind = jxl_wp3::batch_ptr(kind);
  first = jxl_wp3::batch_ptr(first);
  out = jxl_wp3::batch_ptr(out);

  const int anchor = blockIdx.x, c = blockIdx.y;
  if (threadIdx.x || !first[anchor] || kind[anchor] != 2) return;
  const int x = (anchor % 64) * 8, y = (anchor / 64) * 8;
  float t[64], u[64];
  for (int j = 0; j < 8; ++j)
    for (int k = 0; k < 8; ++k)
      t[j * 8 + k] = a[c * pixels + (y + j) * 512 + x + k];
  for (int n = 2; n <= 8; n *= 2) {
    int h = n / 2;
    for (int j = 0; j < h; ++j)
      for (int k = 0; k < h; ++k) {
        float v0 = t[j * 8 + k], v1 = t[j * 8 + h + k], v2 = t[(j + h) * 8 + k],
              v3 = t[(j + h) * 8 + h + k];
        u[j * 16 + 2 * k] = ((v0 + v1) + v2) + v3;
        u[j * 16 + 2 * k + 1] = ((v0 + v1) - v2) - v3;
        u[(2 * j + 1) * 8 + 2 * k] = ((v0 - v1) + v2) - v3;
        u[(2 * j + 1) * 8 + 2 * k + 1] = ((v0 - v1) - v2) + v3;
      }
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) t[j * 8 + k] = u[j * 8 + k];
  }
  for (int j = 0; j < 8; ++j)
    for (int k = 0; k < 8; ++k)
      out[c * pixels + (y + j) * 512 + x + k] = t[j * 8 + k];
}

template <int N>
__device__ inline void dct(float* a) {
  if constexpr (N == 1)
    return;
  else if constexpr (N == 2) {
    float x = a[0], y = a[1];
    a[0] = x + y;
    a[1] = x - y;
  } else {
    float t[N];
    for (int k = 0; k < N / 2; ++k) {
      t[k] = a[k] + a[N - 1 - k];
      t[N / 2 + k] = (a[k] - a[N - 1 - k]) * wc(N, k);
    }
    dct<N / 2>(t);
    dct<N / 2>(t + N / 2);
    t[N / 2] = fmaf(t[N / 2], 1.41421356237f, t[N / 2 + 1]);
    for (int k = 1; k + 1 < N / 2; ++k) t[N / 2 + k] += t[N / 2 + k + 1];
    for (int k = 0; k < N / 2; ++k) {
      a[2 * k] = t[k];
      a[2 * k + 1] = t[N / 2 + k];
    }
  }
}
__device__ inline float resample(int n, int k) {
  const float r[4] = {1.0f, 1.025760096781116015f, 1.108937353592731823f,
                      1.270559368765487251f};
  const float r64[8] = {1.,
                        1.0063534990068217,
                        1.0257600967811158,
                        1.0593017296817173,
                        1.1089373535927318,
                        1.1777765381970435,
                        1.2705593687654873,
                        1.3944898413647777};
  return n == 64 ? r64[k] : n == 16 ? (k ? 1.108937353592731823f : 1.0f) : r[k];
}
__device__ inline float quant_bias(int v, int c, const Params& p) {
  if (v == 0) return 0.0f;
  if (v == 1) return p.biases[c];
  if (v == -1) return -p.biases[c];
  return fmaf(-p.biases[3], 1.0f / float(v), float(v));
}
__global__ void ac_dequant(const int32_t* ac, const int32_t* qf,
                           const uint8_t* kind, const uint8_t* first,
                           const float* cfl, const float* matrices,
                           const Params* params, float* out) {
  const Params& p = *jxl_wp3::batch_ptr(params);
  ac = jxl_wp3::batch_ptr(ac);
  qf = jxl_wp3::batch_ptr(qf);
  kind = jxl_wp3::batch_ptr(kind);
  first = jxl_wp3::batch_ptr(first);
  cfl = jxl_wp3::batch_ptr(cfl);
  matrices = jxl_wp3::batch_ptr(matrices);
  out = jxl_wp3::batch_ptr(out);

  const int anchor = blockIdx.x;
  if (!first[anchor]) return;
  const int t = kind[anchor], w = physical_width(t), h = physical_height(t),
            size = w * h;
  const float scale = p.inverse / float(qf[anchor]), sx = scale * p.xmul,
              sb = scale * p.bmul;
  for (int k = threadIdx.x; k < size; k += blockDim.x) {
    const int at = coefficient_at(anchor, k, w);
    const float dx = quant_bias(ac[at], 0, p) * (matrices[at] * sx);
    const float dy =
        quant_bias(ac[pixels + at], 1, p) * (matrices[pixels + at] * scale);
    const float db = quant_bias(ac[2 * pixels + at], 2, p) *
                     (matrices[2 * pixels + at] * sb);
    out[at] = fmaf(cfl[anchor], dy, dx);
    out[pixels + at] = dy;
    out[2 * pixels + at] = fmaf(cfl[blocks + anchor], dy, db);
  }
}
// One CTA owns the small DC transform. The transpose is explicitly shared
// across the column and row passes, including the 8x8 LLF of a DCT64.
__global__ void lowest_frequencies(const float* dc, const uint8_t* kind,
                                   const uint8_t* first, float* out) {
  dc = jxl_wp3::batch_ptr(dc);
  kind = jxl_wp3::batch_ptr(kind);
  first = jxl_wp3::batch_ptr(first);
  out = jxl_wp3::batch_ptr(out);
  const int anchor = blockIdx.x, c = blockIdx.y, k = threadIdx.x;
  if (!first[anchor]) return;
  const int t = kind[anchor], w = physical_width(t), h = physical_height(t);
  const int cx = w / 8, cy = h / 8, bx = anchor % 64, by = anchor / 64;
  if (cx == 1 && cy == 1) {
    if (k == 0)
      out[c * pixels + by * 8 * 512 + bx * 8] = dc[c * blocks + anchor];
    return;
  }
  __shared__ float transpose[64];
  if (k < cx) {
    float column[8];
    for (int y = 0; y < cy; ++y)
      column[y] = dc[c * blocks + (by + y) * 64 + bx + k];
    if (cy == 2)
      dct<2>(column);
    else if (cy == 4)
      dct<4>(column);
    else if (cy == 8)
      dct<8>(column);
    for (int y = 0; y < cy; ++y) transpose[k * cy + y] = column[y] * (1.f / cy);
  }
  __syncthreads();
  if (k < cy) {
    float row[8];
    for (int x = 0; x < cx; ++x) row[x] = transpose[x * cy + k];
    if (cx == 2)
      dct<2>(row);
    else if (cx == 4)
      dct<4>(row);
    else if (cx == 8)
      dct<8>(row);
    for (int x = 0; x < cx; ++x) {
      const int raw = w <= h ? x * h + k : k * w + x;
      const float s0 = w <= h ? resample(w, x) : resample(h, k);
      const float s1 = w <= h ? resample(h, k) : resample(w, x);
      out[c * pixels + coefficient_at(anchor, raw, w)] =
          ((row[x] * (1.f / cx)) * s0) * s1;
    }
  }
}
__global__ void compute_sigma(const int32_t* qf, const uint8_t* kind,
                              const uint8_t* first, const uint8_t* sharp,
                              float* out, const Params* params) {
  const Params& p = *jxl_wp3::batch_ptr(params);
  qf = jxl_wp3::batch_ptr(qf);
  kind = jxl_wp3::batch_ptr(kind);
  first = jxl_wp3::batch_ptr(first);
  sharp = jxl_wp3::batch_ptr(sharp);
  out = jxl_wp3::batch_ptr(out);

  const int anchor = blockIdx.x * blockDim.x + threadIdx.x;
  if (anchor >= blocks || !first[anchor]) return;
  const int cx = physical_width(kind[anchor]) / 8,
            cy = physical_height(kind[anchor]) / 8;
  const float sq =
      p.epf_quant / ((p.scale * float(qf[anchor])) * -1.1715728752538099024f);
  for (int y = 0; y < cy; ++y)
    for (int x = 0; x < cx; ++x) {
      int at = anchor + y * 64 + x;
      float sig = fminf(-1e-4f, sq * p.sharp[sharp[at]]);
      out[at] = 1.0f / sig;
    }
}

__device__ inline float srgb_transfer(float value, bool inverse) {
  float x = fabsf(value), v;
  if (inverse) {
    const float p[5] = {2.200248328e-4f, 1.043637593e-2f, 1.624820318e-1f,
                        7.961564959e-1f, 8.210152774e-1f};
    const float q[5] = {2.631846970e-1f, 1.076976492f, 4.987528350e-1f,
                        -5.512498495e-2f, 6.521209011e-3f};
    float a = p[4], b = q[4];
    for (int j = 3; j >= 0; --j) {
      a = fmaf(a, x, p[j]);
      b = fmaf(b, x, q[j]);
    }
    v = x > .04045f ? a / b : x * (1.0f / 12.92f);
  } else {
    const float p[5] = {-5.135152395e-4f, 5.287254571e-3f, 3.903842876e-1f,
                        1.474205315f, 7.352629620e-1f};
    const float q[5] = {1.004519624e-2f, 3.036675394e-1f, 1.340816930f,
                        9.258482155e-1f, 2.424867759e-2f};
    float a = p[4], b = q[4], z = sqrtf(x);
    for (int j = 3; j >= 0; --j) {
      a = fmaf(a, z, p[j]);
      b = fmaf(b, z, q[j]);
    }
    v = x > .0031308f ? a / b : x * 12.92f;
  }
  return copysignf(v, value);
}
__global__ void transfer(const float* in, float* out, bool inverse) {
  in = jxl_wp3::batch_ptr(in);
  out = jxl_wp3::batch_ptr(out);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 3 * pixels) out[i] = srgb_transfer(in[i], inverse);
}
}  // namespace satellite_reconstruction_gpu

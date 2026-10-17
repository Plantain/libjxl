// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Experimental pinned-reference comparator. No encoder integration yet.
#pragma once
#include "batch.cuh"
#include "butteraugli_malta.cuh"
#include "butteraugli_preprocess.cuh"
namespace satellite_butteraugli {
__global__ void malta_difference(const float* a, const float* b, int n,
                                 float norm, float norm_gt, float norm_lt,
                                 float* diffs) {
  a = jxl_wp3::batch_ptr(a);
  b = jxl_wp3::batch_ptr(b);
  diffs = jxl_wp3::batch_ptr(diffs);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  const float absval = .5f * (fabsf(a[p]) + fabsf(b[p]));
  const float scale = __fdiv_rn(norm_gt, norm + absval),
              scale2 = __fdiv_rn(norm_lt, norm + absval);
  float value = scale * (a[p] - b[p]);
  const double absolute = fabs(double(a[p])), small = .55 * absolute,
               big = 1.05 * absolute;
  if (a[p] < 0) {
    if (double(b[p]) > -small)
      value = float(double(value) - double(scale2) * (double(b[p]) + small));
    else if (double(b[p]) < -big)
      value = float(double(value) + double(scale2) * (-double(b[p]) - big));
  } else {
    if (double(b[p]) < small)
      value = float(double(value) + double(scale2) * (small - double(b[p])));
    else if (double(b[p]) > big)
      value = float(double(value) - double(scale2) * (double(b[p]) - big));
  }
  diffs[p] = value;
}
__global__ void malta_accumulate(const float* diffs, int w, int h, bool low,
                                 float* ac) {
  diffs = jxl_wp3::batch_ptr(diffs);
  ac = jxl_wp3::batch_ptr(ac);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= w * h) return;
  ac[p] += low ? malta_lf(diffs, w, h, p % w, p / w)
               : malta_hf(diffs, w, h, p % w, p / w);
}
__global__ void l2_errors(const float* a, const float* b, int n,
                          float asymmetry, float* ac, float* dc) {
  a = jxl_wp3::batch_ptr(a);
  b = jxl_wp3::batch_ptr(b);
  ac = jxl_wp3::batch_ptr(ac);
  dc = jxl_wp3::batch_ptr(dc);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  const float weights[9] = {400.f,
                            float(1.50815703118),
                            0.f,
                            2150.f,
                            float(10.6195433239),
                            float(16.2176043152),
                            float(29.2353797994),
                            float(.844626970982),
                            float(.703646627719)};
  for (int c = 0; c < 3; ++c) {
    float total = ac[c * n + p];
    if (c < 2) {
      // Actual API receives FP32 weights before multiplying by double 0.8.
      const double original_weight = c == 0 ? 400. : 1.50815703118;
      const float wg = float(original_weight * double(asymmetry));
      const float wl = float(original_weight / double(asymmetry));
      const float weight_gt = float(double(wg) * .8),
                  weight_lt = float(double(wl) * .8);
      const float original = a[(6 + c) * n + p], candidate = b[(6 + c) * n + p];
      const float d = original - candidate, absolute = fabsf(original),
                  small = .4f * absolute;
      total = __fmaf_rn(d * d, weight_gt, total);
      const float v = original < 0
                          ? (candidate > -small      ? candidate + small
                             : candidate < -absolute ? -candidate - absolute
                                                     : 0.f)
                      : candidate < small    ? small - candidate
                      : candidate > absolute ? candidate - absolute
                                             : 0.f;
      total = __fmaf_rn(weight_lt, v * v, total);
    }
    const float mf = a[(3 + c) * n + p] - b[(3 + c) * n + p];
    ac[c * n + p] = __fmaf_rn(mf * mf, weights[3 + c], total);
    const float lf = a[c * n + p] - b[c * n + p];
    dc[c * n + p] = lf * lf * weights[6 + c];
  }
}
__global__ void masking_precompute(const float* psycho, int n, float* mask) {
  psycho = jxl_wp3::batch_ptr(psycho);
  mask = jxl_wp3::batch_ptr(mask);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  const float x = (psycho[8 * n + p] + psycho[6 * n + p]) * 2.5f;
  const float y = __fmaf_rn(psycho[9 * n + p], .4f, psycho[7 * n + p] * .4f);
  const float combined = sqrtf(__fmaf_rn(x, x, y * y));
  constexpr float mul = float(6.19424080439),
                  bias = mul * float(12.61050594197);
  mask[p] = sqrtf(__fmaf_rn(mul, fabsf(combined), bias)) - sqrtf(bias);
}
__device__ inline void min3(float v, float& a, float& b, float& c) {
  if (v < c) {
    if (v < a) {
      c = b;
      b = a;
      a = v;
    } else if (v < b) {
      c = b;
      b = v;
    } else
      c = v;
  }
}
__global__ void erode_mask(const float* from, int w, int h, float* to) {
  from = jxl_wp3::batch_ptr(from);
  to = jxl_wp3::batch_ptr(to);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= w * h) return;
  const int x = p % w, y = p / w;
  float a = from[p], b = 2 * a, c = b;
  if (x >= 3) {
    min3(from[p - 3], a, b, c);
    if (y >= 3) min3(from[p - 3 * w - 3], a, b, c);
    if (y < h - 3) min3(from[p + 3 * w - 3], a, b, c);
  }
  if (x < w - 3) {
    min3(from[p + 3], a, b, c);
    if (y >= 3) min3(from[p - 3 * w + 3], a, b, c);
    if (y < h - 3) min3(from[p + 3 * w + 3], a, b, c);
  }
  if (y >= 3) min3(from[p - 3 * w], a, b, c);
  if (y < h - 3) min3(from[p + 3 * w], a, b, c);
  to[p] = __fmaf_rn(.25f, c, __fmaf_rn(.45f, a, .3f * b));
}
__global__ void combine_error(const float* mask, const float* mask0,
                              const float* mask1, const float* ac,
                              const float* dc, int n, float xmul, float* out) {
  mask = jxl_wp3::batch_ptr(mask);
  mask0 = jxl_wp3::batch_ptr(mask0);
  mask1 = jxl_wp3::batch_ptr(mask1);
  ac = jxl_wp3::batch_ptr(ac);
  dc = jxl_wp3::batch_ptr(dc);
  out = jxl_wp3::batch_ptr(out);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  constexpr float normalization = 17.83f * .79079917404f;
  constexpr float global = float(1.0 / normalization);
  const double value = mask[p];
  const double av =
      double(global) *
      (1.0 + 2.5485944793 / __fma_rn(.451936922203, value, .829591754942));
  const double dv =
      double(global) *
      (1.0 + .505054525019 / __fma_rn(3.87449418804, value, .20025578522));
  const float am = float(av * av), dm = float(dv * dv);
  const float delta = mask0[p] - mask1[p];
  const float ay = __fmaf_rn(10.f * delta, delta, ac[n + p]);
  const float dcv =
      __fmaf_rn(dc[2 * n + p], dm, __fmaf_rn(dc[p] * xmul, dm, dc[n + p] * dm));
  const float acv =
      __fmaf_rn(ac[2 * n + p], am, __fmaf_rn(ac[p] * xmul, am, ay * am));
  out[p] = sqrtf(dcv + acv);
}

// Reference caches own 12*n floats: psycho[10], blurred mask, eroded mask.
inline cudaError_t preprocess(const float* rgb, int w, int h, float intensity,
                              float* psycho, float* scratch,
                              cudaStream_t stream, int batch) {
  const int n = w * h;
  float *blurred = scratch, *xyb = scratch + 3 * n,
        *transpose = scratch + 6 * n;
  for (int c = 0; c < 3; ++c)
    blur(rgb + c * n, w, h, kernel(1.2f), transpose, blurred + c * n, stream,
         batch);
  opsin_dynamics<<<jxl_wp3::grid((n + 255) / 256, batch), 256, 0, stream>>>(
      rgb, blurred, n, intensity, xyb);
  return separate_frequencies(xyb, w, h, psycho, transpose, stream, batch);
}
inline cudaError_t setup_reference(const float* rgb, int w, int h,
                                   float intensity, float* reference,
                                   float* scratch, cudaStream_t stream,
                                   int batch) {
  const int n = w * h;
  preprocess(rgb, w, h, intensity, reference, scratch, stream, batch);
  masking_precompute<<<jxl_wp3::grid((n + 255) / 256, batch), 256, 0, stream>>>(
      reference, n, scratch);
  blur(scratch, w, h, kernel(2.7f), scratch + n, reference + 10 * n, stream,
       batch);
  erode_mask<<<jxl_wp3::grid((n + 255) / 256, batch), 256, 0, stream>>>(
      reference + 10 * n, w, h, reference + 11 * n);
  return cudaGetLastError();
}
// Scratch owns 17*n floats, reused between full and half resolution.
// Preprocess: blur[0:3], XYB[3:6], transpose[6:7], psycho[7:17].
// Compare: diff/mask[0:1], AC[1:4], DC[4:7], psycho[7:17].
// After masking_precompute, psycho is dead; [7:9] holds the mask blur.
inline cudaError_t compare_scale(const float* reference, const float* rgb,
                                 int w, int h, float intensity, float asymmetry,
                                 float xmul, float* scratch, float* output,
                                 cudaStream_t stream, int batch) {
  const int n = w * h, blocks = (n + 255) / 256;
  const float* pa = reference;
  float *pb = scratch + 7 * n, *diff = scratch;
  float *ac = scratch + n, *dc = scratch + 4 * n;
  float *mask1 = scratch, *blur1 = scratch + 7 * n,
        *transpose = scratch + 8 * n;
  preprocess(rgb, w, h, intensity, pb, scratch, stream, batch);
  jxl_wp3::
      zero_plane<<<jxl_wp3::grid((3 * n + 255) / 256, batch), 256, 0, stream>>>(
          ac, 3 * n);
  cudaError_t code;
  const double weights[6] = {1.10039032555, 173.5,         18.7237414387,
                             6923.99476109, 37.0819870399, 8246.75321353};
  const double norms[6] = {71.7800275169, 5.,
                           4498534.45232, 8051.15833247,
                           130262059.556, 1009002.70582};
  const int planes[6] = {9, 8, 7, 6, 4, 3};
  for (int i = 0; i < 6; ++i) {
    const double asym = i < 2   ? asymmetry
                        : i < 4 ? double(std::sqrt(asymmetry))
                                : 1.;
    const double mult = i < 2 ? .39905817637 : .611612573796;
    const float ng =
        float(mult * std::sqrt(.5f * (weights[i] * asym)) / 8.5 * norms[i]);
    const float nl = float(
        mult * std::sqrt(double(.33f) * (weights[i] / asym)) / 8.5 * norms[i]);
    malta_difference<<<jxl_wp3::grid(blocks, batch), 256, 0, stream>>>(
        pa + planes[i] * n, pb + planes[i] * n, n, float(norms[i]), ng, nl,
        diff);
    malta_accumulate<<<jxl_wp3::grid(blocks, batch), 256, 0, stream>>>(
        diff, w, h, i >= 2, ac + (i % 2 == 0 ? 1 : 0) * n);
    code = cudaGetLastError();
    if (code != cudaSuccess) return code;
  }
  l2_errors<<<jxl_wp3::grid(blocks, batch), 256, 0, stream>>>(
      pa, pb, n, asymmetry, ac, dc);
  masking_precompute<<<jxl_wp3::grid(blocks, batch), 256, 0, stream>>>(pb, n,
                                                                       mask1);
  blur(mask1, w, h, kernel(2.7f), transpose, blur1, stream, batch);
  combine_error<<<jxl_wp3::grid(blocks, batch), 256, 0, stream>>>(
      reference + 11 * n, reference + 10 * n, blur1, ac, dc, n, xmul, output);
  return cudaGetLastError();
}

__global__ void subsample(const float* input, int w, int h, float* output) {
  input = jxl_wp3::batch_ptr(input);
  output = jxl_wp3::batch_ptr(output);

  const int n = w * h, sw = w / 2, sh = h / 2, sn = sw * sh;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= 3 * sn) return;
  const int c = p / sn, k = p % sn, x = (k % sw) * 2, y = (k / sw) * 2;
  const int at = c * n + y * w + x;
  float sum = 0.f;
  sum += .25f * input[at];
  sum += .25f * input[at + 1];
  sum += .25f * input[at + w];
  sum += .25f * input[at + w + 1];
  output[p] = sum;
}
__global__ void supersample_add(const float* sub, int w, int h, float* root) {
  sub = jxl_wp3::batch_ptr(sub);
  root = jxl_wp3::batch_ptr(root);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= w * h) return;
  root[p] = float(double(root[p]) * (1.0 - .3 * .5));
  root[p] += .5f * sub[(p / w / 2) * (w / 2) + (p % w / 2)];
}
}  // namespace satellite_butteraugli

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Standalone CUDA port of pinned e8ff097 Butteraugli blur and opsin dynamics.
// This is comparator preprocessing, not a complete GPU perceptual comparator.
#pragma once
#include <cuda_runtime.h>

#include <cmath>

#include "batch.cuh"

namespace satellite_butteraugli {
struct Kernel {
  int radius;
  float raw[33];
  float scaled[17];
};

inline Kernel kernel(float sigma) {
  Kernel result{};
  result.radius = int(2.25f * std::fabs(sigma));
  if (result.radius < 1) result.radius = 1;
  if (result.radius > 16) return result;
  const double scaler = -1.0 / (2.0 * sigma * sigma);
  float sum = 0.f;
  for (int i = -result.radius; i <= result.radius; ++i) {
    result.raw[i + result.radius] = float(std::exp(scaler * i * i));
    sum += result.raw[i + result.radius];
  }
  const float scale = 1.f / sum;
  for (int i = 0; i <= result.radius; ++i)
    result.scaled[i] = result.raw[i] * scale;
  return result;
}

__device__ inline int mirror(int p, int count) {
  if (p < 0) return -1 - p;
  if (p >= count) return 2 * count - 1 - p;
  return p;
}

// Reference Separable5 uses explicit fused operations and mirrored edges.
__device__ inline float horizontal5(const float* input, int w, int h, int x,
                                    int y, Kernel k) {
  const float* row = input + mirror(y, h) * w;
  const float center = row[x] * k.scaled[2];
  const float near = row[mirror(x - 1, w)] + row[mirror(x + 1, w)];
  const float far = row[mirror(x - 2, w)] + row[mirror(x + 2, w)];
  return __fmaf_rn(far, k.scaled[0], __fmaf_rn(near, k.scaled[1], center));
}

__global__ void separable5(const float* input, int w, int h, Kernel k,
                           float* output) {
  input = jxl_wp3::batch_ptr(input);
  output = jxl_wp3::batch_ptr(output);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= w * h) return;
  const int x = p % w, y = p / w;
  const float center = horizontal5(input, w, h, x, y, k) * k.scaled[2];
  const float near = horizontal5(input, w, h, x, y - 1, k) +
                     horizontal5(input, w, h, x, y + 1, k);
  const float far = horizontal5(input, w, h, x, y - 2, k) +
                    horizontal5(input, w, h, x, y + 2, k);
  output[p] = __fmaf_rn(far, k.scaled[0], __fmaf_rn(near, k.scaled[1], center));
}

// Other Butteraugli kernels normalize truncated borders. Keep the exact
// reference summation groups; these scalar operations must not contract.
__global__ void convolve_axis(const float* input, int w, int h, Kernel k,
                              float* output, bool vertical) {
  input = jxl_wp3::batch_ptr(input);
  output = jxl_wp3::batch_ptr(output);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= w * h) return;
  const int x = vertical ? p / w : p % w, r = k.radius;
  const int count = vertical ? h : w, stride = vertical ? w : 1;
  const float* row = input + (vertical ? p % w : (p / w) * w);
  float value;
  if (x < r || x >= count - r) {
    const int begin = max(0, x - r), end = min(count - 1, x + r);
    float weight = 0.f, sum = 0.f;
    for (int j = begin; j <= end; ++j)
      weight = __fadd_rn(weight, k.raw[j - x + r]);
    const float scale = __fdiv_rn(1.f, weight);
    for (int j = begin; j <= end; ++j)
      sum = __fadd_rn(sum, __fmul_rn(row[j * stride], k.raw[j - x + r]));
    value = __fmul_rn(sum, scale);
  } else {
    const float* start = row + (x - r) * stride;
    float sums[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < r; ++j) {
      const float term =
          __fmul_rn(__fadd_rn(start[j * stride], start[(2 * r - j) * stride]),
                    k.scaled[j]);
      // len7 has three independent terms; other supported sizes use
      // four accumulator chains exactly as ConvolutionWithTranspose.
      if (j < 4)
        sums[j] = term;
      else
        sums[j % 4] = __fadd_rn(sums[j % 4], term);
    }
    value = __fmul_rn(start[r * stride], k.scaled[r]);
    for (int j = 0; j < min(r, 4); ++j) value = __fadd_rn(value, sums[j]);
  }
  output[p] = value;
}

inline cudaError_t blur(const float* input, int w, int h, Kernel k,
                        float* scratch, float* output, cudaStream_t stream,
                        int batch) {
  if (k.radius == 2 && input != output) {
    separable5<<<jxl_wp3::grid((w * h + 255) / 256, batch), 256, 0, stream>>>(
        input, w, h, k, output);
  } else {
    convolve_axis<<<jxl_wp3::grid((w * h + 255) / 256, batch), 256, 0,
                    stream>>>(input, w, h, k, scratch, false);
    convolve_axis<<<jxl_wp3::grid((w * h + 255) / 256, batch), 256, 0,
                    stream>>>(scratch, w, h, k, output, true);
  }
  return cudaGetLastError();
}

__device__ inline float fast_log2(float x) {
  const int bits = __float_as_int(x);
  const int exponent = (bits - 0x3f2aaaab) >> 23;
  const float mantissa = __int_as_float(bits - (exponent << 23));
  const float v = mantissa - 1.f;
  const float p =
      __fmaf_rn(__fmaf_rn(7.4245873327820566E-01f, v, 1.4287160470083755E+00f),
                v, -1.8503833400518310E-06f);
  const float q =
      __fmaf_rn(__fmaf_rn(1.7409343003366853E-01f, v, 1.0096718572241148E+00f),
                v, 9.9032814277590719E-01f);
  return __fdiv_rn(p, q) + float(exponent);
}

__device__ inline float gamma(float value) {
  // The reference folds the natural logarithm conversion into this constant.
  constexpr float kInvLog2e = 0.693147180559945309417232121458176568f;
  return __fmaf_rn(19.245013259874995f * kInvLog2e,
                   fast_log2(fmaxf(value, 0.f) + float(9.9710635769299145)),
                   float(-23.16046239805755));
}

__device__ inline void absorb(float r, float g, float b, bool clamp,
                              float* values) {
  values[0] = __fmaf_rn(float(.29956550340058319), r,
                        __fmaf_rn(float(.63373087833825936), g,
                                  __fmaf_rn(float(.077705617820981968), b,
                                            float(1.7557483643287353))));
  values[1] = __fmaf_rn(float(.22158691104574774), r,
                        __fmaf_rn(float(.69391388044116142), g,
                                  __fmaf_rn(float(.0987313588422), b,
                                            float(1.7557483643287353))));
  values[2] = __fmaf_rn(float(.02), r,
                        __fmaf_rn(float(.02), g,
                                  __fmaf_rn(float(.20480129041026129), b,
                                            float(12.226454707163354))));
  if (clamp) {
    values[0] = fmaxf(values[0], float(1.7557483643287353));
    values[1] = fmaxf(values[1], float(1.7557483643287353));
    values[2] = fmaxf(values[2], float(12.226454707163354));
  }
}

__global__ void opsin_dynamics(const float* rgb, const float* blurred, int n,
                               float intensity, float* output) {
  rgb = jxl_wp3::batch_ptr(rgb);
  blurred = jxl_wp3::batch_ptr(blurred);
  output = jxl_wp3::batch_ptr(output);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  float sensitivity[3], current[3];
  absorb(blurred[p] * intensity, blurred[n + p] * intensity,
         blurred[2 * n + p] * intensity, true, sensitivity);
  absorb(rgb[p] * intensity, rgb[n + p] * intensity, rgb[2 * n + p] * intensity,
         false, current);
  for (int c = 0; c < 3; ++c) {
    sensitivity[c] = fmaxf(sensitivity[c], 1e-4f);
    sensitivity[c] =
        fmaxf(__fdiv_rn(gamma(sensitivity[c]), sensitivity[c]), 1e-4f);
    current[c] =
        fmaxf(current[c] * sensitivity[c],
              c == 2 ? float(12.226454707163354) : float(1.7557483643287353));
  }
  output[p] = current[0] - current[1];
  output[n + p] = current[0] + current[1];
  output[2 * n + p] = current[2];
}

__device__ inline float remove_range(float v, float limit) {
  return v > limit ? v - limit : v < -limit ? v + limit : 0.f;
}
__device__ inline float amplify_range(float v, float limit) {
  return v > limit ? v + limit : v < -limit ? v - limit : v + v;
}
__device__ inline float maximum_clamp(float v, float limit) {
  constexpr float mul = float(.724216145665);
  return v < -limit   ? __fmaf_rn(v + limit, mul, -limit)
         : v >= limit ? __fmaf_rn(v - limit, mul, limit)
                      : v;
}

// PsychoImage packed order: LF[3], MF[3], HF[2], UHF[2].
__global__ void separate_lf(const float* xyb, int n, float* psycho) {
  xyb = jxl_wp3::batch_ptr(xyb);
  psycho = jxl_wp3::batch_ptr(psycho);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  for (int c = 0; c < 3; ++c)
    psycho[(3 + c) * n + p] = xyb[c * n + p] - psycho[c * n + p];
  const float x = psycho[p], y = psycho[n + p], b = psycho[2 * n + p];
  psycho[p] = x * float(33.832837186260);
  psycho[n + p] = y * float(14.458268100570);
  psycho[2 * n + p] =
      __fmaf_rn(float(-.362267051518), y, b) * float(49.87984651440);
}

__global__ void separate_mf(int n, float* psycho) {
  psycho = jxl_wp3::batch_ptr(psycho);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  float x = psycho[6 * n + p] - psycho[3 * n + p],
        y = psycho[7 * n + p] - psycho[4 * n + p];
  psycho[3 * n + p] = remove_range(psycho[3 * n + p], .29f);
  psycho[4 * n + p] = amplify_range(psycho[4 * n + p], .1f);
  const float scaler =
      __fmaf_rn(__fdiv_rn(46.f, __fmaf_rn(y, y, 46.f)),
                float(1.0 - .653020556257), float(.653020556257));
  psycho[6 * n + p] = x * scaler;
  psycho[7 * n + p] = y;
}

__global__ void separate_hf(int n, float* psycho) {
  psycho = jxl_wp3::batch_ptr(psycho);

  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  float hf = psycho[6 * n + p], uhf = psycho[8 * n + p] - hf;
  psycho[6 * n + p] = remove_range(hf, 1.5f);
  psycho[8 * n + p] = remove_range(uhf, .04f);
  hf = maximum_clamp(psycho[7 * n + p], float(28.4691806922));
  uhf = maximum_clamp(psycho[9 * n + p] - hf, float(5.19175294647));
  psycho[9 * n + p] = uhf * float(2.69313763794);
  psycho[7 * n + p] = amplify_range(hf * float(2.155), .132f);
}

inline cudaError_t separate_frequencies(const float* xyb, int w, int h,
                                        float* psycho, float* scratch,
                                        cudaStream_t stream, int batch) {
  const int n = w * h;
  cudaError_t code;
  const Kernel lf = kernel(float(7.15593339443)),
               mf = kernel(float(3.22489901262)),
               hf = kernel(float(1.56416327805));
  for (int c = 0; c < 3; ++c) {
    code = blur(xyb + c * n, w, h, lf, scratch, psycho + c * n, stream, batch);
    if (code != cudaSuccess) return code;
  }
  separate_lf<<<jxl_wp3::grid((n + 255) / 256, batch), 256, 0, stream>>>(
      xyb, n, psycho);
  code = cudaGetLastError();
  if (code != cudaSuccess) return code;
  jxl_wp3::
      copy_plane<<<jxl_wp3::grid((2 * n + 255) / 256, batch), 256, 0, stream>>>(
          psycho + 3 * n, psycho + 6 * n, 2 * n);
  code = cudaGetLastError();
  if (code != cudaSuccess) return code;
  for (int c = 0; c < 3; ++c) {
    code = blur(psycho + (3 + c) * n, w, h, mf, scratch, psycho + (3 + c) * n,
                stream, batch);
    if (code != cudaSuccess) return code;
  }
  separate_mf<<<jxl_wp3::grid((n + 255) / 256, batch), 256, 0, stream>>>(
      n, psycho);
  code = cudaGetLastError();
  if (code != cudaSuccess) return code;
  jxl_wp3::
      copy_plane<<<jxl_wp3::grid((2 * n + 255) / 256, batch), 256, 0, stream>>>(
          psycho + 6 * n, psycho + 8 * n, 2 * n);
  code = cudaGetLastError();
  if (code != cudaSuccess) return code;
  for (int c = 0; c < 2; ++c) {
    code = blur(psycho + (6 + c) * n, w, h, hf, scratch, psycho + (6 + c) * n,
                stream, batch);
    if (code != cudaSuccess) return code;
  }
  separate_hf<<<jxl_wp3::grid((n + 255) / 256, batch), 256, 0, stream>>>(
      n, psycho);
  return cudaGetLastError();
}
}  // namespace satellite_butteraugli

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Experimental device port of pinned e8ff097 TileDistMap. This standalone
// numerical component is not wired into the production encoder.
#pragma once
#include <cuda_runtime.h>

#include <cmath>

#include "batch.cuh"

namespace satellite_feedback {

// One lane owns each transform footprint. Sequential row-major accumulation
// deliberately preserves reference FP32 rounding before the double root.
__global__ void tile_distance(const float* diff, int width, int height,
                              const unsigned char* covered_x,
                              const unsigned char* covered_y, float* output,
                              int* error) {
  diff = jxl_wp3::batch_ptr(diff);
  covered_x = jxl_wp3::batch_ptr(covered_x);
  covered_y = jxl_wp3::batch_ptr(covered_y);
  output = jxl_wp3::batch_ptr(output);
  error = jxl_wp3::batch_ptr(error);

  const int bw = (width + 7) / 8, bh = (height + 7) / 8;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= bw * bh || covered_x[p] == 0) return;
  const int bx = p % bw, by = p / bw;
  const int cx = covered_x[p], cy = covered_y[p];
  if (cx > 32 || cy < 1 || cy > 32 || bx + cx > bw || by + cy > bh) {
    atomicExch(error, 1);
    return;
  }
  const int x_end = min(width, 8 * (bx + cx));
  const int y_end = min(height, 8 * (by + cy));
  float sum = 0.f;
  int count = 0;
  for (int y = by * 8; y < y_end; ++y) {
    for (int x = bx * 8; x < x_end; ++x) {
      float value = diff[y * width + x];
      if (!isfinite(value) || value < 0.f) {
        atomicExch(error, 2);
        return;
      }
      value = __fmul_rn(value, value);
      value = __fmul_rn(value, value);
      value = __fmul_rn(value, value);
      value = __fmul_rn(value, value);
      sum = __fadd_rn(sum, value);
      ++count;
    }
  }
  const double mean = __ddiv_rn(double(sum), double(count ? count : 1));
  const float distance =
      __double2float_rn(__dmul_rn(double(1.2f), pow(mean, 1.0 / 16.0)));
  if (!isfinite(distance)) {
    atomicExch(error, 3);
    return;
  }
  for (int y = 0; y < cy; ++y)
    for (int x = 0; x < cx; ++x) output[(by + y) * bw + bx + x] = distance;
}

}  // namespace satellite_feedback

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cuda_runtime.h>

#include "spike_format.h"
namespace jxl_wp3 {
constexpr size_t kStrideBytes = 47ull * 1024 * 1024;
struct Tile {
  Input input;
  float original[3 * kPixels], decoded[3 * kPixels], diff[kPixels];
  float reference[12 * kPixels], reference_half[3 * kPixels];
  float original_half[3 * kPixels / 4], decoded_half[3 * kPixels / 4],
      diff_half[kPixels / 4];
  float dist[kBlocks];
  int error;
  float scratch[17 * kPixels];
};
static_assert(sizeof(Tile) <= kStrideBytes, "arena cap");
template <class T>
__device__ inline T* batch_ptr(T* ptr) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) +
                              blockIdx.z * kStrideBytes);
}
template <class T>
__device__ inline const T* batch_ptr(const T* ptr) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) +
                                    blockIdx.z * kStrideBytes);
}
inline dim3 grid(int blocks, int batch) { return dim3(blocks, 1, batch); }
__global__ void copy_plane(const float* src, float* dst, int n) {
  src = batch_ptr(src);
  dst = batch_ptr(dst);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
__global__ void zero_plane(float* dst, int n) {
  dst = batch_ptr(dst);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = 0;
}
}  // namespace jxl_wp3

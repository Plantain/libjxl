// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "afv_basis.cuh"
#include "reconstruction.cuh"
namespace satellite_reconstruction_gpu {
template <int H, int W>
__device__ void inverse2(const float* a, float* out, int stride) {
  float tmp[H * W], v[W > H ? W : H];
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) v[x] = a[W <= H ? x * H + y : y * W + x];
    idct<W>(v);
    for (int x = 0; x < W; ++x) tmp[y * W + x] = v[x];
  }
  for (int x = 0; x < W; ++x) {
    for (int y = 0; y < H; ++y) v[y] = tmp[y * W + x];
    idct<H>(v);
    for (int y = 0; y < H; ++y) out[y * stride + x] = v[y];
  }
}
__global__ void inverse_small(const float* input, const uint8_t* kind,
                              const uint8_t* first, float* output) {
  input = jxl_wp3::batch_ptr(input);
  kind = jxl_wp3::batch_ptr(kind);
  first = jxl_wp3::batch_ptr(first);
  output = jxl_wp3::batch_ptr(output);
  int anchor = blockIdx.x, c = blockIdx.y, t = kind[anchor];
  if (threadIdx.x || !first[anchor] || t == 2 || supported_kind(t)) return;
  float a[64], out[64], block[32];
  int bx = anchor % 64 * 8, by = anchor / 64 * 8;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      a[y * 8 + x] = input[c * pixels + (by + y) * 512 + bx + x];
  if (t == 1 || t == 3) {
    float dcs[4] = {
        ((a[0] + a[1]) + a[8]) + a[9], ((a[0] + a[1]) - a[8]) - a[9],
        ((a[0] - a[1]) + a[8]) - a[9], ((a[0] - a[1]) - a[8]) + a[9]};
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        if (t == 1) {
          float sum = 0;
          for (int iy = 0; iy < 4; ++iy)
            for (int ix = 0; ix < 4; ++ix)
              if (ix || iy) sum += a[(y + iy * 2) * 8 + x + ix * 2];
          out[(4 * y + 1) * 8 + 4 * x + 1] = dcs[y * 2 + x] - sum * (1.f / 16);
          for (int iy = 0; iy < 4; ++iy)
            for (int ix = 0; ix < 4; ++ix)
              if (ix != 1 || iy != 1)
                out[(y * 4 + iy) * 8 + x * 4 + ix] =
                    a[(y + iy * 2) * 8 + x + ix * 2] +
                    out[(4 * y + 1) * 8 + 4 * x + 1];
          out[y * 32 + x * 4] =
              a[(y + 2) * 8 + x + 2] + out[(4 * y + 1) * 8 + 4 * x + 1];
        } else {
          block[0] = dcs[y * 2 + x];
          for (int iy = 0; iy < 4; ++iy)
            for (int ix = 0; ix < 4; ++ix)
              if (ix || iy)
                block[iy * 4 + ix] = a[(y + iy * 2) * 8 + x + ix * 2];
          inverse2<4, 4>(block, out + y * 32 + x * 4, 8);
        }
      }
  } else if (t == 12 || t == 13) {
    for (int half = 0; half < 2; ++half) {
      block[0] = half ? a[0] - a[8] : a[0] + a[8];
      for (int iy = 0; iy < 4; ++iy)
        for (int ix = 0; ix < 8; ++ix)
          if (ix || iy) block[iy * 8 + ix] = a[(half + iy * 2) * 8 + ix];
      if (t == 12)
        inverse2<4, 8>(block, out + half * 32, 8);
      else
        inverse2<8, 4>(block, out + half * 4, 8);
    }
  } else if (t >= 14 && t <= 17) {
    int ax = (t - 14) & 1, ay = (t - 14) / 2;
    float cf[16];
    cf[0] = (a[0] + a[8] + a[1]) * 4.f;
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        if (x || y) cf[y * 4 + x] = a[y * 16 + x * 2];
    for (int i = 0; i < 16; ++i) {
      float v = 0;
      for (int j = 0; j < 16; ++j) v = fmaf(cf[j], kAFV[j][i], v);
      block[i] = v;
    }
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        out[(y + ay * 4) * 8 + ax * 4 + x] =
            block[(ay ? 3 - y : y) * 4 + (ax ? 3 - x : x)];
    block[0] = a[0] + a[8] - a[1];
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        if (x || y) block[y * 4 + x] = a[y * 16 + x * 2 + 1];
    inverse2<4, 4>(block, out + ay * 32 + (ax ? 0 : 4), 8);
    block[0] = a[0] - a[8];
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 8; ++x)
        if (x || y) block[y * 8 + x] = a[(1 + y * 2) * 8 + x];
    inverse2<4, 8>(block, out + (ay ? 0 : 32), 8);
  }
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x)
      output[c * pixels + (by + y) * 512 + bx + x] = out[y * 8 + x];
}
}  // namespace satellite_reconstruction_gpu

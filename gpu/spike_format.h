// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
namespace jxl_wp3 {
constexpr int kPixels = 512 * 512, kBlocks = 64 * 64;
// A private, versioned, little-endian spike format; not the I1/I3 API.
struct Params {
  float gab[6], channel[3], sharp[8], border, epf_quant, scale, inverse;
  float biases[4], opsin_bias[3], opsin_cbrt[3], matrix[9];
  float dc_mul[3], dc_cfl[3], dc_precision;
  int32_t skip_dc_smoothing;
  float xmul, bmul;
  uint32_t gab_enabled, epf_iters, global_scale, quant_dc;
  float intensity, distance;
};
struct Input {
  uint32_t magic = 0x3350574a, version = 1;
  Params p{};
  int32_t ac[3*kPixels]{}, dc[3*kBlocks]{}, quant[kBlocks]{};
  uint8_t kind[kBlocks]{}, first[kBlocks]{}, cx[kBlocks]{}, cy[kBlocks]{}, sharp[kBlocks]{};
  float cfl[2*kBlocks]{}, matrices[3*kPixels]{};
};
struct Oracle {
  float original[3*kPixels], linear[3*kPixels], diff[kPixels], dist[kBlocks];
};
static_assert(sizeof(Params) == 224, "bump dump version if Params changes");
static_assert(sizeof(Input) == 6410472, "bump dump version if Input changes");
static_assert(sizeof(Oracle) == 7356416, "bump dump version if Oracle changes");
// Capture during single-threaded dump encodes. No production API or ABI.
inline int32_t* CapturedDC() { static int32_t dc[3*kBlocks]; return dc; }
inline float& DCPrecision() { static float p; return p; }
}

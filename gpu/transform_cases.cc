// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Independent pinned CPU transform oracle, including types absent in the
// corpus.
#include <algorithm>
#include <cstdint>
#include <hwy/highway.h>
#include <vector>

#include "lib/jxl/dec_transforms-inl.h"
#include "spike_format.h"
extern "C" void Wp3TransformCase(int kind, jxl_wp3::Input* in,
                                 float* expected) {
  using namespace jxl_wp3;
  const auto acs = jxl::AcStrategy::FromRawStrategy(kind);
  const int w = 8 * acs.covered_blocks_x(), h = 8 * acs.covered_blocks_y();
  uint32_t rng = 0x23081995;
  alignas(64) float coeff[3 * 64 * 64], scratch[4 * 64 * 64];
  for (int y = 0; y < 512; y += h)
    for (int x = 0; x < 512; x += w) {
      const int anchor = (y / 8) * 64 + x / 8;
      in->kind[anchor] = kind;
      in->first[anchor] = 1;
      for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < w * h; ++k) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          coeff[k] = float(int(rng % 2001) - 1000) * .000001f;
          if (k == 0) coeff[k] = .3f;
          in->matrices[c * kPixels + (y + k / w) * 512 + x + k % w] = coeff[k];
        }
        jxl::HWY_NAMESPACE::TransformToPixels(
            acs.Strategy(), coeff, expected + c * kPixels + y * 512 + x, 512,
            scratch);
      }
    }
}

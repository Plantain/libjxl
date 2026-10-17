// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <cmath>
#include <cstdint>
#include <limits>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/hybrid/internal.h"

namespace jxl {
namespace hybrid {
namespace {
bool QuantizerValid(int32_t scale, int32_t dc) {
  return scale >= 1 && scale <= 73728 && dc >= 1 && dc <= 65536;
}

bool FloatsValid(const float* p, size_t size, float low, float high) {
  if (!p) return false;
  for (size_t i = 0; i < size; ++i) {
    if (!std::isfinite(p[i]) || p[i] < low || p[i] > high) return false;
  }
  return true;
}

bool ColorValid(const JxlHybridColorMap& c) {
  return c.ytox_map && c.ytob_map && c.ytox_dc >= -128 && c.ytox_dc <= 127 &&
         c.ytob_dc >= -128 && c.ytob_dc <= 127 && c.color_factor >= 2 &&
         c.color_factor <= 257 && std::isfinite(c.base_correlation_x) &&
         std::isfinite(c.base_correlation_b) &&
         std::abs(c.base_correlation_x) <= 4 &&
         std::abs(c.base_correlation_b) <= 4;
}

bool StrategiesValid(const uint8_t* p) {
  if (!p) return false;
  bool covered[JXL_HYBRID_BLOCKS] = {};
  for (size_t y = 0; y < 64; ++y) {
    for (size_t x = 0; x < 64; ++x) {
      const size_t i = y * 64 + x;
      if (p[i] == 255) {
        if (!covered[i]) return false;  // hole
        continue;
      }
      // The e8 seam supports transforms through 64x64. They must fit one
      // 64x64 color region, as required by CfL and the stage-override tiling.
      if (p[i] > static_cast<uint8_t>(AcStrategyType::DCT32X64) || covered[i]) {
        return false;
      }
      const AcStrategy acs = AcStrategy::FromRawStrategy(p[i]);
      const size_t w = acs.covered_blocks_x(), h = acs.covered_blocks_y();
      if (x % 8 + w > 8 || y % 8 + h > 8) return false;
      for (size_t dy = 0; dy < h; ++dy) {
        for (size_t dx = 0; dx < w; ++dx) {
          const size_t j = (y + dy) * 64 + x + dx;
          if (covered[j] || ((dx || dy) && p[j] != 255)) return false;
          covered[j] = true;
        }
      }
    }
  }
  return true;
}

bool EpfValid(const uint8_t* p) {
  if (!p) return false;
  for (size_t i = 0; i < JXL_HYBRID_BLOCKS; ++i) {
    if (p[i] > 7) return false;
  }
  return true;
}
}  // namespace

// strategies, when present, must already have passed StrategiesValid.
bool ValidateQuant(const int32_t* q, const uint8_t* strategies) {
  if (!q) return false;
  for (size_t i = 0; i < JXL_HYBRID_BLOCKS; ++i) {
    if (q[i] < 1 || q[i] > 256) return false;
    if (!strategies || strategies[i] == 255) continue;
    const AcStrategy acs = AcStrategy::FromRawStrategy(strategies[i]);
    for (size_t dy = 0; dy < acs.covered_blocks_y(); ++dy) {
      for (size_t dx = 0; dx < acs.covered_blocks_x(); ++dx) {
        if (q[i + dy * 64 + dx] != q[i]) return false;
      }
    }
  }
  return true;
}

bool ValidateInput(const JxlHybridTileInput& in) {
  if (!in.rgb || in.stride < 512 * 3 ||
      in.stride > (std::numeric_limits<size_t>::max() - 512 * 3) / 511) {
    return false;
  }
  if (in.decisions) {
    const auto& d = *in.decisions;
    JxlHybridColorMap c{
        d.ytox_map,     d.ytob_map,           d.ytox_dc,           d.ytob_dc,
        d.color_factor, d.base_correlation_x, d.base_correlation_b};
    if (d.xsize_blocks != 64 || d.ysize_blocks != 64 ||
        !QuantizerValid(d.global_scale, d.quant_dc) ||
        !StrategiesValid(d.ac_strategy) ||
        !ValidateQuant(d.raw_quant_field, d.ac_strategy) || !ColorValid(c) ||
        !EpfValid(d.epf_sharpness) || d.x_qm_scale > 7 || d.b_qm_scale > 7)
      return false;
    if (in.overrides && in.overrides->stages) return false;
  }
  if (in.opsin[0] || in.opsin[1] || in.opsin[2]) {
    if (!in.decisions) return false;
    for (size_t c = 0; c < 3; ++c) {
      if (!FloatsValid(in.opsin[c], 512 * 512, -16, 16)) return false;
    }
  }
  if (!in.overrides) return true;
  const auto& o = *in.overrides;
  if (o.stages & ~JXL_HYBRID_ALL_STAGES) return false;
  if (o.stages & JXL_HYBRID_STAGE(1)) {
    if (!o.s1 || !FloatsValid(o.s1->quant_field, 4096, 1e-8f, 1e4f) ||
        !FloatsValid(o.s1->masking, 4096, 0, 1e4f) ||
        !FloatsValid(o.s1->masking1x1, 512 * 512, 0, 1e4f))
      return false;
  }
  if ((o.stages & JXL_HYBRID_STAGE(2)) && (!o.s2 || !ColorValid(*o.s2)))
    return false;
  const uint8_t* strategies = nullptr;
  if (o.stages & JXL_HYBRID_STAGE(3)) {
    if (!StrategiesValid(o.s3)) return false;
    strategies = o.s3;
  }
  if (o.stages & JXL_HYBRID_STAGE(4)) {
    if (!o.s4 || !FloatsValid(o.s4->quant_field, 4096, 1e-8f, 1e4f) ||
        !QuantizerValid(o.s4->global_scale, o.s4->quant_dc) ||
        !ValidateQuant(o.s4->raw_quant_field, strategies) ||
        !ColorValid(o.s4->cmap))
      return false;
  }
  if (o.stages & JXL_HYBRID_STAGE(5)) {
    if (!o.s5 || !QuantizerValid(o.s5->global_scale, o.s5->quant_dc) ||
        !ValidateQuant(o.s5->raw_quant_field, strategies))
      return false;
  }
  if ((o.stages & JXL_HYBRID_STAGE(6)) && !EpfValid(o.s6)) return false;
  return true;
}
}  // namespace hybrid
}  // namespace jxl

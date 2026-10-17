// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <algorithm>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_gaborish.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/hybrid/internal.h"
#include "lib/jxl/image_ops.h"

namespace jxl {
namespace hybrid {
namespace {
template <typename T>
void Load(const T* packed, size_t stride, const Rect& rect, Plane<T>* plane) {
  for (size_t y = rect.y0(); y < rect.y1(); ++y) {
    std::copy_n(packed + y * stride + rect.x0(), rect.xsize(),
                plane->Row(y) + rect.x0());
  }
}

template <typename T>
void Save(const Plane<T>& plane, const Rect& rect, size_t stride, T* packed) {
  for (size_t y = rect.y0(); y < rect.y1(); ++y) {
    std::copy_n(plane.Row(y) + rect.x0(), rect.xsize(),
                packed + y * stride + rect.x0());
  }
}

void LoadColor(const JxlHybridColorMap& in, const Rect& r,
               ColorCorrelationMap* out) {
  Rect cr(r.x0() / 8, r.y0() / 8, r.xsize() / 8, r.ysize() / 8);
  Load(in.ytox_map, 8, cr, &out->ytox_map);
  Load(in.ytob_map, 8, cr, &out->ytob_map);
  auto& base = out->mutable_base();
  base.SetBaseCorrelation(in.base_correlation_x, in.base_correlation_b);
  base.SetColorFactor(in.color_factor);
  base.SetYToXDC(in.ytox_dc);
  base.SetYToBDC(in.ytob_dc);
}

void SaveColor(const ColorCorrelationMap& in, const Rect& r,
               JxlHybridColorMap* out) {
  Rect cr(r.x0() / 8, r.y0() / 8, r.xsize() / 8, r.ysize() / 8);
  Save(in.ytox_map, cr, 8, out->ytox_map);
  Save(in.ytob_map, cr, 8, out->ytob_map);
  const auto& base = in.base();
  out->base_correlation_x = base.GetBaseCorrelationX();
  out->base_correlation_b = base.GetBaseCorrelationB();
  out->color_factor = base.GetColorFactor();
  out->ytox_dc = base.GetYToXDC();
  out->ytob_dc = base.GetYToBDC();
}

Status LoadStrategies(const uint8_t* in, const Rect& r, AcStrategyImage* out) {
  for (size_t y = r.y0(); y < r.y1(); ++y) {
    for (size_t x = r.x0(); x < r.x1(); ++x) {
      if (in[y * 64 + x] != 255) {
        JXL_RETURN_IF_ERROR(
            out->Set(x, y, static_cast<AcStrategyType>(in[y * 64 + x])));
      }
    }
  }
  return true;
}

void SaveStrategies(const AcStrategyImage& in, const Rect& r, uint8_t* out) {
  for (size_t y = r.y0(); y < r.y1(); ++y) {
    auto row = in.ConstRow(y);
    for (size_t x = r.x0(); x < r.x1(); ++x) {
      out[y * 64 + x] = row[x].IsFirstBlock() ? row[x].RawStrategy() : 255;
    }
  }
}

// Cross-stage validation against strategies actually selected by the CPU.
Status CheckQuantRect(const ImageI& q, const AcStrategyImage& acs,
                      const Rect& r) {
  for (size_t y = r.y0(); y < r.y1(); ++y) {
    auto row = acs.ConstRow(y);
    for (size_t x = r.x0(); x < r.x1(); ++x) {
      if (!row[x].IsFirstBlock()) continue;
      for (size_t dy = 0; dy < row[x].covered_blocks_y(); ++dy) {
        for (size_t dx = 0; dx < row[x].covered_blocks_x(); ++dx) {
          if (q.Row(y)[x] != q.Row(y + dy)[x + dx]) {
            return JXL_FAILURE(
                "Hybrid quant field is not constant within varblock");
          }
        }
      }
    }
  }
  return true;
}
}  // namespace

Status TileHooks::LoadOpsin(Image3F* opsin) {
  for (size_t c = 0; c < 3; ++c) {
    Load(input_.opsin[c], 512, Rect(*opsin), &opsin->Plane(c));
  }
  return true;
}

Status TileHooks::OnOpsin(const Image3F& opsin) {
  if (dump_) {
    for (size_t c = 0; c < 3; ++c) {
      Save(opsin.Plane(c), Rect(opsin), 512, dump_->opsin[c]);
    }
  }
  return true;
}

Status TileHooks::OnFrameHeader(FrameHeader* header) {
  if (input_.decisions) {
    header->x_qm_scale = input_.decisions->x_qm_scale;
    header->b_qm_scale = input_.decisions->b_qm_scale;
  }
  if (dump_) {
    dump_->decisions.x_qm_scale = header->x_qm_scale;
    dump_->decisions.b_qm_scale = header->b_qm_scale;
  }
  return true;
}

Status TileHooks::Replay(const FrameHeader& header, PassesEncoderState* state,
                         Image3F* opsin, const Rect& rect, ThreadPool* pool) {
  if (!HasOpsin() && header.loop_filter.gab) {
    const float weight[3] = {1, 1, 1};
    JXL_RETURN_IF_ERROR(GaborishInverse(opsin, rect, weight, pool));
  }
  auto& s = state->shared;
  const Rect blocks(s.raw_quant_field);
  const uint8_t* acs;
  if (input_.decisions) {
    const auto& d = *input_.decisions;
    acs = d.ac_strategy;
    Load(d.raw_quant_field, 64, blocks, &s.raw_quant_field);
    s.quantizer.SetRawParams(d.global_scale, d.quant_dc);
    JxlHybridColorMap c{
        d.ytox_map,     d.ytob_map,           d.ytox_dc,           d.ytob_dc,
        d.color_factor, d.base_correlation_x, d.base_correlation_b};
    LoadColor(c, blocks, &s.cmap);
  } else {
    const auto& o = *input_.overrides;
    acs = o.s3;
    Load(o.s5->raw_quant_field, 64, blocks, &s.raw_quant_field);
    s.quantizer.SetRawParams(o.s5->global_scale, o.s5->quant_dc);
    LoadColor(o.s4->cmap, blocks, &s.cmap);
  }
  JXL_RETURN_IF_ERROR(LoadStrategies(acs, blocks, &s.ac_strategy));
  uint32_t used = 0;
  for (size_t i = 0; i < JXL_HYBRID_BLOCKS; ++i) {
    if (acs[i] != 255) used |= uint32_t{1} << acs[i];
  }
  // Default dequant matrices are already initialized by PassesSharedState.
  JXL_RETURN_IF_ERROR(s.matrices.EnsureComputed(state->memory_manager(), used));
  FillImage(static_cast<uint8_t>(4), &s.epf_sharpness);
  FindBestBlockEntropyModel(state->cparams, s.raw_quant_field, s.ac_strategy,
                            &s.block_ctx_map);
  return true;
}

Status TileHooks::Stage(unsigned stage, PassesEncoderState* state,
                        const Rect& r, ImageF* quant_field, ImageF* masking) {
  auto& s = state->shared;
  const auto* o = input_.overrides;
  const bool replace = SkipStage(stage);
  switch (stage) {
    case 1:
      if (replace) {
        JXL_ASSIGN_OR_RETURN(*quant_field,
                             ImageF::Create(state->memory_manager(), 64, 64));
        JXL_ASSIGN_OR_RETURN(*masking,
                             ImageF::Create(state->memory_manager(), 64, 64));
        JXL_ASSIGN_OR_RETURN(state->initial_quant_masking1x1,
                             ImageF::Create(state->memory_manager(), 512, 512));
        Load(o->s1->quant_field, 64, r, quant_field);
        Load(o->s1->masking, 64, r, masking);
        Load(o->s1->masking1x1, 512, Rect(0, 0, 512, 512),
             &state->initial_quant_masking1x1);
      }
      if (dump_) {
        Save(*quant_field, r, 64, dump_->s1.quant_field);
        Save(*masking, r, 64, dump_->s1.masking);
        Save(state->initial_quant_masking1x1, Rect(0, 0, 512, 512), 512,
             dump_->s1.masking1x1);
      }
      break;
    case 2:
      if (replace) LoadColor(*o->s2, r, &s.cmap);
      if (dump_) SaveColor(s.cmap, r, &dump_->s2);
      break;
    case 3:
      if (replace)
        JXL_RETURN_IF_ERROR(LoadStrategies(o->s3, r, &s.ac_strategy));
      if (dump_) SaveStrategies(s.ac_strategy, r, dump_->s3);
      break;
    case 4:
      if (replace) {
        Load(o->s4->quant_field, 64, r, quant_field);
        Load(o->s4->raw_quant_field, 64, r, &s.raw_quant_field);
        s.quantizer.SetRawParams(o->s4->global_scale, o->s4->quant_dc);
        LoadColor(o->s4->cmap, r, &s.cmap);
        JXL_RETURN_IF_ERROR(
            CheckQuantRect(s.raw_quant_field, s.ac_strategy, r));
      }
      if (dump_) {
        Save(*quant_field, r, 64, dump_->s4.quant_field);
        Save(s.raw_quant_field, r, 64, dump_->s4.raw_quant_field);
        const auto params = s.quantizer.GetParams();
        dump_->s4.global_scale = params.global_scale;
        dump_->s4.quant_dc = params.quant_dc;
        SaveColor(s.cmap, r, &dump_->s4.cmap);
      }
      break;
    case 5:
      if (replace) {
        Load(o->s5->raw_quant_field, 64, r, &s.raw_quant_field);
        s.quantizer.SetRawParams(o->s5->global_scale, o->s5->quant_dc);
        FillImage(static_cast<uint8_t>(4), &s.epf_sharpness);
        JXL_RETURN_IF_ERROR(
            CheckQuantRect(s.raw_quant_field, s.ac_strategy, r));
      }
      // I1 does not carry patches or splines. Fail explicitly instead of
      // silently making a decision dump that cannot reproduce the codestream.
      if (s.image_features.patches.HasAny() ||
          s.image_features.splines.HasAny()) {
        return JXL_FAILURE("Hybrid seam cannot represent patches or splines");
      }
      if (dump_) {
        Save(s.raw_quant_field, r, 64, dump_->s5.raw_quant_field);
        const auto params = s.quantizer.GetParams();
        dump_->s5.global_scale = dump_->decisions.global_scale =
            params.global_scale;
        dump_->s5.quant_dc = dump_->decisions.quant_dc = params.quant_dc;
        auto& d = dump_->decisions;
        const auto& c = dump_->s4.cmap;
        d.ytox_dc = c.ytox_dc;
        d.ytob_dc = c.ytob_dc;
        d.color_factor = c.color_factor;
        d.base_correlation_x = c.base_correlation_x;
        d.base_correlation_b = c.base_correlation_b;
      }
      break;
    case 6:
      if (replace) {
        const uint8_t* epf =
            input_.decisions ? input_.decisions->epf_sharpness : o->s6;
        Load(epf, 64, r, &s.epf_sharpness);
      }
      if (dump_) Save(s.epf_sharpness, r, 64, dump_->s6);
      break;
    default:
      return JXL_FAILURE("Unknown hybrid stage");
  }
  return true;
}
}  // namespace hybrid
}  // namespace jxl

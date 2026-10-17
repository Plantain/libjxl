// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef LIB_JXL_HYBRID_INTERNAL_H_
#define LIB_JXL_HYBRID_INTERNAL_H_

#include <jxl/hybrid.h>

#include "lib/jxl/hybrid/hooks.h"

namespace jxl {
namespace hybrid {
bool ValidateInput(const JxlHybridTileInput& input);
bool ValidateQuant(const int32_t* q, const uint8_t* strategies = nullptr);

class TileHooks final : public HybridHooks {
 public:
  TileHooks(const JxlHybridTileInput& input, JxlHybridStageDump* dump)
      : input_(input), dump_(dump) {}
  bool FullReplay() const override {
    return input_.decisions || (input_.overrides && input_.overrides->stages ==
                                                        JXL_HYBRID_ALL_STAGES);
  }
  bool HasFrameHeader() const override { return input_.decisions != nullptr; }
  bool SkipStage(unsigned stage) const override {
    return input_.decisions || (input_.overrides && (input_.overrides->stages &
                                                     JXL_HYBRID_STAGE(stage)));
  }
  bool HasOpsin() const override { return input_.opsin[0] != nullptr; }
  Status LoadOpsin(Image3F* opsin) override;
  Status OnOpsin(const Image3F& opsin) override;
  Status OnFrameHeader(FrameHeader* header) override;
  Status Replay(const FrameHeader& header, PassesEncoderState* state,
                Image3F* opsin, const Rect& rect, ThreadPool* pool) override;
  Status Stage(unsigned stage, PassesEncoderState* state, const Rect& rect,
               ImageF* quant_field, ImageF* masking) override;

 private:
  const JxlHybridTileInput& input_;
  JxlHybridStageDump* dump_;
};
}  // namespace hybrid
}  // namespace jxl
#endif  // LIB_JXL_HYBRID_INTERNAL_H_

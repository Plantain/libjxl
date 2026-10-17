// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef LIB_JXL_HYBRID_HOOKS_H_
#define LIB_JXL_HYBRID_HOOKS_H_

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"

namespace jxl {
struct PassesEncoderState;

// No dependency on the optional hybrid library: normal libjxl pays only the
// null checks. Hook implementations live in lib/jxl/hybrid/.
class HybridHooks {
 public:
  virtual ~HybridHooks() = default;
  virtual bool FullReplay() const = 0;
  virtual bool HasFrameHeader() const = 0;
  virtual bool SkipStage(unsigned stage) const = 0;
  virtual bool HasOpsin() const = 0;
  virtual Status LoadOpsin(Image3F* opsin) = 0;
  virtual Status OnOpsin(const Image3F& opsin) = 0;
  virtual Status OnFrameHeader(FrameHeader* header) = 0;
  virtual Status Replay(const FrameHeader& header, PassesEncoderState* state,
                        Image3F* opsin, const Rect& rect, ThreadPool* pool) = 0;
  virtual Status Stage(unsigned stage, PassesEncoderState* state,
                       const Rect& rect, ImageF* quant_field = nullptr,
                       ImageF* masking = nullptr) = 0;
};

// FindBestQuantizer and reconstruction read state->cparams themselves. Change
// it only for the duration of a heuristic; backend calls see the original tier.
class HybridHeuristicScope {
 public:
  explicit HybridHeuristicScope(CompressParams* params)
      : params_(params), saved_(params->speed_tier) {
    if (params->hybrid_hooks) params->speed_tier = SpeedTier::kKitten;
  }
  ~HybridHeuristicScope() { Restore(); }
  void Restore() { params_->speed_tier = saved_; }

 private:
  CompressParams* params_;
  SpeedTier saved_;
};
}  // namespace jxl
#endif  // LIB_JXL_HYBRID_HOOKS_H_

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/hybrid.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "lib/jxl/encode_internal.h"
#include "lib/jxl/hybrid/internal.h"

namespace {
struct DumpStorage {
  JxlHybridStageDump view{};
  std::vector<float> floats = std::vector<float>(4 * 512 * 512 + 3 * 4096);
  std::vector<int32_t> quant = std::vector<int32_t>(2 * 4096);
  std::vector<uint8_t> blocks = std::vector<uint8_t>(2 * 4096);
  std::vector<int8_t> colors = std::vector<int8_t>(4 * 64);
  std::vector<uint8_t> codestream;
  DumpStorage() {
    auto& d = view;
    d.s1.quant_field = floats.data();
    d.s1.masking = d.s1.quant_field + 4096;
    d.s4.quant_field = d.s1.masking + 4096;
    d.s1.masking1x1 = d.s4.quant_field + 4096;
    for (size_t c = 0; c < 3; ++c)
      d.opsin[c] = d.s1.masking1x1 + (c + 1) * 512 * 512;
    d.s4.raw_quant_field = quant.data();
    d.s5.raw_quant_field = quant.data() + 4096;
    d.s3 = blocks.data();
    d.s6 = blocks.data() + 4096;
    d.s2.ytox_map = colors.data();
    d.s2.ytob_map = colors.data() + 64;
    d.s4.cmap.ytox_map = colors.data() + 128;
    d.s4.cmap.ytob_map = colors.data() + 192;
    d.decisions.xsize_blocks = d.decisions.ysize_blocks = 64;
    d.decisions.raw_quant_field = d.s5.raw_quant_field;
    d.decisions.ac_strategy = d.s3;
    d.decisions.ytox_map = d.s4.cmap.ytox_map;
    d.decisions.ytob_map = d.s4.cmap.ytob_map;
    d.decisions.epf_sharpness = d.s6;
    d.opaque = this;
  }
};
struct Worker {
  JxlEncoderPtr encoder = JxlEncoderMake(nullptr);
  std::vector<uint8_t> output;
};

int Encode(Worker& worker, const JxlHybridOptions& options,
           const JxlHybridTileInput& input, JxlHybridStageDump* dump) {
  if (!jxl::hybrid::ValidateInput(input)) return JXL_HYBRID_INVALID;
  JxlEncoder* enc = worker.encoder.get();
  if (!enc) return JXL_HYBRID_ENCODE_ERROR;
  JxlEncoderReset(enc);
  jxl::hybrid::TileHooks hooks(input, dump);
  if (JxlEncoderUseContainer(enc, JXL_FALSE) != JXL_ENC_SUCCESS)
    return JXL_HYBRID_ENCODE_ERROR;
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = info.ysize = 512;
  info.bits_per_sample = 8;
  info.num_color_channels = 3;
  info.uses_original_profile = JXL_FALSE;
  if (JxlEncoderSetBasicInfo(enc, &info) != JXL_ENC_SUCCESS)
    return JXL_HYBRID_ENCODE_ERROR;
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  if (JxlEncoderSetColorEncoding(enc, &color) != JXL_ENC_SUCCESS)
    return JXL_HYBRID_ENCODE_ERROR;
  JxlEncoderFrameSettings* settings =
      JxlEncoderFrameSettingsCreate(enc, nullptr);
  if (!settings ||
      JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT,
                                       options.effort) != JXL_ENC_SUCCESS ||
      JxlEncoderSetFrameDistance(settings, options.distance) !=
          JXL_ENC_SUCCESS) {
    return JXL_HYBRID_ENCODE_ERROR;
  }
  settings->values.cparams.hybrid_hooks = &hooks;
  const JxlPixelFormat format{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN,
                              input.stride};
  const size_t input_size = 511 * input.stride + 512 * 3;
  if (JxlEncoderAddImageFrame(settings, &format, input.rgb, input_size) !=
      JXL_ENC_SUCCESS) {
    return JXL_HYBRID_ENCODE_ERROR;
  }
  JxlEncoderCloseInput(enc);
  auto& out = worker.output;
  out.resize(64 * 1024);
  uint8_t* next = out.data();
  size_t avail = out.size();
  for (;;) {
    const auto status = JxlEncoderProcessOutput(enc, &next, &avail);
    if (status == JXL_ENC_SUCCESS) break;
    if (status != JXL_ENC_NEED_MORE_OUTPUT) return JXL_HYBRID_ENCODE_ERROR;
    const size_t offset = next - out.data();
    out.resize(out.size() * 2);
    next = out.data() + offset;
    avail = out.size() - offset;
  }
  out.resize(next - out.data());
  return JXL_HYBRID_OK;
}

int EncodeOutput(Worker& worker, const JxlHybridOptions& options,
                 const JxlHybridTileInput& input, JxlHybridOutput* output) {
  *output = {};
  try {
    output->status = Encode(worker, options, input, nullptr);
    if (output->status) return output->status;
    output->data = static_cast<uint8_t*>(malloc(worker.output.size()));
    if (!output->data) return output->status = JXL_HYBRID_ENCODE_ERROR;
    output->size = worker.output.size();
    memcpy(output->data, worker.output.data(), output->size);
  } catch (...) {
    output->status = JXL_HYBRID_ENCODE_ERROR;
  }
  return output->status;
}
}  // namespace

struct JxlHybridEncoder {
  JxlHybridOptions options;
  std::mutex api_mutex;
  std::vector<Worker> contexts;
  std::vector<std::thread> threads;
  std::mutex mutex;
  std::condition_variable ready, finished;
  bool stop = false;
  size_t generation = 0, remaining = 0, count = 0;
  std::atomic<size_t> next{0};
  const JxlHybridTileInput* inputs = nullptr;
  JxlHybridOutput* outputs = nullptr;
  explicit JxlHybridEncoder(JxlHybridOptions o)
      : options(o), contexts(o.threads) {}
  void Start() {
    // Start after construction so partial startup is safely destructible.
    if (options.threads == 1) return;
    for (size_t t = 0; t < options.threads; ++t) {
      threads.emplace_back([this, t] {
        size_t seen = 0;
        for (;;) {
          std::unique_lock<std::mutex> lock(mutex);
          ready.wait(lock, [&] { return stop || generation != seen; });
          if (stop) return;
          seen = generation;
          lock.unlock();
          for (;;) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= count) break;
            EncodeOutput(contexts[t], options, inputs[i], &outputs[i]);
          }
          lock.lock();
          if (--remaining == 0) finished.notify_one();
        }
      });
    }
  }
  ~JxlHybridEncoder() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    ready.notify_all();
    for (auto& t : threads) t.join();
  }
};

extern "C" {
void JxlHybridDefaultOptions(JxlHybridOptions* options) {
  if (options) *options = {1.2f, 7, 16};
}
JxlHybridEncoder* JxlHybridEncoderCreate(const JxlHybridOptions* options) {
  JxlHybridOptions o;
  JxlHybridDefaultOptions(&o);
  if (options) o = *options;
  if (!std::isfinite(o.distance) || o.distance < 0.1f || o.distance > 25 ||
      o.effort < 7 || o.effort > 8 || o.threads < 1 || o.threads > 16)
    return nullptr;
  try {
    std::unique_ptr<JxlHybridEncoder> enc(new JxlHybridEncoder(o));
    enc->Start();
    return enc.release();
  } catch (...) {
    return nullptr;
  }
}
void JxlHybridEncoderDestroy(JxlHybridEncoder* encoder) { delete encoder; }
void JxlHybridFree(void* buffer) { free(buffer); }
JxlHybridStageDump* JxlHybridStageDumpCreate(void) {
  try {
    return &(new DumpStorage)->view;
  } catch (...) {
    return nullptr;
  }
}
void JxlHybridStageDumpDestroy(JxlHybridStageDump* dump) {
  if (dump) delete static_cast<DumpStorage*>(dump->opaque);
}
int JxlHybridRecord(JxlHybridEncoder* encoder, const uint8_t* rgb,
                    size_t stride, JxlHybridStageDump* out) {
  if (!encoder || !out || !out->opaque) return JXL_HYBRID_INVALID;
  std::lock_guard<std::mutex> lock(encoder->api_mutex);
  out->codestream = nullptr;
  out->codestream_size = 0;
  JxlHybridTileInput input{};
  input.rgb = rgb;
  input.stride = stride;
  try {
    const int status =
        Encode(encoder->contexts[0], encoder->options, input, out);
    if (status) return status;
    auto& stored = static_cast<DumpStorage*>(out->opaque)->codestream;
    stored = encoder->contexts[0].output;
    out->codestream = stored.data();
    out->codestream_size = stored.size();
    return JXL_HYBRID_OK;
  } catch (...) {
    return JXL_HYBRID_ENCODE_ERROR;
  }
}
int JxlHybridEncodeTile(JxlHybridEncoder* encoder, const uint8_t* rgb,
                        size_t stride, const JxlHybridTileDecisions* decisions,
                        const float* const opsin[3],
                        const JxlHybridStageOverrides* overrides, uint8_t** out,
                        size_t* out_size) {
  if (out) *out = nullptr;
  if (out_size) *out_size = 0;
  if (!encoder || !out || !out_size) return JXL_HYBRID_INVALID;
  std::lock_guard<std::mutex> lock(encoder->api_mutex);
  JxlHybridTileInput input{};
  input.rgb = rgb;
  input.stride = stride;
  input.decisions = decisions;
  input.overrides = overrides;
  if (opsin)
    for (size_t c = 0; c < 3; ++c) input.opsin[c] = opsin[c];
  JxlHybridOutput output;
  const int status =
      EncodeOutput(encoder->contexts[0], encoder->options, input, &output);
  *out = output.data;
  *out_size = output.size;
  return status;
}
int JxlHybridEncodeBatch(JxlHybridEncoder* encoder,
                         const JxlHybridTileInput* tiles, size_t count,
                         JxlHybridOutput* outputs) {
  if (!encoder || (count && (!tiles || !outputs))) return JXL_HYBRID_INVALID;
  std::lock_guard<std::mutex> api_lock(encoder->api_mutex);
  if (count == 0) return JXL_HYBRID_OK;
  if (encoder->threads.empty()) {
    for (size_t i = 0; i < count; ++i) {
      EncodeOutput(encoder->contexts[0], encoder->options, tiles[i],
                   &outputs[i]);
    }
  } else {
    std::unique_lock<std::mutex> lock(encoder->mutex);
    encoder->inputs = tiles;
    encoder->outputs = outputs;
    encoder->count = count;
    encoder->next = 0;
    encoder->remaining = encoder->threads.size();
    ++encoder->generation;
    encoder->ready.notify_all();
    encoder->finished.wait(lock, [&] { return encoder->remaining == 0; });
  }
  for (size_t i = 0; i < count; ++i)
    if (outputs[i].status) return outputs[i].status;
  return JXL_HYBRID_OK;
}
}  // extern "C"

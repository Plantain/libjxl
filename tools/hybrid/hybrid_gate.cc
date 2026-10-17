// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/hybrid.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
using Encoder =
    std::unique_ptr<JxlHybridEncoder, decltype(&JxlHybridEncoderDestroy)>;
using Dump =
    std::unique_ptr<JxlHybridStageDump, decltype(&JxlHybridStageDumpDestroy)>;
constexpr size_t kRGBSize = 512 * 512 * 3;
void Require(bool ok, const std::string& what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what.c_str());
    std::exit(1);
  }
}
Encoder MakeEncoder(float d, unsigned threads = 1, unsigned effort = 7) {
  JxlHybridOptions o{d, effort, threads};
  Encoder enc(JxlHybridEncoderCreate(&o), JxlHybridEncoderDestroy);
  Require(bool(enc), "create encoder");
  return enc;
}
Dump MakeDump() {
  Dump d(JxlHybridStageDumpCreate(), JxlHybridStageDumpDestroy);
  Require(bool(d), "create stage dump");
  return d;
}
std::vector<uint8_t> Read(const char* path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> rgb(kRGBSize);
  Require(bool(f.read(reinterpret_cast<char*>(rgb.data()), rgb.size())), path);
  Require(f.peek() == EOF,
          std::string(path) + ": expected exactly 512x512 RGB8");
  return rgb;
}
JxlHybridStageOverrides Overrides(const JxlHybridStageDump& d, uint32_t mask) {
  return {mask, &d.s1, &d.s2, d.s3, &d.s4, &d.s5, d.s6};
}
std::vector<uint8_t> Encode(JxlHybridEncoder* enc,
                            const std::vector<uint8_t>& rgb,
                            const JxlHybridTileDecisions* decisions = nullptr,
                            const JxlHybridStageOverrides* overrides = nullptr,
                            const float* const* opsin = nullptr,
                            size_t stride = 1536) {
  uint8_t* data = nullptr;
  size_t size = 0;
  const int status = JxlHybridEncodeTile(enc, rgb.data(), stride, decisions,
                                         opsin, overrides, &data, &size);
  Require(status == 0,
          "encode tile status=" + std::to_string(status) + " scale=" +
              std::to_string(decisions ? decisions->global_scale : 0) +
              " dc=" + std::to_string(decisions ? decisions->quant_dc : 0) +
              " stages=" + std::to_string(overrides ? overrides->stages : 0));
  std::vector<uint8_t> out(data, data + size);
  JxlHybridFree(data);
  return out;
}
std::vector<uint8_t> Native(const std::vector<uint8_t>& rgb, float distance) {
  auto enc = JxlEncoderMake(nullptr);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = info.ysize = 512;
  info.bits_per_sample = 8;
  info.num_color_channels = 3;
  info.uses_original_profile = JXL_FALSE;
  Require(JxlEncoderSetBasicInfo(enc.get(), &info) == JXL_ENC_SUCCESS,
          "native info");
  JxlColorEncoding color;
  JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  Require(JxlEncoderSetColorEncoding(enc.get(), &color) == JXL_ENC_SUCCESS,
          "native color");
  auto* s = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  Require(JxlEncoderFrameSettingsSetOption(s, JXL_ENC_FRAME_SETTING_EFFORT,
                                           8) == JXL_ENC_SUCCESS,
          "native effort");
  Require(JxlEncoderSetFrameDistance(s, distance) == JXL_ENC_SUCCESS,
          "native distance");
  JxlPixelFormat f{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  Require(
      JxlEncoderAddImageFrame(s, &f, rgb.data(), rgb.size()) == JXL_ENC_SUCCESS,
      "native input");
  JxlEncoderCloseInput(enc.get());
  std::vector<uint8_t> out(65536);
  uint8_t* next = out.data();
  size_t avail = out.size();
  for (;;) {
    auto status = JxlEncoderProcessOutput(enc.get(), &next, &avail);
    if (status == JXL_ENC_SUCCESS) break;
    Require(status == JXL_ENC_NEED_MORE_OUTPUT, "native output");
    size_t n = next - out.data();
    out.resize(out.size() * 2);
    next = out.data() + n;
    avail = out.size() - n;
  }
  out.resize(next - out.data());
  return out;
}
std::vector<float> Decode(const std::vector<uint8_t>& encoded) {
  auto dec = JxlDecoderMake(nullptr);
  Require(JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_FULL_IMAGE) ==
              JXL_DEC_SUCCESS,
          "decoder events");
  Require(JxlDecoderSetInput(dec.get(), encoded.data(), encoded.size()) ==
              JXL_DEC_SUCCESS,
          "decoder input");
  JxlDecoderCloseInput(dec.get());
  std::vector<float> rgb(kRGBSize);
  Require(JxlDecoderProcessInput(dec.get()) == JXL_DEC_NEED_IMAGE_OUT_BUFFER,
          "decoder buffer request");
  JxlPixelFormat format{3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  Require(JxlDecoderSetImageOutBuffer(dec.get(), &format, rgb.data(),
                                      rgb.size() * sizeof(float)) ==
              JXL_DEC_SUCCESS,
          "decoder output buffer");
  Require(JxlDecoderProcessInput(dec.get()) == JXL_DEC_FULL_IMAGE,
          "decoder full image");
  return rgb;
}

template <typename F>
void Parallel(size_t n, unsigned threads, F func) {
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      for (;;) {
        size_t i = next++;
        if (i >= n) break;
        func(i);
      }
    });
  for (auto& t : pool) t.join();
}
void Identity(const std::vector<std::vector<uint8_t>>& tiles, float distance,
              unsigned threads) {
  std::atomic<size_t> bytes{0}, native_bytes{0}, checks{0};
  Parallel(tiles.size(), threads, [&](size_t i) {
    auto enc = MakeEncoder(distance);
    auto dump = MakeDump();
    const auto& rgb = tiles[i];
    Require(JxlHybridRecord(enc.get(), rgb.data(), 1536, dump.get()) == 0,
            "record tile " + std::to_string(i));
    std::vector<uint8_t> expected(dump->codestream,
                                  dump->codestream + dump->codestream_size);
    auto check = [&](const std::vector<uint8_t>& actual,
                     const std::string& mode) {
      Require(actual == expected,
              "identity tile " + std::to_string(i) + " " + mode);
      ++checks;
    };
    check(Encode(enc.get(), rgb), "plain");
    check(Encode(enc.get(), rgb, &dump->decisions), "full");
    const float* opsin[3] = {dump->opsin[0], dump->opsin[1], dump->opsin[2]};
    check(Encode(enc.get(), rgb, &dump->decisions, nullptr, opsin), "opsin");
    for (unsigned s = 1; s <= 6; ++s) {
      auto o = Overrides(*dump, JXL_HYBRID_STAGE(s));
      check(Encode(enc.get(), rgb, nullptr, &o), "S" + std::to_string(s));
    }
    auto all = Overrides(*dump, JXL_HYBRID_ALL_STAGES);
    check(Encode(enc.get(), rgb, nullptr, &all), "all stages");
    // Independent oracle: unhooked public libjxl at effort 8. This detects a
    // shared mistake in recording and self-overrides, including the effort
    // split.
    auto native = Native(rgb, distance);
    auto enc8 = MakeEncoder(distance, 1, 8);
    auto recorded8 = MakeDump();
    Require(JxlHybridRecord(enc8.get(), rgb.data(), 1536, recorded8.get()) == 0,
            "record e8 backend");
    Require(native.size() == recorded8->codestream_size &&
                std::equal(native.begin(), native.end(), recorded8->codestream),
            "native e8 oracle");
    Require(Encode(enc8.get(), rgb, &recorded8->decisions) == native,
            "e8 backend replay");
    const auto decoded = Decode(expected), ref = Decode(native);
    Require(decoded == ref, "decoded float RGB equals e8 (quality parity)");
    checks += 3;
    bytes += expected.size();
    native_bytes += native.size();
  });
  printf(
      "{\"mode\":\"identity\",\"distance\":%.3f,\"tiles\":%zu,\"checks\":%zu,"
      "\"failures\":0,\"backend7_bytes\":%zu,\"native_e8_bytes\":%zu,\"decoded_"
      "float_equal\":true}\n",
      distance, tiles.size(), checks.load(), bytes.load(), native_bytes.load());
}
void Bench(const std::vector<std::vector<uint8_t>>& tiles, float distance,
           unsigned threads, unsigned repeats) {
  Require(threads == 16 && tiles.size() * repeats >= 2000,
          "throughput gate requires 16 threads and at least 2000 tiles");
  std::vector<Dump> dumps;
  for (size_t i = 0; i < tiles.size(); ++i) dumps.push_back(MakeDump());
  Parallel(tiles.size(), threads, [&](size_t i) {
    auto enc = MakeEncoder(distance);
    Require(
        JxlHybridRecord(enc.get(), tiles[i].data(), 1536, dumps[i].get()) == 0,
        "bench record");
  });
  auto enc = MakeEncoder(distance, threads);
  std::vector<JxlHybridTileInput> inputs(tiles.size() * repeats);
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs[i].rgb = tiles[i % tiles.size()].data();
    inputs[i].stride = 1536;
    inputs[i].decisions = &dumps[i % tiles.size()]->decisions;
  }
  std::vector<JxlHybridOutput> outputs(inputs.size());
  bool passed = true;
  auto run = [&](bool timed, unsigned round) {
    const auto start = std::chrono::steady_clock::now();
    Require(JxlHybridEncodeBatch(enc.get(), inputs.data(),
                                 timed ? inputs.size() : tiles.size(),
                                 outputs.data()) == 0,
            "encode batch");
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    size_t bytes = 0;
    for (size_t i = 0; i < (timed ? inputs.size() : tiles.size()); ++i) {
      bytes += outputs[i].size;
      Require(outputs[i].size == dumps[i % tiles.size()]->codestream_size &&
                  std::equal(outputs[i].data, outputs[i].data + outputs[i].size,
                             dumps[i % tiles.size()]->codestream),
              "batch identity");
      JxlHybridFree(outputs[i].data);
    }
    if (timed) {
      passed = passed && inputs.size() / secs >= 900;
      printf(
          "{\"mode\":\"bench\",\"round\":%u,\"distance\":%.3f,\"threads\":%u,"
          "\"tiles\":%zu,\"encodes\":%zu,\"seconds\":%.6f,\"tiles_per_s\":%.3f,"
          "\"bytes\":%zu}\n",
          round, distance, threads, tiles.size(), inputs.size(), secs,
          inputs.size() / secs, bytes);
      fflush(stdout);
    }
  };
  run(false, 0);
  for (unsigned round = 1; round <= 3; ++round) run(true, round);
  Require(passed, "backend throughput below 900 tiles/s");
}
void Robustness() {
  auto enc = MakeEncoder(1.2f, 4);
  auto dump = MakeDump();
  std::vector<uint8_t> rgb(kRGBSize);
  std::mt19937 rng(0x48594231);
  for (auto& v : rgb) v = rng() & 255;
  Require(JxlHybridRecord(enc.get(), rgb.data(), 1536, dump.get()) == 0,
          "robustness record");
  auto expected = Encode(enc.get(), rgb, &dump->decisions);
  size_t rejected = 0, valid = 0, extreme_errors = 0;
  auto reject = [&](const JxlHybridTileDecisions* d,
                    const JxlHybridStageOverrides* o = nullptr,
                    const float* const* opsin = nullptr, size_t stride = 1536) {
    uint8_t* out = reinterpret_cast<uint8_t*>(1);
    size_t size = 100;
    int rc = JxlHybridEncodeTile(enc.get(), rgb.data(), stride, d, opsin, o,
                                 &out, &size);
    Require(rc != 0 && out == nullptr && size == 0,
            "invalid input must fail with empty output");
    ++rejected;
  };
  for (unsigned i = 0; i < 2000; ++i) {
    auto d = dump->decisions;
    const size_t p = rng() % 4096;
    int32_t q = d.raw_quant_field[p];
    uint8_t acs = d.ac_strategy[p], epf = d.epf_sharpness[p];
    switch (i % 14) {
      case 0:
        d.global_scale = (rng() & 1) ? 0 : 73729;
        break;
      case 1:
        d.quant_dc = (rng() & 1) ? 0 : 65537;
        break;
      case 2:
        d.raw_quant_field[p] = (rng() & 1) ? -1 : 257;
        break;
      case 3:
        d.ac_strategy[p] = 254;
        break;
      case 4:
        d.epf_sharpness[p] = 8;
        break;
      case 5:
        d.ytox_dc = 128;
        break;
      case 6:
        d.color_factor = 0;
        break;
      case 7:
        d.base_correlation_b = std::numeric_limits<float>::quiet_NaN();
        break;
      case 8:
        d.base_correlation_x = 5;
        break;
      case 9:
        d.xsize_blocks = 63;
        break;
      case 10:
        d.raw_quant_field = nullptr;
        break;
      case 11:
        d.ytob_map = nullptr;
        break;
      case 12:
        d.b_qm_scale = 8;
        break;
      case 13:
        d.ytob_dc = -129;
        break;
    }
    reject(&d);
    dump->decisions.raw_quant_field[p] = q;
    dump->decisions.ac_strategy[p] = acs;
    dump->decisions.epf_sharpness[p] = epf;
  }
  // Coverage, overlap, boundary crossing, and a valid rectangular transform.
  std::vector<uint8_t> strategies(4096, 0);
  auto d = dump->decisions;
  d.ac_strategy = strategies.data();
  strategies[0] = 255;
  reject(&d);
  strategies[0] = 4;
  reject(&d);
  strategies[0] = 0;
  strategies[63] = 4;
  reject(&d);
  strategies[63] = 0;
  std::vector<int32_t> quant(4096, 8);
  d.raw_quant_field = quant.data();
  std::vector<uint8_t> sharp(4096, 4);
  d.epf_sharpness = sharp.data();
  // Exercise every supported strategy with a repeated 8x8 color-region tiling.
  const size_t widths[] = {1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2,
                           4, 1, 1, 1, 1, 1, 1, 8, 4, 8};
  const size_t heights[] = {1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4,
                            2, 1, 1, 1, 1, 1, 1, 8, 8, 4};
  for (unsigned raw = 0; raw < 21; ++raw) {
    std::fill(strategies.begin(), strategies.end(), 255);
    for (size_t y = 0; y < 64; y += heights[raw])
      for (size_t x = 0; x < 64; x += widths[raw]) strategies[y * 64 + x] = raw;
    for (auto& v : sharp) v = rng() % 8;
    Decode(Encode(enc.get(), rgb, &d));
    ++valid;
    if (raw == 4) {
      ++quant[1];
      reject(&d);
      --quant[1];
    }
  }
  for (unsigned i = 0; i < 32; ++i) {
    std::fill(strategies.begin(), strategies.end(), 0);
    for (auto& q : quant) q = 1 + rng() % 256;
    d.global_scale = 1 + rng() % (i < 16 ? 8192 : 73728);
    d.quant_dc = 1 + rng() % (i < 16 ? 256 : 65536);
    d.ytox_dc = int(rng() % 256) - 128;
    d.ytob_dc = int(rng() % 256) - 128;
    d.color_factor = 2 + rng() % 256;
    uint8_t* data = nullptr;
    size_t size = 0;
    const int status = JxlHybridEncodeTile(enc.get(), rgb.data(), 1536, &d,
                                           nullptr, nullptr, &data, &size);
    // Randomized extremes may encode or return an error; both paths must be
    // memory safe, including when reusing the encoder after an error.
    if (i >= 16 && status == JXL_HYBRID_ENCODE_ERROR) {
      Require(!data && size == 0, "extreme decisions fail without output");
      ++extreme_errors;
    } else {
      Require(status == 0, "random ordinary decisions encode");
      Decode(std::vector<uint8_t>(data, data + size));
      JxlHybridFree(data);
      ++valid;
    }
  }
  for (unsigned s = 1; s <= 6; ++s) {
    JxlHybridStageOverrides o{};
    o.stages = JXL_HYBRID_STAGE(s);
    reject(nullptr, &o);
  }
  for (uint32_t factor : {1u, 258u, 65793u, 0xffffffffu}) {
    auto invalid = dump->decisions;
    invalid.color_factor = factor;
    reject(&invalid);
  }
  auto o = Overrides(*dump, 64);
  reject(nullptr, &o);
  o = Overrides(*dump, 1);
  reject(&dump->decisions, &o);
  float saved = dump->s1.quant_field[0];
  dump->s1.quant_field[0] = std::numeric_limits<float>::infinity();
  reject(nullptr, &o);
  dump->s1.quant_field[0] = saved;
  const float* partial[3] = {dump->opsin[0], nullptr, dump->opsin[2]};
  reject(&dump->decisions, nullptr, partial);
  const float* opsin[3] = {dump->opsin[0], dump->opsin[1], dump->opsin[2]};
  reject(nullptr, nullptr, opsin);
  saved = dump->opsin[0][0];
  dump->opsin[0][0] = NAN;
  reject(&dump->decisions, nullptr, opsin);
  dump->opsin[0][0] = saved;
  reject(&dump->decisions, nullptr, nullptr, 1535);
  reject(&dump->decisions, nullptr, nullptr,
         std::numeric_limits<size_t>::max());
  // Padded input rows, all 64 subsets, batch errors, and encoder reuse.
  std::vector<uint8_t> padded(1600 * 512);
  for (size_t y = 0; y < 512; ++y)
    memcpy(padded.data() + y * 1600, rgb.data() + y * 1536, 1536);
  Require(Encode(enc.get(), padded, &dump->decisions, nullptr, nullptr, 1600) ==
              expected,
          "padded stride");
  for (uint32_t mask = 0; mask < 64; ++mask) {
    auto subset = Overrides(*dump, mask);
    Require(Encode(enc.get(), rgb, nullptr, &subset) == expected,
            "override subset identity");
    ++valid;
  }
  JxlHybridTileInput inputs[3]{};
  for (auto& in : inputs) {
    in.rgb = rgb.data();
    in.stride = 1536;
    in.decisions = &dump->decisions;
  }
  inputs[1].stride = 1;
  JxlHybridOutput out[3]{};
  Require(JxlHybridEncodeBatch(enc.get(), inputs, 3, out) != 0,
          "batch reports invalid tile");
  for (size_t i = 0; i < 3; ++i) {
    Require(out[i].status == (i == 1 ? JXL_HYBRID_INVALID : JXL_HYBRID_OK),
            "per-tile batch status");
    JxlHybridFree(out[i].data);
  }
  Require(JxlHybridEncodeBatch(enc.get(), nullptr, 0, nullptr) == 0,
          "empty batch");
  Require(Encode(enc.get(), rgb, &dump->decisions) == expected,
          "reuse after errors");
  printf(
      "{\"mode\":\"robustness\",\"seed\":1213809201,\"rejected\":%zu,\"valid\":"
      "%zu,\"extreme_errors\":%zu,\"failures\":0}\n",
      rejected, valid, extreme_errors);
}
}  // namespace
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "robustness") {
    Robustness();
    return 0;
  }
  if (argc < 6) {
    fprintf(stderr,
            "usage: hybrid_gate identity|bench DISTANCE THREADS REPEATS "
            "tile.rgb...\n       hybrid_gate robustness\n");
    return 2;
  }
  const std::string mode = argv[1];
  const float distance = std::stof(argv[2]);
  const unsigned threads = std::stoul(argv[3]), repeats = std::stoul(argv[4]);
  Require(threads > 0 && threads <= 16 && repeats > 0, "thread/repeat limits");
  std::vector<std::vector<uint8_t>> tiles;
  for (int i = 5; i < argc; ++i) tiles.push_back(Read(argv[i]));
  if (mode == "identity")
    Identity(tiles, distance, threads);
  else if (mode == "bench")
    Bench(tiles, distance, threads, repeats);
  else
    return 2;
  return 0;
}

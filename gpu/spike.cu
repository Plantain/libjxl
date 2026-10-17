// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cuda_profiler_api.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch.cuh"
#include "butteraugli_compare.cuh"
#include "small_transforms.cuh"
#include "tile_distance.cuh"
using namespace jxl_wp3;
namespace ba = satellite_butteraugli;
namespace rc = satellite_reconstruction_gpu;
void Check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
class ProcessLock {
 public:
  ProcessLock() {
    if (std::getenv("WP3_GPU_LOCK_HELD")) return;
    fd_ = open("/w/wp3/gpu.lock", O_CREAT | O_RDWR, 0666);
    if (fd_ < 0) throw std::runtime_error("WP3 GPU lock open failed");
    if (flock(fd_, LOCK_EX)) {
      close(fd_);
      throw std::runtime_error("WP3 GPU lock failed");
    }
  }
  ~ProcessLock() {
    if (fd_ >= 0) close(fd_);
  }
  ProcessLock(const ProcessLock&) = delete;
  ProcessLock& operator=(const ProcessLock&) = delete;

 private:
  int fd_ = -1;
};
void Reconstruct(Tile* t, int b, cudaStream_t stream) {
  auto& in = t->input;
  auto* p = &in.p;
  float *dc0 = t->scratch, *dc1 = dc0 + 3 * kBlocks, *sigma = dc1 + 3 * kBlocks;
  float *p0 = sigma + kBlocks, *p1 = p0 + 3 * kPixels, *p2 = p1 + 3 * kPixels;
  rc::dc_dequant<<<grid(16, b), 256, 0, stream>>>(in.dc, dc0, p);
  rc::dc_smooth<<<grid(16, b), 256, 0, stream>>>(dc0, dc1, p);
  rc::ac_dequant<<<grid(kBlocks, b), 128, 0, stream>>>(
      in.ac, in.quant, in.kind, in.first, in.cfl, in.matrices, p, p0);
  rc::lowest_frequencies<<<dim3(kBlocks, 3, b), 8, 0, stream>>>(dc1, in.kind,
                                                                in.first, p0);
  rc::inverse_dct<<<dim3(kBlocks, 3, b), 64, 0, stream>>>(p0, in.kind, in.first,
                                                          p2);
  rc::inverse_dct2<<<dim3(kBlocks, 3, b), 1, 0, stream>>>(p0, in.kind, in.first,
                                                          p2);
  rc::inverse_small<<<dim3(kBlocks, 3, b), 1, 0, stream>>>(p0, in.kind,
                                                           in.first, p2);
  rc::compute_sigma<<<grid(16, b), 256, 0, stream>>>(
      in.quant, in.kind, in.first, in.sharp, sigma, p);
  rc::gaborish<<<grid(1024, b), 256, 0, stream>>>(p2, p0, p);
  rc::epf1<<<grid(1024, b), 256, 0, stream>>>(p0, sigma, p1, p);
  rc::xyb_linear<<<grid(1024, b), 256, 0, stream>>>(p1, p0, p);
  // RoundtripImage outputs the frame's sRGB encoding. CompareWith transforms
  // that float image back to linear sRGB with these pinned transfer functions.
  rc::transfer<<<grid(3072, b), 256, 0, stream>>>(p0, p1, false);
  rc::transfer<<<grid(3072, b), 256, 0, stream>>>(p1, t->decoded, true);
}
void Setup(Tile* t, int b, cudaStream_t s) {
  Check(ba::setup_reference(t->original, 512, 512, 80.f, t->reference,
                            t->scratch, s, b));
  ba::subsample<<<grid(768, b), 256, 0, s>>>(t->original, 512, 512,
                                             t->original_half);
  Check(ba::setup_reference(t->original_half, 256, 256, 80.f, t->reference_half,
                            t->scratch, s, b));
}
void Compare(Tile* t, int b, cudaStream_t s) {
  Check(ba::compare_scale(t->reference, t->decoded, 512, 512, 80.f, 1.f, 1.f,
                          t->scratch, t->diff, s, b));
  ba::subsample<<<grid(768, b), 256, 0, s>>>(t->decoded, 512, 512,
                                             t->decoded_half);
  Check(ba::compare_scale(t->reference_half, t->decoded_half, 256, 256, 80.f,
                          1.f, 1.f, t->scratch, t->diff_half, s, b));
  ba::supersample_add<<<grid(1024, b), 256, 0, s>>>(t->diff_half, 512, 512,
                                                    t->diff);
  satellite_feedback::tile_distance<<<grid(16, b), 256, 0, s>>>(
      t->diff, 512, 512, t->input.cx, t->input.cy, t->dist, &t->error);
}
struct Sample {
  Input in;
  Oracle oracle;
};
std::unique_ptr<Sample> Read(const char* name) {
  auto x = std::make_unique<Sample>();
  std::ifstream f(name, std::ios::binary);
  f.read(reinterpret_cast<char*>(&x->in), sizeof(Input));
  if (!f) throw std::runtime_error("input read failed");
  f.read(reinterpret_cast<char*>(&x->oracle), sizeof(Oracle));
  if (!f || f.peek() != EOF) throw std::runtime_error("oracle read failed");
  if (x->in.magic != 0x3350574a || x->in.version != 1)
    throw std::runtime_error("dump version");
  if (x->in.p.epf_iters != 1 || !x->in.p.gab_enabled)
    throw std::runtime_error("unsupported loop filter");
  for (int i = 0; i < kBlocks; ++i)
    if (x->in.kind[i] > 20)
      throw std::runtime_error("strategy > 64 unsupported");
  return x;
}
struct Metrics {
  double rec_sum = 0, rec_max = 0;
  size_t pixels = 0, blocks = 0, agree1 = 0, agreetarget = 0, tiles = 0,
         argmax = 0;
  unsigned long long kinds[27] = {};
  std::vector<float> differr, disterr;
  double dm_max = 0;
  void Add(const Sample& sample, const std::vector<float>& rgb,
           const std::vector<float>& diff, const std::vector<float>& dist,
           bool comparator_only) {
    const auto& o = sample.oracle;
    for (int i = 0; i < 3 * kPixels; ++i) {
      double e = std::abs(double(rgb[i]) - o.linear[i]);
      if (!std::isfinite(e))
        throw std::runtime_error("nonfinite reconstruction");
      rec_sum += e;
      rec_max = std::max(rec_max, e);
    }
    pixels += 3 * kPixels;
    for (int i = 0; i < kPixels; ++i) {
      double e = std::abs(double(diff[i]) - o.diff[i]) /
                 std::max(1e-12, double(o.diff[i]));
      if (!std::isfinite(e)) throw std::runtime_error("nonfinite diffmap");
      differr.push_back(e);
      dm_max = std::max(dm_max, e);
    }
    for (int i = 0; i < kBlocks; ++i) {
      double e = std::abs(double(dist[i]) - o.dist[i]) /
                 std::max(1e-12, double(o.dist[i]));
      if (!std::isfinite(e)) throw std::runtime_error("nonfinite distance");
      disterr.push_back(e);
      agree1 += (dist[i] > 1) == (o.dist[i] > 1);
      agreetarget += (dist[i] > sample.in.p.distance) ==
                     (o.dist[i] > sample.in.p.distance);
      ++blocks;
      if (sample.in.first[i]) ++kinds[sample.in.kind[i]];
    }
    argmax += (std::max_element(dist.begin(), dist.end()) - dist.begin()) ==
              (std::max_element(o.dist, o.dist + kBlocks) - o.dist);
    ++tiles;
  }
  float Percentile(std::vector<float>& a, double p) {
    size_t i = static_cast<size_t>(std::ceil(p * a.size())) - 1;
    std::nth_element(a.begin(), a.begin() + i, a.end());
    return a[i];
  }
  bool Report(int batch, bool comparator_only) {
    const double dp = Percentile(differr, .99), tp = Percentile(disterr, .99);
    const bool pass = rec_sum / pixels <= 1e-4 && dp <= 1e-3 &&
                      double(agreetarget) / blocks >= .995 &&
                      double(agree1) / blocks >= .995;
    std::printf(
        "{\"mode\":\"%s\",\"batch\":%d,\"tiles\":%zu,\"linear_mae\":%.9g,"
        "\"linear_max_abs\":%.9g,\"diff_relative_p99\":%.9g,\"diff_relative_"
        "max\":%.9g,\"tile_dist_relative_p99\":%.9g,\"threshold_1_agreement\":%"
        ".9g,\"update_threshold_agreement\":%.9g,\"tile_argmax_agreement\":%."
        "9g,\"numeric_gate_pass\":%s,\"strategy_counts\":[",
        comparator_only ? "comparator_only" : "full", batch, tiles,
        rec_sum / pixels, rec_max, dp, dm_max, tp, double(agree1) / blocks,
        double(agreetarget) / blocks, double(argmax) / tiles,
        pass ? "true" : "false");
    for (int i = 0; i < 27; ++i) std::printf("%s%llu", i ? "," : "", kinds[i]);
    std::puts("]}");
    return pass;
  }
};
extern "C" void Wp3TransformCase(int kind, Input* in, float* expected);
bool TransformTests(Tile* t, cudaStream_t stream) {
  bool pass = true;
  std::printf("{\"transform_cases\":[");
  for (int kind = 0; kind <= 20; ++kind) {
    auto in = std::make_unique<Input>();
    std::vector<float> expected(3 * kPixels), got(3 * kPixels);
    Wp3TransformCase(kind, in.get(), expected.data());
    Check(cudaMemcpyAsync(&t->input, in.get(), sizeof(Input),
                          cudaMemcpyHostToDevice, stream));
    rc::inverse_dct<<<dim3(kBlocks, 3), 64, 0, stream>>>(
        t->input.matrices, t->input.kind, t->input.first, t->decoded);
    rc::inverse_dct2<<<dim3(kBlocks, 3), 1, 0, stream>>>(
        t->input.matrices, t->input.kind, t->input.first, t->decoded);
    rc::inverse_small<<<dim3(kBlocks, 3), 1, 0, stream>>>(
        t->input.matrices, t->input.kind, t->input.first, t->decoded);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(got.data(), t->decoded, got.size() * 4,
                     cudaMemcpyDeviceToHost));
    double maximum = 0, total = 0;
    for (size_t k = 0; k < got.size(); ++k) {
      double e = std::abs(double(got[k]) - expected[k]);
      if (!std::isfinite(e)) throw std::runtime_error("transform nonfinite");
      maximum = std::max(maximum, e);
      total += e;
    }
    pass &= maximum <= 1e-5;
    std::printf("%s{\"strategy\":%d,\"mae\":%.9g,\"max_abs\":%.9g}",
                kind ? "," : "", kind, total / got.size(), maximum);
  }
  std::printf("],\"pass\":%s}\n", pass ? "true" : "false");
  return pass;
}
int main(int argc, char** argv) try {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: wp3_spike validate|compare|bench|profile|transforms B "
                 "REPEATS file.wp3...\n");
    return 2;
  }
  const std::string mode = argv[1];
  bool bench = mode == "bench", profile = mode == "profile",
       comparison = mode == "compare";
  int b = std::stoi(argv[2]), repeats = std::stoi(argv[3]);
  if ((mode != "validate" && !comparison && !bench && !profile &&
       mode != "transforms") ||
      b < 1 || b > 32 || repeats < 1)
    throw std::runtime_error("invalid arguments");
  // Serialize spike processes even when invoked without the wrapper. The
  // wrapper can pass its inherited host-side lock through this explicit flag.
  ProcessLock process_lock;
  size_t free0, total;
  Check(cudaMemGetInfo(&free0, &total));
  if (free0 < 6ull * 1024 * 1024 * 1024)
    throw std::runtime_error("less than 6 GiB free");
  Tile* t;
  const size_t allocated = b * kStrideBytes;
  Check(cudaMalloc(&t, allocated));
  Check(cudaMemset(t, 0, allocated));
  cudaStream_t stream;
  Check(cudaStreamCreate(&stream));
  Metrics m;
  size_t free1;
  Check(cudaMemGetInfo(&free1, &total));
  if (mode == "transforms") {
    bool ok = TransformTests(t, stream);
    Check(cudaStreamDestroy(stream));
    Check(cudaFree(t));
    return ok ? 0 : 1;
  }
  for (int base = 4; base < argc; base += b) {
    int count = std::min(b, argc - base);
    std::vector<std::unique_ptr<Sample>> samples;
    for (int j = 0; j < count; ++j)
      samples.push_back(Read(
          argv[(bench || profile) ? 4 + j * (argc - 4) / count : base + j]));
    for (int j = 0; j < b; ++j) {
      const auto& v = *samples[j % count];
      auto* dst = reinterpret_cast<Tile*>(reinterpret_cast<char*>(t) +
                                          j * kStrideBytes);
      Check(cudaMemcpyAsync(&dst->input, &v.in, sizeof(Input),
                            cudaMemcpyHostToDevice, stream));
      Check(cudaMemcpyAsync(dst->original, v.oracle.original,
                            sizeof(v.oracle.original), cudaMemcpyHostToDevice,
                            stream));
      Check(cudaMemcpyAsync(dst->decoded, v.oracle.linear,
                            sizeof(v.oracle.linear), cudaMemcpyHostToDevice,
                            stream));
    }
    Setup(t, b, stream);
    if (profile) {
      for (int i = 0; i < 3; ++i) {
        Reconstruct(t, b, stream);
        Compare(t, b, stream);
      }
      Check(cudaStreamSynchronize(stream));
      Check(cudaProfilerStart());
      for (int i = 0; i < repeats; ++i) {
        Reconstruct(t, b, stream);
        Compare(t, b, stream);
      }
      Check(cudaStreamSynchronize(stream));
      Check(cudaProfilerStop());
      break;
    }
    if (bench) {
      auto timing = [&](auto fn) {
        for (int i = 0; i < 3; ++i) fn();
        cudaEvent_t a, z;
        Check(cudaEventCreate(&a));
        Check(cudaEventCreate(&z));
        Check(cudaEventRecord(a, stream));
        for (int i = 0; i < repeats; ++i) fn();
        Check(cudaEventRecord(z, stream));
        Check(cudaEventSynchronize(z));
        float ms;
        Check(cudaEventElapsedTime(&ms, a, z));
        Check(cudaEventDestroy(a));
        Check(cudaEventDestroy(z));
        return ms / repeats / b;
      };
      double setup = timing([&] { Setup(t, b, stream); });
      double reconstruction = timing([&] { Reconstruct(t, b, stream); });
      double compare = timing([&] { Compare(t, b, stream); });
      double evaluation = timing([&] {
        Reconstruct(t, b, stream);
        Compare(t, b, stream);
      });
      size_t free2;
      Check(cudaMemGetInfo(&free2, &total));
      free1 = std::min(free1, free2);
      std::printf(
          "{\"batch\":%d,\"repeats\":%d,\"setup_ms_per_tile\":%.9g,"
          "\"reconstruction_ms_per_tile\":%.9g,\"compare_ms_per_tile\":%.9g,"
          "\"evaluation_ms_per_tile\":%.9g,\"allocated_bytes\":%zu,\"observed_"
          "device_delta_bytes\":%zu,\"timing_gate_pass\":%s}\n",
          b, repeats, setup, reconstruction, compare, evaluation, allocated,
          free0 > free1 ? free0 - free1 : 0,
          (b >= 8 && setup <= .15 && evaluation <= .4) ? "true" : "false");
      break;
    }
    if (!comparison) Reconstruct(t, b, stream);
    Compare(t, b, stream);
    Check(cudaStreamSynchronize(stream));
    for (int j = 0; j < count; ++j) {
      auto* dst = reinterpret_cast<Tile*>(reinterpret_cast<char*>(t) +
                                          j * kStrideBytes);
      std::vector<float> rgb(3 * kPixels), diff(kPixels), dist(kBlocks);
      int error = 0;
      Check(cudaMemcpy(rgb.data(), dst->decoded, rgb.size() * 4,
                       cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(diff.data(), dst->diff, diff.size() * 4,
                       cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(dist.data(), dst->dist, dist.size() * 4,
                       cudaMemcpyDeviceToHost));
      Check(cudaMemcpy(&error, &dst->error, 4, cudaMemcpyDeviceToHost));
      if (error) {
        double sum = 0, maximum = 0;
        int bad = 0;
        for (int k = 0; k < 3 * kPixels; ++k) {
          double e = std::abs(double(rgb[k]) - samples[j]->oracle.linear[k]);
          if (!std::isfinite(e))
            ++bad;
          else {
            sum += e;
            maximum = std::max(maximum, e);
          }
        }
        std::fprintf(stderr,
                     "device error=%d reconstruction mae=%g max=%g bad=%d "
                     "dc_precision=%g dc0=%d matrix0=%g inv=%g\n",
                     error, sum / (3 * kPixels), maximum, bad,
                     samples[j]->in.p.dc_precision, samples[j]->in.dc[0],
                     samples[j]->in.p.matrix[0], samples[j]->in.p.inverse);
        throw std::runtime_error("TileDistMap device error");
      }
      m.Add(*samples[j], rgb, diff, dist, comparison);
    }
    std::fprintf(stderr, "validated %d/%d\n", base - 4 + count, argc - 4);
  }
  Check(cudaGetLastError());
  Check(cudaStreamDestroy(stream));
  Check(cudaFree(t));
  if (!bench && !profile) return m.Report(b, comparison) ? 0 : 1;
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "WP3: %s\n", e.what());
  return 2;
}

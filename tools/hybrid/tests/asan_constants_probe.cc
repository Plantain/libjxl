// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.
// Isolated from lib/jxl/headers.cc; GCC 13 + ASan + -fmerge-all-constants
// reproduces a false lifetime error. Run with and without the optimization.
#include <cstdint>
#include <cstdlib>
struct Rational {
  constexpr Rational(uint32_t n, uint32_t d) : num(n), den(d) {}
  uint32_t MulTruncate(uint32_t m) const { return uint64_t(m) * num / den; }
  uint32_t num, den;
};
Rational FixedAspectRatios(uint32_t ratio) {
  constexpr Rational ratios[7] = {{1, 1},  {12, 10}, {4, 3}, {3, 2},
                                  {16, 9}, {5, 4},   {2, 1}};
  return ratios[ratio - 1];
}
uint32_t FindAspectRatio(uint32_t x, uint32_t y) {
  for (uint32_t r = 1; r < 8; ++r)
    if (x == FixedAspectRatios(r).MulTruncate(y)) return r;
  return 0;
}
int main(int argc, char** argv) {
  uint32_t size = argc > 1 ? std::atoi(argv[1]) : 512;
  uint32_t ratio = FindAspectRatio(size, size);
  return FixedAspectRatios(ratio).MulTruncate(size) == size ? 0 : 1;
}

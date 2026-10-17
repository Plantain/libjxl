// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Mechanically adapted pinned MaltaUnit expressions; actualCPUoracle
// isindependent.
#pragma once
#include <cuda_runtime.h>

#include "batch.cuh"
namespace satellite_butteraugli {
__device__ inline float Sum(float a, float b, float c, float d) {
  return (a + b) + (c + d);
}
__device__ inline float Sum(float a, float b, float c, float d, float e) {
  return Sum(a, b, c, d + e);
}
__device__ inline float Sum(float a, float b, float c, float d, float e,
                            float f, float g) {
  return Sum(a, b, c, Sum(d, e, f, g));
}
__device__ inline float Sum(float a, float b, float c, float d, float e,
                            float f, float g, float h, float i) {
  return (Sum(a, b, c, d) + Sum(e, f, g, h)) + i;
}
__device__ inline float malta_sample(const float* input, int w, int h, int x,
                                     int y, int offset) {
  // Offsets encode dy*1000+dx with abs(dx)<=4; no invalid pointer arithmetic.
  const int dy = (offset + 4500) / 1000 - 4, dx = offset - dy * 1000;
  const int px = x + dx, py = y + dy;
  return px < 0 || py < 0 || px >= w || py >= h ? 0.f : input[py * w + px];
}
__device__ inline float malta_lf(const float* input, int w, int h, int x,
                                 int y) {
  const int xs = 1000, xs3 = 3000;

  const auto center = malta_sample(input, w, h, x, y, 0);

  // x grows, y constant
  const auto sum_yconst = Sum(malta_sample(input, w, h, x, y, 0 - 4),
                              malta_sample(input, w, h, x, y, 0 - 2), center,
                              malta_sample(input, w, h, x, y, 0 + 2),
                              malta_sample(input, w, h, x, y, 0 + 4));
  // Will return this, sum of all line kernels
  auto retval = __fmul_rn(sum_yconst, sum_yconst);
  {
    // y grows, x constant
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs),
                   malta_sample(input, w, h, x, y, 0 - xs - xs), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // both grow
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - 3),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 2), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 2),
                   malta_sample(input, w, h, x, y, 0 + xs3 + 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // y grows, x shrinks
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 + 3),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 2), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 2),
                   malta_sample(input, w, h, x, y, 0 + xs3 - 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs + 1),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs - 1));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs - 1),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs + 1));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - 4 - xs),
                   malta_sample(input, w, h, x, y, 0 - 2 - xs), center,
                   malta_sample(input, w, h, x, y, 0 + 2 + xs),
                   malta_sample(input, w, h, x, y, 0 + 4 + xs));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - 4 + xs),
                   malta_sample(input, w, h, x, y, 0 - 2 + xs), center,
                   malta_sample(input, w, h, x, y, 0 + 2 - xs),
                   malta_sample(input, w, h, x, y, 0 + 4 - xs));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7______*__
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - 2),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + 2));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7__*______
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 + 2),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 - 2));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__*______
       4____0____
       5______*__
       6_______*_
       7_________
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs - xs - 3),
                   malta_sample(input, w, h, x, y, 0 - xs - 2), center,
                   malta_sample(input, w, h, x, y, 0 + xs + 2),
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_______*_
       3______*__
       4____0____
       5__*______
       6_*_______
       7_________
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs - xs + 3),
                   malta_sample(input, w, h, x, y, 0 - xs + 2), center,
                   malta_sample(input, w, h, x, y, 0 + xs - 2),
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2________*
       3______*__
       4____0____
       5__*______
       6*________
       7_________
       8_________ */

    auto sum = Sum(malta_sample(input, w, h, x, y, 0 + xs + xs - 4),
                   malta_sample(input, w, h, x, y, 0 + xs - 2), center,
                   malta_sample(input, w, h, x, y, 0 - xs + 2),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 4));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2*________
       3__*______
       4____0____
       5______*__
       6________*
       7_________
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs - xs - 4),
                   malta_sample(input, w, h, x, y, 0 - xs - 2), center,
                   malta_sample(input, w, h, x, y, 0 + xs + 2),
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 4));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0__*______
       1_________
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7_________
       8______*__ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs - 2),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs + 2));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0______*__
       1_________
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7_________
       8__*______ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs + 2),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs - 2));
    retval = __fmaf_rn(sum, sum, retval);
  }
  return retval;
}
__device__ inline float malta_hf(const float* input, int w, int h, int x,
                                 int y) {
  const int xs = 1000, xs3 = 3000;

  const auto center = malta_sample(input, w, h, x, y, 0);

  // x grows, y constant
  const auto sum_yconst = Sum(malta_sample(input, w, h, x, y, 0 - 4),
                              malta_sample(input, w, h, x, y, 0 - 3),
                              malta_sample(input, w, h, x, y, 0 - 2),
                              malta_sample(input, w, h, x, y, 0 - 1), center,
                              malta_sample(input, w, h, x, y, 0 + 1),
                              malta_sample(input, w, h, x, y, 0 + 2),
                              malta_sample(input, w, h, x, y, 0 + 3),
                              malta_sample(input, w, h, x, y, 0 + 4));
  // Will return this, sum of all line kernels
  auto retval = __fmul_rn(sum_yconst, sum_yconst);

  {
    // y grows, x constant
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs),
                   malta_sample(input, w, h, x, y, 0 - xs3),
                   malta_sample(input, w, h, x, y, 0 - xs - xs),
                   malta_sample(input, w, h, x, y, 0 - xs), center,
                   malta_sample(input, w, h, x, y, 0 + xs),
                   malta_sample(input, w, h, x, y, 0 + xs + xs),
                   malta_sample(input, w, h, x, y, 0 + xs3),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // both grow
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - 3),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 2),
                   malta_sample(input, w, h, x, y, 0 - xs - 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 2),
                   malta_sample(input, w, h, x, y, 0 + xs3 + 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // y grows, x shrinks
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 + 3),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 2),
                   malta_sample(input, w, h, x, y, 0 - xs + 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 2),
                   malta_sample(input, w, h, x, y, 0 + xs3 - 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // y grows -4 to 4, x shrinks 1 -> -1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs + 1),
                   malta_sample(input, w, h, x, y, 0 - xs3 + 1),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 1),
                   malta_sample(input, w, h, x, y, 0 - xs), center,
                   malta_sample(input, w, h, x, y, 0 + xs),
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs - 1));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    //  y grows -4 to 4, x grows -1 -> 1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs - 1),
                   malta_sample(input, w, h, x, y, 0 - xs3 - 1),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 1),
                   malta_sample(input, w, h, x, y, 0 - xs), center,
                   malta_sample(input, w, h, x, y, 0 + xs),
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs + 1));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y grows -1 to 1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - 4 - xs),
                   malta_sample(input, w, h, x, y, 0 - 3 - xs),
                   malta_sample(input, w, h, x, y, 0 - 2 - xs),
                   malta_sample(input, w, h, x, y, 0 - 1), center,
                   malta_sample(input, w, h, x, y, 0 + 1),
                   malta_sample(input, w, h, x, y, 0 + 2 + xs),
                   malta_sample(input, w, h, x, y, 0 + 3 + xs),
                   malta_sample(input, w, h, x, y, 0 + 4 + xs));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    // x grows -4 to 4, y shrinks 1 to -1
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - 4 + xs),
                   malta_sample(input, w, h, x, y, 0 - 3 + xs),
                   malta_sample(input, w, h, x, y, 0 - 2 + xs),
                   malta_sample(input, w, h, x, y, 0 - 1), center,
                   malta_sample(input, w, h, x, y, 0 + 1),
                   malta_sample(input, w, h, x, y, 0 + 2 - xs),
                   malta_sample(input, w, h, x, y, 0 + 3 - xs),
                   malta_sample(input, w, h, x, y, 0 + 4 - xs));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1__*______
       2___*_____
       3___*_____
       4____0____
       5_____*___
       6_____*___
       7______*__
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - 2),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 1),
                   malta_sample(input, w, h, x, y, 0 - xs - 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + 2));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1______*__
       2_____*___
       3_____*___
       4____0____
       5___*_____
       6___*_____
       7__*______
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 + 2),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 1),
                   malta_sample(input, w, h, x, y, 0 - xs + 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 - 2));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_*_______
       3__**_____
       4____0____
       5_____**__
       6_______*_
       7_________
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs - xs - 3),
                   malta_sample(input, w, h, x, y, 0 - xs - 2),
                   malta_sample(input, w, h, x, y, 0 - xs - 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs + 2),
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_______*_
       3_____**__
       4____0____
       5__**_____
       6_*_______
       7_________
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs - xs + 3),
                   malta_sample(input, w, h, x, y, 0 - xs + 2),
                   malta_sample(input, w, h, x, y, 0 - xs + 1), center,
                   malta_sample(input, w, h, x, y, 0 + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs - 2),
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 3));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_________
       3______***
       4___*0*___
       5***______
       6_________
       7_________
       8_________ */

    auto sum = Sum(malta_sample(input, w, h, x, y, 0 + xs - 4),
                   malta_sample(input, w, h, x, y, 0 + xs - 3),
                   malta_sample(input, w, h, x, y, 0 + xs - 2),
                   malta_sample(input, w, h, x, y, 0 - 1), center,
                   malta_sample(input, w, h, x, y, 0 + 1),
                   malta_sample(input, w, h, x, y, 0 - xs + 2),
                   malta_sample(input, w, h, x, y, 0 - xs + 3),
                   malta_sample(input, w, h, x, y, 0 - xs + 4));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_________
       1_________
       2_________
       3***______
       4___*0*___
       5______***
       6_________
       7_________
       8_________ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs - 4),
                   malta_sample(input, w, h, x, y, 0 - xs - 3),
                   malta_sample(input, w, h, x, y, 0 - xs - 2),
                   malta_sample(input, w, h, x, y, 0 - 1), center,
                   malta_sample(input, w, h, x, y, 0 + 1),
                   malta_sample(input, w, h, x, y, 0 + xs + 2),
                   malta_sample(input, w, h, x, y, 0 + xs + 3),
                   malta_sample(input, w, h, x, y, 0 + xs + 4));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0___*_____
       1___*_____
       2___*_____
       3____*____
       4____0____
       5____*____
       6_____*___
       7_____*___
       8_____*___ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs - 1),
                   malta_sample(input, w, h, x, y, 0 - xs3 - 1),
                   malta_sample(input, w, h, x, y, 0 - xs - xs - 1),
                   malta_sample(input, w, h, x, y, 0 - xs), center,
                   malta_sample(input, w, h, x, y, 0 + xs),
                   malta_sample(input, w, h, x, y, 0 + xs + xs + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs + 1));
    retval = __fmaf_rn(sum, sum, retval);
  }
  {
    /* 0_____*___
       1_____*___
       2____ *___
       3____*____
       4____0____
       5____*____
       6___*_____
       7___*_____
       8___*_____ */
    auto sum = Sum(malta_sample(input, w, h, x, y, 0 - xs3 - xs + 1),
                   malta_sample(input, w, h, x, y, 0 - xs3 + 1),
                   malta_sample(input, w, h, x, y, 0 - xs - xs + 1),
                   malta_sample(input, w, h, x, y, 0 - xs), center,
                   malta_sample(input, w, h, x, y, 0 + xs),
                   malta_sample(input, w, h, x, y, 0 + xs + xs - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 - 1),
                   malta_sample(input, w, h, x, y, 0 + xs3 + xs - 1));
    retval = __fmaf_rn(sum, sum, retval);
  }
  return retval;
}
}  // namespace satellite_butteraugli

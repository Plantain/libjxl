/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. */
#ifndef JXL_HYBRID_H_
#define JXL_HYBRID_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(JXL_HYBRID_STATIC)
#define JXL_HYBRID_EXPORT
#elif defined(JXL_HYBRID_BUILD)
#define JXL_HYBRID_EXPORT __declspec(dllexport)
#else
#define JXL_HYBRID_EXPORT __declspec(dllimport)
#endif
#else
#define JXL_HYBRID_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Experimental, pinned-fork ABI v1. All arrays are tightly packed row-major.
 * Only 512x512 RGB8 sRGB, bare codestreams, no extra channels are supported.
 * Input pointers must cover the documented sizes and remain valid until the
 * synchronous call returns. Each encoder serializes calls; independent encoders
 * can be used concurrently. Destroy only after outstanding calls have returned.
 */
#define JXL_HYBRID_ABI_VERSION 1
#define JXL_HYBRID_DIM 512
#define JXL_HYBRID_BLOCKS 4096
#define JXL_HYBRID_COLOR_TILES 64
#define JXL_HYBRID_ALL_STAGES 63u
#define JXL_HYBRID_STAGE(n) (1u << ((n) - 1))

typedef struct JxlHybridEncoder JxlHybridEncoder;

typedef struct {
  uint32_t xsize_blocks, ysize_blocks; /* must both be 64 */
  int32_t global_scale, quant_dc;
  int32_t* raw_quant_field; /* 4096 values in 1..256, constant per varblock */
  uint8_t* ac_strategy;     /* 4096: type 0..20 at origin, 0xFF elsewhere */
  int8_t *ytox_map, *ytob_map; /* 64 entries per map */
  int32_t ytox_dc, ytob_dc;
  uint32_t color_factor; /* 2..257 (pinned backend's short CfL encoding) */
  float base_correlation_x, base_correlation_b;
  uint8_t* epf_sharpness; /* 4096 entries in 0..7 */
  /* Additional frame decisions needed to bypass pre-Gaborish XYB statistics. */
  uint32_t x_qm_scale, b_qm_scale; /* each 0..7 */
} JxlHybridTileDecisions;

typedef struct {
  int8_t *ytox_map, *ytob_map;
  int32_t ytox_dc, ytob_dc;
  uint32_t color_factor;
  float base_correlation_x, base_correlation_b;
} JxlHybridColorMap;

typedef struct {
  float* quant_field; /* 4096 positive finite floats */
  float* masking;     /* 4096 nonnegative finite floats */
  float* masking1x1;  /* 512*512 nonnegative finite floats */
} JxlHybridStage1;

typedef struct {
  float* quant_field;       /* 4096 adjusted positive finite floats */
  int32_t* raw_quant_field; /* 4096 entries */
  int32_t global_scale, quant_dc;
  JxlHybridColorMap cmap;
} JxlHybridStage4;

typedef struct {
  int32_t* raw_quant_field; /* 4096 entries */
  int32_t global_scale, quant_dc;
} JxlHybridStage5;

/* Create/Destroy own all storage, including the codestream. Record overwrites
 * that storage. Views and overrides may borrow it until the next Record or
 * Destroy. Do not free individual pointers or copy the owning struct by value.
 * Arrays may be edited to construct overrides. Persist array contents, never
 * native pointers, in binary dumps. */
typedef struct {
  JxlHybridTileDecisions decisions;
  JxlHybridStage1 s1;
  JxlHybridColorMap s2;
  uint8_t* s3;
  JxlHybridStage4 s4;
  JxlHybridStage5 s5;
  uint8_t* s6;
  float* opsin[3];           /* CPU post-Gaborish planes, each 512*512 */
  const uint8_t* codestream; /* plain e8 heuristics + selected backend effort */
  size_t codestream_size;
  void* opaque;
} JxlHybridStageDump;

typedef struct {
  uint32_t stages; /* bit n-1 selects Sn; unselected pointers are ignored */
  const JxlHybridStage1* s1;
  const JxlHybridColorMap* s2;
  const uint8_t* s3;
  const JxlHybridStage4* s4;
  const JxlHybridStage5* s5;
  const uint8_t* s6;
} JxlHybridStageOverrides;

typedef struct {
  float distance;  /* finite, 0.1..25; default 1.2 */
  uint32_t effort; /* backend effort: 7 or 8; heuristics always use 8 */
  uint32_t
      threads; /* 1..16 persistent workers, one single-threaded tile each */
} JxlHybridOptions;

typedef struct {
  const uint8_t* rgb;
  size_t stride; /* >= 512*3; input includes 511*stride + 512*3 bytes */
  const JxlHybridTileDecisions* decisions; /* null: run e8 heuristics */
  const float* opsin[3]; /* all null or 3 post-Gaborish planes */
  const JxlHybridStageOverrides* overrides;
} JxlHybridTileInput;

typedef struct {
  uint8_t* data; /* caller owns, release with JxlHybridFree */
  size_t size;
  int status;
} JxlHybridOutput;

enum { JXL_HYBRID_OK = 0, JXL_HYBRID_INVALID = 1, JXL_HYBRID_ENCODE_ERROR = 2 };

JXL_HYBRID_EXPORT void JxlHybridDefaultOptions(JxlHybridOptions* options);
JXL_HYBRID_EXPORT JxlHybridEncoder* JxlHybridEncoderCreate(
    const JxlHybridOptions* options);
JXL_HYBRID_EXPORT void JxlHybridEncoderDestroy(JxlHybridEncoder* encoder);
JXL_HYBRID_EXPORT JxlHybridStageDump* JxlHybridStageDumpCreate(void);
JXL_HYBRID_EXPORT void JxlHybridStageDumpDestroy(JxlHybridStageDump* dump);
JXL_HYBRID_EXPORT void JxlHybridFree(void* buffer);
JXL_HYBRID_EXPORT int JxlHybridRecord(JxlHybridEncoder* encoder,
                                      const uint8_t* rgb, size_t stride,
                                      JxlHybridStageDump* out);
/* Full decisions and nonempty overrides are mutually exclusive. Opsin requires
 * full decisions (S1 and S6 otherwise require the pre-Gaborish image). Full
 * decisions skip all heuristics, XYB is also skipped when opsin is supplied.
 * Any subset of stage overrides is supported, including all six. */
JXL_HYBRID_EXPORT int JxlHybridEncodeTile(
    JxlHybridEncoder* encoder, const uint8_t* rgb, size_t stride,
    const JxlHybridTileDecisions* decisions, const float* const opsin[3],
    const JxlHybridStageOverrides* overrides, uint8_t** out, size_t* out_size);
/* Returns nonzero if any tile failed. All outputs are initialized, successful
 * outputs remain caller-owned even if another tile failed. Empty batches work.
 * The output array must be fresh (free previous data before reusing it). */
JXL_HYBRID_EXPORT int JxlHybridEncodeBatch(JxlHybridEncoder* encoder,
                                           const JxlHybridTileInput* tiles,
                                           size_t count,
                                           JxlHybridOutput* outputs);

#ifdef __cplusplus
}
#endif
#endif /* JXL_HYBRID_H_ */

# libjxl_hybrid (experimental ABI v1)

This pinned-fork library encodes independent 512x512 RGB8 sRGB tiles. It
records e8 decisions, substitutes any subset of S1–S6, or runs only the
backend from full decisions. The backend uses effort 7 by default; effort 8
is also supported. Codestreams use the standard decoder.

Enable `-DJPEGXL_ENABLE_HYBRID=ON` in the normal CMake configuration. Targets
are `jxl_hybrid` (shared), `jxl_hybrid-static`, and `hybrid_gate`. The static
archive includes libjxl core objects and links the normal Highway, Brotli and
jxl_cms dependencies. The option defaults to OFF; ordinary libjxl does not
link any hybrid implementation code. Core changes are a borrowed internal
hook pointer, stage call sites, an effort scope, and small setters.

## C API and stage ownership

See `lib/include/jxl/hybrid.h`. Initialize options with
`JxlHybridDefaultOptions`, create an encoder and a stage dump, then call
`JxlHybridRecord`. The dump owns all arrays and the reference codestream.
`dump->decisions` is an I1 view over its S3, S4, S5 and S6 arrays. Destroy the
dump only after all calls borrowing its views have returned. `Record` can
reuse a dump; recording replaces its contents.

`JxlHybridEncodeTile` accepts either full decisions or stage overrides.
Null decisions and null overrides run CPU e8 heuristics with the selected
backend effort. Record uses that same path and additionally collects outputs.
Stage override pointers are borrowed; only selected stages are read. Stage
outputs can be constructed in caller buffers, including GPU pinned buffers.
Every stage input is validated before use, with additional checks against the
CPU-selected strategies for S4/S5. Arrays have fixed sizes, so their allocated
length and lifetime remain the caller's responsibility.

The S2/S3/S4 hooks run per 64x64-pixel region in the original order. Full
replay and the all-six-stage override shortcut bypass those stages entirely.
The block entropy context model is always computed by the backend. S1 and
S6 need pre-Gaborish XYB, so supplied post-Gaborish opsin is accepted only
with full decisions. That path skips RGB conversion, XYB and Gaborish.

I1 includes two appended frame-header values, `x_qm_scale` and `b_qm_scale`.
These come from pre-Gaborish chromacity statistics, and are necessary for
exact replay with supplied opsin. Earlier I1 fields retain their planned
order/types. Integrators must use this version of the header and wrapper.
The supported AC strategy set is e8's types 0..20; transforms must fit one
64x64-pixel color region, with no overlap or uncovered blocks. Raw quant
fields must be constant inside each varblock. Patches/splines are not part
of I1: CPU record/override calls explicitly fail if they are discovered.
CfL color factors are limited to 2..257: the pinned encoder's CfL bit-writer
budget only accommodates the short representation (the default is 84).
Larger factors are rejected before entering the backend.

The C calls return 0 on success, 1 for invalid arguments, and 2 for encoding
failure. Encode outputs are owned by the caller and released with
`JxlHybridFree`. Batch has per-tile status and preserves successful outputs
when another tile fails. All outputs, even on error, must be inspected/freed.
The internal pool persists for the lifetime of the encoder, with one tile per
worker and no within-tile threading. An encoder serializes external calls;
independent encoders may run concurrently. Full-decision replay does no
file I/O and uses no environment-variable hooks.

## Python

```python
from jxl_hybrid import Encoder

with Encoder(distance=1.2, effort=7, threads=16) as encoder:
    with encoder.record(rgb_bytes) as dump:
        assert encoder.encode(rgb_bytes, dump.decisions) == dump.codestream
        assert encoder.encode(rgb_bytes, dump.decisions,
                              opsin=dump.opsin) == dump.codestream
        assert encoder.encode(rgb_bytes, overrides=dump.overrides([3])) == dump.codestream
        streams = encoder.encode_batch([rgb_bytes] * 16, [dump.decisions] * 16)
```

Set `PYTHONPATH=tools/hybrid` and optionally `JXL_HYBRID_LIBRARY` to the shared
library path. The default lookup finds `build-hybrid/lib/libjxl_hybrid.so`.
The wrapper needs only the standard library and also accepts contiguous numpy
uint8 arrays. Module-level `record`, `encode`, and `encode_batch` are provided.
Use `dump.fields` for C stage-array views; for example numpy callers can use
`numpy.ctypeslib.as_array(dump.fields.s1.quant_field, shape=(4096,))`.
Keep the dump open while using such array views. Binary dumps must serialize
array contents and scalar fields, never native pointer values.

WP0's encoder spec is `py:jxl_hybrid:scoreboard_encode`. That adapter takes
`(rgb_bytes, distance)` and runs the CPU oracle. It does not claim GPU speed.
The backend-only benchmark precomputes CPU decisions outside the timed region.

## Gate on sat-5090

After copying the source base once, sync this worktree excluding `.git`,
`build*`, and `third_party` to `/root/jxl-hybrid/wp1/libjxl/`. Run there:

```sh
bash tools/hybrid/remote.sh build
for d in 1.0 1.2 1.45; do
  bash tools/hybrid/remote.sh identity "$d" 8 1
  JXL_HYBRID_CORPUS=dev bash tools/hybrid/remote.sh identity "$d" 8 1
done
bash tools/hybrid/remote.sh sanitize
bash tools/hybrid/remote.sh robustness
bash tools/hybrid/remote.sh without
bash tools/hybrid/remote.sh bench 1.2 16 20
```

The script limits builds to eight CPUs at nice 10. Benchmarks acquire the
shared timing lock and run in a `--cpus=32` container with 16 encoder workers.
Every container uses `--rm`. The default corpus is the original 120 tiles;
`JXL_HYBRID_CORPUS=dev` selects WP0's 400-tile dev set. Explicit container
paths can instead follow the repeat count.

Identity checks plain encoding, full replay, supplied opsin, each single
stage, and all six stages together. It additionally compares the e8 backend
against unhooked native e8 and verifies exact decoded float RGB equality
between the e7 and e8 backends. Robustness exercises malformed decisions,
random legal decisions, every supported transform, all 64 stage subsets,
padded strides, invalid batches and reuse after errors, under ASan+UBSan.
Benchmarking reports three rounds of 2400 tiles, after a warmup; the timer
covers `JxlHybridEncodeBatch`, including validation and output allocation.
Output byte comparison and freeing are outside the timer.
The throughput command requires 16 workers and at least 2000 tiles per round,
and exits nonzero if any of its three rounds is below 900 tiles/s.

The ctypes integration suite runs inside the same build image:

```sh
docker run --rm --cpus=4 -v /root/jxl-hybrid:/w -w /w/wp1/libjxl \
  -e PYTHONPATH=tools/hybrid --entrypoint /usr/bin/env \
  satellite-cujpegxl-build:cuda12.8 \
  python3 -m unittest discover -s tools/hybrid/tests -v
```

See the branch-root `REPORT.md` and `results/*.json` for measured results.

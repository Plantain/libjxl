# WP1: libjxl_hybrid

Status: all correctness, sanitizer and build gates pass. First throughput attempt failed; one further attempt remains.
Branch: `wp1-hybrid-api`. Work is local; Claude reviews and pushes.

The optional `JPEGXL_ENABLE_HYBRID` target builds `libjxl_hybrid.so`,
`libjxl_hybrid.a` and `hybrid_gate`. The implementation provides I1–I3,
S1–S6 record/override hooks, full-decision replay, optional post-Gaborish
opsin, separate e8 heuristics and e7/e8 backend effort, a persistent pool
(maximum 16 workers), and the ctypes wrapper. Core edits are hook call sites,
a scoped effort change, and small accessors. Normal libjxl links no hybrid
implementation. API/ownership details are in `tools/hybrid/README.md` and
`lib/include/jxl/hybrid.h`. WP0 can use `py:jxl_hybrid:scoreboard_encode`.

Run on sat-5090 from `/root/jxl-hybrid/wp1/libjxl` after syncing this tree,
excluding `.git`, `build*`, and `third_party`:

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

Identity: all **20,280 checks passed** across the original 120 tiles and
WP0's 400 dev tiles at three distances. Each tile checks plain encoding,
full replay, supplied opsin, each single-stage override, all six stages,
e8-backend recording/replay against unhooked native e8, and decoded float RGB
equality between the e7 and e8 backends. There is no decoded-pixel quality
change on this set. The brief's three-distance gate takes precedence over
PLAN's five-distance table.

| Corpus | Distance | Tiles | Checks | Backend e7 bytes | Native e8 bytes | Size delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| poc120 | 1.0 | 120 | 1,560 | 3,300,991 | 3,299,946 | +0.0317% |
| poc120 | 1.2 | 120 | 1,560 | 2,893,338 | 2,893,518 | -0.0062% |
| poc120 | 1.45 | 120 | 1,560 | 2,524,034 | 2,526,943 | -0.1151% |
| dev | 1.0 | 400 | 5,200 | 6,298,729 | 6,280,194 | +0.2951% |
| dev | 1.2 | 400 | 5,200 | 5,547,734 | 5,531,139 | +0.3000% |
| dev | 1.45 | 400 | 5,200 | 4,893,365 | 4,885,878 | +0.1532% |

The poc120 d1.2 totals exactly reproduce the prototype. The new dev set is
+0.3000% in bytes at d1.2 with e7 entropy settings; the broader project's
size target still needs to account for that. WP1's identity gate compares
against the same e8-heuristic/e7-backend path, and passes.

Release robustness: **2,022 invalid inputs rejected, 117 valid cases passed**,
including all 21 supported transforms, 32 randomized quantizer/CfL cases,
all 64 override subsets, padded stride, invalid batch cleanup and reuse.
Successful randomized outputs are decoded. **ASan + UBSan passed the same
2,022 rejected / 117 valid cases, with zero diagnostics and leak detection
enabled** (`results/robustness-sanitized.json`).

Shared-library ctypes tests: **8 passed in 6.614 s**, including the WP0
adapter, borrowed lifetimes, opsin and batches. C11 public-header syntax
check passed. Exact command (inside a `--cpus=4` build-image container):

```sh
cc -std=c11 -Werror -fsyntax-only lib/include/jxl/hybrid.h
PYTHONPATH=tools/hybrid python3 -m unittest discover -s tools/hybrid/tests -v
```

Builds with hybrid ON (static/shared/driver) and OFF (`jxl`) passed. Builds
use nice 10 and at most eight CPUs. Benchmarks acquire
`/root/jxl-hybrid/locks/timing.lock`, use a `--cpus=32` container and 16
single-tile workers. Each of the three rounds encodes 2,400 tiles after
warmup, including validation and output allocation in the timer; all bytes
are checked outside it. First throughput attempt: **547.872, 562.522, 566.953 tiles/s**
(4.380584, 4.266502, 4.233158 s respectively). Each round wrote 57,866,760
bytes. The 900 tiles/s gate failed; see `results/throughput-attempt1.json`.
Investigating build/runtime differences from the prototype before a second
and final throughput gate attempt.

Deviations and resolved issues:

- I1 appends `x_qm_scale` and `b_qm_scale` (each 0..7). They are frame-header
  decisions from pre-Gaborish chromacity statistics, needed to bypass XYB
  exactly with supplied opsin. Earlier I1 fields retain their order/types.
- Only e8's transforms 0..20 within 64x64 color regions are supported.
  Patches/splines are not in I1; CPU record/override returns an error if any
  are found. Backend effort is 7 or 8.
- CfL color factors are limited to 2..257. Isolation of a release robustness
  failure showed that `ColorCorrelationEncodeDC` budgets 12 bits for the
  factor, while values >=258 need 18. Factors 84/257 encode; 258/60000 fail.
  Validation now rejects them before the backend; production factor 84 is
  unaffected. The quantizer itself passed the previously suspect values.
- Sanitizer attempt 1 failed before stage processing in existing
  `headers.cc:FixedAspectRatios`. GCC 13 with `-fmerge-all-constants` promotes
  local constexpr arrays to globals which ASan then poisons at scope exit.
  The isolated `tools/hybrid/tests/asan_constants_probe.cc` reproduces this
  (exit 1 with merging, 0 without). The hybrid CMake file disables only this
  optimization in GCC sanitizer builds. Lifetime checking, leak detection
  and UBSan remain enabled. Production flags/heuristics are unchanged.
  GCC [documents the optimization as non-conforming](https://gcc.gnu.org/onlinedocs/gcc-13.5.0/gcc/Optimize-Options.html#index-fmerge-all-constants).

Small numeric evidence is in `results/*.json`; full remote build/test logs
are `/root/jxl-hybrid/wp1/*.log`. No GPU code or allocations, no operational
service changes, no pushed commits. Remote disk use is 2,242,700 KiB (about 2.14 GiB), below 50 GB.

Blockers: none currently.

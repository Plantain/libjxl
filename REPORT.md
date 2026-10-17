# WP3 GPU reconstruction and Butteraugli spike

Status: **numerical gate passes; initial B=8 timing gate passes; final batch sweep in progress**.
Branch `wp3-gpu-spike`, baseline `d7c57e62` (production `e8ff0976` plus the
unused-evaluation removal). All WP3 work is uncommitted for Claude's review;
the CPU dump hook is staged separately. An earlier local hook commit was
returned to the index without changing file contents to follow COMMON.md's
no-commit workflow. No pushes. Intended commits:

1. `Add optional WP3 dumps of actual quantization-loop reconstruction state`:
   staged changes in `CMakeLists.txt`, `gpu/CMakeLists.txt`, `gpu/dump-inl.h`,
   `gpu/dump_main.cc`, `gpu/spike_format.h`,
   `lib/jxl/enc_adaptive_quantization.cc` and `lib/jxl/enc_modular.cc`.
2. `Add batched CUDA reconstruction and cached Butteraugli spike`:
   remaining `gpu/CMakeLists.txt` and `gpu/spike_format.h` changes, all new
   `gpu/*.cuh`, `gpu/spike.cu`, `gpu/transform_cases.cc`, `gpu/*.py`,
   `gpu/remote.sh`, `REPORT.md` and `results/*.json`.

Built an opt-in `JPEGXL_GPU_SPIKE` CMake target for CUDA 12.8 / SM120. The
batched FP32 path performs AC/DC dequantization, CfL, DC smoothing and LLF
synthesis, inverse transforms, Gaborish, EPF1, XYB-to-linear (including the
CPU roundtrip's sRGB transfer), cached full/half-resolution Butteraugli and
TileDistMap. Tile index is on the grid's z axis; there are no host syncs inside
an evaluation. Blurs use coalesced writes; both DCT axes share one CTA.

From this worktree, synchronize and build with:

```sh
rsync -a --exclude=.git --exclude='build*' --exclude=third_party --exclude=__pycache__ ./ root@sat-5090:/root/jxl-hybrid/wp3/libjxl/
ssh -o BatchMode=yes -o ConnectTimeout=10 root@sat-5090 sh -s -- build < gpu/remote.sh
```

The remote checkout was initially copied from `/root/jxl-hybrid/src-base/libjxl`.
Inputs are the 120 `.rgb` tiles copied from `/root/claude-gpu-hybrid-poc/tiles/`
to `/root/jxl-hybrid/wp3/tiles/`; the WP0 corpus was empty at launch. Dump both
quantization-loop iterations and run the correctness gates:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 root@sat-5090 sh -s -- dump < gpu/remote.sh
ssh -o BatchMode=yes -o ConnectTimeout=10 root@sat-5090 sh -s -- compare 8 1 < gpu/remote.sh
ssh -o BatchMode=yes -o ConnectTimeout=10 root@sat-5090 sh -s -- transforms 1 1 < gpu/remote.sh
ssh -o BatchMode=yes -o ConnectTimeout=10 root@sat-5090 sh -s -- validate 32 1 < gpu/remote.sh
```

At d1.2, the CPU oracle encodes **2,893,518 bytes**, exactly reproducing the
README-EVIDENCE total. This is an oracle check, not a GPU encoder size claim.
On all 120 tiles, B=8 and B=32 produce the same aggregate errors:

| Measurement | Observed | Gate |
| --- | ---: | ---: |
| Reconstructed linear mean absolute error | 1.28139724e-7 | <= 1e-4 |
| Reconstructed linear max absolute error | 6.61611557e-6 | report |
| Diffmap p99 relative error | 5.29011741e-5 | <= 1e-3 |
| Diffmap max relative error | 0.0044397764 | report |
| TileDistMap p99 relative error | 1.24239959e-5 | report |
| Raw `>1` classification agreement | 100% | >= 99.5% |
| Update-rule `>distance` classification agreement | 100% | >= 99.5% |
| Per-tile TileDistMap argmax agreement | 100% | >= 99% |

Relative errors use `max(abs(CPU), 1e-12)` as denominator. The comparator alone
is bit-exact on all 120 CPU reconstructions. Eighteen strategy types occur;
all are implemented on the GPU. Independent pinned-CPU transform checks pass
for every type 0–20, including the absent 4x4 and 8x32/32x8 types, with maximum
absolute error **4.47034836e-7**. No CPU fallback is used. Numerical evidence is
in `results/numerics-d1.2*.json`, `results/comparator-d1.2.json`,
`results/transforms.json`, `results/coverage-d1.2.json` and `results/dump.json`.

Timing and profile commands:

```sh
for b in 4 8 16 32; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 root@sat-5090 sh -s -- bench "$b" 100 < gpu/remote.sh > "results/timing-b$b.json"
  scp "root@sat-5090:/root/jxl-hybrid/wp3/memory-$b.json" "results/memory-b$b.json"
done
ssh -o BatchMode=yes -o ConnectTimeout=10 root@sat-5090 sh -s -- profile 8 20 < gpu/remote.sh
scp root@sat-5090:/root/jxl-hybrid/wp3/kernels.csv /tmp/wp3-kernels.csv
python3 gpu/summarize_profile.py /tmp/wp3-kernels.csv 8 20 > results/nsys.json
python3 gpu/check_gate.py results > results/gate.json
```

The wrapper holds `/root/jxl-hybrid/locks/timing.lock` for all timing/profiling,
checks >=6 GiB free, and serializes this package's GPU jobs. Builds use
`nice -n 10`, `--cpus=8`; all containers use `--rm`. No operational service
changes. CUDA-event timings use three warmups and 100 repetitions on B evenly
spaced entries from the sorted 120-tile corpus. They include reconstruction,
both comparator scales and TileDistMap, with cached reference setup measured
separately. Dump generation, allocation and PCIe upload are excluded.

Explicit device allocation is **47 MiB/tile**, **1504 MiB at B=32**. The measured
B=32 process peak is **2004 MiB**, including **500 MiB** of CUDA overhead, below
the 2048 MiB cap. The initial 56 MiB/tile layout measured 2292 MiB and was
rejected. Reusing the dead preprocessing buffers for comparator errors and
masking reduced scratch from 26 to 17 MiB/tile; numerical results are unchanged.
The wrapper samples process VRAM into `/root/jxl-hybrid/wp3/memory-B.json`.
The first locked B=8 timing measured **0.307924807 ms/evaluation/tile** and
**0.0942361653 ms/reference/tile**, passing both time limits. The compact layout's
B=4 run measured **0.360431671 ms/evaluation/tile** and
**0.11200504 ms/reference/tile**, with **688 MiB** process VRAM. B=8/16/32 timings
remain queued. Remote inputs,
builds, logs and a private Nsight Systems 2025.1.3 installation are under
`/root/jxl-hybrid/wp3/`, currently about 5 GiB. No binary dumps or profiler
archives are committed.

The final compact-layout Nsight capture covers 20 B=8 evaluations and sums to
**0.3087887125 ms of kernel time per tile**. Profiling overhead and gaps between
kernels make this distinct from the unprofiled CUDA-event gate measurement.
The largest kernel totals, each divided by 160 evaluated tiles, are:

| Kernel | ms/tile |
| --- | ---: |
| `convolve_axis` | 0.070819725 |
| `malta_difference` | 0.060150294 |
| `malta_accumulate` | 0.033400875 |
| `tile_distance` | 0.027842050 |
| `inverse_dct` | 0.019972438 |
| `l2_errors` | 0.014830819 |

All kernel totals are in `results/nsys.json`; the native capture is
`/root/jxl-hybrid/wp3/profile.nsys-rep`. Nsight's initial stats export rejected
its stale SQLite cache after the new capture. Re-exporting with
`--force-export=true` succeeded; the wrapper now supplies that flag.

Scope limits: private dump input expands matrix weights into coefficient
footprints and exports decoded CfL ratios instead of the production I1 layout.
Types 21–26 (128/256 transforms), absent from e8 output here, are rejected.
The accepted filter profile is Gaborish+EPF1: the pinned encoder's thresholds
0.7/1.5/4.0 select EPF1 throughout d1.0–1.45. This is not the production engine
API. Forward quantization, AC search and the update rule remain out of scope.

A development error in the reused local transpose for the 8x8 DC transform was
removed by explicit shared-memory staging between the two passes; the final
corpus and transform checks above include that fix. No unresolved numerical
blockers. The remaining B=8/16/32 measurements are queued behind another project
job's timing lock. No implementation blockers; the profile is complete.

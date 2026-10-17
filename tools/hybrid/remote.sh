#!/usr/bin/env bash
# Run inside /root/jxl-hybrid/wp1/libjxl on sat-5090. Containers are removed.
set -euo pipefail
mode=${1:-build}
image=satellite-cujpegxl-build:cuda12.8
root=/root/jxl-hybrid
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_HYBRID=ON
  -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_TOOLS=OFF -DJPEGXL_ENABLE_DEVTOOLS=OFF
  -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF
  -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF
  -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_TCMALLOC=OFF)
case "$mode" in
  build|sanitize|without)
    build=build-hybrid
    if [[ $mode == sanitize ]]; then
      build=build-hybrid-asan
      common+=(-DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DSANITIZER=asan
        '-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer'
        '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer'
        '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined'
        '-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address,undefined')
    elif [[ $mode == without ]]; then
      build=build-without-hybrid
      common+=(-DJPEGXL_ENABLE_HYBRID=OFF)
    fi
    docker run --rm --cpus=8 -v "$root:/w" -w /w/wp1/libjxl --entrypoint /usr/bin/env "$image" \
      nice -n 10 cmake -S . -B "$build" "${common[@]}"
    targets=(jxl_hybrid jxl_hybrid-static hybrid_gate)
    if [[ $mode == without ]]; then targets=(jxl); fi
    docker run --rm --cpus=8 -v "$root:/w" -w /w/wp1/libjxl --entrypoint /usr/bin/env "$image" \
      nice -n 10 cmake --build "$build" --parallel 8 --target "${targets[@]}"
    ;;
  identity|bench|robustness)
    shift
    cpus=8
    build=build-hybrid
    if [[ $mode == robustness ]]; then build=build-hybrid-asan; fi
    command=(docker run --rm --cpus="$cpus" -v "$root:/w"
      -v /root/claude-gpu-hybrid-poc/tiles:/tiles:ro -w /w/wp1/libjxl
      -e ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
      -e UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
      --entrypoint /usr/bin/env "$image" "$build/tools/hybrid_gate" "$mode" "$@")
    if [[ $mode != robustness && $# == 3 ]]; then
      host_tiles=/root/claude-gpu-hybrid-poc/tiles
      container_tiles=/tiles
      if [[ ${JXL_HYBRID_CORPUS:-poc120} == dev ]]; then
        host_tiles=$root/corpus/dev
        container_tiles=/w/corpus/dev
      fi
      for tile in "$host_tiles"/*.rgb; do
        command+=("$container_tiles/${tile##*/}")
      done
    fi
    if [[ $mode == bench ]]; then
      command[3]=--cpus=32
      flock "$root/locks/timing.lock" "${command[@]}"
    else
      "${command[@]}"
    fi
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

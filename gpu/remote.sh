#!/bin/sh
# Run on sat-5090; exclusively owns wp3 paths and ephemeral wp3 containers.
set -eu
root=/root/jxl-hybrid
image=satellite-cujpegxl-build:cuda12.8
case ${1:-} in
 build)
 docker run --rm --name wp3-build --cpus=8 --entrypoint /bin/sh -v "$root:/w" "$image" -c 'nice -n 10 cmake -S /w/wp3/libjxl -B /w/wp3/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_TOOLS=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_JPEGLI=OFF -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_GPU_SPIKE=ON > /w/wp3/configure.log 2>&1 && nice -n 10 cmake --build /w/wp3/build --target wp3_dump wp3_spike -j8 > /w/wp3/build.log 2>&1'
 ;;
 dump)
 shift
 flock "$root/locks/timing.lock" docker run --rm --name wp3-dump --cpus=8 --entrypoint /bin/sh -v "$root:/w" "$image" -c '/w/wp3/build/gpu/wp3_dump 1.2 /w/wp3/dumps /w/wp3/tiles/*.rgb > /w/wp3/dump.json 2> /w/wp3/dump.log'
 ;;
 smoke|validate|compare|bench|profile|transforms)
 action=$1; batch=${2:-8}; repeats=${3:-20}
 if [ "$action" = bench ] || [ "$action" = profile ]; then
   exec 9>"$root/locks/timing.lock"; flock 9
 fi
 exec 8>"$root/wp3/gpu.lock"; flock 8
 free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
 test "$free" -ge 6144 || { echo 'Less than 6 GiB free' >&2; exit 1; }
 if [ "$action" = bench ] || [ "$action" = validate ]; then
   python3 "$root/wp3/libjxl/gpu/monitor_memory.py" "$root/wp3/memory-$batch.json" &
   monitor=$!
   trap 'kill -TERM "$monitor"; wait "$monitor"' EXIT
 fi
 docker run --rm --name wp3-gpu --env WP3_GPU_LOCK_HELD=1 --gpus all --cpus=8 --entrypoint /bin/sh -v "$root:/w" "$image" -c '
 set -e
 action=$1; batch=$2; repeats=$3
 if [ "$action" = smoke ]; then
   /w/wp3/build/gpu/wp3_spike compare "$batch" 1 /w/wp3/dumps/ahi-022-001.i1.wp3
   /w/wp3/build/gpu/wp3_spike validate "$batch" 1 /w/wp3/dumps/ahi-022-001.i1.wp3
 elif [ "$action" = profile ]; then
   /w/wp3/nsys/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys profile --sample=none --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o /w/wp3/profile /w/wp3/build/gpu/wp3_spike profile "$batch" "$repeats" /w/wp3/dumps/*.i1.wp3
   /w/wp3/nsys/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv /w/wp3/profile.nsys-rep > /w/wp3/kernels.csv
 else
   /w/wp3/build/gpu/wp3_spike "$action" "$batch" "$repeats" /w/wp3/dumps/*.i1.wp3
 fi' sh "$action" "$batch" "$repeats"
 ;;
 *) echo 'usage: remote.sh build|dump|smoke|validate|compare|bench|profile|transforms [batch] [repeats]' >&2;exit 2;;
esac

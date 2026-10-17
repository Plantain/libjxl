# Optional spike. All normal libjxl builds are unchanged.
add_executable(wp3_dump dump_main.cc)
target_compile_features(wp3_dump PRIVATE cxx_std_17)
target_link_libraries(wp3_dump PRIVATE jxl-internal)

enable_language(CUDA)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "WP3 requires CUDA 12.8 or newer")
endif()
add_executable(wp3_spike spike.cu transform_cases.cc)
target_link_libraries(wp3_spike PRIVATE jxl-internal)
set_target_properties(wp3_spike PROPERTIES CUDA_ARCHITECTURES 120 CUDA_STANDARD 17)
target_compile_options(wp3_spike PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)

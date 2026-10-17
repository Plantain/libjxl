# Experimental seam: optional public static and shared libraries.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND SANITIZER MATCHES "asan|ubsan")
  # GCC 13 promotes local constexpr arrays to merged globals, then poisons
  # them at scope exit under ASan. See tools/hybrid/tests/asan_constants_probe.cc.
  # Preserve lifetime instrumentation; disable this non-conforming optimization.
  target_compile_options(jxl_dec-obj PRIVATE -fno-merge-all-constants)
  target_compile_options(jxl_enc-obj PRIVATE -fno-merge-all-constants)
endif()
configure_file(../../include/jxl/hybrid.h
               ${PROJECT_BINARY_DIR}/lib/include/jxl/hybrid.h COPYONLY)
add_library(jxl_hybrid-obj OBJECT api.cc hooks.cc validate.cc)
target_link_libraries(jxl_hybrid-obj PUBLIC jxl-internal)
target_compile_definitions(jxl_hybrid-obj PRIVATE JXL_HYBRID_BUILD)
set_target_properties(jxl_hybrid-obj PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
add_library(jxl_hybrid SHARED $<TARGET_OBJECTS:jxl_hybrid-obj>)
add_library(jxl_hybrid-static STATIC $<TARGET_OBJECTS:jxl_hybrid-obj>
            ${JPEGXL_INTERNAL_OBJECTS})
set_target_properties(jxl_hybrid-static PROPERTIES OUTPUT_NAME jxl_hybrid)
target_compile_definitions(jxl_hybrid-static INTERFACE JXL_HYBRID_STATIC JXL_STATIC_DEFINE)
foreach(target jxl_hybrid jxl_hybrid-static)
  target_link_libraries(${target} PUBLIC jxl_base jxl_cms
                         PRIVATE ${JPEGXL_INTERNAL_LIBS})
  set_target_properties(${target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endforeach()
target_link_libraries(jxl_hybrid PRIVATE jxl-internal)
if(UNIX AND NOT APPLE)
  target_link_options(jxl_hybrid PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/hybrid.version")
  set_property(TARGET jxl_hybrid APPEND PROPERTY LINK_DEPENDS
               ${CMAKE_CURRENT_SOURCE_DIR}/hybrid.version)
endif()
install(TARGETS jxl_hybrid jxl_hybrid-static
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
add_executable(hybrid_gate ${PROJECT_SOURCE_DIR}/tools/hybrid/hybrid_gate.cc)
target_link_libraries(hybrid_gate PRIVATE jxl_hybrid-static)
set_target_properties(hybrid_gate PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tools)

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(_NAME ireecompiler)

project(${_NAME} VERSION 1.0 LANGUAGES CXX C)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
include(CheckCCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(LLVM_DIR "${IREE_INSTALL_PREFIX}/llvm-project/lib/cmake/llvm")
# set(MLIR_DIR "${IREE_INSTALL_PREFIX}/lib/cmake/mlir")
# set(LLD_DIR "${IREE_INSTALL_PREFIX}/lib/cmake/lld")
# set(Clang_DIR "${IREE_INSTALL_PREFIX}/lib/cmake/clang")

set(LLVM_ABI_BREAKING_CHECKS FORCE_OFF)

set(IREE_INPUT_STABLEHLO ON)
# set(IREE_BUILD_BUNDLED_LLVM ON)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)

set(IREE_HAL_DRIVER_DEFAULTS ON)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON)

if(CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  set(IREE_SIZE_OPTIMIZED ON)
endif()

set(C_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../exla/iree")
file(GLOB iree_runtime_sources CONFIGURE_DEPENDS "${C_SRC}/*.cc" "${C_SRC}/*.h" "${C_SRC}/../exla_nif_util.cc" "${C_SRC}/../exla_nif_util.h")


if(IREE_BUILD_FOR_IOS)
  set(BUILD_IREE_COMPILER OFF)
  file(GLOB iree_runtime_sources CONFIGURE_DEPENDS "${C_SRC}/runtime.cc" "${C_SRC}/runtime.h")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_FOR_IOS")
endif()

add_library(${_NAME} SHARED ${iree_runtime_sources})
set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD 17)

if (IREE_BUILD_FOR_IOS)
  set(__BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/ios_runtime/build")
else()
  set(__BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/build")
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree" ${__BUILD_DIR} EXCLUDE_FROM_ALL)

install(
    TARGETS ${_NAME}
    DESTINATION "."
)

set_target_properties(${_NAME} PROPERTIES SUFFIX ".so")

if (NOT IREE_BUILD_FOR_IOS)
set_target_properties(${_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)
endif()

if(NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared")
  set_target_properties(${_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/${IREE_COMPILER_DIR}")
else()
  # Although the compiler complains about not using these,
  # things only work with them set
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
  check_c_compiler_flag("-arch arm64" ARM64_SUPPORTED)
  if(ARM64_SUPPORTED)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAC_ARM64")
  endif()

  if(IREE_BUILD_FOR_IOS)
  else()
    set_target_properties(${_NAME} PROPERTIES INSTALL_RPATH "@loader_path/${IREE_COMPILER_DIR}")
  endif()
endif()

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -Wno-unused-function -Wno-sign-compare -Wno-comment -Wno-macro-redefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers -DLLVM_VERSION_STRING= -std=c++17")

if($ENV{DEBUG})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

add_definitions(-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

target_link_libraries(${_NAME} iree_runtime_runtime)

if(IREE_BUILD_FOR_IOS)
else()
  if(NOT APPLE)
    target_link_libraries(${_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/build/lib/libIREECompiler.so")
  else()
    target_link_libraries(${_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/build/lib/libIREECompiler.dylib")
  endif()

  target_include_directories(${_NAME} PUBLIC $ENV{ERTS_INCLUDE_DIR})
  target_include_directories(${_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../${XLA_INCLUDE_PATH}")
  target_include_directories(${_NAME} SYSTEM
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../../${IREE_COMPILER_INCLUDE_PATH}/iree/compiler"
  )
  target_include_directories(${_NAME} SYSTEM
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/third_party/llvm-project/mlir/include"
  )

  set(XLA_EXTENSION_LIB_PATH ${XLA_EXTENSION_LIB})
  set(XLA_EXTENSION_INCLUDE_PATH ${XLA_INCLUDE_PATH})
  include_directories(${XLA_EXTENSION_INCLUDE_PATH})
  target_link_libraries(${_NAME} "${XLA_EXTENSION_LIB}/libxla_extension.so")
  target_link_libraries(${_NAME} iree_compiler_bindings_c_loader)
endif()

if($ENV{DEBUG})
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/third_party/tracy" "${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/third_party/tracy/build")
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../cache/iree/third_party/tracy/public")
  target_link_libraries(${_NAME} TracyClient)
endif()

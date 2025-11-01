cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(torchx)
include(CheckCCompilerFlag)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
  set(CMAKE_TOOLCHAIN_FILE $ENV{CMAKE_TOOLCHAIN_FILE})
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(C_SRC "${CMAKE_CURRENT_SOURCE_DIR}/c_src")
set(CMAKE_INSTALL_PREFIX "$ENV{PRIV_DIR}")
if(NOT IS_DIRECTORY "$ENV{PRIV_DIR}")
  file(MAKE_DIRECTORY "$ENV{PRIV_DIR}")
endif()

set(Torch_DIR "$ENV{LIBTORCH_DIR}/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH $ENV{LIBTORCH_DIR})

message(STATUS "CMAKE_TOOLCHAIN_FILE: $ENV{CMAKE_TOOLCHAIN_FILE}")
message(STATUS "C_SRC: ${C_SRC}")
message(STATUS "PRIV_DIR: $ENV{PRIV_DIR}")
message(STATUS "LIBTORCH_DIR: $ENV{LIBTORCH_DIR}")
message(STATUS "ERTS_INCLUDE_DIR: $ENV{ERTS_INCLUDE_DIR}")
message(STATUS "FINE_INCLUDE_DIR: $ENV{FINE_INCLUDE_DIR}")
message(STATUS "LIBTORCH_BASE: $ENV{LIBTORCH_BASE}")
message(STATUS "MIX_BUILD_EMBEDDED $ENV{MIX_BUILD_EMBEDDED}")
message(STATUS "LIBTORCH_LINK $ENV{LIBTORCH_LINK}")

find_package(Torch REQUIRED)

STRING(REGEX REPLACE "\\\\" "/" C_SRC ${C_SRC})

file(GLOB torchx_sources CONFIGURE_DEPENDS "${C_SRC}/*.cpp" "${C_SRC}/*.hpp")

add_library(torchx SHARED ${torchx_sources})

target_link_libraries(torchx "${TORCH_LIBRARIES}")
set_property(TARGET torchx PROPERTY CXX_STANDARD 17)

target_include_directories(torchx PUBLIC $ENV{ERTS_INCLUDE_DIR} $ENV{FINE_INCLUDE_DIR})

install(
    TARGETS torchx
    DESTINATION "$ENV{PRIV_DIR}"
)
SET_TARGET_PROPERTIES(torchx PROPERTIES PREFIX "")
if(NOT WIN32)
    set_target_properties(torchx PROPERTIES SUFFIX ".so")
endif()

if(WIN32)
  file(GLOB torchx_dll_loader_sources CONFIGURE_DEPENDS "${C_SRC}/dll_loader/*.cpp")
  add_library(torchx_dll_loader SHARED ${torchx_dll_loader_sources})
  set_property(TARGET torchx_dll_loader PROPERTY CXX_STANDARD 17)
  target_include_directories(torchx_dll_loader PUBLIC $ENV{ERTS_INCLUDE_DIR})
  install(
    TARGETS torchx_dll_loader
    DESTINATION "$ENV{PRIV_DIR}"
  )
  SET_TARGET_PROPERTIES(torchx_dll_loader PROPERTIES PREFIX "")
endif()

if (Torch_VERSION_MAJOR EQUAL 1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSING_TORCH_V1")
endif()

if(UNIX)
  set_target_properties(torchx PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
  )

  if(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared")
    set_target_properties(torchx PROPERTIES INSTALL_RPATH "\$ORIGIN/$ENV{LIBTORCH_BASE}")
  else()
    # Although the compiler complains about not using these,
    # things only work with them set
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
    check_c_compiler_flag("-arch arm64" ARM64_SUPPORTED)
    if(ARM64_SUPPORTED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAC_ARM64")
    endif()
    # set(CMAKE_SHARED_LINKER_FLAGS "-bundle -flat_namespace -undefined suppress")
    set_target_properties(torchx PROPERTIES INSTALL_RPATH "@loader_path/$ENV{LIBTORCH_BASE}")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers")
else()
  # On Windows
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4624")
endif()

# https://pytorch.org/cppdocs/installing.html
#
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if(MSVC)
  file(GLOB TORCH_DLLS "$ENV{LIBTORCH_DIR}/lib/*.dll")
  if(NOT IS_DIRECTORY "$ENV{PRIV_DIR}/$ENV{LIBTORCH_BASE}")
   file(MAKE_DIRECTORY "$ENV{PRIV_DIR}/$ENV{LIBTORCH_BASE}")
  endif()
  add_custom_command(TARGET torchx
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    "$ENV{PRIV_DIR}/$ENV{LIBTORCH_BASE}")
else()
  if(ENV{MIX_BUILD_EMBEDDED} STREQUAL "true")
    set(EMBEDDED_TYPE copy_directory)
  else()
    set(EMBEDDED_TYPE create_symlink)
  endif()
  add_custom_command(TARGET torchx
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E "${EMBEDDED_TYPE}"
                      "$ENV{LIBTORCH_DIR}/lib"
                      "$ENV{PRIV_DIR}/$ENV{LIBTORCH_BASE}")
endif()

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(torchx)
include(CheckCCompilerFlag)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(Torch REQUIRED)

STRING(REGEX REPLACE "\\\\" "/" C_SRC ${C_SRC})

file(GLOB torchx_sources CONFIGURE_DEPENDS "${C_SRC}/*.cpp" "${C_SRC}/*.hpp")

if (NOT APPLE)
add_library(torchx SHARED ${torchx_sources})
else()
add_library(torchx MODULE ${torchx_sources})
endif()

target_link_libraries(torchx "${TORCH_LIBRARIES}")
set_property(TARGET torchx PROPERTY CXX_STANDARD 14)

target_include_directories(torchx PUBLIC ${ERTS_INCLUDE_DIR})

SET_TARGET_PROPERTIES(torchx PROPERTIES PREFIX "")

if(UNIX)
set_target_properties(torchx PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared")
    set_target_properties(torchx PROPERTIES INSTALL_RPATH "\$ORIGIN/${LIBTORCH_BASE}")
else()
    # Although the compiler complains about not using these,
    # things only work with them set
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup")
    check_c_compiler_flag("-arch arm64" ARM64_SUPPORTED)
    if(ARM64_SUPPORTED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAC_ARM64")
    endif()
    # set(CMAKE_SHARED_LINKER_FLAGS "-bundle -flat_namespace -undefined suppress")
    set_target_properties(torchx PROPERTIES INSTALL_RPATH "@loader_path/${LIBTORCH_BASE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers")
else()
# On Windows
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4624")
endif()


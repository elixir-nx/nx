cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(torchx)

find_package(Torch REQUIRED)

file(GLOB torchx_sources CONFIGURE_DEPENDS "${C_SRC}/*.cpp" "${C_SRC}/*.hpp")

add_executable(torchx.so ${torchx_sources})
target_link_libraries(torchx.so "${TORCH_LIBRARIES}")
set_property(TARGET torchx.so PROPERTY CXX_STANDARD 14)

set_target_properties(torchx.so PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(torchx.so PROPERTIES INSTALL_RPATH "\$ORIGIN/libtorch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -I${ERTS_INCLUDE_DIR} -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers")

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared")
endif()
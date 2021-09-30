cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchx)

find_package(Torch REQUIRED)

add_executable(torchx.so ../c_src/torchx.cpp ../c_src/nx_nif_utils.hpp)
target_link_libraries(torchx.so "${TORCH_LIBRARIES}")
set_property(TARGET torchx.so PROPERTY CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -I${ERTS_INCLUDE_DIR} -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -shared")
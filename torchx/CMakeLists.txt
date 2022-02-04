cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(torchx)

find_package(Torch REQUIRED)

STRING(REGEX REPLACE "\\\\" "/" C_SRC ${C_SRC}) 

file(GLOB torchx_sources CONFIGURE_DEPENDS "${C_SRC}/*.cpp" "${C_SRC}/*.hpp")

add_library(torchx SHARED ${torchx_sources})
target_link_libraries(torchx "${TORCH_LIBRARIES}")
set_property(TARGET torchx PROPERTY CXX_STANDARD 14)

set_target_properties(torchx PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_include_directories(torchx PUBLIC ${ERTS_INCLUDE_DIR}) 

if(UNIX)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
if(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared")
    set_target_properties(torchx PROPERTIES INSTALL_RPATH "\$ORIGIN/${LIBTORCH_BASE}")
else()
    # Although the compiler complains about not using these,
    # things only work with them set
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup -dynamiclib")
    set(CMAKE_SHARED_LINKER_FLAGS "-bundle -flat_namespace -undefined suppress")
    set_target_properties(torchx PROPERTIES INSTALL_RPATH "@loader_path/${LIBTORCH_BASE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC  -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers")
else()
# On Windows
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()


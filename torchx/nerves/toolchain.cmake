# maybe the easiest way to identify a nerves build?
if(NOT "$ENV{TARGET_CPU}" STREQUAL "" AND NOT "$ENV{TARGET_ARCH}" STREQUAL "" AND NOT "$ENV{TARGET_OS}" STREQUAL "" AND NOT "$ENV{TARGET_ABI}" STREQUAL "")
    if("$ENV{MIX_TARGET}" MATCHES "^(rpi4|rpi3|rpi3a|rpi2|rpi0|rpi|bbb|x86_64|osd32mp1)$")
        # get_filename_component required at least 3.4
        cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

        set(CMAKE_SYSTEM "Linux")
        set(CMAKE_SYSTEM_NAME "Linux")

        # set nerves sysroot
        find_program(NERVES_CC "${CMAKE_C_COMPILER}")
        string(REPLACE "-gcc" "" TOOLCHAIN_PREFIX "${CMAKE_C_COMPILER}")
        get_filename_component(NERVES_CC_PATH "${NERVES_CC}" PATH)
        set(CMAKE_FIND_ROOT_PATH "${NERVES_CC_PATH}/../${TOOLCHAIN_PREFIX}/sysroot;${LIBTORCH_DIR}")

        # disable nepn opt in libpng as the compiler was not compiled with neon enabled
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPNG_ARM_NEON_OPT=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPNG_ARM_NEON_OPT=0")

        # adjust the default behavior of the find commands:
        # search headers and libraries in the target environment
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
        set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

        # search programs in the host environment
        set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    endif()
endif()

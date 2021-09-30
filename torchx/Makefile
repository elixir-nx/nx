LIBTORCH_DIR ?= "$(shell pwd)/_build/libtorch"
LIBTORCH_ZIP ?= "$(LIBTORCH_DIR).zip"
CMAKE_BUILD_DIR ?= $(shell pwd)/_build/cmake

PRIV_DIR = $(MIX_APP_PATH)/priv
TORCHX_SO = $(PRIV_DIR)/torchx.so

C_SRC = $(shell pwd)/c_src

OS=$(shell uname -s)

ifeq ($(OS), Linux)
	LIBTORCH_DOWNLOAD_URL ?= https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcpu.zip
endif

ifeq ($(OS), Darwin)
	LIBTORCH_DOWNLOAD_URL ?= https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.1.zip
endif

.DEFAULT_GLOBAL := build

build: libtorch $(TORCHX_SO)

libtorch:
	@ if [ ! -d "$(LIBTORCH_DIR)" ]; then \
		wget $(LIBTORCH_DOWNLOAD_URL) --no-check-certificate -O $(LIBTORCH_ZIP) && \
		unzip $(LIBTORCH_ZIP) -d /tmp/torchx && \
		mv -T /tmp/torchx/libtorch $(LIBTORCH_DIR); \
	fi

SOURCES_CPP = $(wildcard $(C_SRC)/*.cpp)
SOURCES_HPP = $(wildcard $(C_SRC)/*.hpp)

$(TORCHX_SO): $(SOURCES_CPP) $(SOURCES_HPP)
	mkdir -p $(CMAKE_BUILD_DIR)
	mkdir -p $(PRIV_DIR)
	cd $(CMAKE_BUILD_DIR) && \
		cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DC_SRC=$(C_SRC) -S $(shell pwd) && \
		cmake --build .
	mv $(CMAKE_BUILD_DIR)/torchx.so $(TORCHX_SO)

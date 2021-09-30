LIBTORCH_DIR ?= "./_build/libtorch"

PRIV_DIR = $(MIX_APP_PATH)/priv
TORCHX_SO = $(PRIV_DIR)/torchx.so

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
		wget $(LIBTORCH_DOWNLOAD_URL) --no-check-certificate -O _build/libtorch.zip && \
		unzip _build/libtorch.zip -d $(LIBTORCH_DIR); \
	else \
		echo "libtorch found at $(LIBTORCH_DIR)"; \
	fi

$(TORCHX_SO): c_src/torchx.cpp c_src/nx_nif_utils.hpp
	mkdir -p $(PRIV_DIR)
	cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -B ./cmake ./cmake
	cmake --build ./cmake --config Release
	mv ./cmake/torchx.so $(TORCHX_SO)

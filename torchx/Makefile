LIBTORCH_INCLUDE_PATH = $(LIBTORCH_DIR)/include/
LIBTORCH_API_INCLUDE_PATH = $(LIBTORCH_DIR)/include/torch/csrc/api/include
LIBTORCH_LINK_PATH = $(LIBTORCH_DIR)/lib/

PRIV_DIR = $(MIX_APP_PATH)/priv
LIBTORCH_LIB_DIR = $(PRIV_DIR)/lib
TORCHX_SO = $(PRIV_DIR)/torchx.so

.DEFAULT_GLOBAL := build

build: check_libtorch $(TORCHX_SO)

check_libtorch:
	@ if [ -z "$(LIBTORCH_DIR)" ]; then \
		echo "LIBTORCH_DIR env var is not set. It should point to the directory, where LibTorch is installed."; \
		echo "You can download LibTorch here: https://pytorch.org/get-started/locally/"; \
		exit 1; \
	fi

$(TORCHX_SO): c_src/torchx.cpp c_src/nx_nif_utils.hpp
	mkdir -p $(PRIV_DIR)
	cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) ./cmake -B ./cmake
	cmake --build ./cmake --config Release
	mv ./cmake/torchx.so $(TORCHX_SO)

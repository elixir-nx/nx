PRIV_DIR = $(MIX_APP_PATH)/priv
TORCHX_SO = $(PRIV_DIR)/torchx.so

# If you want verbose output for debugging
# CMAKE_BUILD_FLAGS = --verbose
CMAKE_BUILD_DIR ?= $(MIX_BUILD_PATH)/cmake
C_SRC = $(shell pwd)/c_src

.DEFAULT_GLOBAL := build

build: check_libtorch $(TORCHX_SO)

check_libtorch:
	@ if [ ! -d "$(LIBTORCH_DIR)" ]; then \
		echo "LIBTORCH_DIR is not a valid directory. \
		It should either point to the directory, where LibTorch is installed, \
		or it should be unset so we download it automatically."; \
		echo "You can download LibTorch here: https://pytorch.org/get-started/locally/"; \
		exit 1; \
	fi

$(TORCHX_SO): c_src/torchx.cpp c_src/nx_nif_utils.hpp
	@ mkdir -p $(PRIV_DIR)
	@ mkdir -p $(CMAKE_BUILD_DIR)
	@ ln -sf $(abspath $(LIBTORCH_DIR)/lib) $(PRIV_DIR)/libtorch
	@ cd $(CMAKE_BUILD_DIR) && \
		cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DC_SRC=$(C_SRC) \
      	-DERTS_INCLUDE_DIR=$(ERTS_INCLUDE_DIR) -S $(shell pwd) && \
		cmake --build . $(CMAKE_BUILD_FLAGS)
	@ mv $(CMAKE_BUILD_DIR)/torchx.so $(TORCHX_SO)

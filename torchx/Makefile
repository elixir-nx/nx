PRIV_DIR = $(MIX_APP_PATH)/priv
TORCHX_SO = $(PRIV_DIR)/torchx.so

CMAKE_BUILD_DIR ?= $(MIX_APP_PATH)/cmake
CMAKE_BUILD_TYPE ?= Release

.DEFAULT_GLOBAL := build

build: $(TORCHX_SO)

$(TORCHX_SO): c_src/torchx.cpp c_src/nx_nif_utils.hpp
	@ mkdir -p $(CMAKE_BUILD_DIR)
	@ cd $(CMAKE_BUILD_DIR) && \
		cmake -S $(shell pwd) && \
		cmake --build . --config $(CMAKE_BUILD_TYPE) && \
		cmake --install . --config $(CMAKE_BUILD_TYPE)

PRIV_DIR = $(MIX_APP_PATH)/priv
TORCHX_SO = $(PRIV_DIR)/torchx.so

# If you want verbose output for debugging
# CMAKE_BUILD_FLAGS = --verbose
CMAKE_BUILD_DIR ?= $(MIX_APP_PATH)/cmake
C_SRC = $(shell pwd)/c_src
NERVES_BUILD ?= nerves_build_no

# this .cmake file makes this project compatible with nerves and it should have no effect on normal build
TOOLCHAIN_FILE ?= $(shell pwd)/nerves/toolchain.cmake

.DEFAULT_GLOBAL := build

build: check_libtorch $(TORCHX_SO)

nerves_build_yes:
	@ mkdir -p $(PRIV_DIR)
	@ if [ "$(PRIV_DIR)/libtorch" != "$(abspath $(PRIV_DIR)/libtorch)" ]; then \
		rm -rf "$(PRIV_DIR)/libtorch" ; \
		cp -a "$(abspath $(LIBTORCH_DIR)/lib)" "$(PRIV_DIR)/libtorch" ; \
    fi

nerves_build_no:
	@ mkdir -p $(PRIV_DIR)
	@ if [ "$(PRIV_DIR)/libtorch" == "$(abspath $(PRIV_DIR)/libtorch)" ]; then \
		rm -rf "$(PRIV_DIR)/libtorch" ; \
		ln -sf $(abspath $(LIBTORCH_DIR)/lib) $(PRIV_DIR)/libtorch ; \
	fi

check_libtorch:
	@ if [ ! -d "$(LIBTORCH_DIR)" ]; then \
		echo "LIBTORCH_DIR is not a valid directory. \
		It should either point to the directory, where LibTorch is installed, \
		or it should be unset so we download it automatically."; \
		echo "You can download LibTorch here: https://pytorch.org/get-started/locally/"; \
		exit 1; \
	fi

$(TORCHX_SO): $(NERVES_BUILD) c_src/torchx.cpp c_src/nx_nif_utils.hpp
	@ mkdir -p $(CMAKE_BUILD_DIR)
	@ cd $(CMAKE_BUILD_DIR) && \
		cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DC_SRC=$(C_SRC) \
      	-DERTS_INCLUDE_DIR=$(ERTS_INCLUDE_DIR) -S $(shell pwd) \
		-DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_FILE) && \
		cmake --build . $(CMAKE_BUILD_FLAGS)
	@ mv $(CMAKE_BUILD_DIR)/torchx.so $(TORCHX_SO)

LIBTORCH_INCLUDE_PATH = $(LIBTORCH_DIR)/include/
LIBTORCH_API_INCLUDE_PATH = $(LIBTORCH_DIR)/include/torch/csrc/api/include
LIBTORCH_LINK_PATH = $(LIBTORCH_DIR)/lib/
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(LIBTORCH_INCLUDE_PATH) -I$(LIBTORCH_API_INCLUDE_PATH) \
	 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -shared -std=c++14
LDFLAGS = -L$(LIBTORCH_LINK_PATH) -ltorch

PRIV_DIR = $(MIX_APP_PATH)/priv
LIBTORCH_LIB_DIR = $(PRIV_DIR)/lib
TORCHX_SO = $(PRIV_DIR)/torchx.so

CMAKE_BUILD_DIR ?= $(shell pwd)/_build/cmake
C_SRC = $(shell pwd)/c_src

# MacOS needs extra flags to link successfully
ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
	POST_INSTALL = install_name_tool -add_rpath @loader_path/lib $(TORCHX_SO)
else  # Linux
	CFLAGS += -fopenmp
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib' -lpthread -lc10 -ltorch_cpu -lkineto
	POST_INSTALL = $(NOOP)
endif

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
	mkdir -p $(PRIV_DIR)
	mkdir -p $(CMAKE_BUILD_DIR)
	cd $(CMAKE_BUILD_DIR) && \
		cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DC_SRC=$(C_SRC) \
      	-DERTS_INCLUDE_DIR=$(ERTS_INCLUDE_DIR) -S $(shell pwd) && \
		# add --verbose to show compilation commands. Useful for debugging \
		cmake --build .
	mv $(CMAKE_BUILD_DIR)/torchx.so $(TORCHX_SO)

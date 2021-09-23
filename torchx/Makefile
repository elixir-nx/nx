LIBTORCH_INCLUDE_PATH = $(LIBTORCH_DIR)/include/
LIBTORCH_API_INCLUDE_PATH = $(LIBTORCH_DIR)/include/torch/csrc/api/include
LIBTORCH_LINK_PATH = $(LIBTORCH_DIR)/lib/
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(LIBTORCH_INCLUDE_PATH) -I$(LIBTORCH_API_INCLUDE_PATH) \
	 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -shared \
	 -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS = -L$(LIBTORCH_LINK_PATH) -ltorch

PRIV_DIR = $(MIX_APP_PATH)/priv
LIBTORCH_LIB_DIR = $(PRIV_DIR)/lib
TORCHX_SO = $(PRIV_DIR)/torchx.so

# MacOS needs extra flags to link successfully
ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
	POST_INSTALL = install_name_tool -add_rpath @loader_path/lib $(TORCHX_SO)
else  # Linux
    CFLAGS += -fopenmp
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib'
	POST_INSTALL = $(NOOP)
endif

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
	ln -sf $(abspath $(LIBTORCH_LINK_PATH)) $(LIBTORCH_LIB_DIR)
	$(CXX) $(CFLAGS) c_src/torchx.cpp -o $(TORCHX_SO) $(LDFLAGS)
	$(POST_INSTALL)

LIBTORCH_DIR ?= $(abspath ./_build/libtorch)
LIBTORCH_ZIP ?= "$(LIBTORCH_DIR).zip"
LIBTORCH_INCLUDE_PATH = $(LIBTORCH_DIR)/include
LIBTORCH_API_INCLUDE_PATH = $(LIBTORCH_DIR)/include/torch/csrc/api/include
LIBTORCH_LINK_PATH = $(LIBTORCH_DIR)/lib
CFLAGS = -fPIC -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -shared -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=1

PRIV_DIR = $(MIX_APP_PATH)/priv
LIBTORCH_LIB_DIR = $(PRIV_DIR)/lib
TORCHX_SO = $(PRIV_DIR)/torchx.so

# MacOS needs extra flags to link successfully
ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
	POST_INSTALL = install_name_tool -add_rpath @loader_path/lib $(TORCHX_SO)
	LIBTORCH_DOWNLOAD_URL ?= https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.1.zip
else  # Linux
    CFLAGS += -isystem $(LIBTORCH_INCLUDE_PATH) -isystem $(LIBTORCH_API_INCLUDE_PATH) -isystem $(ERTS_INCLUDE_DIR)
	LDFLAGS += -Wl,-rpath,$(LIBTORCH_LINK_PATH) $(LIBTORCH_LINK_PATH)/libtorch.so $(LIBTORCH_LINK_PATH)/libc10.so $(LIBTORCH_LINK_PATH)/libkineto.a -Wl,--no-as-needed,"$(LIBTORCH_LINK_PATH)/libtorch_cpu.so" -Wl,--as-needed $(LIBTORCH_LINK_PATH)/libc10.so -lpthread -Wl,--no-as-needed,"$(LIBTORCH_LINK_PATH)/libtorch.so" -Wl,--as-needed
	POST_INSTALL = $(NOOP)
	LIBTORCH_DOWNLOAD_URL ?= https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcpu.zip
endif

.DEFAULT_GLOBAL := build

build: libtorch $(TORCHX_SO)

libtorch:
	@ if [ ! -d "$(LIBTORCH_DIR)" ]; then \
		wget $(LIBTORCH_DOWNLOAD_URL) --no-check-certificate -O $(LIBTORCH_ZIP) && \
		unzip $(LIBTORCH_ZIP) -d /tmp/torchx && \
		mv -T /tmp/torchx/libtorch $(LIBTORCH_DIR); \
	fi

$(TORCHX_SO): c_src/torchx.cpp c_src/nx_nif_utils.hpp
	mkdir -p $(PRIV_DIR)
	ln -sf $(abspath $(LIBTORCH_LINK_PATH)) $(LIBTORCH_LIB_DIR)
	$(CXX) $(CFLAGS) -c c_src/torchx.cpp -o $(PRIV_DIR)/torchx.cpp.o
	$(CXX) $(CFLAGS) $(PRIV_DIR)/torchx.cpp.o -o $(TORCHX_SO) $(LDFLAGS)
	$(POST_INSTALL)

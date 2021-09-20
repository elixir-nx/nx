LIBTORCH_INCLUDE_PATH = $(LIBTORCH_DIR)/include/
LIBTORCH_API_INCLUDE_PATH = $(LIBTORCH_DIR)/include/torch/csrc/api/include
LIBTORCH_LINK_PATH = $(LIBTORCH_DIR)/lib/
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(LIBTORCH_INCLUDE_PATH) -I$(LIBTORCH_API_INCLUDE_PATH) -O3 -Wall -Wextra \
				 -Wno-unused-parameter -Wno-missing-field-initializers -shared -std=c++14
LDFLAGS = -L$(LIBTORCH_LINK_PATH) -Wl,-rpath,$(LIBTORCH_LINK_PATH) -ltorch -lc10

# MacOS needs extra flags to link successfully
ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
else  # Linux
	LDFLAGS += -lm
endif

.DEFAULT_GLOBAL := build

build: check_libtorch $(MIX_APP_PATH)/priv/torchx.so

check_libtorch:
	@ if [ -z "$(LIBTORCH_DIR)" ]; then \
		echo "LIBTORCH_DIR env var is not set. It should point to the directory, where LibTorch is installed."; \
		echo "You can download LibTorch here: https://pytorch.org/get-started/locally/"; \
		exit 1; \
	fi

$(MIX_APP_PATH)/priv/torchx.so: c_src/torchx.cpp c_src/nx_nif_utils.hpp
	@echo 'Compiling: '$<
	@mkdir -p $(MIX_APP_PATH)/priv
	@$(CXX) $(CFLAGS) c_src/torchx.cpp -o $(MIX_APP_PATH)/priv/torchx.so $(LDFLAGS)

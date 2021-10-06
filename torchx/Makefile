CUDA_DIR ?= /usr/local/cuda
CUDA_VERSION_DIR ?= /usr/local/cuda-11.4
CUDA_INCLUDE_PATH = $(CUDA_DIR)/include/
CUDA_LIB64 = $(CUDA_DIR)/lib64
CUDA_VERSION_INCLUDE_PATH = $(CUDA_VERSION_DIR)/include/
CUDA_VERSION_LIB64 = $(CUDA_VERSION_DIR)/lib64

LIBTORCH_INCLUDE_PATH = $(LIBTORCH_DIR)/include/
LIBTORCH_API_INCLUDE_PATH = $(LIBTORCH_DIR)/include/torch/csrc/api/include
LIBTORCH_LINK_PATH = $(LIBTORCH_DIR)/lib/
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(LIBTORCH_INCLUDE_PATH) -I$(LIBTORCH_API_INCLUDE_PATH) \
	 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -shared -std=c++14
TORCHX_TARGET ?= cpu
LDFLAGS = -L$(LIBTORCH_LINK_PATH) -ltorch



PRIV_DIR = $(MIX_APP_PATH)/priv
LIBTORCH_LIB_DIR = $(PRIV_DIR)/lib
CUDA_LIB_DIR = $(PRIV_DIR)/libcuda
CUDA64_LIB_DIR = $(PRIV_DIR)/libcuda64
CUDA64_EXTRA_LIB_DIR = $(PRIV_DIR)/libcuda64_extra
TORCHX_SO = $(PRIV_DIR)/torchx.so

CMAKE_BUILD_DIR ?= $(shell pwd)/_build/cmake
C_SRC = $(shell pwd)/c_src

# MacOS needs extra flags to link successfully
ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
	POST_INSTALL = install_name_tool -add_rpath @loader_path/lib $(TORCHX_SO)
else  # Linux
	CFLAGS += -fopenmp
	ifeq ($(TORCHX_TARGET), cuda)
		CFLAGS += -I$(CUDA_VERSION_INCLUDE_PATH) -I$(CUDA_INCLUDE_PATH)
		LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib:$(CUDA_VERSION_LIB64):$(CUDA_VERSION_LIB64)/stubs:$(CUDA_LIB64)' -ltorch\
		-lc10 -lkineto -lcuda -lnvrtc -lnvToolsExt -lcudart\
		-lc10_cuda -ltorch_cuda -ltorch_cuda_cpp -ltorch_cpu\
		-lpthread -lc10_cuda -lc10 -lcufft -lcurand -lcublas\
		$(CUDA_VERSION_LIB64)/libcudnn.so -ltorch_cuda_cu
	else
		LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib' -lpthread -lc10 -ltorch_cpu -lkineto
	endif
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
	ln -sf $(abspath $(LIBTORCH_LINK_PATH)) $(LIBTORCH_LIB_DIR)
	$(CXX) $(CFLAGS) c_src/torchx.cpp -o $(TORCHX_SO) $(LDFLAGS)
	$(POST_INSTALL)

# Commands below can be uncommented to compile with CMake, which makes it
# easier to discover which links are missing for standard gcc compilation
# $(TORCHX_SO): c_src/torchx.cpp c_src/nx_nif_utils.hpp
# 	mkdir -p $(PRIV_DIR)
# 	mkdir -p $(CMAKE_BUILD_DIR)
# 	cd $(CMAKE_BUILD_DIR) && \
# 		cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DC_SRC=$(C_SRC) \
# 		-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDNN_ROOT=/usr/local/cuda-11.4/ \
#       -DERTS_INCLUDE_DIR=$(ERTS_INCLUDE_DIR) -S $(shell pwd) && \
# 		cmake --build . --verbose
# 	mv $(CMAKE_BUILD_DIR)/torchx.so $(TORCHX_SO)

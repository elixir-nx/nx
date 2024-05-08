# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH

# XLA extension installation location
XLA_EXTENSION_DIR = cache/xla_extension
XLA_EXTENSION_LIB = $(XLA_EXTENSION_DIR)/lib
XLA_INCLUDE_PATH = $(XLA_EXTENSION_DIR)/include

IREE_COMPILER_DIR = iree/build/lib
IREE_COMPILER_LIB = cache/$(IREE_COMPILER_DIR)
IREE_COMPILER_INCLUDE_PATH = cache/iree/compiler/bindings/c

LLVM_MLIR_INCLUDES = -Icache/iree/third_party/llvm-project/mlir/include

# Cache configuration
EXLA_CACHE_SO = cache/libexla.so
EXLA_CACHE_OBJ_DIR = cache/objs
EXLA_CACHE_IREE_COMPILER_SO = cache/libireecompiler.so

# Private configuration
EXLA_DIR = c_src/exla
PRIV_DIR = $(MIX_APP_PATH)/priv
EXLA_SO = $(PRIV_DIR)/libexla.so
EXLA_IREE_COMPILER_SO = $(PRIV_DIR)/libireecompiler.so
EXLA_LIB_DIR = $(PRIV_DIR)/xla_extension/lib
EXLA_IREE_LIB_DIR = $(PRIV_DIR)/$(IREE_COMPILER_DIR)

# Link paths
XLA_EXTENSION_LIB_LINK_PATH = ../$(CWD_RELATIVE_TO_PRIV_PATH)/$(XLA_EXTENSION_DIR)/$(XLA_EXTENSION_LIB)
IREE_COMPILER_LIB_LINK_PATH = ../../$(CWD_RELATIVE_TO_PRIV_PATH)/$(IREE_COMPILER_LIB)
EXLA_CACHE_SO_LINK_PATH = $(CWD_RELATIVE_TO_PRIV_PATH)/$(EXLA_CACHE_SO)
EXLA_CACHE_IREE_COMPILER_SO_LINK_PATH = $(CWD_RELATIVE_TO_PRIV_PATH)/$(EXLA_CACHE_IREE_COMPILER_SO)

# Build flags
# c++17 is needed, otherwise xla headers
# break on some conflicting llvm/std definitions
# Note: this is on :xla 0.5.0 -- things can change with later versions
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(XLA_INCLUDE_PATH) -Wall -Wno-sign-compare \
	 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
	 -std=c++17 -w -DLLVM_VERSION_STRING=

IREE_CFLAGS = $(CFLAGS) -I$(IREE_COMPILER_INCLUDE_PATH)

NVCCFLAGS = -shared -Xcompiler -fPIC

ifdef DEBUG
	CFLAGS += -g
	NVCCFLAGS += -g
else
	CFLAGS += -O3
endif

LDFLAGS = -L$(XLA_EXTENSION_LIB) -lxla_extension -shared
IREE_LDFLAGS = $(LDFLAGS) -lIREECompiler

ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress -rpath @loader_path/xla_extension/lib -rpath @loader_path/$(IREE_COMPILER_DIR)
else
	# Use a relative RPATH, so at runtime libexla.so looks for libxla_extension.so
	# in ./lib regardless of the absolute location. This way priv can be safely
	# packed into an Elixir release. Also, we use $$ to escape Makefile variable
	# and single quotes to escape shell variable
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/xla_extension/lib' -Wl,-rpath,'$$ORIGIN/$(IREE_COMPILER_DIR)'
endif

$(EXLA_SO): $(EXLA_CACHE_SO) $(EXLA_CACHE_IREE_COMPILER_SO)
	@ mkdir -p $(PRIV_DIR)
	@ mkdir -p $(PRIV_DIR)/xla_extension
	@ mkdir -p $(PRIV_DIR)/iree/build
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ]; then \
		cp -a $(abspath $(XLA_EXTENSION_LIB)) $(EXLA_LIB_DIR) ; \
		cp -a $(abspath $(IREE_COMPILER_LIB)) $(EXLA_IREE_LIB_DIR) ; \
		cp -a $(abspath $(EXLA_CACHE_SO)) $(EXLA_SO) ; \
		cp -a $(abspath $(EXLA_CACHE_IREE_COMPILER_SO)) $(EXLA_IREE_COMPILER_SO) ; \
	else \
		ln -sf $(XLA_EXTENSION_LIB_LINK_PATH) $(EXLA_LIB_DIR) ; \
		ln -sf $(IREE_COMPILER_LIB_LINK_PATH) $(EXLA_IREE_LIB_DIR) ; \
		ln -sf $(EXLA_CACHE_SO_LINK_PATH) $(EXLA_SO) ; \
		ln -sf $(EXLA_CACHE_IREE_COMPILER_SO_LINK_PATH) $(EXLA_IREE_COMPILER_SO) ; \
	fi

SOURCES = $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_mlir.cc $(EXLA_DIR)/custom_calls.cc $(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_nif_util.cc
HEADERS = $(EXLA_DIR)/exla_mlir.h $(EXLA_DIR)/custom_calls.h $(EXLA_DIR)/exla_client.h $(EXLA_DIR)/exla_nif_util.h $(EXLA_DIR)/exla_log_sink.h
OBJECTS = $(patsubst $(EXLA_DIR)/%.cc,$(EXLA_CACHE_OBJ_DIR)/%.o,$(SOURCES)) $(EXLA_CACHE_OBJ_DIR)/exla_cuda.o

IREE_SOURCES = $(EXLA_DIR)/iree/iree.cc $(EXLA_DIR)/iree/compiler.cc $(EXLA_DIR)/iree/runtime.cc
IREE_HEADERS = $(EXLA_DIR)/iree/compiler.h $(EXLA_DIR)/iree/runtime.h $(EXLA_DIR)/exla_nif_util.h
IREE_OBJECTS = $(patsubst $(EXLA_DIR)/%.cc,$(EXLA_CACHE_OBJ_DIR)/%.o,$(IREE_SOURCES))

NVCC_RESULT := $(shell which nvcc 2> /dev/null)
NVCC_TEST := $(notdir $(NVCC_RESULT))

ifeq ($(NVCC_TEST),nvcc)
  NVCC := nvcc
	NVCCFLAGS += -DCUDA_ENABLED
else
  NVCC := $(CXX)
	NVCCFLAGS = $(CFLAGS)
endif

$(EXLA_CACHE_OBJ_DIR)/iree/%.o:  $(EXLA_DIR)/iree/%.cc $(IREE_HEADERS)
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)/iree
	$(CXX) $(IREE_CFLAGS) $(LLVM_MLIR_INCLUDES) -c $< -o $@

$(EXLA_CACHE_OBJ_DIR)/exla_cuda.o: $(EXLA_DIR)/exla_cuda.cc $(EXLA_DIR)/exla_cuda.h
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXLA_CACHE_OBJ_DIR)/%.o: $(EXLA_DIR)/%.cc $(HEADERS)
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)/iree
	$(CXX) $(CFLAGS) -c $< -o $@

$(EXLA_CACHE_SO): $(XLA_EXTENSION_DIR) $(IREE_COMPILER_LIB) $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXLA_CACHE_SO) $(LDFLAGS)

$(EXLA_CACHE_IREE_COMPILER_SO): $(EXLA_CACHE_OBJ_DIR)/iree/iree.o $(EXLA_CACHE_OBJ_DIR)/iree/compiler.o $(EXLA_CACHE_OBJ_DIR)/iree/runtime.o $(EXLA_CACHE_OBJ_DIR)/exla_nif_util.o
	$(CXX) $^ -o $@ $(IREE_LDFLAGS)

$(IREE_COMPILER_LIB):
  # TO-DO: setup proper download and caching of the iree compiler
	@ln -s $(HOME)/coding/iree cache/iree
	cmake -G Ninja -B cache/iree/build -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-fvisibility=hidden" cache/iree
	cmake --build cache/iree/build

clean:
	rm -rf cache

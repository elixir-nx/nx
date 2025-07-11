# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH

# XLA extension installation location
XLA_EXTENSION_DIR = cache/xla_extension
XLA_EXTENSION_LIB = $(XLA_EXTENSION_DIR)/lib
XLA_INCLUDE_PATH = $(XLA_EXTENSION_DIR)/include

# Cache configuration
EXLA_CACHE_SO = cache/libexla.so
EXLA_CACHE_OBJ_DIR = cache/$(EXLA_VERSION)/objs

# Private configuration
EXLA_DIR = c_src/exla
PRIV_DIR = $(MIX_APP_PATH)/priv
EXLA_SO = $(PRIV_DIR)/libexla.so
EXLA_LIB_DIR = $(PRIV_DIR)/xla_extension/lib

# Link paths
XLA_EXTENSION_LIB_LINK_PATH = ../$(CWD_RELATIVE_TO_PRIV_PATH)/$(XLA_EXTENSION_LIB)
EXLA_CACHE_SO_LINK_PATH = $(CWD_RELATIVE_TO_PRIV_PATH)/$(EXLA_CACHE_SO)

# Build flags
# Note that XLA requires c++17, Fine as well
CFLAGS += -fPIC -I$(ERTS_INCLUDE_DIR) -I$(FINE_INCLUDE_DIR) -I$(XLA_INCLUDE_PATH) -Wall -Wno-sign-compare \
	 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
	 -std=c++17 -w

ifdef DEBUG
	CFLAGS += -g
else
	CFLAGS += -O3
endif

NVCC = $(CXX)
NVCCFLAGS = $(CFLAGS)
LDFLAGS += -L$(XLA_EXTENSION_LIB) -lxla_extension -shared -fvisibility=hidden

ifeq ($(CROSSCOMPILE),)
	# Interrogate the system for local compilation
	UNAME_S = $(shell uname -s)

ifndef EXLA_CPU_ONLY
$(info EXLA_CPU_ONLY is not set, checking for nvcc availability)
	NVCC_RESULT := $(shell which nvcc 2> /dev/null)
	NVCC_TEST := $(notdir $(NVCC_RESULT))

	ifeq ($(NVCC_TEST),nvcc)
$(info CUDA is available.)
		NVCC = nvcc
		NVCCFLAGS = -Xcompiler "$(CFLAGS)" -DCUDA_ENABLED
	else
$(info CUDA is not available.)
	endif
else
$(info EXLA_CPU_ONLY is set, skipping nvcc step)
endif
else
	# Determine settings for cross-compiled builds like for Nerves
	UNAME_S = Linux
endif

ifeq ($(UNAME_S), Darwin)
	LDFLAGS += -flat_namespace -undefined dynamic_lookup -rpath @loader_path/xla_extension/lib
else
	# Use a relative RPATH, so at runtime libexla.so looks for libxla_extension.so
	# in ./lib regardless of the absolute location. This way priv can be safely
	# packed into an Elixir release. Also, we use $$ to escape Makefile variable
	# and single quotes to escape shell variable
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/xla_extension/lib'
endif

$(EXLA_SO): $(EXLA_CACHE_SO)
	@ mkdir -p $(PRIV_DIR)
	@ mkdir -p $(PRIV_DIR)/xla_extension
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ]; then \
		cp -a $(abspath $(XLA_EXTENSION_LIB)) $(EXLA_LIB_DIR) ; \
		cp -a $(abspath $(EXLA_CACHE_SO)) $(EXLA_SO) ; \
	else \
		ln -sf $(XLA_EXTENSION_LIB_LINK_PATH) $(EXLA_LIB_DIR) ; \
		ln -sf $(EXLA_CACHE_SO_LINK_PATH) $(EXLA_SO) ; \
	fi

SOURCES = $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_mlir.cc $(EXLA_DIR)/ipc.cc
SOURCES += $(wildcard $(EXLA_DIR)/custom_calls/*.cc)
HEADERS = $(EXLA_DIR)/exla_mlir.h $(EXLA_DIR)/custom_calls/qr.h $(EXLA_DIR)/custom_calls/eigh.h $(EXLA_DIR)/exla_client.h $(EXLA_DIR)/exla_nif_util.h $(EXLA_DIR)/exla_log_sink.h $(EXLA_DIR)/ipc.h
OBJECTS = $(patsubst $(EXLA_DIR)/%.cc,$(EXLA_CACHE_OBJ_DIR)/%.o,$(SOURCES)) $(EXLA_CACHE_OBJ_DIR)/exla_cuda.o


$(EXLA_CACHE_OBJ_DIR)/exla_cuda.o: $(EXLA_DIR)/exla_cuda.cc $(EXLA_DIR)/exla_cuda.h
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXLA_CACHE_OBJ_DIR)/%.o: $(EXLA_DIR)/%.cc $(HEADERS)
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)
	@ mkdir -p $(EXLA_CACHE_OBJ_DIR)/custom_calls
	$(CXX) $(CFLAGS) -c $< -o $@

$(EXLA_CACHE_SO): $(XLA_EXTENSION_DIR) $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(EXLA_CACHE_SO) $(LDFLAGS)

clean:
	rm -rf cache

# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH

# XLA extension installation location
XLA_EXTENSION_DIR = cache/xla_extension
XLA_EXTENSION_LIB = $(XLA_EXTENSION_DIR)/lib
XLA_INCLUDE_PATH = $(XLA_EXTENSION_DIR)/include

# Cache configuration
EXLA_CACHE_SO = cache/libexla.so

# Private configuration
EXLA_DIR = c_src/exla
PRIV_DIR = $(MIX_APP_PATH)/priv
EXLA_SO = $(PRIV_DIR)/libexla.so
EXLA_LIB_DIR = $(PRIV_DIR)/lib

# Build flags
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(XLA_INCLUDE_PATH) -O3 -Wall -Wno-sign-compare \
	 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
	 -shared -std=c++17 -w -DLLVM_ON_UNIX=1

LDFLAGS = -L$(XLA_EXTENSION_LIB) -lxla_extension

ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
	POST_INSTALL = install_name_tool \
		-change bazel-out/darwin_arm64-opt/bin/tensorflow/compiler/xla/extension/libxla_extension.so @loader_path/xla_extension/lib/libxla_extension.so \
		-change bazel-out/darwin-opt/bin/tensorflow/compiler/xla/extension/libxla_extension.so @loader_path/xla_extension/lib/libxla_extension.so \
		$(EXLA_CACHE_SO)
else
	# Use a relative RPATH, so at runtime libexla.so looks for libxla_extension.so
	# in ./lib regardless of the absolute location. This way priv can be safely
	# packed into an Elixir release. Also, we use $$ to escape Makefile variable
	# and single quotes to escape shell variable
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib'
	POST_INSTALL = $(NOOP)
endif

XLA_EXTENSION_LIB_LINK_PATH = $(CWD_RELATIVE_TO_PRIV_PATH)/$(XLA_EXTENSION_LIB)
EXLA_CACHE_SO_LINK_PATH = $(CWD_RELATIVE_TO_PRIV_PATH)/$(EXLA_CACHE_SO)

$(EXLA_SO): $(EXLA_CACHE_SO)
	@ mkdir -p $(PRIV_DIR)
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ]; then \
		cp -a $(abspath $(XLA_EXTENSION_LIB)) $(EXLA_LIB_DIR) ; \
		cp -a $(abspath $(EXLA_CACHE_SO)) $(EXLA_SO) ; \
	else \
		ln -sf $(XLA_EXTENSION_LIB_LINK_PATH) $(EXLA_LIB_DIR) ; \
		ln -sf $(EXLA_CACHE_SO_LINK_PATH) $(EXLA_SO) ; \
	fi

$(EXLA_CACHE_SO): $(XLA_EXTENSION_DIR) $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_client.h $(EXLA_DIR)/exla_nif_util.cc $(EXLA_DIR)/exla_nif_util.h $(EXLA_DIR)/exla_log_sink.h
	$(CXX) $(CFLAGS) $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_nif_util.cc $(EXLA_DIR)/exla_client.cc -o $(EXLA_CACHE_SO) $(LDFLAGS)
	$(POST_INSTALL)

clean:
	rm -rf cache

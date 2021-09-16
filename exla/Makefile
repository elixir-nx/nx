# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR

# XLA Extension Installation Location
XLA_EXTENSION_DIR ?= cache/xla_extension

XLA_EXTENSION_LIB = $(XLA_EXTENSION_DIR)/lib
XLA_INCLUDE_PATH = $(XLA_EXTENSION_DIR)/include

# Private configuration
EXLA_DIR = c_src/exla
PRIV_DIR = priv
EXLA_SO = $(PRIV_DIR)/libexla.so
EXLA_LIB_DIR = $(PRIV_DIR)/lib

# Build Flags
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -isystem $(XLA_INCLUDE_PATH) -O3 -Wall -Wextra \
				 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
				 -shared -std=c++14

# Use a relative RPATH, so at runtime libexla.so looks for libxla_extension.so
# in ./lib regardless of the absolute location. This way priv can be safely
# packed into an Elixir release. Also, we use $$ to escape Makefile variable
# and single quotes to escape shell variable
LDFLAGS = -L$(EXLA_LIB_DIR) -Wl,-rpath,'$$ORIGIN/lib' -lxla_extension

$(EXLA_SO): $(XLA_EXTENSION_DIR) $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_client.h $(EXLA_DIR)/exla_nif_util.cc $(EXLA_DIR)/exla_nif_util.h $(EXLA_DIR)/exla_log_sink.h
	mkdir -p $(PRIV_DIR) && \
		ln -sf $(abspath $(XLA_EXTENSION_LIB)) $(EXLA_LIB_DIR) && \
		$(CXX) $(CFLAGS) $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_nif_util.h \
											$(EXLA_DIR)/exla_nif_util.cc $(EXLA_DIR)/exla_client.h \
											$(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_log_sink.h \
											-o $(EXLA_SO) $(LDFLAGS)

clean:
	rm -rf $(EXLA_SO)

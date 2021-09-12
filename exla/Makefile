# mix.exs vars
# ERTS_INCLUDE_DIR
# ERTS_VERSION

# Private configuration
PRIV_DIR = priv
EXLA_SO = $(PRIV_DIR)/libexla.so
EXLA_DIR = c_src/exla

# XLA Extension Installation Location
XLA_EXTENSION_DIR ?= $(PRIV_DIR)/xla_extension
XLA_EXTENSION_LIB ?= $(XLA_EXTENSION_DIR)/lib
XLA_INCLUDE_PATH ?= $(XLA_EXTENSION_DIR)/include

# Build Flags
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(XLA_INCLUDE_PATH) -O3 -Wall -Wextra \
				 -Wno-unused-parameter -Wno-missing-field-initializers -Wno-comment \
				 -shared -std=c++14

LDFLAGS = -L$(XLA_EXTENSION_LIB) -Wl,-rpath,$(XLA_EXTENSION_LIB) -lxla_extension

$(EXLA_SO): $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_client.h $(EXLA_DIR)/exla_nif_util.cc $(EXLA_DIR)/exla_nif_util.h $(EXLA_DIR)/exla_log_sink.h
	@ mkdir -p $(PRIV_DIR)
	@ $(CXX) $(CFLAGS) $(EXLA_DIR)/exla.cc $(EXLA_DIR)/exla_nif_util.h \
										$(EXLA_DIR)/exla_nif_util.cc $(EXLA_DIR)/exla_client.h \
										$(EXLA_DIR)/exla_client.cc $(EXLA_DIR)/exla_log_sink.h \
										-o $(EXLA_SO) $(LDFLAGS)

clean:
	rm -rf $(EXLA_SO)

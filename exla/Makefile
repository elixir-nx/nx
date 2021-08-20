# System vars
TEMP ?= $(HOME)/.cache

# mix.exs vars
# ERTS_INCLUDE_DIR
ERTS_VERSION ?= $(shell erl -eval "io:fwrite(erlang:system_info(version))." -noshell -s erlang halt)

# Public configuration
EXLA_MODE ?= opt # can also be dbg
EXLA_CACHE ?= $(TEMP)/exla
EXLA_TENSORFLOW_GIT_REPO ?= https://github.com/tensorflow/tensorflow.git

# Replicated public configuration
# When changing this, also change:
# * Makefile
# * Makefile.win
# * lib/exla/aot/compiler.ex
EXLA_TENSORFLOW_GIT_REV ?= 54dee6dd8d47b6e597f4d3f85b6fb43fd5f50f82

# Private configuration
EXLA_SO = priv/libexla.so
EXLA_DIR = c_src/exla
ERTS_SYM_DIR = $(EXLA_DIR)/erts
BAZEL_FLAGS = --define "framework_shared_object=false" -c $(EXLA_MODE)

TENSORFLOW_NS = tf-$(EXLA_TENSORFLOW_GIT_REV)
TENSORFLOW_DIR = $(EXLA_CACHE)/$(TENSORFLOW_NS)/erts-$(ERTS_VERSION)
TENSORFLOW_EXLA_NS = tensorflow/compiler/xla/exla
TENSORFLOW_EXLA_DIR = $(TENSORFLOW_DIR)/$(TENSORFLOW_EXLA_NS)

XLA_EXTENSION_LIB = /home/sean/projects/xla_extension/lib
XLA_INCLUDE_PATH = /home/sean/projects/xla_extension/include

CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(XLA_INCLUDE_PATH) -I${Protobuf_INCLUDE_PATH} -O3 -Wall -Wextra \
				 -Wno-unused-parameter -Wno-missing-field-initializers -shared -std=c++14
LDFLAGS = -L$(XLA_EXTENSION_LIB) -Wl,-rpath,$(XLA_EXTENSION_LIB) -lxla_extension

$(EXLA_SO):
	@mkdir -p $(MIX_APP_PATH)/priv
	@$(CXX) $(CFLAGS) c_src/exla/exla.cc c_src/exla/exla_nif_util.h \
										c_src/exla/exla_nif_util.cc c_src/exla/exla_client.h \
										c_src/exla/exla_client.cc c_src/exla/exla_log_sink.h \
										-o $(MIX_APP_PATH)/priv/libexla.so $(LDFLAGS)

$(TENSORFLOW_DIR)/bazel-bin/$(TENSORFLOW_EXLA_NS)/xla_extension: symlinks
	cd $(TENSORFLOW_DIR) && \
		bazel build $(BAZEL_FLAGS) $(EXLA_FLAGS) //$(TENSORFLOW_EXLA_NS):xla_extension


symlinks: $(TENSORFLOW_DIR)
	@ rm -f $(TENSORFLOW_EXLA_DIR)
	@ ln -s "$(MIX_CURRENT_PATH)/$(EXLA_DIR)" $(TENSORFLOW_EXLA_DIR)
	@ rm -f $(ERTS_SYM_DIR)
	@ ln -s "$(ERTS_INCLUDE_DIR)" $(ERTS_SYM_DIR)

# Print Tensorflow Dir
PTD:
	@ echo $(TENSORFLOW_DIR)

# Clones tensorflow
$(TENSORFLOW_DIR):
	mkdir -p $(TENSORFLOW_DIR)

	cd $(TENSORFLOW_DIR) && \
		git init && \
		git remote add origin $(EXLA_TENSORFLOW_GIT_REPO) && \
		git fetch --depth 1 origin $(EXLA_TENSORFLOW_GIT_REV) && \
		git checkout FETCH_HEAD

clean:
	cd $(TENSORFLOW_DIR) && bazel clean --expunge
	rm -f $(ERTS_SYM_DIR) $(TENSORFLOW_EXLA_DIR)
	rm -rf $(EXLA_SO) $(TENSORFLOW_DIR)

# System vars
HOME ?= $(USERPROFILE)
TMP ?= $(HOME)/.cache

# mix.exs vars
# ERTS_INCLUDE_DIR
# ERTS_VSN
# EXLA_VSN

# Public configuration
EXLA_TARGET ?= cpu # can also be cuda
EXLA_MODE ?= opt # can also be dbg
EXLA_CACHE ?= $(TMP)/exla
EXLA_TENSORFLOW_GIT_REPO ?= git@github.com:tensorflow/tensorflow.git
EXLA_TENSORFLOW_GIT_REV ?= 8a1bb87da5b8dcec1d7f0bf8baa43565c095830e

# Private configuration
EXLA_SO = priv/libexla.so
EXLA_DIR = c_src/exla
ERTS_SYM_DIR = $(EXLA_DIR)/erts
BAZEL_FLAGS = --define "framework_shared_object=false" -c $(EXLA_MODE)

TENSORFLOW_NS = tf-$(EXLA_TENSORFLOW_GIT_REV)
TENSORFLOW_DIR = c_src/$(TENSORFLOW_NS)
TENSORFLOW_EXLA_NS = tensorflow/compiler/xla/exla
TENSORFLOW_EXLA_DIR = $(TENSORFLOW_DIR)/$(TENSORFLOW_EXLA_NS)
TENSORFLOW_CACHE_DIR = $(EXLA_CACHE)/$(TENSORFLOW_NS)

all: $(EXLA_TARGET)

cpu: symlinks
	cd $(TENSORFLOW_DIR) && \
		bazel build $(BAZEL_FLAGS) //$(TENSORFLOW_EXLA_NS):libexla_cpu.so
	mkdir -p priv
	cp -f $(TENSORFLOW_DIR)/bazel-bin/$(TENSORFLOW_EXLA_NS)/libexla_cpu.so $(EXLA_SO)

cuda: symlinks
	cd $(TENSORFLOW_DIR) && \
		bazel build $(BAZEL_FLAGS) --config=cuda //$(TENSORFLOW_EXLA_NS):libexla_gpu.so
	mkdir -p priv
	cp -f $(TENSORFLOW_DIR)/bazel-bin/$(TENSORFLOW_EXLA_NS)/libexla_gpu.so $(EXLA_SO)

symlinks: $(TENSORFLOW_DIR)
	rm -f $(TENSORFLOW_EXLA_DIR)
	ln -s "$(PWD)/$(EXLA_DIR)" $(TENSORFLOW_EXLA_DIR)
	rm -f $(ERTS_SYM_DIR)
	ln -s "$(ERTS_INCLUDE_DIR)" $(ERTS_SYM_DIR)

# Sets up a tensorflow symlink inside this repo
$(TENSORFLOW_DIR): $(TENSORFLOW_CACHE_DIR)
	ln -s "$(TENSORFLOW_CACHE_DIR)" $(TENSORFLOW_DIR)

# Clones tensorflow
$(TENSORFLOW_CACHE_DIR):
	mkdir -p $(TENSORFLOW_CACHE_DIR)
	cd $(TENSORFLOW_CACHE_DIR) && \
		git init && \
		git remote add origin $(EXLA_TENSORFLOW_GIT_REPO) && \
		git fetch --depth 1 origin $(EXLA_TENSORFLOW_GIT_REV) && \
		git checkout FETCH_HEAD

clean:
	cd $(TENSORFLOW_CACHE_DIR) && bazel clean --expunge
	rm -f $(ERTS_SYM_DIR) $(TENSORFLOW_EXLA_DIR) $(TENSORFLOW_DIR)
	rm -rf $(EXLA_SO) $(TENSORFLOW_CACHE_DIR)

# System vars
TEMP ?= $(HOME)/.cache

# mix.exs vars
# ERTS_INCLUDE_DIR

# Public configuration
EXLA_TARGET ?= host # can also be cuda
EXLA_MODE ?= opt # can also be dbg
EXLA_CACHE ?= $(TEMP)/exla
EXLA_TENSORFLOW_GIT_REPO ?= https://github.com/tensorflow/tensorflow.git
EXLA_TENSORFLOW_GIT_REV ?= 6af836f407f546cf2f9ab3b5fcb7a8285bda5c96

# Private configuration
EXLA_SO = priv/libexla.so
EXLA_DIR = c_src/exla
ERTS_SYM_DIR = $(EXLA_DIR)/erts
BAZEL_FLAGS = --define "framework_shared_object=false" -c $(EXLA_MODE)

TENSORFLOW_NS = tf-$(EXLA_TENSORFLOW_GIT_REV)
TENSORFLOW_DIR = $(EXLA_CACHE)/$(TENSORFLOW_NS)
TENSORFLOW_EXLA_NS = tensorflow/compiler/xla/exla
TENSORFLOW_EXLA_DIR = $(TENSORFLOW_DIR)/$(TENSORFLOW_EXLA_NS)

all: symlinks
	cd $(TENSORFLOW_DIR) && \
		bazel build $(BAZEL_FLAGS) $(EXLA_FLAGS) //$(TENSORFLOW_EXLA_NS):libexla.so
	mkdir -p priv
	cp -f $(TENSORFLOW_DIR)/bazel-bin/$(TENSORFLOW_EXLA_NS)/libexla.so $(EXLA_SO)

symlinks: $(TENSORFLOW_DIR)
	rm -f $(TENSORFLOW_EXLA_DIR)
	ln -s "$(PWD)/$(EXLA_DIR)" $(TENSORFLOW_EXLA_DIR)
	rm -f $(ERTS_SYM_DIR)
	ln -s "$(ERTS_INCLUDE_DIR)" $(ERTS_SYM_DIR)

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

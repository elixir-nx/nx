# Public configuration
EXLA_TARGET ?= cpu # can also be cuda
EXLA_MODE ?= opt # can also be dbg
EXLA_SO = priv/libexla.so

# Tensorflow flags and config
ERTS_SYM_DIR = c_src/exla/erts
TENSORFLOW_DIR = c_src/tensorflow
TENSORFLOW_EXLA_NS = tensorflow/compiler/xla/exla
TENSORFLOW_EXLA_DIR = $(TENSORFLOW_DIR)/$(TENSORFLOW_EXLA_NS)
BAZEL_FLAGS = --define "framework_shared_object=false" -c $(EXLA_MODE)

all: $(EXLA_TARGET)

cpu: $(TENSORFLOW_EXLA_DIR)
	rm -f $(ERTS_SYM_DIR)
	ln -s "$(ERTS_INCLUDE_DIR)" $(ERTS_SYM_DIR)
	cd $(TENSORFLOW_DIR) && \
		bazel build $(BAZEL_FLAGS) //$(TENSORFLOW_EXLA_NS):libexla_cpu.so
	mkdir -p priv
	cp -f $(TENSORFLOW_DIR)/bazel-bin/$(TENSORFLOW_EXLA_NS)/libexla_cpu.so $(EXLA_SO)

cuda: $(TENSORFLOW_EXLA_DIR)
	rm -f $(ERTS_SYM_DIR)
	ln -s "$(ERTS_INCLUDE_DIR)" $(ERTS_SYM_DIR)
	cd $(TENSORFLOW_DIR) && \
		bazel build $(BAZEL_FLAGS) --config=cuda //$(TENSORFLOW_EXLA_NS):libexla_gpu.so
	mkdir -p priv
	cp -f $(TENSORFLOW_DIR)/bazel-bin/$(TENSORFLOW_EXLA_NS)/libexla_gpu.so $(EXLA_SO)

$(TENSORFLOW_EXLA_DIR):
	git submodule init
	git submodule update
	rm -f $(TENSORFLOW_EXLA_DIR)
	ln -s ../../../../exla $(TENSORFLOW_EXLA_DIR)

clean:
	cd $(TENSORFLOW_DIR) && bazel clean --expunge
	rm -f $(EXLA_SO)

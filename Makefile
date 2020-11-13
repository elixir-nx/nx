# Public configuration
EXLA_TARGET ?= cpu # can also be cuda
EXLA_MODE ?= opt # can also be dbg
EXLA_SO = priv/libexla.so

# Tensorflow flags and config
TENSORFLOW_NS = tensorflow/compiler/xla/exla
TENSORFLOW_SRC = c_src/tensorflow
TENSORFLOW_EXLA = $(TENSORFLOW_SRC)/$(TENSORFLOW_NS)
BAZEL_FLAGS = --define "framework_shared_object=false" -c $(EXLA_MODE)

all: $(EXLA_TARGET)

cpu: $(TENSORFLOW_EXLA)
	ln -sf $(ERTS_INCLUDE_PATH) c_src/exla/erts
	cd $(TENSORFLOW_SRC) && \
		bazel build $(BAZEL_FLAGS) //$(TENSORFLOW_NS):libexla_cpu.so
	mkdir priv
	cp $(TENSORFLOW_SRC)/bazel-bin/$(TENSORFLOW_NS)/libexla_cpu.so $(EXLA_SO)

cuda: $(TENSORFLOW_EXLA)
	ln -sf $(ERTS_INCLUDE_PATH) c_src/exla/erts
	cd $(TENSORFLOW_SRC) && \
		bazel build $(BAZEL_FLAGS) //$(TENSORFLOW_NS):libexla_gpu.so
	mkdir priv
	cp $(TENSORFLOW_SRC)/bazel-bin/$(TENSORFLOW_NS)/libexla_gpu.so $(EXLA_SO)

$(TENSORFLOW_EXLA):
	git submodule init
	git submodule update
	ln -sf ../../../../exla $(TENSORFLOW_EXLA)

clean:
	cd $(TENSORFLOW_SRC) && bazel clean --expunge
	rm -f $(EXLA_SO)

# TODO: Move the generate_compilation_db.sh to makefile
compile-db:
	cd src/ && \
	sh ./generate_compilation_db.sh && \
	mv bazel-bin/exla/compile_commands.json ../compile_commands.json

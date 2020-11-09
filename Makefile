all: cpu compile-db

cuda:
	cd src/ && \
	bazel build //exla:libexla_gpu.so --config=cuda && \
	cp bazel-bin/exla/libexla_gpu.so ../priv/libexla.so

cpu:
	cd src/ && \
	bazel build //exla:libexla_cpu.so && \
	cp bazel-bin/exla/libexla_cpu.so ../priv/libexla.so

clean:
	cd src/ && \
	bazel clean --expunge && \
	mkdir -p ../priv && \
	rm ../priv/libexla.so | true && \
	rm ../compile_commands.json | true

compile-db:
	cd src/ && \
	sh ./generate_compilation_db.sh && \
	mv bazel-bin/exla/compile_commands.json ../compile_commands.json
ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)

all: compile-db
	cd src/ && \
	bazel build //exla:libexla.so --config=cuda && \
	cp bazel-bin/exla/libexla.so ../priv/libexla.so

clean:
	cd src/ && \
	bazel clean --expunge && \
	mkdir -p ../priv && \
	rm ../priv/libexla.so

compile-db:
	./src/generate_compile_db.sh
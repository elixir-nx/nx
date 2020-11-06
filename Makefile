ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)

# TODO: Shouldn't explicitly pass cuda as a config option
all: # compile-db
	cd src/ && \
	bazel build //exla:libexla.so --config=cuda && \
	cp bazel-bin/exla/libexla.so ../priv/libexla.so

clean:
	cd src/ && \
	bazel clean --expunge && \
	mkdir -p ../priv && \
	rm ../priv/libexla.so && \
	rm ../compile_commands.json

compile-db:
	cd src/ && \
	sh ./generate_compilation_db.sh && \
	mv bazel-bin/exla/compile_commands.json ../compile_commands.json
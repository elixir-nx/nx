ERLANG_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)

all:
	cd src/ && \
	bazel build //exla:libexla.so && \
	cp -r bazel-bin/ ../priv/

clean:
	cd src/ && \
	bazel clean --expunge
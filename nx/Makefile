default_target: all

ERL_INCLUDE_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)
LIBTORCH_INCLUDE_PATH = /Users/catalyst/Projects/C/libtorch/include/
LIBTORCH_LINK_PATH = /Users/catalyst/Projects/C/libtorch/lib/
CFLAGS = -fPIC -I$(ERL_INCLUDE_PATH) -I$(LIBTORCH_INCLUDE_PATH) -O3 -Wall -Wextra -Wno-unused-parameter -shared -std=c++14
LDFLAGS = -L$(LIBTORCH_LINK_PATH) -Wl,-rpath,$(LIBTORCH_LINK_PATH) -ltorch -lc10

# MacOS needs extra flags to link successfully
ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
else  # Linux
	CFLAGS += -shared
	LDFLAGS += -lm
endif


all: c_src/pytorch.cpp
	@$(CXX) $(CFLAGS) c_src/pytorch.cpp -o priv/pytorch.so $(LDFLAGS)


clean:
	rm priv/*.so
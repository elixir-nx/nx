SUBDIRS=	nx exla torchx

all:	setup
	@for _dir in ${SUBDIRS}; do \
		(cd $${_dir} && mix compile); \
	done

setup:
	@for _dir in ${SUBDIRS}; do \
		(cd $${_dir} && mix deps.get); \
	done

test:
	@for _dir in ${SUBDIRS}; do \
		(cd $${_dir} && mix test); \
	done

clean:
	@for _dir in ${SUBDIRS}; do \
		(cd $${_dir} && echo "Cleaning in $${_dir}" && mix clean); \
	done

# Convenient shorthand for doing all of the appropriate steps in CUDA
# environment since it's easy to miss the extra flag setting in the docs.
cuda:
	@env XLA_TARGET=cuda ${MAKE} all test

# Convenient shorthand for doing all of the appropriate steps in ROCm
# environment since it's easy to miss the extra flag setting in the docs.
rocm:
	@env XLA_TARGET=rocm ${MAKE} all test

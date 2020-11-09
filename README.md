# Exla

Elixir XLA Client for compiling and running Elixir code on CPU/GPU/TPU.

## Building

The easiest way to build is with [Docker](https://docs.docker.com/get-docker/). You'll also need to set up the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

To build, clone this repo and run:

```shell
docker build --rm -t exla:cuda10.1 .
```

Then to run:

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/bazel_cache \
  -w $PWD \
  --gpus=all \
  --rm exla:cuda10.1 bash
```

Inside the container you can interact with the API from IEX using:

```shell
iex -S mix
```

Or you can run an example:

```shell
mix run examples/basic_addition.exs
```

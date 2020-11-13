# EXLA

Elixir XLA Client for compiling and running Elixir code on CPU/GPU/TPU.

## Building locally

EXLA is a regular Elixir project, therefore, to run it locally:

```shell
mix deps.get
mix compile
```

Notice [you will need Bazel installed locally](https://bazel.build/).

### Environment variables

  * `EXLA_TARGET` - controls to compile with CPU-only (default) or CUDA-enabled, example: `EXLA_TARGET=cuda`

  * `EXLA_MODE` - controls to compile `opt` (default) artifacts or `dbg`, example: `EXLA_MODE=dbg`

## Building with Docker

The easiest way to build is with [Docker](https://docs.docker.com/get-docker/). You'll also need to set up the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

To build, clone this repo and run:

```shell
docker build --rm -t exla:cuda10.1 .
```

Then to run (without Cuda):

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -w $PWD \
  --gpus=all \
  --rm exla:cuda10.1 bash
```

With Cuda enabled:

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e EXLA_TARGET=cuda \
  -w $PWD \
  --gpus=all \
  --rm exla:cuda10.1 bash
```

Inside the container you can interact with the API from IEx using:

```shell
iex -S mix
```

Or you can run an example:

```shell
mix run examples/basic_addition.exs
```

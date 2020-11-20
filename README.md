# EXLA

Elixir XLA Client for compiling and running Elixir code on CPU/GPU/TPU.

## Usage

Add EXLA as a dependency in your project:

```elixir
def deps do
  {:exla, "~> 0.1"}
end
```

The first compilation will take a long time, as it needs to compile parts of Tensorflow + XLA. You will need the following installed in your system to compile them:

  * [Git](https://git-scm.com/) for checking out Tensorflow
  * [Bazel](https://bazel.build/) for compiling Tensorflow
  * [Python3](https://python.org) with numpy installed (`pip3 install numpy`) for compiling Tensorflow

Subsequent commands should be much faster.

### Environment variables

You can use the following env vars to customize your build:

  * `EXLA_TARGET` - controls to compile with CPU-only (default) or CUDA-enabled, example: `EXLA_TARGET=cuda`

  * `EXLA_MODE` - controls to compile `opt` (default) artifacts or `dbg`, example: `EXLA_MODE=dbg`

  * `EXLA_CACHE` - control where to store Tensorflow checkouts and builds

Note those variables can be set directly in the dependency:

```elixir
def deps do
  {:exla, "~> 0.1", system_env: %{"EXLA_TARGET" => "CUDA"}}
end
```

### GPU Support

To run EXLA on a GPU, you need both CUDA and CUDNN. EXLA has been tested with combinations of CUDA 10.1, 10.2, 11.0, and 11.1 and CUDNN 7 and 8. You can check the [CUDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) to ensure your drivers and versions are compatible.

## Contributing

### Building locally

EXLA is a regular Elixir project, therefore, to run it locally:

```shell
mix deps.get
mix test
```

### Building with Docker

The easiest way to build is with [Docker](https://docs.docker.com/get-docker/). For GPU support, you'll also need to set up the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

To build, clone this repo and run:

```shell
docker build --rm -t exla:cuda10.1 .
```

Then to run (without Cuda):

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e EXLA_CACHE=$PWD/tmp/exla_cache \
  -w $PWD \
  --rm exla:cuda10.1 bash
```

With Cuda enabled:

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e EXLA_CACHE=$PWD/tmp/exla_cache \
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

To run tests:

```shell
mix test
```

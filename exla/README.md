<h1><img src="https://github.com/elixir-nx/nx/raw/main/exla/exla.png" alt="EXLA" width="350"></h1>

Elixir client for Google's XLA (Accelerated Linear Algebra). It includes integration with the `Nx` library to compile numerical definitions (`defn`) to the CPU/GPU.

## Installation

In order to use `Nx`, you will need Elixir installed. Then create an Elixir project via the `mix` build tool:

```
$ mix new my_app
```

Then you can add `EXLA` as dependency in your `mix.exs`. At the moment you will have to use a Git dependency while we work on our first release:

```elixir
def deps do
  [
    {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
    {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
  ]
end
```

You will need the following installed in your system to compile EXLA:

  * [Git](https://git-scm.com/) for checking out Tensorflow
  * [Bazel v3.1](https://bazel.build/) for compiling Tensorflow (note Bazel v4 is available but it is not compatible)
  * [Python3](https://python.org) for compiling Tensorflow

If running on Windows, you will also need:

  * [MSYS2](https://www.msys2.org/)
  * [Microsoft Build Tools 2019](https://visualstudio.microsoft.com/downloads/)
  * [Microsoft Visual C++ 2019 Redistributable](https://visualstudio.microsoft.com/downloads/)

The first compilation will take a long time, as it needs to compile parts of Tensorflow + XLA. Subsequent commands should be much faster.

#### Common Installation Issues

  * Bazel
    * Use `bazel --version` to check your Bazel version, make sure you are using v3.1
    * Most binaries are also available on [Github](https://github.com/bazelbuild/bazel/releases)
    * It can also be installed with `asdf`:
      * asdf plugin-add bazel
      * asdf install bazel 3.1.0
      * asdf global bazel 3.1.0
  * ElixirLS on VSCode
    * Make sure that your Python installation is available globally, as ElixirLS won't know how to activate Python

### GPU Support

To run EXLA on a GPU, you need either ROCm or CUDA/cuDNN. EXLA has been tested with combinations of CUDA 10.1, 10.2, 11.0, and 11.1. You need either cuDNN 7 or 8 installed. [See the installation instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) and check the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) to ensure your drivers and versions are compatible. EXLA has been tested only on ROCm 3.9.0.

In order to link-in the appropriate dependencies for your platform's accelerator, you need to set the appropriate configuration flags in the `EXLA_FLAGS` environment variable.

To link in CUDA dependencies:

```
export EXLA_FLAGS=--config=cuda
```

To link in ROCm dependencies:

```
export EXLA_FLAGS=--config=rocm --action_env=HIP_PLATFORM=hcc
```

When building EXLA locally, it's recommended you set these flags in `.bash_profile` or a similar configuration file so you don't need to export them every time you need to build EXLA.

### Environment variables

You can use the following env vars to customize your build:

  * `EXLA_FLAGS` - controls compilation with GPU support

  * `EXLA_MODE` - controls to compile `opt` (default) artifacts or `dbg`, example: `EXLA_MODE=dbg`

  * `EXLA_CACHE` - controls where to store Tensorflow checkouts and builds

  * `XLA_FLAGS` - controls XLA-specific options, see: [tensorflow/compiler/xla/debug_options_flags.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/debug_options_flags.cc) for list of flags

## Contributing

### Building locally

EXLA is a regular Elixir project, therefore, to run it locally:

```shell
mix deps.get
mix test
```

In order to run tests on a specific device, use the `EXLA_TARGET` environment variable, which is a dev-only variable for this project (it has no effect when using EXLA as a dependency). For example, `EXLA_TARGET=cuda` or `EXLA_TARGET=rocm`.

### Building with Docker

The easiest way to build is with [Docker](https://docs.docker.com/get-docker/). For GPU support, you'll also need to set up the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

To build, clone this repo, select your preferred Dockerfile, and run:

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

With CUDA enabled:

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e EXLA_CACHE=$PWD/tmp/exla_cache \
  -e EXLA_FLAGS=--config=cuda \
  -e EXLA_TARGET=cuda \
  -w $PWD \
  --gpus=all \
  --rm exla:cuda10.1 bash
```

With ROCm enabled:

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e EXLA_CACHE=$PWD/tmp/exla_cache \
  -e EXLA_FLAGS="--config=rocm --action_env=HIP_PLATFORM=hcc" \
  -e EXLA_TARGET=rocm \
  -w $PWD \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --rm exla:rocm bash
```

Inside the container you can interact with the API from IEx using:

```shell
iex -S mix
```

Or you can run an example:

```shell
mix run examples/regression.exs
```

To run tests:

```shell
mix test
```

## License

Copyright (c) 2020 Sean Moriarity

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

<h1><img src="https://github.com/elixir-nx/nx/raw/main/exla/exla.png" alt="EXLA" width="350"></h1>

Elixir client for Google's XLA (Accelerated Linear Algebra). It includes integration with the `Nx` library to compile numerical definitions (`defn`) to the CPU/GPU/TPU.

## Installation

In order to use `EXLA`, you will need Elixir installed. Then create an Elixir project via the `mix` build tool:

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
  * [Bazel v3.7.2](https://bazel.build/) for compiling Tensorflow
  * [Python3](https://python.org) with NumPy installed for compiling Tensorflow

If running on Windows, you will also need:

  * [MSYS2](https://www.msys2.org/)
  * [Microsoft Build Tools 2019](https://visualstudio.microsoft.com/downloads/)
  * [Microsoft Visual C++ 2019 Redistributable](https://visualstudio.microsoft.com/downloads/)

The first compilation will take a long time, as it needs to compile parts of Tensorflow + XLA. Subsequent commands should be much faster.

### Common Installation Issues

  * Missing Dependencies
    * Some Erlang installs do not include some of the dependencies needed to compile the EXLA NIF. You may need to install `erlang-dev` separately.
  * EXLA
    * Make sure you use `:exla` as a `:github` dependency and not as a `:path` dependency to avoid rebuilds
  * Bazel
    * Use `bazel --version` to check your Bazel version, make sure you are using v3.7.2
    * Most binaries are also available on [Github](https://github.com/bazelbuild/bazel/releases)
    * It can also be installed with `asdf`:
      * asdf plugin-add bazel
      * asdf install bazel 3.7.2
      * asdf global bazel 3.7.2
  * GCC
    * You may have issues with newer and older versions of GCC. TensorFlow builds are known to work with GCC versions between 7.5 and 9.3. If your system uses a newer GCC version, you can install an older version and tell Bazel to use it with `export CC=/path/to/gcc-{version}` where version is the GCC version you installed. 
  * ElixirLS on VSCode
    * Make sure that your Python installation is available globally, as ElixirLS won't know how to activate Python

#### Compiling in ElixirLS

ElixirLS will need to run its own compile of `:exla`, so if you want to use ElixirLS, be prepared to possibly wait another 2 hours for it complete. As soon as you open VSCode with ElixirLS enabled and `:exla` as a dependency, let the ElixirLS compile complete (watch the output tab -> ElixirlS) *before* clicking around to any other files in your project, or else ElixirLS will rapid fire queue compiles that will all need to complete before you will be able to use your project. If no output appears in the ElixirLS output, you may need to trigger the compile by opening a file and saving it. Proceed slowly, one step at a time, and as soon as you see a build kick off in the ElixirLS output panel, walk away from the editor until it is done.

#### Python and asdf

`Bazel` cannot find `python` installed via the `asdf` version manager by default. `asdf` uses a function to lookup the specified version of a given binary, this approach prevents `Bazel` from being able to correctly build `EXLA`. The error is `unknown command: python. Perhaps you have to reshim?`. There are two known workarounds:

1. Use a separate installer or explicitly change your `$PATH` to point to a Python installation (note the build process looks for `python`, not `python3`). For example, on Homebrew on macOS, you would do:

    ```
    export PATH=/usr/local/opt/python@3.9/libexec/bin:/usr/local/bin:$PATH
    mix deps.compile
    ```

2. Use the [`asdf direnv`](https://github.com/asdf-community/asdf-direnv) plugin to install [`direnv 2.20.0`](https://direnv.net). `direnv` along with the `asdf-direnv` plugin will explicitly set the paths for any binary specified in your project's `.tool-versions` files.

After doing any of the steps above, it may be necessary to clear the build cache by removing ` ~/.cache/exla`.

### GPU Support

To run EXLA on a GPU, you need either ROCm or CUDA/cuDNN. EXLA has been tested with combinations of CUDA 10.1, 10.2, 11.0, and 11.1. You need either cuDNN 7 or 8 installed. **NOTE: There are known [issues](https://github.com/elixir-nx/nx/issues/197) with cuDNN 8 at this time. Consider using the CUDA Dockerfile which has been tested and known to work.** [See the installation instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) and check the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) to ensure your drivers and versions are compatible. EXLA has been tested only on ROCm 3.9.0.

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

### TPU Support

EXLA supports GCP TPU VMs. If you have acess to a TPU VM, you need only to install Elixir and OTP24. Then, set the appropriate environment variables:

```
export EXLA_FLAGS=--config=tpu
export EXLA_TARGET=tpu
```

### Environment variables

You can use the following env vars to customize your build:

  * `EXLA_FLAGS` - controls compilation with accelerator support

  * `EXLA_MODE` - controls to compile `opt` (default) artifacts or `dbg`, example: `EXLA_MODE=dbg`

  * `EXLA_CACHE` - controls where to store Tensorflow checkouts and builds

  * `XLA_FLAGS` - controls XLA-specific options, see: [tensorflow/compiler/xla/debug_options_flags.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/debug_options_flags.cc) for list of flags

## Usage

The main mechanism to use EXLA is by setting it as the `@defn_compiler` for your numerical definitions:

```elixir
@defn_compiler EXLA
defn softmax(tensor) do
  Nx.exp(tensor) / Nx.sum(Nx.exp(tensor))
end
```

You can also pass `EXLA` as a compiler to `Nx.Defn.jit/4/` and friends:

```elixir
# JIT
Nx.Defn.jit(&some_function/2, [Nx.tensor(1), Nx.tensor(2)], EXLA)

# Async/await
async = Nx.Defn.async(&some_function/2, [Nx.tensor(1), Nx.tensor(2)], EXLA)
Nx.Async.await(async)
```

Those functions are also aliased in the `EXLA` module for your convenience:

```elixir
# JIT
EXLA.jit(&some_function/2, [Nx.tensor(1), Nx.tensor(2)])

# Async/await
async = EXLA.async(&some_function/2, [Nx.tensor(1), Nx.tensor(2)])
Nx.Async.await(async)
```

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
docker build --rm -t exla:host . # Host Docker image
docker build --rm -t exla:cuda10.2 . # CUDA 10.2 Docker image
docker build --rm -t exla:rocm . # ROCm Docker image
```

Then to run (without Cuda):

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e EXLA_CACHE=$PWD/tmp/exla_cache \
  -w $PWD \
  --rm exla:host bash
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
  --rm exla:cuda10.2 bash
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

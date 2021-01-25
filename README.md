*Note: This repository currently holds two projects, `Nx` and `EXLA`, which are described below. The main project, `Nx`, will be extracted to its own repository before the first release.*

<img src="https://github.com/elixir-nx/exla/raw/main/nx.png" alt="Nx" width="400">

Nx is a multi-dimensional tensors library for Elixir. Its main features are:

  * Typed multi-dimensional tensors, where the tensors can be unsigned integers (sizes 8, 16, 32, 64), signed integers (sizes 8, 16, 32, 64), floats (sizes 32, 64) and brain floats (sizes 16);

  * Named tensors, allowing developers to give names to each dimension of the tensors, leading to more readable code and less error prone codebases;

  * Automatic differentiation, also known as autograd. The `grad` function which provides reverse-mode differentiation, extremely useful for linear regression, machine learning algorithms, and more;

  * Tensors backends, which allow the main `Nx` API to be used to manipulate binary tensors, SIMD tensors, sparse matrices, and more;

  * Compiled definitions, known as `defn`, provides multi-stage programming and allow tensor operations to be compiled to multiple targets, such as highly specialized CPU code or the GPU. The compilation can happen either ahead-of-time (AOT) or just-in-time (JIT);

Other features include broadcasting, multi-device support, etc. You can find planned enhancements and features in the issues tracker. If you need one particular feature to move forward, don't hesitate to let us know.

For Python developers, `Nx` takes its main inspirations from [`Numpy`](https://numpy.org/) and [`Jax`](https://github.com/google/jax) but packaged into a single unified library. There are also plans to [support labelled coordinates](https://github.com/elixir-nx/exla/issues/167).

## Examples

Let's create a tensor:

```elixir
iex> t = Nx.tensor([[1, 2], [3, 4]])
iex> Nx.shape(t)
{2, 2}
```

To implement [the Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
using this library:

```elixir
iex> t = Nx.tensor([[1, 2], [3, 4]])
iex> Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t)))
#Nx.Tensor<
  f64[2][2]
  [
    [0.03205860328008499, 0.08714431874203257],
    [0.23688281808991013, 0.6439142598879722]
  ]
>
```

See the `bench` and `examples` directory for some use cases.

## Numerical definitions

By default, `Nx` uses pure Elixir code. Since Elixir is a functional and immutable language, each operation above makes a copy of the tensor, which is quite innefficient.

However, `Nx` also comes with numerical definitions, called `defn`, which is a subset of Elixir tailored for numerical computations. For example, it overrides Elixir's default operators so they are tensor-aware:

```elixir
defmodule MyModule do
  import Nx.Defn

  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end
end
```

`defn` supports multiple compiler backends, which can compile said functions to run on the CPU or in the GPU. For example, using the `EXLA` compiler:

```elixir
@defn_compiler {EXLA, platform: :host}
defn softmax(t) do
  Nx.exp(t) / Nx.sum(Nx.exp(t))
end
```

Once `softmax` is called, `Nx.Defn` will invoke `EXLA` to emit a just-in-time and high-specialized compiled version of the code, tailored to the tensor type and shape. By passing `platform: :cuda` or `platform: :rocm`, the code can be compiled for the GPU.

`defn` relies on a technique called multi-stage programming, which is built on top of Elixir functional and meta-prgramming capabilities: we transform Elixir code to emit an AST that is then transformed to run on the CPU/GPU.

Many of Elixir features are supported inside `defn`, such as the pipe operator, aliases, conditionals, pattern-matching, and more. Other features such as loops, updates, and access (generally known as slicing) are on the roadmap. `defn` also support `transforms`, which allows numerical definitions to be transformed at runtime. Automatic differentiations, via the `grad` function, is one example of transforms.

# EXLA

Elixir XLA Client for compiling and running Elixir code on CPU/GPU/TPU. It also provides compilers for the `Nx` library.

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

If running on Windows, you will also need:
  
  * [MSYS2](https://www.msys2.org/)
  * [Microsoft Build Tools 2019](https://visualstudio.microsoft.com/downloads/)
  * [Microsoft Visual C++ 2019 Redistributable](https://visualstudio.microsoft.com/downloads/)

Subsequent commands should be much faster.

### Environment variables

You can use the following env vars to customize your build:

  * `EXLA_TARGET` - controls to compile with CPU-only (default) or CUDA-enabled, example: `EXLA_TARGET=cuda`

  * `EXLA_MODE` - controls to compile `opt` (default) artifacts or `dbg`, example: `EXLA_MODE=dbg`

  * `EXLA_CACHE` - control where to store Tensorflow checkouts and builds

  * `XLA_FLAGS` - controls XLA-specific options, see: [tensorflow/compiler/xla/debug_options_flags.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/debug_options_flags.cc) for list of flags

### GPU Support

To run EXLA on a GPU, you need either ROCm or CUDA/cuDNN. EXLA has been tested with combinations of CUDA 10.1, 10.2, 11.0, and 11.1 and cuDNN 7 and 8. You can check the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) to ensure your drivers and versions are compatible. EXLA has been tested only on ROCm 3.9.0.

## Contributing

### Building locally

EXLA is a regular Elixir project, therefore, to run it locally:

```shell
mix deps.get
mix test
```

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
  -e EXLA_TARGET=cuda \
  -w $PWD \
  --gpus=all \
  --rm exla:cuda10.1 bash
```

With ROCm enables:

```shell
docker run -it \
  -v $PWD:$PWD \
  -e TEST_TMPDIR=$PWD/tmp/bazel_cache \
  -e EXLA_CACHE=$PWD/tmp/exla_cache \
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
mix run examples/basic_addition.exs
```

To run tests:

```shell
mix test
```

## License

Copyright (c) 2020 Dashbit

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

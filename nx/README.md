<h1><img src="https://github.com/elixir-nx/nx/raw/main/nx/nx.png" alt="Nx" width="400"></h1>

Nx is a multi-dimensional tensors library for Elixir with multi-staged compilation to the CPU/GPU. Its high-level features are:

  * Typed multi-dimensional tensors, where the tensors can be unsigned integers (`u8`, `u16`, `u32`, `u64`), signed integers (`s8`, `s16`, `s32`, `s64`), floats (`f32`, `f64`) and brain floats (`bf16`);

  * Named tensors, allowing developers to give names to each dimension, leading to more readable and less error prone codebases;

  * Automatic differentiation, also known as autograd. The `grad` function provides reverse-mode differentiation, useful for simulations, training probabilistic models, etc;

  * Tensors backends, which enables the main `Nx` API to be used to manipulate binary tensors, GPU-backed tensors, sparse matrices, and more;

  * Numerical definitions, known as `defn`, provide multi-stage compilation of tensor operations to multiple targets, such as highly specialized CPU code or the GPU. The compilation can happen either ahead-of-time (AOT) or just-in-time (JIT) with a compiler of your choice;

You can find planned enhancements and features in the issues tracker. If you need one particular feature to move forward, don't hesitate to let us know and give us feedback.

For Python developers, `Nx` currently takes its main inspirations from [`Numpy`](https://numpy.org/) and [`JAX`](https://github.com/google/jax) but packaged into a single unified library.

## Community

Developers interested in Numerical Elixir can join the community and interact in the following places:

  * For general discussion on Numerical Elixir and Machine Learning, [join the #machine-learning channel in the Erlang Ecosystem Foundation Slack](https://erlef.org/wg/machine-learning) (click on the link on the sidebar on the right)

  * For bugs and pull requests, use the [issues tracker](https://github.com/elixir-nx/nx)

  * For feature requests and Nx-specific discussion, [join the Nx mailing list](https://groups.google.com/g/elixir-nx)

Nx discussion is also welcome on any of the Elixir-specific forums and chats maintained by the community.

## Support

In order to support Nx, you might:

  * Become a supporting member or a sponsor of the Erlang Ecosystem Foundation. The Nx project is part of the [Machine Learning WG](https://erlef.org/wg/machine-learning).

  * Nx's mascot is the Numbat, a marsupial native to southern Australia. Unfortunately the Numbat are endangered and it is estimated to be fewer than 1000 left. If you enjoy this project, consider donating to Numbat conservation efforts, such as [Project Numbat](https://www.numbat.org.au/) and [Australian Wildlife Conservancy](https://www.australianwildlife.org). The Project Numbat website also contains Numbat related swag.

## Resources

Here are some introductory resources with more information on Nx as a whole:

  * [A post by José Valim on Dashbit's blog announcing Nx, outlining some of the design decisions, benchmarks, and general direction](https://dashbit.co/blog/nx-numerical-elixir-is-now-publicly-available) (text)

  * [Sean Moriarity's blog](https://seanmoriarity.com/) containing tips on how to use Nx (text)

  * [The ThinkingElixir podcast where José Valim unveiled Nx](https://thinkingelixir.com/podcast-episodes/034-jose-valim-reveals-project-nx/) (audio - starts around 8 minutes in)

  * [A talk by José Valim at Lambda Days 2021 where he builds a neural network from scratch with Nx](https://www.youtube.com/watch?v=fPKMmJpAGWc) (video)

## Installation

In order to use `Nx`, you will need Elixir installed. Then create an Elixir project via the `mix` build tool:

```
$ mix new my_app
```

Then you can add `Nx` as dependency in your `mix.exs`. At the moment you will have to use a Git dependency while we work on our first release:

```elixir
def deps do
  [
    {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "nx"}
  ]
end
```

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
  f32[2][2]
  [
    [0.032058604061603546, 0.08714432269334793],
    [0.23688282072544098, 0.6439142227172852]
  ]
>
```

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

`defn` supports multiple compiler backends, which can compile said functions to run on the CPU or in the GPU. For example, [using the `EXLA` compiler](https://github.com/elixir-nx/nx/tree/main/exla), which provides bindings to Google's XLA:

```elixir
@defn_compiler {EXLA, client: :host}
defn softmax(t) do
  Nx.exp(t) / Nx.sum(Nx.exp(t))
end
```

Once `softmax` is called, `Nx.Defn` will invoke `EXLA` to emit a just-in-time and high-specialized compiled version of the code, tailored to the input tensors type and shape. By passing `client: :cuda` or `client: :rocm`, the code can be compiled for the GPU. For reference, here are some benchmarks of the function above when called with a tensor of one million random float values:

```
Name                       ips        average  deviation         median         99th %
xla gpu f32 keep      15308.14      0.0653 ms    ±29.01%      0.0638 ms      0.0758 ms
xla gpu f64 keep       4550.59        0.22 ms     ±7.54%        0.22 ms        0.33 ms
xla cpu f32             434.21        2.30 ms     ±7.04%        2.26 ms        2.69 ms
xla gpu f32             398.45        2.51 ms     ±2.28%        2.50 ms        2.69 ms
xla gpu f64             190.27        5.26 ms     ±2.16%        5.23 ms        5.56 ms
xla cpu f64             168.25        5.94 ms     ±5.64%        5.88 ms        7.35 ms
elixir f32                3.22      311.01 ms     ±1.88%      309.69 ms      340.27 ms
elixir f64                3.11      321.70 ms     ±1.44%      322.10 ms      328.98 ms

Comparison:
xla gpu f32 keep      15308.14
xla gpu f64 keep       4550.59 - 3.36x slower +0.154 ms
xla cpu f32             434.21 - 35.26x slower +2.24 ms
xla gpu f32             398.45 - 38.42x slower +2.44 ms
xla gpu f64             190.27 - 80.46x slower +5.19 ms
xla cpu f64             168.25 - 90.98x slower +5.88 ms
elixir f32                3.22 - 4760.93x slower +310.94 ms
elixir f64                3.11 - 4924.56x slower +321.63 ms
```

See the [`bench`](https://github.com/elixir-nx/nx/tree/main/exla/bench) and [`examples`](https://github.com/elixir-nx/nx/tree/main/exla/examples) directory inside the EXLA project for more information.

`defn` relies on a technique called multi-stage programming, which is built on top of Elixir functional and meta-programming capabilities: we transform Elixir code to emit an AST that is then transformed to run on the CPU/GPU. Ultimately, the `defn` compiler is pluggable, which means developers can implement bindings for different tensor compiler technologies and choose the most appropriate one.

Many of Elixir features are supported inside `defn`, such as the pipe operator, aliases, conditionals, pattern-matching, and more. Other features such as loops and in-place updates are on the roadmap. `defn` also support `transforms`, which allows numerical definitions to be transformed at runtime. Automatic differentiation, via the `grad` function, is one example of transforms.

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

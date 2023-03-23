<h1><img src="https://github.com/elixir-nx/nx/raw/main/nx/nx.png" alt="Nx" width="400"></h1>

[![Package](https://img.shields.io/badge/-Package-important)](https://hex.pm/packages/nx) [![Documentation](https://img.shields.io/badge/-Documentation-blueviolet)](https://hexdocs.pm/nx)

Nx is a multi-dimensional tensors library for Elixir with multi-staged compilation to the CPU/GPU. Its high-level features are:

  * Typed multi-dimensional tensors, where the tensors can be unsigned integers (`u8`, `u16`, `u32`, `u64`), signed integers (`s8`, `s16`, `s32`, `s64`), floats (`f16`, `f32`, `f64`), brain floats (`bf16`), and complex numbers (`c64`, `c128`);

  * Named tensors, allowing developers to give names to each dimension, leading to more readable and less error prone codebases;

  * Automatic differentiation, also known as autograd. The `grad` function provides reverse-mode differentiation, useful for simulations, training probabilistic models, etc;

  * Numerical definitions, known as `defn`, is a subset of Elixir that is compilable to multiple targets, including GPUs. See [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) for just-in-time (JIT) compilation for CPUs/GPUs/TPUs and [Torchx](https://github.com/elixir-nx/nx/tree/main/torchx) for CPUs/GPUs support;

  * Built-in distributed² serving: encapsulate complex numerical pipelines into `Nx.Serving`. Servings provide batching, up to a given size or time, distribution over multiple CPU cores and GPU devices, as well as distribution over a cluster of machines;

  * Support for data streaming and hooks, allowing developers to send and receive data from CPUs/GPUs/TPUs while computations are running;

  * Support for linear algebra primitives via `Nx.LinAlg`;

You can find planned enhancements and features in the issues tracker. If you need one particular feature to move forward, don't hesitate to let us know and give us feedback.

For Python developers, `Nx` packages features from [`Numpy`](https://numpy.org/), [`JAX`](https://github.com/google/jax), [HuggingFace Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines), and TorchServing/TensorServing, but packaged into a single unified library and developer experience.

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

  * [A post by José Valim on Nx v0.1 release, discussing its goals, showing benchmarks, and general direction](https://dashbit.co/blog/elixir-and-machine-learning-nx-v0.1) (text)

  * [Sean Moriarity's blog](https://seanmoriarity.com/) containing tips on how to use Nx (text)

  * [A talk by José Valim at Lambda Days 2021 where he builds a neural network from scratch with Nx](https://www.youtube.com/watch?v=fPKMmJpAGWc) (video)

  * [A screencast by José Valim announcing Livebook, where he also showcases Nx and Axon (a neural network library built on top of Nx)](https://www.youtube.com/watch?v=RKvqc-UEe34) (video)

  * [An article by Philip Brown showing an end-to-end example of running a Machine Learning model with Elixir in production](https://fly.io/phoenix-files/recognize-digits-using-ml-in-elixir/) (text)

  * [The announcement of Bumblebee, which provides pre-trained machine learning models such BERT, StableDiffusion, and others](https://news.livebook.dev/announcing-bumblebee-gpt2-stable-diffusion-and-more-in-elixir-3Op73O) (text+video)

## Installation

In order to use `Nx`, you will need Elixir installed. Then create an Elixir project via the `mix` build tool:

```
$ mix new my_app
```

Then you can add `Nx` as dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:nx, "~> 0.2"}
  ]
end
```

If you are using Livebook or IEx, you can instead run:

```elixir
Mix.install([
  {:nx, "~> 0.2"}
])
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

By default, `Nx` uses pure Elixir code. Since Elixir is a functional and immutable language, each operation above makes a copy of the tensor, which is quite innefficient. You can use either [EXLA](https://github.com/elixir-nx/nx/tree/main/exla) or [Torchx](https://github.com/elixir-nx/nx/tree/main/torchx) backends for an improvement in performance, often over 3 orders of magnitude, as well as the ability to work on the data in the GPU. See the README of those projects for more information.

## Numerical definitions

`Nx` also comes with numerical definitions, called `defn`, which is a subset of Elixir tailored for numerical computations. For example, it overrides Elixir's default operators so they are tensor-aware:

```elixir
defmodule MyModule do
  import Nx.Defn

  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end
end
```

You can now invoke it as:

```elixir
MyModule.softmax(Nx.tensor([1, 2, 3]))
```

`defn` relies on a technique called multi-stage programming, which is built on top of Elixir functional and meta-programming capabilities: we transform Elixir code to build a graph of your numerical definitions. This brings two important capabilities:

  1. We can transform this graph to provide features such as automatic differentiation, type lowering, and more

  2. We support custom compilers, which can compile said definitions to run on the CPU and GPU just-in-time

For example, [using the `EXLA` compiler](https://github.com/elixir-nx/nx/tree/main/exla), which provides bindings to Google's XLA:

```elixir
will_jit = EXLA.jit(&MyModule.softmax/1)
will_jit.(some_tensor)
```

Once `softmax` is called, `EXLA` will emit a just-in-time and high-specialized compiled version of the code, tailored to the input tensors type and shape. By setting the `XLA_TARGET` environment variable to `cuda` or `rocm`, the code can be compiled for the GPU. For reference, here are some benchmarks of the function above when called with a tensor of one million random float values:

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

Many of Elixir's features are supported inside `defn`, such as the pipe operator, aliases, conditionals, pattern-matching, and more. It also brings exclusive features to numerical definitions, such as `while` loops, automatic differentiation via the `grad` function, hooks to inspect data running on the GPU, and more.

## Working with processes

Elixir runs on the Erlang Virtual Machine, which runs your code inside lightweight thread of executions called "processes". Sending tensors between processes is done by sharing, no copying is required. The tensors are then [refcounted](https://en.wikipedia.org/wiki/Reference_counting) and garbage collected once all processes no longer hold a reference to them.

Nx also allows developers to run different Erlang VM processes, each against a different GPU device. For using, with EXLA, one could do:

```elixir
Nx.default_backend({EXLA.Backend, client: :cuda, device_id: 1})
```

or:

```elixir
will_jit = EXLA.jit(&MyModule.softmax/1, client: :cuda, device_id: 1)
```

And, from that moment on, all operations will happen within a particular GPU instance. You can then use Elixir's message-passing abilities to coordinate the necessary work across processes.

## Why Elixir?

The goal of the Nx project is to marry the power of numerical computing with the Erlang VM capabilities for building concurrent, scalable, and fault-tolerant systems.

Elixir is a functional programming language that runs on the Erlang VM. And, at this point, you might ask: is functional programming a good fit for numerical computing? One of the main concerns is that immutability can lead to high memory usage when working with large blobs of memory. And that's true!

However, it turns out that the most efficient way of executing numerical computations is by first building a graph of all computations, then compiling that graph to run on your CPUs/GPUs just-in-time. At this point, your numerical computing code becomes a function:

    input -> [compiled numerical computing graph] -> output

The `input` is an Elixir data-structure. Inside the function, the algorithm is highly optimized and free to mutate the data in any way it seems fit. Then we get an output that once again must obey Elixir semantics.

To build those graphs, immutability becomes an indispensable tool both in terms of implementation and reasoning. As an example, the JAX library for Python, which has been one of the inspirations for Nx, also promotes functional and immutable principles:

> JAX is intended to be used with a functional style of programming
>
> — JAX Docs

> Unlike NumPy arrays, JAX arrays are always immutable
>
> — JAX Docs

At the end of the day, Elixir provides the functional foundation and a powerful macro system that allows us to compile a subset of Elixir to the CPU/GPU.

With the addition of `Nx.Serving`, we started to marry the benefits of the Erlang VM with numerical computing. With `Nx.Serving`, you can batch numerical computing requests, as well as load balance requests over a cluster of machines. This makes it easy to embed and scale Nx code within your existing Elixir systems, both horizontally and vertically, and without a need for third-party services.

We also expect numerical computing to complement the Elixir ecosystem in different ways, such as:

  * running Machine Learning models in real-time within your [Phoenix web application](https://phoenixframework.org/)

  * deploying models, signal processing, and data modelling inside embedded systems [via Nerves](https://www.nerves-project.org/)

  * incorporating data analysis and classification algorithms inside concurrent data pipelines powered [by Broadway](https://www.elixir-broadway.org/)

  * adding audio and video processing and AI capabilities to media systems [through Membrane](https://membrane.stream/)

We are also excited to explore how Nx and Elixir can be used under distinct domains, such as federated learning, and leverage the Erlang VM ability to handle data in and out of hundreds of thousands of devices concurrently.

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

# Torchx

Elixir client for LibTorch (from PyTorch). It includes a backend for `Nx` for native execution of tensor operations (inside and outside of `defn`).

This project is currently alpha and it supports only a fraction of `Nx`'s API.

## Installation

In order to use `Torchx`, you will need Elixir installed. Then create an Elixir project via the `mix` build tool:

```
$ mix new my_app
```

Then you can add `Torchx` as dependency in your `mix.exs`. At the moment you will have to use a Git dependency while we work on our first release:

```elixir
def deps do
  [
    {:torchx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "torchx"},
    {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
  ]
end
```

To compile `Torchx`, you will need to download a precompiled version of `LibTorch` and tell `Torchx` how to use it:

  1. [Download `LibTorch` from PyTorch's official website](https://pytorch.org/get-started/locally/) - make sure you select "LibTorch" as the package option alongside your Operating System
  2. Unpackage LibTorch to a directory of your choice
  3. Set the `LIBTORCH_DIR` [environment variable](https://en.wikipedia.org/wiki/Environment_variable) to point to the `LibTorch` directory

If running on Windows, you will also need `make` and a `C++` compiler:

  * [Microsoft Build Tools 2019](https://visualstudio.microsoft.com/downloads/)
  * [Microsoft Visual C++ 2019 Redistributable](https://visualstudio.microsoft.com/downloads/)

## Usage

The main mechanism to use `Torchx` is by setting it as a backend to your tensors:

```elixir
Nx.tensor([1, 2, 3], backend: Torchx.Backend)
Nx.iota({100, 100}, backend: Torchx.Backend)
```

Then you can proceed to use `Nx` functions as usual!

You can also set `Torchx` as a default backend, which will apply to all tensors created by the current Elixir process:

```elixir
Nx.default_backend(Torchx.Backend)
Nx.tensor([1, 2, 3])
Nx.iota({100, 100})
```

See `Nx.default_backend/2` for more information.

## License

Copyright (c) 2021 Stas Versilov, Dashbit

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

<h1><img src="https://github.com/elixir-nx/nx/raw/main/torchx/torchx.png" alt="Torchx" width="400"></h1>

Elixir client for PyTorch (through the LibTorch C++ frontend).
It includes a backend for `Nx` for native execution of tensor
operations (inside and outside of `defn`).

## Installation

In order to use `Torchx`, you will need Elixir installed. Then create an Elixir project
via the `mix` build tool:

```
$ mix new my_app
```

Then you can add `Torchx` as dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:torchx, "~> 0.9"}
  ]
end
```

If you are using Livebook or IEx, you can instead run:

```elixir
Mix.install([
  {:torchx, "~> 0.9"}
])
```

We will automatically download a precompiled version of `LibTorch` that
runs on the CPU. If you want to use another version, you can set `LIBTORCH_VERSION`
to `2.1.0` or later.

If you want torch with CUDA support, please use `LIBTORCH_TARGET` to choose
CUDA versions. The current supported targets are:

- `cpu` default CPU only version
- `cu118` CUDA 11.8 and CPU version (no macOS support)
- `cu126` CUDA 12.6 and CPU version (no macOS support)
- `cu128` CUDA 12.8 and CPU version (no macOS support)

Once downloaded, we will compile `Torchx` bindings. You will need `make`/`nmake`,
`cmake` (3.12+) and a `C++` compiler. If building on Windows, you will need:

- [Microsoft Build Tools 2019](https://visualstudio.microsoft.com/downloads/)
- [Microsoft Visual C++ 2019 Redistributable](https://visualstudio.microsoft.com/downloads/)
- [CMake](https://cmake.org/)

You can also compile LibTorch from scratch, using the official instructions,
and configure this library to use it by pointing the `LIBTORCH_DIR` environment
variable to the LibTorch directory.

## Usage

The main mechanism to use `Torchx` is by setting it as a backend to your tensors:

```elixir
Nx.tensor([1, 2, 3], backend: Torchx.Backend)
Nx.iota({100, 100}, backend: Torchx.Backend)
```

Then you can proceed to use `Nx` functions as usual!

You can also set `Torchx` as a default backend, which will apply to all tensors created
by the current Elixir process:

```elixir
Nx.default_backend(Torchx.Backend)
Nx.tensor([1, 2, 3])
Nx.iota({100, 100})
```

See `Nx.default_backend/1` for more information.

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

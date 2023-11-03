# Candlex

[![ci](https://github.com/mimiquate/candlex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mimiquate/candlex/actions?query=branch%3Amain)
[![Hex.pm](https://img.shields.io/hexpm/v/candlex.svg)](https://hex.pm/packages/candlex)
[![Docs](https://img.shields.io/badge/docs-gray.svg)](https://hexdocs.pm/candlex)

An `Nx` [backend](https://hexdocs.pm/nx/Nx.html#module-backends) for [candle](https://huggingface.github.io/candle) machine learning minimalist framework

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `candlex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:candlex, "~> 0.1.2"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/candlex>.

## Usage

Just configure Nx to default to Candlex backend in your configuration:

```elixir
# Possibly config/runtime.exs

config :nx, default_backend: Candlex.Backend
```

or in your scripts, precede all your Nx operations with:

```elixir
Nx.default_backend(Candlex.Backend)
```

More details in [Nx backends](https://hexdocs.pm/nx/Nx.html#module-backends)

#### `CANDLEX_NIF_BUILD`

Defaults to `false`. If `true` the native binary is built locally, which may be useful
if no precompiled binary is available for your target environment. Once set, you
must run `mix deps.clean candlex --build` explicitly to force to recompile.
Building has a number of dependencies, see *Building from source* below.

## Building from source

To build the native binary locally you need to set `CANDLEX_NIF_BUILD=true`.
Keep in mind that the compilation usually takes time.

You will need the following installed in your system for the compilation:

  * [Git](https://git-scm.com) for fetching candle-core source
  * [Rust](https://www.rust-lang.org) with cargo to compile rustler NIFs

## Releasing

To publish a new version of this package:

1. Update `@version` in `mix.exs` and `project-version` in `.github/workflows/binaries.yml`.
1. `git tag -s <tag-version>` to create new signed tag.
1. `git push origin <tag-version>` to push the tag.
1. Wait for the `binaries.yml` GitHub workflow to build all the NIF binaries.
1. `mix rustler_precompiled.download Candlex.Native --all --print` to generate binaries checksums locally.
1. `rm -r native/candlex/target` to leave out rust crate build artifacts from published elixir package.
1. `mix hex.build --unpack` to check the package includes the correct files.
1. Publish the release from draft in GitHub.
1. `mix hex.publish` to publish package to Hex.pm.

## License

Copyright 2023 Mimiquate

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

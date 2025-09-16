# Installation

The only prerequisite for installing Nx is Elixir itself. If you don't have Elixir installed
in your machine you can visit this [installation page](https://elixir-lang.org/install.html).

There are several ways to install Nx (Numerical Elixir), depending on your project type and needs.

## Using Mix in a standardElixir Project

If you are working inside a Mix project, the recommended way to install Nx is by adding it to your mix.exs dependencies:

1. Open mix.exs and modify the deps function:

```elixir
defp deps do
  [
    {:nx, "~> 0.9"}  # Install the latest stable version
  ]
end
```

2. Fetch the dependencies, run on the terminal:

```sh
mix deps.get
```

## Installing Nx from GitHub (Latest Development Version)

If you need the latest, unreleased features, install Nx directly from the GitHub repository.

1. Modify `mix.exs`:

```elixir
defp deps do
  [
    {:nx, github: "elixir-nx/nx", branch: "main", sparse: "nx"}
  ]
end
```

2. Fetch dependencies:

```sh
mix deps.get
```

## Installing Nx in a Standalone Script (Without a Mix Project)

If you don’t have a Mix project and just want to run a standalone script, use Mix.install/1 to dynamically fetch and install Nx.

```elixir
Mix.install([:nx])

require Nx

tensor = Nx.tensor([1, 2, 3])
IO.inspect(tensor)
```

Run the script with:

```sh
elixir my_script.exs
```

Best for: Quick experiments, small scripts, or one-off computations.

## Installing the Latest Nx from GitHub in a Standalone Script

To use the latest development version in a script (without a Mix project):

```elixir
Mix.install([
  {:nx, github: "elixir-nx/nx", branch: "main", sparse: "nx"}
])

require Nx

tensor = Nx.tensor([1, 2, 3])
IO.inspect(tensor)
```

Run:

```sh
elixir my_script.exs
```

Best for: Trying new features from Nx without creating a full project.

## Installing Nx with EXLA for GPU Acceleration

To enable GPU/TPU acceleration with Google’s XLA backend, install Nx along with EXLA:

1. Modify mix.exs:

```elixir
defp deps do
  [
    {:nx, "~> 0.9"},
    {:exla, "~> 0.9"}  # EXLA (Google XLA Backend)
  ]
end
```

2. Fetch dependencies:

```sh
mix deps.get
```

3. Run with EXLA enabled:

```elixir
Nx.default_backend(EXLA.Backend)
Nx.Defn.default_options(compiler: EXLA)
```

Best for: Running Nx on GPUs or TPUs using Google’s XLA compiler.

## Installing Nx with Torchx for PyTorch Acceleration

To run Nx operations on PyTorch’s backend (LibTorch):

1. Modify mix.exs:

```elixir
defp deps do
  [
    {:nx, "~> 0.9"},
    {:torchx, "~> 0.9"}  # PyTorch Backend
  ]
end

```

2. Fetch dependencies:

```sh
mix deps.get
```

3. Run with Torchx enabled:

```elixir
Nx.default_backend(Torchx.Backend)
```

Best for: Deep learning applications with PyTorch acceleration.

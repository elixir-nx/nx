defmodule EXLA do
  @moduledoc """
  Bindings and Nx integration for [Google's XLA](https://www.tensorflow.org/xla/).

  ## defn compiler

  Most often, this library will be used as `Nx.Defn` compiler, like this:

      @defn_compiler EXLA
      defn softmax(tensor) do
        Nx.exp(n) / Nx.sum(Nx.exp(n))
      end

  Then, every time `softmax/1` is called, EXLA will just-in-time (JIT)
  compile a native implementation of the function above, tailored for the
  type and shape of the given tensor. Ahead-of-time (AOT) compilation is
  planned for future versions.

  EXLA is able to compile to the CPU or GPU, by customizing the default
  client or specifying your own client:

      @defn_compiler {EXLA, client: :cuda}
      defn softmax(tensor) do
        Nx.exp(n) / Nx.sum(Nx.exp(n))
      end

  Read the "Client" section below for more information.

  ### Options

  The options accepted by the EXLA compiler are:

    * `:client` - an atom representing the client to use. Defaults
      to `:default`. See "Clients" section

    * `:run_options` - options given when running the computation:

      * `:keep_on_device` - if the data should be kept on the device,
        useful if multiple computations are done in a row. See
        "Device allocation" section

  ## Clients

  The `EXLA` library uses a client for compiling and executing code.
  Those clients are typically bound to a platform, such as CPU or
  GPU.

  Those clients are singleton resources on Google's XLA library,
  therefore they are treated as a singleton resource on this library
  too. You can configure a client via the application environment.
  For example, to configure the default client:

      config :exla, :clients,
        default: [platform: :host]

  `platform: :host` is the default value. You can configure it to
  use the GPU though:

      config :exla, :clients,
        default: [platform: :cuda]

  You can also specify multiple clients for different platforms:

      config :exla, :clients,
        default: [platform: :host],
        cuda: [platform: :cuda]

  While specifying multiple clients is possible, keep in mind you
  want a single client per platform. If you have multiple clients
  per platform, they can race each other and fight for resources,
  such as memory. Therefore, we recommend developers to use the
  `:default` client as much as possible.

  ## Device allocation

  EXLA also ships with a `EXLA.DeviceBackend` that allows data
  to be either be explicitly allocated or kept on the EXLA device
  after a computation. For example:

      @defn_compiler {EXLA, run_options: [keep_on_device: true]}
      defn softmax(tensor) do
        Nx.exp(n) / Nx.sum(Nx.exp(n))
      end

  Will keep the computation on the device, either the CPU or GPU.
  For CPU, this is actually detrimental, as allocating an Elixir
  binary has the same cost as keeping it on CPU, but this yields
  important performance benefits on the GPU.

  If data is kept on the device, you can pipe it into other `defn`
  computations running on the same compiler (in this case, the
  `EXLA` compiler) but you cannot use the regular `Nx` operations,
  unless you transfer it back:

      Nx.tensor([1, 2, 3, 4])
      |> softmax()
      |> Nx.backend_transfer() # bring the data back to Elixir

  You can also use `Nx.backend_transfer/1` to put data on a given
  device before invoking a `defn` function:

      # Explicitly move data to the device, useful for GPU
      Nx.backend_transfer(Nx.tensor([1, 2, 3, 4]), EXLA.DeviceBackend)

  If instead you want to make a copy of the data, you can use
  `Nx.backend_copy/1` instead. However, when working with large
  data, be mindful of memory allocations.

  ## Docker considerations

  EXLA should run fine on Docker with one important consideration:
  you must not start the Erlang VM as the root process in Docker.
  That's because when the Erlang VM runs as root, it has to manage
  all child programs.

  At the same time, Google XLA's shells out to child program during
  compilation and it must retain control over how child programs
  terminate.

  To address this, simply make sure you wrap the Erlang VM in
  another process, such as the shell one. In other words, if you
  are using releases, instead of this:

      RUN path/to/release start

  do this:

      RUN sh -c "path/to/release start"

  If you are using Mix inside your Docker containers, instead of this:

      RUN mix run

  do this:

      RUN sh -c "mix run"

  Alternatively, you can pass the `--init` flag to `docker run`, so
  it runs an `init` inside the container that forwards signals and
  reaps processes.
  """

  @behaviour Nx.Defn.Compiler

  @doc """
  A shortcut for `Nx.Defn.jit/4` with the EXLA compiler.

      iex> EXLA.jit(&Nx.add(&1, &1), [Nx.tensor([1, 2, 3])])
      #Nx.Tensor<
        s64[3]
        [2, 4, 6]
      >

  See the moduledoc for options.
  """
  def jit(function, args, options \\ []) do
    Nx.Defn.jit(function, args, Keyword.put(options, :compiler, EXLA))
  end

  @doc """
  A shortcut for `Nx.Defn.aot/4` with the EXLA compiler.

      iex> functions = [{:exp, &Nx.exp/1, [Nx.template({3}, {:s, 64})]}]
      iex> EXLA.aot(ExpAotDemo, functions)
      iex> ExpAotDemo.exp(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32[3]
        [2.7182817459106445, 7.389056205749512, 20.08553695678711]
      >

  See `export_aot/4` for options.
  """
  def aot(module, functions, options \\ []) do
    Nx.Defn.aot(module, functions, Keyword.put(options, :compiler, EXLA))
  end

  @doc """
  A shortcut for `Nx.Defn.export_aot/5` with the EXLA compiler.

      iex> functions = [{:exp, &Nx.exp/1, [Nx.template({3}, {:s, 64})]}]
      iex> :ok = EXLA.export_aot("tmp", ExpExportDemo, functions)
      iex> defmodule Elixir.ExpExportDemo do
      ...>   Nx.Defn.import_aot("tmp", __MODULE__)
      ...> end
      iex> ExpExportDemo.exp(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32[3]
        [2.7182817459106445, 7.389056205749512, 20.08553695678711]
      >

  Note ahead-of-time compilation for EXLA only runs on the CPU.
  It is slightly less performant than the JIT variant for the
  CPU but it compensates the loss in performance by removing
  the need to have EXLA up and running in production.

  ## Options

    * `:target_features` - the default executable makes
      no assumption about the target runtime, so special
      instructions such as SIMD are not leveraged. But you
      can specify those flags if desired:

          target_features: "+sse4.1 +sse4.2 +avx +avx2 +fma"

  The following options might be used for cross compilation:

    * `:bazel_flags` - flags that customize `bazel build` command

    * `:bazel_env` - flags that customize `bazel build` environment.
      It must be a list of tuples where the env key and env value
      are binaries

    * `:target_triple` - the target triple to compile to.
      It defaults to the current target triple but one
      can be set for cross-compilation. A list is available
      [on Tensorflow repo](https://github.com/tensorflow/tensorflow/blob/e687cab61615a95b8aea79120c9f168d4cc30955/tensorflow/compiler/aot/tfcompile.bzl).
      Note this configures only how the tensor expression is
      compiled but not the underlying NIF. For cross compilation,
      one has to set the proper `:bazel_flags` and `:bazel_env`.

          target_triple: "x86_64-pc-linux"

  ## AOT export with Mix

  Ahead-of-time exports with Mix are useful because you only need
  EXLA when exporting. In practice, you can do this:

    1. Add `{:exla, ..., only: :export_aot}` as a dependency

    2. Define an exporting script at `script/export_my_module.exs`

    3. Run the script to export the AOT `mix run script/export_my_module.exs`

    4. Now inside `lib/my_module.ex` you can import it:

          if File.exists?("priv/MyModule.nx.aot") do
            defmodule MyModule do
              Nx.Defn.import_aot("priv", __MODULE__)
            end
          else
            IO.warn "Skipping MyModule because aot definition was not found"
          end

  """
  def export_aot(dir, module, functions, options \\ []) do
    Nx.Defn.export_aot(dir, module, functions, Keyword.put(options, :compiler, EXLA))
  end

  @impl true
  defdelegate __jit__(key, vars, fun, opts), to: EXLA.Defn

  @impl true
  defdelegate __aot__(output_dir, module, tuples, opts), to: EXLA.Defn
end

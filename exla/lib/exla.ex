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
  type and shape of the given tensor.

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

    * `:device_id` - the default device id to run the computation
        on. Defaults to the `:default_device_id` on the client

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

  ### Client options

  Each client configuration accepts the following options:

    * `:platform` - the platform the client runs on. It can be
      `:host` (CPU), `:cuda`, `:rocm`, or `:tpu`

    * `:default_device_id` - the default device ID to run on.
      For example, if you have two GPUs, you can choose one as
      the default

    * `:preallocate`- if the memory should be preallocated on
      GPU devices. Defaults to `true`

    * `:memory_fraction` - how much memory of a GPU device to
      allocate. Defaults to `0.9.`

  ### GPU Runtime Issues

  GPU Executions run in dirty IO threads, which have a considerable smaller
  stack size than regular scheduler threads. This may lead to problems with
  certain CUDA or cuDNN versions, leading to segmentation fails. In a development
  environment, it is suggested to set:

      ELIXIR_ERL_OPTIONS="+sssdio 128"

  To increase the stack size of dirty IO threads from 40 kilowords to
  128 kilowords. In a release, you can set this flag in your `vm.args`.

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

  > **Important!** EXLA operations and the `defn` compiler do not
  take the input devices into account when executing. So, if you
  transfer a tensor to the GPU, by explicitly passing the client
  to be CUDA, but then your default client runs on the CPU, the
  tensors will be transferred back to CPU before execution. That's
  why it is important to configure the `:default` client with your
  desired specifications.

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

  @impl true
  defdelegate __jit__(key, vars, fun, opts), to: EXLA.Defn

  @impl true
  def __stream__(_, _, _, _, _, _), do: raise("not implemented")
end

defmodule EXLA do
  @moduledoc """
  Bindings and Nx integration for [Google's XLA](https://www.tensorflow.org/xla/).

  EXLA works both as a backend for Nx tensors and an optimized
  `Nx.Defn` compiler. To enable both globally, add a `config/config.exs`
  (or `config/ENV.exs`) with the following:

      import Config
      config :nx, :default_backend, EXLA.Backend
      config :nx, :default_defn_options, [compiler: EXLA]

  Then, every time you call a numerical definition, EXLA will just-in-time
  (JIT) compile a native implementation of the function above, tailored for
  the type and shape of the given tensor.

  EXLA is able to compile to the CPU/GPU/TPU, by specifying another client:

      import Config
      config :nx, :default_backend, {EXLA.Backend, client: :cuda}
      config :nx, :default_defn_options, [compiler: EXLA, client: :cuda]

  To use the GPU or TPUs, don't forget to also set the appropriate value
  for the [`XLA_TARGET`](https://github.com/elixir-nx/xla#xla_target)
  environment variable.

  Read the "Client" section below for more information.

  ### Options

  The options accepted by the EXLA compiler are:

    * `:client` - an atom representing the client to use. Defaults
      to `:host`. See "Clients" section

    * `:device_id` - the default device id to run the computation
        on. Defaults to the `:default_device_id` on the client

  ## Clients

  The `EXLA` library uses a client for compiling and executing code.
  Those clients are typically bound to a platform, such as CPU or
  GPU.

  Those clients are singleton resources on Google's XLA library,
  therefore they are treated as a singleton resource on this library
  too. EXLA ships with the client configuration for each supported
  platform, which would be the equivalent to this:

      config :exla, :clients,
        host: [platform: :host],
        cuda: [platform: :cuda],
        rocm: [platform: :rocm],
        tpu: [platform: :tpu]

  In scripts and code notebooks, you can call
  `EXLA.set_as_nx_default/1`, which will traverse the list
  of clients and enable the `EXLA` compiler with the first client
  available as the default `defn` options:

      EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

  To use the GPU or TPUs, don't forget to also set the appropriate value
  for the [`XLA_TARGET`](https://github.com/elixir-nx/xla#xla_target)
  environment variable.

  > **Important!** you should avoid using multiple clients for the
  > same platform. If you have multiple clients per platform, they
  > can race each other and fight for resources, such as memory.
  > Therefore, we recommend developers to stick with the default
  > clients above.

  ### Client options

  Each client configuration accepts the following options:

    * `:platform` - the platform the client runs on. It can be
      `:host` (CPU), `:cuda`, `:rocm`, or `:tpu`.

    * `:default_device_id` - the default device ID to run on.
      For example, if you have two GPUs, you can choose a different
      one as the default. Defaults to device 0 (the first device).

    * `:preallocate`- if the memory should be preallocated on
      GPU devices. Defaults to `true`.

    * `:memory_fraction` - how much memory of a GPU device to
      allocate. Defaults to `0.9`.

  ### GPU Runtime Issues

  GPU Executions run in dirty IO threads, which have a considerable smaller
  stack size than regular scheduler threads. This may lead to problems with
  certain CUDA or cuDNN versions, leading to segmentation fails. In a development
  environment, it is suggested to set:

      ELIXIR_ERL_OPTIONS="+sssdio 128"

  To increase the stack size of dirty IO threads from 40 kilowords to
  128 kilowords. In a release, you can set this flag in your `vm.args`.

  ## Device allocation

  EXLA also ships with a `EXLA.Backend` that allows data to be explicitly
  allocated on the EXLA device. You can create tensors with `EXLA.Backend`
  directly:

      Nx.tensor([1, 2, 3, 4], backend: EXLA.Backend)

  or you can configure `EXLA.Backend` as the default backend, so that
  all tensors are allocated on the EXLA device by default.

  In some cases you may want to explicitly move an existing tensor to
  the device:

      tensor = Nx.tensor([1, 2, 3, 4], backend: Nx.BinaryBackend)
      Nx.backend_transfer(tensor, EXLA.Backend)

  Note that you can use regular `Nx` operations, so the following works:

      tensor = Nx.tensor([1, 2, 3, 4], backend: EXLA.Backend)
      Nx.sum(tensor)

  Under the hood, EXLA will create a computation for the sum operation
  and invoke it on the device. This is essentially an "eager mode"
  that provides acceleration during prototyping. However, eventually
  you should wrap your computations in a `defn` to utilize the full
  performance of JIT.

  To bring the tensor data back to Elixir you need an explicit transfer:

      Nx.tensor([1, 2, 3, 4])
      |> softmax()
      |> Nx.backend_transfer()

  If instead you want to make a copy of the data, you can use
  `Nx.backend_copy/1` instead. However, when working with large
  data, be mindful of memory allocations.

  `EXLA.Backend` will use the same client as the one configured for
  `Nx.Defn` by default.

  > **Important!** EXLA operations and the `defn` compiler do not
  > take the input devices into account when executing. So, if you
  > transfer a tensor to the GPU, by explicitly passing the client
  > to be CUDA, but then your default client runs on the CPU, the
  > tensors will be transferred back to CPU before execution.

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
  Sets the global defn options to the EXLA compiler with the preferred
  client based on their availability.

  This function is typically invoked at the top of scripts and code
  notebooks which might be potentially executed from multiple platforms.
  Do not invoke this function during runtime, as it changes `Nx.Defn`
  options globally. If you have a specific client that you want to use
  throughout your project, use configuration files instead:

      config :nx, :default_defn_options, [compiler: EXLA, client: :cuda]

  ## Examples

      EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

  The above will try to find the first client available and set
  the `EXLA` compiler with the client as the compilers for `Nx.Defn`.
  If no client is found, `EXLA` is not set as compiler at all,
  therefore it is common to add `:host` as the last option.

  If additional options are given, they are given as compiler options:

      EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

  To use the GPU or TPUs, don't forget to also set the appropriate value
  for the [`XLA_TARGET`](https://github.com/elixir-nx/xla#xla_target)
  environment variable.
  """
  def set_as_nx_default(clients, opts \\ []) do
    supported_platforms = EXLA.Client.get_supported_platforms()
    all_clients = Application.fetch_env!(:exla, :clients)

    chosen =
      Enum.find(clients, fn client ->
        client_config = all_clients[client]
        client_platform = client_config[:platform] || :host
        client_config && Map.has_key?(supported_platforms, client_platform)
      end)

    if chosen do
      opts = Keyword.put(opts, :client, chosen)
      Nx.default_backend({EXLA.Backend, opts})
      Nx.Defn.global_default_options([compiler: EXLA] ++ opts)
      chosen
    end
  end

  @doc false
  @deprecated "Use set_as_nx_default/2 instead"
  def set_preferred_defn_options(clients, opts \\ []) do
    set_as_nx_default(clients, opts)
  end

  @doc """
  A shortcut for `Nx.Defn.jit/3` with the EXLA compiler.

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
  Starts streaming the given anonymous function with just-in-time
  compilation.

  At least two arguments are expected:

    1. The first argument is a tensor template of the data to
       be streamed in

    2. The second argument is a tensor with the stream initial state

  The streaming function must return a two element tuple, the
  first element is the data to be sent and the second is the
  accumulator.

  For each streamed chunk, you must call `Nx.Stream.send/2` and
  `Nx.Stream.recv/1`. You don't need to call `recv` immediately
  after `send`, but doing so can be a useful mechanism to provide
  backpressure. Once all chunks are sent, you must use `Nx.Stream.done/1`
  to receive the accumulated result. Let's see an example:

      defmodule Streamed do
        import Nx.Defn

        defn sum(tensor, acc) do
          {acc, tensor + acc}
        end
      end

  Now let's invoke it:

      stream = EXLA.stream(&Streamed.sum/2, [Nx.template({}, {:s, 64}), 0])

      for i <- 1..5 do
        Nx.Stream.send(stream, i)
        IO.inspect {:chunk, Nx.Stream.recv(stream)}
      end

      IO.inspect {:result, Nx.Stream.done(stream)}

  It will print:

      {:chunk, 0}
      {:chunk, 1}
      {:chunk, 2}
      {:chunk, 3}
      {:chunk, 4}
      {:result, 5}

  **Note:** While any process can call `Nx.Stream.send/2`, EXLA
  expects the process that starts the streaming to be the one
  calling `Nx.Stream.recv/1` and `Nx.Stream.done/1`.
  """
  def stream(function, args, options \\ []) do
    Nx.Defn.stream(function, args, Keyword.put(options, :compiler, EXLA))
  end

  @doc """
  Checks if the JIT compilation of function with
  args is cached.

  Note that hooks are part of the cache, and
  therefore they must be included in the options.

  ## Examples

      iex> fun = fn a, b -> Nx.add(a, b) end
      iex> left = Nx.tensor(1, type: {:u, 8})
      iex> right = Nx.tensor([1, 2, 3], type: {:u, 16})
      iex> EXLA.jit(fun, [left, right])
      iex> EXLA.jit_cached?(fun, [left, right])
      true
      iex> EXLA.jit_cached?(fun, [left, Nx.tensor([1, 2, 3, 4], type: {:u, 16})])
      false

  """
  def jit_cached?(function, args, options \\ []) do
    jit(function, args, [{EXLA, cached_check()} | options])
  catch
    {:cached?, bool} -> bool
  end

  @doc """
  Checks if the JIT compilation of stream with
  args is cached.

  Note that hooks are part of the cache, and
  therefore they must be included in the options.

  ## Examples

      iex> left = Nx.tensor(1, type: {:u, 8})
      iex> right = Nx.tensor([1, 2, 3], type: {:u, 16})
      iex> fun = fn x, acc -> {acc, Nx.add(x, acc)} end
      iex> stream = EXLA.stream(fun, [left, right])
      iex> Nx.Stream.done(stream)
      iex> EXLA.stream_cached?(fun, [left, right])
      true
      iex> EXLA.stream_cached?(fun, [left, Nx.tensor([1, 2, 3, 4], type: {:u, 16})])
      false
  """
  def stream_cached?(function, args, options \\ []) do
    stream(function, args, [{EXLA, cached_check()} | options])
  catch
    {:cached?, bool} -> bool
  end

  defp cached_check do
    expr_cache_fun = fn key, _callback ->
      case EXLA.Defn.LockedCache.fetch(key) do
        {:ok, res} -> {nil, res}
        :error -> throw({:cached?, false})
      end
    end

    comp_cache_fun = fn key, _callback ->
      throw({:cached?, EXLA.Defn.LockedCache.fetch(key) != :error})
    end

    {expr_cache_fun, comp_cache_fun}
  end

  @impl true
  defdelegate __jit__(key, vars, fun, args, opts), to: EXLA.Defn

  @impl true
  defdelegate __stream__(key, input, acc, vars, fun, args, opts), to: EXLA.Defn
end

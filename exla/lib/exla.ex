defmodule EXLA do
  @moduledoc """
  [Google's XLA](https://www.tensorflow.org/xla/) (Accelerated Linear Algebra) compiler/backend for Nx.

  It supports just-in-time (JIT) compilation to GPU (both CUDA and ROCm) and TPUs.

  ## XLA binaries

  EXLA relies on the [XLA](https://github.com/elixir-nx/xla) package to
  provide the necessary XLA binaries. Whenever possible it tries to download
  precompiled builds, but you may need to build from source if there is no
  version matching your target environment. For more details, including
  GPU/TPU support see [the usage section](https://github.com/elixir-nx/xla#usage).

  ## Configuration

  EXLA ships with a backend to store tensors and run computations on.
  Generally speaking, the backend is enabled globally in your `config/config.exs`
  (or `config/ENV.exs`) with the following:

      import Config
      config :nx, :default_backend, EXLA.Backend

  In a script/notebook, you would do:

      Mix.install([
        {:exla, "~> 0.2"}
      ])

      Nx.global_default_backend(EXLA.Backend)

  From now on, all created tensors will be allocated directly on the given
  `EXLA.Backend`. You can use functions such as `Nx.backend_transfer/2` to
  explicitly transfer tensors.

  EXLA will pick an available client to allocate and compute tensors, in this
  order: `:cuda`, `:rocm`, `:tpu`, and `:host` (CPU). See the "Clients" section
  below for more information.

  To use GPUs/TPUs, you must also set the appropriate value for the
  [`XLA_TARGET`](https://github.com/elixir-nx/xla#xla_target) environment
  variable. If you have GPU/TPU enabled, we recommend setting the environment
  variable for your machine altogether. For CUDA, setting
  `ELIXIR_ERL_OPTIONS="+sssdio 128"` is also required on more complex operations
  to increase CUDA's compiler stack size.

  Note that setting the `EXLA.Backend` does not enable the EXLA compiler.
  You must still pass the `compiler: EXLA` option to `Nx.Defn` functions
  or call the functions in this module.

  ### Options

  The options accepted by EXLA backend/compiler are:

    * `:client` - an atom representing the client to use. The default
      client is chosen on this order: `:cuda`, `:rocm`, `:tpu`, and `:host`.

    * `:device_id` - the default device id to run the computation
        on. Defaults to the `:default_device_id` on the client

    * `:precision` - control the tradeoff between speed and accuracy for
      array computations on accelerator backends (i.e. TPU and GPU).
      It must be one of:

      * `:default` - Fastest mode, but least accurate. Performs computations
        in bfloat16

      * `:high` - Slower but more accurate. Performs float32 computations in
        3 bfloat16 passes, or using tensorfloat32 where available

      * `:highest` - Slowest but most accurate. Performs computations in float32
        or float64 as applicable

  ## Clients

  The `EXLA` library uses a client for compiling and executing code.
  Those clients are typically bound to a platform, such as CPU or
  GPU.

  Those clients are singleton resources on Google's XLA library,
  therefore they are treated as a singleton resource on this library
  too. EXLA ships with runtime client configuration for each supported
  platform:

      config :exla, :clients,
        cuda: [platform: :cuda],
        rocm: [platform: :rocm],
        tpu: [platform: :tpu],
        host: [platform: :host]

  In a script/notebook, you can set those after `Mix.install/2`,
  but before any tensor operation is performed:

      Application.put_env(:exla, :clients,
        cuda: [platform: :cuda],
        rocm: [platform: :rocm],
        tpu: [platform: :tpu],
        host: [platform: :host]
      )

  You can provide your own list of clients, replacing the list above
  or configuring each client as listed below. You can also specify
  `:default_client` to set a particular client by default or
  `:preferred_clients` to change the order of clients preference,
  but those configurations are rarely set in practice.

  > **Important!** you should avoid using multiple clients for the
  > same platform. If you have multiple clients per platform, they
  > can race each other and fight for resources, such as memory.
  > Therefore, we recommend developers to stick with the default
  > clients above.

  ### Client options

  Each client configuration accepts the following options:

    * `:platform` - the platform the client runs on. It can be
      `:host` (CPU), `:cuda`, `:rocm`, or `:tpu`. Defaults to `:host`.

    * `:default_device_id` - the default device ID to run on.
      For example, if you have two GPUs, you can choose a different
      one as the default. Defaults to device 0 (the first device).

    * `:preallocate`- if the memory should be preallocated on
      GPU devices. Defaults to `true`.

    * `:memory_fraction` - how much memory of a GPU device to
      allocate. Defaults to `0.9`.

  ### Memory preallocation

  XLA preallocates memory in GPU devices. This means that, if you are to
  run multiple notebooks or multiple instances of your application, the
  second, third, and so on instances won't be able to allocate memory.

  You can disable this behaviour by setting `preallocate: false` on the
  client configuration, as specified above. You may also use
  `:memory_fraction` to control how much is preallocated.

  ### GPU Runtime Issues

  GPU Executions run in dirty IO threads, which have a considerable smaller
  stack size than regular scheduler threads. This may lead to problems with
  certain CUDA or cuDNN versions, leading to segmentation fails. In a development
  environment, it is suggested to set:

      ELIXIR_ERL_OPTIONS="+sssdio 128"

  To increase the stack size of dirty IO threads from 40 kilowords to
  128 kilowords. In a release, you can set this flag in your `vm.args`.

  ## Docker considerations

  EXLA should run fine on Docker with one important consideration:
  you must not start the Erlang VM as the root process in Docker.
  That's because when the Erlang VM runs as root, it has to manage
  all child programs.

  At the same time, Google XLA's shells out to child programs and
  must retain control over how child programs terminate.

  To address this, simply make sure you wrap the Erlang VM in
  another process, such as the shell one. In other words, if you
  are using releases, instead of this:

      CMD path/to/release start

  do this:

      CMD sh -c "path/to/release start"

  If you are using Mix inside your Docker containers, instead of this:

      CMD mix run

  do this:

      CMD sh -c "mix run"

  Alternatively, you can pass the `--init` flag to `docker run`,
  so it runs an `init` inside the container that forwards signals
  and reaps processes.

  The `--init` flag uses the [`tini`](https://github.com/krallin/tini)
  project, so for cases where the flag may not available (e.g.
  kubernetes) you may want to install it.

  ## Telemetry events

  EXLA executes a telemetry event every time a function is JIT-compiled.
  The events are named `[:exla, :compilation]` and include the following
  measurements, given in microseconds:

    * `:eval_time` - the time spent on turning the function into XLA
      computation
    * `:compile_time` - the time spent on compiling the XLA computation
      into an executable
    * `:total_time` - the sum of `:eval_time` and `:compile_time`

  The metadata is:

    * `:key` - the compilation key for debugging
  """

  @behaviour Nx.Defn.Compiler

  @doc """
  A shortcut for `Nx.Defn.jit/2` with the EXLA compiler.

      iex> EXLA.jit(&Nx.add(&1, &1)).(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        s64[3]
        [2, 4, 6]
      >

  Results are allocated on the `EXLA.Backend`. Note that the
  `EXLA.Backend` is asynchronous: operations on its tensors
  *may* return immediately, before the tensor data is available.
  The backend will then block only when trying to read the data
  or when passing it to another operation.

  ## Options

  It accepts the same option as `Nx.Defn.jit/2` plus:

    * `:cache` - cache the results of compilation, defaults to `true`.

    * `:client` - an atom representing the client to use. The default
      client is chosen on this order: `:cuda`, `:rocm`, `:tpu`, and `:host`.

    * `:debug` - print compile and debugging information, defaults to `false`.

    * `:device_id` - the default device id to run the computation on.
      Defaults to the `:default_device_id` on the client

    * `:lazy_transfers` - when `:always`, it lazily transfers data to the device
      instead of upfront. This is useful to reduce memory allocation on GPU/TPU
      devices at the cost of increased latency. **It is recommended to only enable
      this if the input tensors are allocated on host and the computation is
      running on GPU/TPU with a limited amount of memory**

  """
  def jit(function, options \\ []) do
    Nx.Defn.jit(function, Keyword.put(options, :compiler, EXLA))
  end

  @doc """
  A shortcut for `Nx.Defn.jit_apply/3` with the EXLA compiler.

      iex> EXLA.jit_apply(&Nx.add(&1, &1), [Nx.tensor([1, 2, 3])])
      #Nx.Tensor<
        s64[3]
        [2, 4, 6]
      >

  See `jit/2` for supported options.
  """
  def jit_apply(function, args, options \\ []) do
    Nx.Defn.jit_apply(function, args, Keyword.put(options, :compiler, EXLA))
  end

  @doc """
  A shortcut for `Nx.Defn.compile/3` with the EXLA compiler.

      iex> fun = EXLA.compile(&Nx.add(&1, &1), [Nx.template({3}, {:s, 64})])
      iex> fun.(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        s64[3]
        [2, 4, 6]
      >

  Results are allocated on the `EXLA.Backend`. Note that the
  `EXLA.Backend` is asynchronous: operations on its tensors
  *may* return immediately, before the tensor data is available.
  The backend will then block only when trying to read the data
  or when passing it to another operation.

  ## Options

  It accepts the same option as `Nx.Defn.compile/3` plus:

    * `:debug` - print compile and debugging information, defaults to `false`.

    * `:cache` - cache the results of compilation, defaults to `true`.
      You can set it to false if you plan to compile the function only
      once and store the compile contents somewhere.

    * `:client` - an atom representing the client to use. The default
      client is chosen on this order: `:cuda`, `:rocm`, `:tpu`, and `:host`.

    * `:device_id` - the default device id to run the computation on.
      Defaults to the `:default_device_id` on the client

  """
  def compile(function, args, options \\ []) do
    Nx.Defn.compile(function, args, Keyword.put(options, :compiler, EXLA))
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

  See `jit/2` for supported options.
  """
  def stream(function, args, options \\ []) do
    Nx.Defn.stream(function, args, Keyword.put(options, :compiler, EXLA))
  end

  @doc """
  Checks if the compilation of function with args is cached.

  Note that hooks are part of the cache, and
  therefore they must be included in the options.

  ## Examples

      iex> fun = fn a, b -> Nx.add(a, b) end
      iex> left = Nx.tensor(1, type: {:u, 8})
      iex> right = Nx.tensor([1, 2, 3], type: {:u, 16})
      iex> EXLA.jit(fun).(left, right)
      iex> EXLA.cached?(fun, [left, right])
      true
      iex> EXLA.cached?(fun, [left, Nx.tensor([1, 2, 3, 4], type: {:u, 16})])
      false

  Compiled functions are also cached, unless cache is set to false:

      iex> fun = fn a, b -> Nx.subtract(a, b) end
      iex> left = Nx.tensor(1, type: {:u, 8})
      iex> right = Nx.tensor([1, 2, 3], type: {:u, 16})
      iex> EXLA.compile(fun, [left, right], cache: false)
      iex> EXLA.cached?(fun, [left, right])
      false
      iex> EXLA.compile(fun, [left, right])
      iex> EXLA.cached?(fun, [left, right])
      true

  """
  def cached?(function, args, options \\ []) do
    function |> jit([{EXLA, cached_check()} | options]) |> apply(args)
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
      if res = EXLA.Defn.LockedCache.get(key) do
        {nil, res}
      else
        throw({:cached?, false})
      end
    end

    comp_cache_fun = fn key, _callback ->
      throw({:cached?, EXLA.Defn.LockedCache.get(key) != nil})
    end

    {expr_cache_fun, comp_cache_fun}
  end

  @impl true
  defdelegate __compile__(key, vars, fun, opts), to: EXLA.Defn

  @impl true
  defdelegate __jit__(key, vars, fun, args, opts), to: EXLA.Defn

  @impl true
  defdelegate __stream__(key, input, acc, vars, fun, args, opts), to: EXLA.Defn

  @impl true
  defdelegate __partitions_options__(opts), to: EXLA.Defn
end

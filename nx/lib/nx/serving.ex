defmodule Nx.Serving do
  @moduledoc """
  Serving encapsulates client and server work to perform batched requests.

  Servings can be executed on the fly, without starting a server, but most
  often they are used to run servers that batch requests until a given size
  or timeout is reached.

  More specifically, servings are a mechanism to apply a computation on a
  `Nx.Batch`, with hooks for preprocessing input from and postprocessing
  output for the client. Thus we can think of an instance of `Nx.Serving.t()`
  (a serving) as something that encapsulates batches of Nx computations.

  ## Inline/serverless workflow

  First, let's define a simple numerical definition function:

      defmodule MyDefn do
        import Nx.Defn

        defnp print_and_multiply(x) do
          print_value({:debug, x})
          x * 2
        end
      end

  The function prints the given tensor and doubles its contents.
  We can use `new/1` to create a serving that will return a JIT
  or AOT compiled function to execute on batches of tensors:

      iex> serving = Nx.Serving.new(fn opts -> Nx.Defn.jit(&print_and_multiply/1, opts) end)
      iex> batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      iex> Nx.Serving.run(serving, batch)
      {:debug, #Nx.Tensor<
        s64[1][3]
        [
          [1, 2, 3]
        ]
      >}
      #Nx.Tensor<
        s64[1][3]
        [
          [2, 4, 6]
        ]
      >

  We started the serving by passing a function that receives
  compiler options and returns a JIT or AOT compiled function.
  We called `Nx.Defn.jit/2` passing the options received as
  argument, which will customize the JIT/AOT compilation.

  You should see two values printed. The former is the result of
  `Nx.Defn.Kernel.print_value/1`, which shows the tensor that was
  actually part of the computation and how it was batched.
  The latter is the result of the computation.

  When defining a `Nx.Serving`, we can also customize how the data is
  batched by using the `client_preprocessing` as well as the result by
  using `client_postprocessing` hooks. Let's give it another try:

      iex> serving = (
      ...>   Nx.Serving.new(fn opts -> Nx.Defn.jit(&print_and_multiply/1, opts) end)
      ...>   |> Nx.Serving.client_preprocessing(fn input -> {Nx.Batch.stack(input), :client_info} end)
      ...>   |> Nx.Serving.client_postprocessing(&{&1, &2, &3})
      ...> )
      iex> Nx.Serving.run(serving, [Nx.tensor([1, 2]), Nx.tensor([3, 4])])
      {:debug, #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >}
      {#Nx.Tensor<
         s64[2][2]
         [
           [2, 4],
           [6, 8]
         ]
       >,
       :server_info,
       :client_info}

  You can see the results are a bit different now. First of all, notice that
  we were able to run the serving passing a list of tensors. Our custom
  `client_preprocessing` function stacks those tensors into a batch of two
  entries and returns a tuple with a `Nx.Batch` struct and additional client
  information which we represent as the atom `:client_info`. The default
  client preprocessing simply enforces a batch was given and returns no client
  information.

  Then the result is a triplet tuple, returned by the client
  postprocessing function, containing the result, the server information
  (which we will later learn how to customize), and the client information.
  From this, we can infer the default implementation of `client_postprocessing`
  simply returns the result, discarding the server and client information.

  So far, `Nx.Serving` has not given us much. It has simply encapsulated the
  execution of a function. Its full power comes when we start running our own
  `Nx.Serving` process. That's when we will also learn why we have a `client_`
  prefix in some of the function names.

  ## Stateful/process workflow

  `Nx.Serving` allows us to define an Elixir process to handle requests.
  This process provides several features, such as batching up to a given
  size or time, partitioning, and distribution over a group of nodes.

  To do so, we need to start a `Nx.Serving` process with a serving inside
  a supervision tree:

      children = [
        {Nx.Serving,
         serving: Nx.Serving.new(Nx.Defn.jit(&print_and_multiply/1)),
         name: MyServing,
         batch_size: 10,
         batch_timeout: 100}
      ]

      Supervisor.start_child(children, strategy: :one_for_one)

  Now you can send batched runs to said process:

      iex> batch = Nx.Batch.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
      iex> Nx.Serving.batched_run(MyServing, batch)
      {:debug, #Nx.Tensor<
        s64[2][3]
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
      >}
      #Nx.Tensor<
        s64[2][3]
        [
          [2, 4, 6],
          [8, 10, 12]
        ]
      >

  In the example, we pushed a batch of 2 and eventually got a reply.
  The process will wait for requests from other processes, for up to
  100 milliseconds or until it gets 10 entries. Then it merges all
  batches together and once the result is computed, it slices and
  distributes those responses to each caller.

  If there is any `client_preprocessing` function, it will be executed
  before the batch is sent to the server. If there is any `client_postprocessing`
  function, it will be executed after getting the response from the
  server.

  ### Partitioning

  You can start several partitions under th same serving by passing
  `partitions: true` when starting the serving. The number of partitions
  will be determined according  your compiler and for which host it is
  compiling.

  For example, when creating the serving, you may pass the following
  `defn_options`:

      Nx.Serving.new(computation, compiler: EXLA, client: :cuda)

  Now when booting up the serving:

      children = [
        {Nx.Serving,
         serving: serving,
         name: MyServing,
         batch_size: 10,
         batch_timeout: 100,
         partitions: true}
      ]

  If you have two GPUs, `batched_run/3` will now gather batches and send
  them to the GPUs as they become available to process requests.

  ### Distribution

  All `Nx.Serving`s are distributed by default. If the current machine
  does not have an instance of `Nx.Serving` running, `batched_run/3` will
  automatically look for one in the cluster. The nodes do not need to run
  the same code and applications. It is only required that they run the
  same `Nx` version.

  The load balancing between servings is done randomly, however, the number
  of partitions are considered if the `partitioned: true` option is also given.
  For example, if you have a node with 2 GPUs and another with 4, the latter
  will receive the double of requests compared to the former.

  `batched_run/3` receives an optional `distributed_preprocessing` callback as
  third argument for preprocessing the input for distributed requests. When
  using libraries like EXLA or Torchx, the tensor is often allocated in memory
  inside a third-party library so it is necessary to either transfer or copy
  the tensor to the binary backend before sending it to another node.
  This can be done by passing either `Nx.backend_transfer/1` or `Nx.backend_copy/1`
  as third argument:

      Nx.Serving.batched_run(MyDistributedServing, input, &Nx.backend_copy/1)

  Use `backend_transfer/1` if you know the input will no longer be used.

  Similarly, the serving has a `distributed_postprocessing` callback which can do
  equivalent before sending the reply to the caller.

  The servings are dispatched using Erlang Distribution. You can use
  `Node.connect/1` to manually connect nodes. In a production setup, this is
  often done with the help of libraries like [`libcluster`](https://github.com/bitwalker/libcluster).

  ## Advanced notes

  ### Module-based serving

  In the examples so far, we have been using the default version of
  `Nx.Serving`, which executes the given function for each batch.

  However, we can also use `new/2` to start a module-based version of
  `Nx.Serving` which gives us more control over both inline and process
  workflows. A simple module implementation of a `Nx.Serving` could look
  like this:

      defmodule MyServing do
        @behaviour Nx.Serving

        defnp print_and_multiply(x) do
          print_value({:debug, x})
          x * 2
        end

        @impl true
        def init(_inline_or_process, :unused_arg, [defn_options]) do
          {:ok, Nx.Defn.jit(&print_and_multiply/1, defn_options)}
        end

        @impl true
        def handle_batch(batch, 0, function) do
          {:execute, fn -> {function.(batch), :server_info} end, function}
        end
      end

  It has two functions. The first, `c:init/3`, receives the type of serving
  (`:inline` or `:process`) and the serving argument. In this step,
  we capture `print_and_multiply/1`as a jitted function.

  The second function is called `handle_batch/3`. This function
  receives a `Nx.Batch` and returns a function to execute.
  The function itself must return a two element-tuple: the batched
  results and some server information. The server information can
  be any value and we set it to the atom `:server_info`.

  Now let's give it a try by defining a serving with our module and
  then running it on a batch:

      iex> serving = Nx.Serving.new(MyServing, :unused_arg)
      iex> batch = Nx.Batch.stack([Nx.tensor([1, 2, 3])])
      iex> Nx.Serving.run(serving, batch)
      {:debug, #Nx.Tensor<
        s64[1][3]
        [
          [1, 2, 3]
        ]
      >}
      #Nx.Tensor<
        s64[1][3]
        [
          [2, 4, 6]
        ]
      >

  From here on, you use `start_link/1` to start this serving in your
  supervision and even customize `client_preprocessing/1` and
  `client_postprocessing/1` callbacks to this serving, as seen in the
  previous sections.

  Note in our implementation above assumes it won't run partitioned.
  In partitioned mode, `c:init/3` may receive multiple `defn_options`
  as the third argument and `handle_batch/3` may receive another partition
  besides 0.
  """

  @doc false
  @enforce_keys [:module, :arg]
  defstruct [
    :module,
    :arg,
    :client_preprocessing,
    :client_postprocessing,
    distributed_postprocessing: &Function.identity/1,
    process_options: [],
    defn_options: []
  ]

  @type metadata() :: term()
  @type client_info() :: term()
  @type client_preprocessing() :: (term() -> {Nx.Batch.t(), client_info()})
  @type client_postprocessing() :: (Nx.Container.t(), metadata(), client_info() -> term())
  @type distributed_preprocessing() :: (term() -> term())
  @type distributed_postprocessing() :: (term() -> term())

  @type t :: %__MODULE__{
          module: atom(),
          arg: term(),
          client_preprocessing: client_preprocessing(),
          client_postprocessing: client_postprocessing(),
          distributed_postprocessing: distributed_postprocessing(),
          process_options: keyword(),
          defn_options: keyword()
        }

  @axis 0

  @doc """
  The callback used to initialize the serving.

  The first argument reveals if the serving is executed inline,
  such as by calling `run/2`, by started with the process.
  The second argument is the serving argument given to `new/2`.
  The third argument option is a list of compiler options to be
  used to compile each partition the serving will run.

  It must return `{:ok, state}`, where the `state` can be any term.
  """
  @callback init(type :: :inline | :process, arg :: term(), [defn_options :: keyword]) ::
              {:ok, state :: term()}

  @doc """
  Receives a batch, a partition, and returns a function to execute the batch.

  In case of serving processes, the function is executed is an
  separate process.
  """
  @callback handle_batch(Nx.Batch.t(), partition :: non_neg_integer(), state) ::
              {:execute, (() -> {Nx.Container.t(), metadata()}), state}
            when state: term()

  @doc """
  Creates a new function serving.

  It expects a function that receives the compiler options and
  returns a JIT (via `Nx.Defn.jit/2`) or AOT compiled (via
  `Nx.Defn.compile/3`) one-arity function as argument.

  The function will be called with the arguments returned by the
  `client_preprocessing` callback.
  """
  def new(function, defn_options \\ [])

  def new(function, defn_options)
      when is_function(function, 1) and is_list(defn_options) do
    new(Nx.Serving.Default, function, defn_options)
  end

  def new(function, process_options)
      when is_function(function, 0) and is_list(process_options) do
    IO.warn(
      "passing a zero-arity function to Nx.Serving.new is deprecated, " <>
        "please pass a single arity function that will receive the compiler options"
    )

    new(Nx.Serving.Default, fn _ -> function.() end, [])
    |> process_options(process_options)
  end

  def new(module, arg) when is_atom(module) do
    new(module, arg, [])
  end

  @doc """
  Creates a new module-based serving.

  It expects a module and an argument that is given to its `init`
  callback.

  A third optional argument called `defn_options` are additional
  compiler options which will be given to the module. Those options
  will be merged into `Nx.Defn.default_options/0`.
  """
  def new(module, arg, defn_options) when is_atom(module) and is_list(defn_options) do
    defn_options = Keyword.merge(Nx.Defn.default_options(), defn_options)
    %Nx.Serving{module: module, arg: arg, defn_options: defn_options}
  end

  @doc """
  Sets the client preprocessing function.

  The default implementation creates a single element batch
  with the given argument and is equivalent to `&Nx.Batch.stack([&1])`.
  """
  def client_preprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 1) or is_nil(function) do
    %{serving | client_preprocessing: function}
  end

  @doc """
  Sets the client postprocessing function.

  The default implementation returns the first element given
  to the function.
  """
  def client_postprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 3) or is_nil(function) do
    %{serving | client_postprocessing: function}
  end

  @doc """
  Sets the distributed postprocessing function.

  The default implementation is `Function.identity/1`.
  """
  def distributed_postprocessing(%Nx.Serving{} = serving, function)
      when is_function(function, 1) do
    %{serving | distributed_postprocessing: function}
  end

  @doc """
  Sets the process options of this serving.

  These are the same options as supported on `start_link/1`,
  except `:name` and `:serving` itself.
  """
  def process_options(%Nx.Serving{} = serving, process_options) when is_list(process_options) do
    %{serving | process_options: process_options}
  end

  @doc """
  Sets the defn options of this serving.

  These are the options supported by `Nx.Defn.default_options/1`.
  """
  def defn_options(%Nx.Serving{} = serving, defn_options) when is_list(defn_options) do
    %{serving | defn_options: defn_options}
  end

  @doc """
  Runs `serving` with the given `input` inline with the current process.
  """
  def run(%Nx.Serving{} = serving, input) do
    %{
      module: module,
      arg: arg,
      client_preprocessing: preprocessing,
      client_postprocessing: postprocessing,
      defn_options: defn_options
    } = serving

    {:ok, state} = handle_init(module, :inline, arg, [defn_options])
    {%{size: size} = batch, info} = handle_preprocessing(preprocessing, input)
    {:execute, function, _} = handle_batch(module, batch, 0, state)

    {output, metadata} =
      :telemetry.span([:nx, :serving, :execute], %{module: module}, fn ->
        {output, metadata} = handle_executed(module, function.())

        {{output, metadata}, %{metadata: metadata, module: module}}
      end)

    output = Nx.Defn.Composite.traverse(output, &Nx.slice_along_axis(&1, 0, size, axis: @axis))
    handle_postprocessing(postprocessing, output, metadata, info)
  end

  ## Process API

  @doc false
  def child_spec(opts) when is_list(opts) do
    name = opts[:name]

    if name == nil or not is_atom(name) do
      raise ArgumentError, ":name option is expected when starting Nx.Serving and must be an atom"
    end

    opts[:serving] ||
      raise ArgumentError, ":serving option is expected when starting Nx.Serving"

    %{
      id: name,
      start: {__MODULE__, :start_link, [opts]},
      type: :supervisor
    }
  end

  @doc """
  Starts a `Nx.Serving` process to batch requests to a given serving.

  ## Options

    * `:name` - an atom with the name of the process

    * `:serving` - a `Nx.Serving` struct with the serving configuration

    * `:batch_size` - the maximum batch size. A value is first read
      from the `Nx.Serving` struct and then it falls back to this option
      (which defaults to `1`)

    * `:batch_timeout` - the maximum time to wait, in milliseconds,
      before executing the batch. A value is first read from the `Nx.Serving`
      struct and then it falls back to this option (which defaults to `100`ms)

    * `:partitions` - when `true`, starts several partitions under this serving.
      The number of partitions will be determined according to your compiler
      and for which host it is compiling. See the module docs for more information

    * `:shutdown` - the maximum time for the serving to shutdown. This will
      block until the existing computation finishes (defaults to `30_000`ms)

    * `:hibernate_after` and `:spawn_opt` - configure the underlying serving
      workers (see `GenServer.start_link/3`)
  """
  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    {%Nx.Serving{process_options: process_options} = serving, opts} = Keyword.pop!(opts, :serving)

    opts =
      Keyword.merge(process_options, opts, fn
        k, v1, v2 when k == :batch_size and v1 != v2 ->
          raise ArgumentError,
                "#{inspect(k)} has been set when starting an Nx.Serving process (#{inspect(v2)}) " <>
                  "but a conflicting value was already set on the Nx.Serving struct (#{inspect(v1)}). " <>
                  "Please remove the option given to the Nx.Serving process"

        _k, _v1, v2 ->
          v2
      end)

    {shutdown, opts} = Keyword.pop(opts, :shutdown, 30_000)
    {partitions, opts} = Keyword.pop(opts, :partitions, false)
    {batch_size, opts} = Keyword.pop(opts, :batch_size, 1)
    {batch_timeout, opts} = Keyword.pop(opts, :batch_timeout, 100)

    supervisor = Module.concat(name, "Supervisor")
    task_supervisor = Module.concat(name, "TaskSupervisor")
    arg = {name, serving, partitions, batch_size, batch_timeout, task_supervisor}

    children = [
      {Task.Supervisor, name: task_supervisor},
      %{
        id: __MODULE__,
        start: {GenServer, :start_link, [__MODULE__, arg, opts]},
        shutdown: shutdown
      }
    ]

    Supervisor.start_link(children, strategy: :one_for_all, max_restarts: 0, name: supervisor)
  end

  @doc """
  Runs the given `input` on the serving process given by `name`.

  `name` is either an atom representing a local or distributed
  serving process. First it will attempt to dispatch locally, then it
  falls back to the distributed serving. You may specify
  `{:local, name}` to force a local lookup or `{:distributed, name}`
  to force a distributed one.

  The `client_preprocessing` callback will be invoked on the `input`
  which is then sent to the server. The server will batch requests
  and send a response either when the batch is full or on timeout.
  Then `client_postprocessing` is invoked on the response. See the
  module documentation for more information. In the distributed case,
  the callbacks are invoked in the distributed node, but still outside of
  the serving process.

  Note that you cannot batch an `input` larger than the configured
  `:batch_size` in the server.

  ## Distributed mode

  To run in distributed mode, the nodes do not need to run the same
  code and applications. It is only required that they run the
  same `Nx` version.

  If the current node is running a serving given by `name` locally
  and `{:distributed, name}` is used, the request will use the same
  distribution mechanisms instead of being handled locally, which
  is useful for testing locally without a need to spawn nodes.

  This function receives an optional `distributed_preprocessing` callback as
  third argument for preprocessing the input for distributed requests. When
  using libraries like EXLA or Torchx, the tensor is often allocated in memory
  inside a third-party library so it is necessary to either transfer or copy
  the tensor to the binary backend before sending it to another node.
  This can be done by passing either `Nx.backend_transfer/1` or `Nx.backend_copy/1`
  as third argument:

      Nx.Serving.batched_run(MyDistributedServing, input, &Nx.backend_copy/1)

  Use `backend_transfer/1` if you know the input will no longer be used.

  Similarly, the serving has a `distributed_postprocessing` callback which can do
  equivalent before sending the reply to the caller.
  """
  def batched_run(name, input, distributed_preprocessing \\ &Function.identity/1)

  def batched_run(name, input, distributed_preprocessing) when is_atom(name) do
    if pid = Process.whereis(name) do
      local_batched_run!(pid, name, input)
    else
      distributed_batched_run!(name, input, distributed_preprocessing)
    end
  end

  def batched_run({:local, name}, input, _distributed_preprocessing) when is_atom(name) do
    pid =
      Process.whereis(name) || exit({:noproc, {__MODULE__, :local_batched_run, [name, input]}})

    local_batched_run!(pid, name, input)
  end

  def batched_run({:distributed, name}, input, distributed_preprocessing) when is_atom(name) do
    distributed_batched_run!(name, input, distributed_preprocessing)
  end

  defp local_batched_run!(pid, name, input) do
    case local_batched_run(pid, name, input) do
      {:ok, result} -> result
      {:DOWN, reason} -> exit({reason, {__MODULE__, :local_batched_run, [name, input]}})
    end
  end

  defp local_batched_run(pid, name, input) do
    %{
      preprocessing: preprocessing,
      postprocessing: postprocessing,
      limit: limit
    } = :persistent_term.get(persistent_key(name))

    {batch, info} = handle_preprocessing(preprocessing, input)

    if batch.size > limit do
      raise ArgumentError,
            "batch size (#{batch.size}) cannot exceed Nx.Serving server batch size of #{limit}"
    end

    # Use Process.monitor/2 on Elixir v1.15+
    ref = :erlang.monitor(:process, pid, alias: :demonitor)
    Process.send(pid, {__MODULE__, :batched_run, ref, batch}, [:noconnect])

    case receive_batched(batch.size, ref, [], nil, name, input) do
      {:ok, tensor, metadata} ->
        {:ok, handle_postprocessing(postprocessing, tensor, metadata, info)}

      {:DOWN, reason} ->
        {:DOWN, reason}
    end
  end

  defp receive_batched(0, ref, acc, {template, metadata}, _name, _input) do
    Process.demonitor(ref, [:flush])

    tensors =
      acc
      |> Enum.reverse()
      |> Enum.zip_with(&Nx.concatenate(&1, axis: @axis))

    {output, []} =
      Nx.Defn.Composite.traverse(template, tensors, fn _template, [tensor | tensors] ->
        {tensor, tensors}
      end)

    {:ok, output, metadata}
  end

  defp receive_batched(total_size, ref, acc, _template_metadata, name, input) do
    receive do
      {^ref, {start, size, output, metadata}} ->
        # If we have a single response, slice and return immediately.
        # Otherwise we collect their contents and build the concatenated result later.
        if acc == [] and size == total_size do
          Process.demonitor(ref, [:flush])

          output =
            Nx.Defn.Composite.traverse(output, &Nx.slice_along_axis(&1, start, size, axis: @axis))

          {:ok, output, metadata}
        else
          funs =
            output
            |> Nx.Defn.Composite.reduce(
              [],
              &[Nx.slice_along_axis(&1, start, size, axis: @axis) | &2]
            )
            |> Enum.reverse()

          receive_batched(total_size - size, ref, [funs | acc], {output, metadata}, name, input)
        end

      {:DOWN, ^ref, _, _, reason} ->
        # We fake monitor messages, so still demonitor and flush.
        Process.demonitor(ref, [:flush])
        {:DOWN, reason}
    end
  end

  defp distributed_batched_run!(name, input, distributed_callback) do
    distributed_batched_run_with_retries!(name, distributed_callback.(input), 3)
  end

  defp distributed_batched_run_with_retries!(name, input, 0) do
    exit({:noproc, {__MODULE__, :distributed_batched_run, [name, input, [retries: 0]]}})
  end

  defp distributed_batched_run_with_retries!(name, input, retries) do
    case :pg.get_members(Nx.Serving.PG, __MODULE__) do
      [] ->
        exit({:noproc, {__MODULE__, :distributed_batched_run, [name, input, [retries: retries]]}})

      entries ->
        pid = Enum.random(entries)
        ref = make_ref()
        args = [ref, name, input]

        {_, monitor_ref} =
          Node.spawn_monitor(node(pid), __MODULE__, :__distributed_batched_run__, args)

        receive do
          {:DOWN, ^monitor_ref, _, _, {^ref, result}} ->
            result

          {:DOWN, _, _, _, :noproc} ->
            distributed_batched_run_with_retries!(name, input, retries - 1)

          {:DOWN, _, _, _, reason} ->
            exit(
              {reason, {__MODULE__, :distributed_batched_run, [name, input, [retries: retries]]}}
            )
        end
    end
  end

  @doc false
  def __distributed_batched_run__(ref, name, input) do
    pid = Process.whereis(name) || exit(:noproc)

    case local_batched_run(pid, name, input) do
      {:ok, result} ->
        result = :persistent_term.get(persistent_key(name)).distributed_postprocessing.(result)
        exit({ref, result})

      {:DOWN, reason} ->
        exit(reason)
    end
  end

  ## Process callbacks

  require Logger
  @behaviour GenServer

  @empty_stack {[], 0}
  @empty_queue :queue.new()
  @timeout_message {__MODULE__, :timeout}

  @impl true
  def init({name, serving, partitions, batch_size, batch_timeout, task_supervisor}) do
    Process.flag(:trap_exit, true)
    partitions = serving_partitions(serving, partitions)
    partitions_count = length(partitions)
    {:ok, module_state} = handle_init(serving.module, :process, serving.arg, partitions)

    :persistent_term.put(
      persistent_key(name),
      %{
        limit: batch_size,
        preprocessing: serving.client_preprocessing,
        postprocessing: serving.client_postprocessing,
        distributed_postprocessing: serving.distributed_postprocessing
      }
    )

    :pg.join(Nx.Serving.PG, __MODULE__, List.duplicate(self(), partitions_count))

    # We keep batches in a stack. Once the stack is full
    # or it times out, we either execute or enqueue it.
    state = %{
      module: serving.module,
      module_state: module_state,
      stack: @empty_stack,
      limit: batch_size,
      timeout: batch_timeout,
      timer: :none,
      in_queue: @empty_queue,
      out_queue: Enum.reduce(0..(partitions_count - 1), :queue.new(), &:queue.in/2),
      tasks: [],
      task_supervisor: task_supervisor
    }

    {:ok, state}
  end

  defp serving_partitions(%Nx.Serving{defn_options: defn_options}, true) do
    compiler = Keyword.get(defn_options, :compiler, Nx.Defn.Evaluator)
    compiler.__partitions_options__(defn_options)
  end

  defp serving_partitions(%Nx.Serving{defn_options: defn_options}, false) do
    [defn_options]
  end

  @impl true
  def handle_info({__MODULE__, :batched_run, ref, %Nx.Batch{} = batch}, state) do
    %{limit: limit, stack: {_, count}} = state

    state =
      cond do
        # Single entry takes the whole batch.
        # Execute what we have (if any) and execute a new one.
        batch.size == limit ->
          state
          |> server_execute()
          |> server_stack(ref, batch)
          |> server_execute()

        # First entry in batch.
        count == 0 ->
          state
          |> server_timer()
          |> server_stack(ref, batch)

        # We don't exceed the limit.
        batch.size + count < limit ->
          server_stack(state, ref, batch)

        # We go over the limit.
        batch.size + count > limit ->
          {current, next} = Nx.Batch.split(batch, limit - count)

          state
          |> server_stack(ref, current)
          |> server_execute()
          |> server_timer()
          |> server_stack(ref, next)

        # Exact match.
        true ->
          state
          |> server_stack(ref, batch)
          |> server_execute()
      end

    {:noreply, state}
  end

  def handle_info(@timeout_message, state) do
    {:noreply, server_timeout(state)}
  end

  def handle_info({ref, reply}, %{tasks: tasks, module: module} = state) do
    case Enum.split_with(tasks, &(elem(&1, 0).ref == ref)) do
      {[{_task, partition, ref_sizes}], tasks} ->
        server_reply_ok(module, ref, reply, ref_sizes)
        {:noreply, server_task_done(state, tasks, partition)}

      _ ->
        {:noreply, state}
    end
  end

  def handle_info({:DOWN, ref, :process, _process, reason}, %{tasks: tasks} = state) do
    case Enum.split_with(tasks, &(elem(&1, 0).ref == ref)) do
      {[{_task, partition, ref_sizes}], tasks} ->
        server_reply_down(reason, ref_sizes)
        {:noreply, server_task_done(state, tasks, partition)}

      _ ->
        {:noreply, state}
    end
  end

  def handle_info(msg, state) do
    Logger.warning("Unknown message in Nx.Serving: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, %{module: module, tasks: tasks, in_queue: in_queue, stack: {stack, _}}) do
    # Emulate the process is gone for entries in the queue
    for {_batch, ref_sizes} <- :queue.to_list(in_queue) do
      server_reply_down(:noproc, ref_sizes)
    end

    # As well as for entries in the stack
    for {ref, _batch} <- stack do
      send(ref, {:DOWN, ref, :process, self(), :noproc})
    end

    # And wait until all current tasks are processed
    for {%Task{ref: ref}, _partition, ref_sizes} <- tasks do
      receive do
        {^ref, reply} -> server_reply_ok(module, ref, reply, ref_sizes)
        {:DOWN, ^ref, :process, _, reason} -> server_reply_down(reason, ref_sizes)
      end
    end

    :ok
  end

  defp server_reply_ok(module, ref, reply, ref_sizes) do
    Process.demonitor(ref, [:flush])
    {output, metadata} = handle_executed(module, reply)

    for {ref, start, size} <- ref_sizes do
      send(ref, {ref, {start, size, output, metadata}})
    end
  end

  defp server_reply_down(reason, ref_sizes) do
    for {ref, _start, _size} <- ref_sizes do
      send(ref, {:DOWN, ref, :process, self(), reason})
    end
  end

  defp server_stack(%{stack: {stack, count}, limit: limit} = state, ref, batch)
       when batch.size + count <= limit do
    %{state | stack: {[{ref, batch} | stack], count + batch.size}}
  end

  defp server_timer(%{timeout: timeout, timer: :none} = state),
    do: %{state | timer: Process.send_after(self(), @timeout_message, timeout)}

  defp server_execute(%{stack: @empty_stack} = state), do: state

  defp server_execute(state) do
    %{stack: {stack, count}, timer: timer} = state

    if is_reference(timer) do
      Process.cancel_timer(timer)

      receive do
        @timeout_message -> :ok
      after
        0 -> :ok
      end
    end

    {ref_sizes, batches, _} =
      Enum.reduce(stack, {[], [], count}, fn {ref, batch}, {ref_sizes, batches, ending} ->
        size = batch.size
        {[{ref, ending - size, size} | ref_sizes], [batch | batches], ending - size}
      end)

    state = %{state | timer: :none, stack: @empty_stack}
    server_task_or_enqueue(state, Nx.Batch.merge(batches), ref_sizes)
  end

  defp server_task_or_enqueue(%{out_queue: out_queue} = state, batch, ref_sizes) do
    case :queue.out(out_queue) do
      {:empty, _out_queue} ->
        %{state | in_queue: :queue.in({batch, ref_sizes}, state.in_queue)}

      {{:value, partition}, out_queue} ->
        %{module: module, module_state: module_state} = state
        {:execute, function, module_state} = handle_batch(module, batch, partition, module_state)

        wrapped_function = fn ->
          :telemetry.span([:nx, :serving, :execute], %{module: module}, fn ->
            {output, metadata} = function.()
            {{output, metadata}, %{metadata: metadata, module: module}}
          end)
        end

        task = Task.Supervisor.async_nolink(state.task_supervisor, wrapped_function)
        tasks = [{task, partition, ref_sizes} | state.tasks]
        %{state | module_state: module_state, tasks: tasks, out_queue: out_queue}
    end
  end

  defp server_task_done(%{in_queue: in_queue, out_queue: out_queue} = state, tasks, partition) do
    out_queue = :queue.in(partition, out_queue)

    case :queue.out(in_queue) do
      # The timer expired while the task was processing, so execute the current batch.
      {:empty, _in_queue} when state.timer == :done ->
        server_execute(%{state | tasks: tasks, out_queue: out_queue})

      # Nothing to do.
      {:empty, _in_queue} ->
        %{state | tasks: tasks, out_queue: out_queue}

      # Execute the next one in queue.
      {{:value, {batch, ref_sizes}}, in_queue} ->
        state = %{state | tasks: tasks, out_queue: out_queue, in_queue: in_queue}
        server_task_or_enqueue(state, batch, ref_sizes)
    end
  end

  # It timed out, the in queue is empty and out queue is not empty, execute it now.
  defp server_timeout(%{out_queue: out_queue, in_queue: @empty_queue} = state)
       when out_queue != @empty_queue,
       do: server_execute(%{state | timer: :done})

  # Otherwise continue batching until the queue is empty or it is full.
  defp server_timeout(state),
    do: %{state | timer: :done}

  ## Shared helpers

  defp persistent_key(name) when is_atom(name) do
    {__MODULE__, name}
  end

  defp handle_init(module, type, arg, [_ | _] = partitions) do
    case module.init(type, arg, partitions) do
      {:ok, _} = pair ->
        pair

      other ->
        raise "#{inspect(module)}.init/3 must return {:ok, state}. Got: #{inspect(other)}"
    end
  end

  defp handle_batch(module, batch, partition, state) do
    case module.handle_batch(batch, partition, state) do
      {:execute, function, _} = pair when is_function(function, 0) ->
        pair

      other ->
        raise "#{inspect(module)}.handle_batch/3 must return {:execute, function, state}, " <>
                "where function is a function that receives no arguments and returns a tuple. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_executed(module, result) do
    case result do
      {output, metadata} ->
        {output, metadata}

      other ->
        raise "the function returned by #{inspect(module)}.handle_batch/3 must return {output, metadata}. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_preprocessing(nil, input) do
    case input do
      %Nx.Batch{} ->
        {no_empty_batch!(input), :client_info}

      _ ->
        raise ArgumentError,
              "the default client_preprocessing expects a `Nx.Batch` as input. " <>
                "Give a batch or use a custom preprocessing"
    end
  end

  defp handle_preprocessing(preprocessing, input) do
    meta = %{input: input}

    :telemetry.span([:nx, :serving, :preprocessing], meta, fn ->
      case preprocessing.(input) do
        {%Nx.Batch{} = batch, info} ->
          {{no_empty_batch!(batch), info}, Map.put(meta, :info, info)}

        other ->
          raise "client_preprocessing function #{inspect(preprocessing)} must return a two element tuple " <>
                  "where the first element is a Nx.Batch and the second is any value. Got: #{inspect(other)}"
      end
    end)
  end

  defp no_empty_batch!(%{size: 0}), do: raise(ArgumentError, "cannot run with empty Nx.Batch")
  defp no_empty_batch!(%{size: _} = batch), do: batch

  defp handle_postprocessing(nil, output, _metadata, _info), do: output

  defp handle_postprocessing(postprocessing, output, metadata, info) do
    meta = %{info: info, metadata: metadata, output: output}

    :telemetry.span([:nx, :serving, :postprocessing], meta, fn ->
      output = postprocessing.(output, metadata, info)

      {output, %{meta | output: output}}
    end)
  end
end

defmodule Nx.Serving.Default do
  @moduledoc false
  @behaviour Nx.Serving

  @impl true
  def init(_type, fun, partitions) do
    batch_funs =
      for defn_options <- partitions do
        case fun.(defn_options) do
          batch_fun when is_function(batch_fun, 1) ->
            batch_fun

          other ->
            raise "anonymous function given to Nx.Serving.new/2 should return an AOT or " <>
                    "JIT compiled function that expects one argument. Got: #{inspect(other)}"
        end
      end

    {:ok, batch_funs}
  end

  @impl true
  def handle_batch(batch, partition, batch_funs) do
    batch_fun = Enum.fetch!(batch_funs, partition)
    {:execute, fn -> {batch_fun.(batch), :server_info} end, batch_funs}
  end
end

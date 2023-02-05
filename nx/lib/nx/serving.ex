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

  `Nx.Serving` allows us to define a process that will batch requests up to
  a certain size or time. To do so, we need to start a `Nx.Serving` process
  with a serving inside a supervision tree:

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

  ## Module-based serving

  In the examples so far, we have been using the default version of
  `Nx.Serving`, which executes the given function for each batch.

  However, we can also use `new/2` to start a module-based version of
  `Nx.Serving`, which acts similar to an Elixir `GenServer` and gives
  us more control over both inline and process workflows. A simple
  module implementation of a `Nx.Serving` could look like this:

      defmodule MyServing do
        @behaviour Nx.Serving

        defnp print_and_multiply(x) do
          print_value({:debug, x})
          x * 2
        end

        @impl true
        def init(_inline_or_process, :unused_arg) do
          {:ok, Nx.Defn.jit(&print_and_multiply/1)}
        end

        @impl true
        def handle_batch(batch, function) do
          {:execute, fn -> {function.(batch), :server_info} end, function}
        end
      end

  It has two functions. The first, `c:init/2`, receives the type of serving
  (`:inline` or `:process`) and the serving argument. In this step,
  we capture `print_and_multiply/1`as a jitted function.

  The second function is called `handle_batch/2`. This function
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
  """

  @doc false
  @enforce_keys [:module, :arg]
  defstruct [
    :module,
    :arg,
    :client_preprocessing,
    :client_postprocessing,
    process_options: [],
    defn_options: []
  ]

  @type metadata() :: term()
  @type client_info() :: term()
  @type client_preprocessing() :: (term() -> {Nx.Batch.t(), client_info()})
  @type client_postprocessing() :: (Nx.Container.t(), metadata(), client_info() -> term())

  @type t :: %__MODULE__{
          module: atom(),
          arg: term(),
          client_preprocessing: client_preprocessing(),
          client_postprocessing: client_postprocessing()
        }

  @axis 0

  @doc """
  The callback used to initialize the serving.

  The first argument reveals if the serving is executed inline,
  such as by calling `run/2`, by started with the process.
  The second argument is the serving argument given to `new/2`,
  and the third argument are the compiler options to be used to
  compile the computation.

  It must return `{:ok, state}`, where the `state` can be any term.
  """
  @callback init(type :: :inline | :process, arg :: term(), defn_options :: Keyword.t()) ::
              {:ok, state :: term()}

  @doc """
  Receives a batch and returns a function to execute the batch.

  In case of serving processes, the function is executed is an
  separate process.
  """
  @callback handle_batch(Nx.Batch.t(), state) ::
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
  Sets the process options of this serving.

  These are the same options as supported on `start_link/1`,
  except `:name` and `:serving` itself.
  """
  def process_options(%Nx.Serving{} = serving, process_options) when is_list(process_options) do
    %{serving | process_options: process_options}
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

    {:ok, state} = handle_init(module, :inline, arg, defn_options)
    {%{size: size} = batch, info} = handle_preprocessing(preprocessing, input)
    {:execute, function, _} = handle_batch(module, batch, state)

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

    * `:shutdown` - the maximum time for the serving to shutdown. This will
      block until the existing computation finishes (defaults to `30_000`ms)

    * `:max_restarts` and `:max_seconds` - the maximum number of restarts
      within max seconds for each serving (see `Supervisor.start_link/3`)

    * `:hibernate_after` and `:spawn_opt` - configure the underlying serving
      workers (see `GenServer.start_link/3`)
  """
  def start_link(opts) do
    {name, opts} = Keyword.pop!(opts, :name)
    {serving, opts} = Keyword.pop!(opts, :serving)
    %Nx.Serving{module: module, arg: arg, process_options: process_options} = serving

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

    {supervisor_opts, opts} = Keyword.split(opts, [:max_restarts, :max_seconds])
    shutdown = Keyword.get(opts, :shutdown, 30_000)
    batch_size = Keyword.get(opts, :batch_size, 1)
    batch_timeout = Keyword.get(opts, :batch_timeout, 100)

    base_name = Atom.to_string(name)
    task_supervisor = Module.concat(base_name, "Supervisor")

    serving_partitions =
      serving
      |> serving_partitions()
      |> Enum.with_index(fn defn_options, index ->
        {:"#{base_name}_#{index + 1}", defn_options}
      end)

    key = persistent_key(name)

    payload = %{
      names: Keyword.keys(serving_partitions),
      limit: batch_size,
      preprocessing: serving.client_preprocessing,
      postprocessing: serving.client_postprocessing
    }

    children =
      for {name, defn_options} <- serving_partitions do
        arg = {task_supervisor, module, arg, defn_options, batch_size, batch_timeout}

        %{
          id: __MODULE__,
          start: {GenServer, :start_link, [__MODULE__, arg, [name: name] ++ opts]},
          shutdown: shutdown
        }
      end

    children = [{Task.Supervisor, name: task_supervisor} | children]

    Supervisor.start_link(
      Nx.Serving.Supervisor,
      {key, payload, children, [strategy: :one_for_one] ++ supervisor_opts},
      name: name
    )
  end

  defp serving_partitions(%Nx.Serving{defn_options: defn_options}) do
    [defn_options]
  end

  @doc """
  Runs the given `input` on the process given by `name`.

  The process `name` will batch requests and send a response
  either when the batch is full or on timeout. See the module
  documentation for more information.

  Note that you cannot batch an `input` larger than the configured
  `:batch_size` in the server.
  """
  def batched_run(name, input) when is_atom(name) do
    %{
      names: [partition_name],
      preprocessing: preprocessing,
      postprocessing: postprocessing,
      limit: limit
    } =
      :persistent_term.get(persistent_key(name), nil) ||
        exit({:noproc, {__MODULE__, :batched_run, [name, input]}})

    {batch, info} = handle_preprocessing(preprocessing, input)

    if batch.size > limit do
      raise ArgumentError,
            "batch size (#{batch.size}) cannot exceed Nx.Serving server batch size of #{limit}"
    end

    # Use Process.monitor/2 on Elixir v1.15+
    ref = :erlang.monitor(:process, partition_name, alias: :demonitor)
    Process.send(partition_name, {__MODULE__, :batched_run, ref, batch}, [:noconnect])

    {tensor, metadata} = receive_batched(batch.size, ref, [], nil, name, input)
    handle_postprocessing(postprocessing, tensor, metadata, info)
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

    {output, metadata}
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

          {output, metadata}
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
        exit({reason, {__MODULE__, :batched_run, [name, input]}})
    end
  end

  ## Process callbacks

  require Logger
  @behaviour GenServer

  @empty_stack {[], 0}
  @empty_queue :queue.new()
  @timeout_message {__MODULE__, :timeout}

  @impl true
  def init({task_supervisor, module, arg, defn_options, batch_size, batch_timeout}) do
    Process.flag(:trap_exit, true)
    {:ok, module_state} = handle_init(module, :process, arg, defn_options)

    # We keep batches in a stack. Once the stack is full
    # or it times out, we either execute or enqueue it.
    state = %{
      module: module,
      module_state: module_state,
      stack: @empty_stack,
      limit: batch_size,
      timeout: batch_timeout,
      timer: :none,
      queue: @empty_queue,
      task: :none,
      task_supervisor: task_supervisor
    }

    {:ok, state}
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

  def handle_info({ref, reply}, %{task: {task, ref_sizes}, module: module} = state)
      when task.ref == ref do
    server_reply_ok(module, ref, reply, ref_sizes)
    {:noreply, server_task_done(state)}
  end

  def handle_info({:DOWN, ref, type, _process, reason}, %{task: {task, ref_sizes}} = state)
      when task.ref == ref do
    server_reply_down(type, reason, ref_sizes)
    {:noreply, server_task_done(state)}
  end

  def handle_info(msg, state) do
    Logger.warning("Unknown message in Nx.Serving: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, state) do
    with %{module: module, task: {%Task{ref: ref}, ref_sizes}} <- state do
      receive do
        {^ref, reply} -> server_reply_ok(module, ref, reply, ref_sizes)
        {:DOWN, ^ref, type, _, reason} -> server_reply_down(type, reason, ref_sizes)
      end
    end

    state
  end

  defp server_reply_ok(module, ref, reply, ref_sizes) do
    Process.demonitor(ref, [:flush])
    {output, metadata} = handle_executed(module, reply)

    for {ref, start, size} <- ref_sizes do
      send(ref, {ref, {start, size, output, metadata}})
    end
  end

  defp server_reply_down(type, reason, ref_sizes) do
    for {ref, _start, _size} <- ref_sizes do
      send(ref, {:DOWN, ref, type, self(), reason})
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
    %{module: module, module_state: module_state, stack: {stack, count}, timer: timer} = state

    if is_reference(timer) and Process.cancel_timer(timer) == false do
      receive do
        @timeout_message -> :ok
      end
    end

    {ref_sizes, batches, _} =
      Enum.reduce(stack, {[], [], count}, fn {ref, batch}, {ref_sizes, batches, ending} ->
        size = batch.size
        {[{ref, ending - size, size} | ref_sizes], [batch | batches], ending - size}
      end)

    batch = Nx.Batch.merge(batches)
    {:execute, function, module_state} = handle_batch(module, batch, module_state)

    wrapped_function = fn ->
      :telemetry.span([:nx, :serving, :execute], %{module: module}, fn ->
        {output, metadata} = function.()

        {{output, metadata}, %{metadata: metadata, module: module}}
      end)
    end

    state = %{state | timer: :none, stack: @empty_stack, module_state: module_state}
    server_task_or_enqueue(state, wrapped_function, ref_sizes)
  end

  defp server_task_or_enqueue(%{task: :none} = state, function, ref_sizes) do
    task = Task.Supervisor.async_nolink(state.task_supervisor, function)
    %{state | task: {task, ref_sizes}}
  end

  defp server_task_or_enqueue(%{task: {_, _}, queue: queue} = state, function, ref_sizes) do
    %{state | queue: :queue.in({function, ref_sizes}, queue)}
  end

  defp server_task_done(%{queue: queue} = state) do
    case :queue.out(queue) do
      # The timer expired while the task was processing, so execute the current batch.
      {:empty, _queue} when state.timer == :done ->
        server_execute(%{state | task: :none})

      # Nothing to do.
      {:empty, _queue} ->
        %{state | task: :none}

      # Execute the next one in queue.
      {{:value, {function, ref_sizes}}, queue} ->
        server_task_or_enqueue(%{state | task: :none, queue: queue}, function, ref_sizes)
    end
  end

  # It timed out and there is no task and the queue is empty, execute it now.
  defp server_timeout(%{task: :none, queue: @empty_queue} = state),
    do: server_execute(%{state | timer: :done})

  # Otherwise continue batching until the queue is empty or it is full.
  defp server_timeout(%{task: {_, _}} = state),
    do: %{state | timer: :done}

  ## Shared helpers

  defp persistent_key(name) when is_atom(name) do
    {__MODULE__, name}
  end

  defp handle_init(module, type, arg, opts) do
    case module.init(type, arg, opts) do
      {:ok, _} = pair ->
        pair

      other ->
        raise "#{inspect(module)}.init/3 must return {:ok, state}. Got: #{inspect(other)}"
    end
  end

  defp handle_batch(module, batch, state) do
    case module.handle_batch(batch, state) do
      {:execute, function, _} = pair when is_function(function, 0) ->
        pair

      other ->
        raise "#{inspect(module)}.handle_batch/2 must return {:execute, function, state}, " <>
                "where function is a function that receives no arguments and returns a tuple. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_executed(module, result) do
    case result do
      {output, metadata} ->
        {output, metadata}

      other ->
        raise "the function returned by #{inspect(module)}.handle_batch/2 must return {output, metadata}. " <>
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

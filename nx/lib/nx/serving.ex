defmodule Nx.Serving do
  @moduledoc """
  Serving encapsulates client and server work to perform batched requests.

  Serving can be executed on the fly, without starting a server, but most
  often it is used to run servers that batch requests until a given size
  or timeout is reached.

  ## Inline/serverless workflow

  First, let's define a simple numerical definition function:

      defmodule MyDefn do
        import Nx.Defn

        defnp print_and_multiply(x) do
          print_value({:debug, x})
          x * 2
        end
      end

  The function prints the given tensor and double its contents.
  We can use `new/1` to create a serving that will return a JITted
  or compiled function to execute on batches of tensors:

      iex> serving = Nx.Serving.new(fn -> Nx.Defn.jit(&print_and_multiply/1) end)
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

  You should see two values printed. The first is the result of
  `Nx.Defn.Kernel.print_value/1`, which shows the tensor that was
  actually part of the computation and how it was batched.
  Then we see the result of the computation.

  When defining a `Nx.Serving`, we can also customize how the data is
  batched by using the `client_preprocessing` as well as the result by
  using `client_postprocessing` hooks. Let's give it another try:

      iex> serving = (
      ...>   Nx.Serving.new(fn -> Nx.Defn.jit(&print_and_multiply/1) end)
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

  You can see the results are a bit different now. First of all, notice we
  were able to run the serving passing a list of tensors. Our custom
  `client_preprocessing` function stacks those tensors into a batch of two
  entries and returns a tuple with a `Nx.Batch` struct and additional client
  information which we represent as the atom `:client_info`. The default
  client preprocessing simply enforces a batch was given and returns no client
  information.

  Then the result is a `{..., ..., ...}` tuple, returned by the client
  postprocessing function, containing the result, the server information
  (which we will learn later how to customize it), and the client information.
  From this, we can infer the default implementation of `client_postprocessing`
  simply returns the result, discarding the server and client information.

  So far, `Nx.Serving` has not given us much. It has simply encapsulated the
  execution of a function. Its full power comes when we start running our own
  `Nx.Serving` process. That's when we will also learn why we have a `client_`
  prefix in some of the function names.

  ## Stateful/process workflow

  `Nx.Serving` allows us to define a process that will batch requests up to
  a certain size or within a certain time. To do so, we need to start a
  `Nx.Serving` process with a serving inside a supervision tree:

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
  The process will wait up for requests from other processes, up to
  100 milliseconds or until it gets 10 entries. Then it merges all
  batches together and, once the result is computed, it slices and
  distributed those responses to each caller.

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

  It has two functions: `c:init/2`, which receives the type of serving
  (`:inline` or `:process`) and the serving argument. In this step,
  we capture `print_and_multiply/1`as a jitted function.

  The second function is called `handle_batch/2`. This function
  receives a `Nx.Batch` and it must return a function to execute.
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
  defstruct [:module, :arg, :client_preprocessing, :client_postprocessing, process_options: []]

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
  The second argument is the serving argument given to `new/2`.

  It must return `{:ok, state}`, where the `state` can be any term.
  """
  @callback init(type :: :inline | :process, arg :: term()) :: {:ok, state :: term()}

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

  It expects a function that returns a JITted (via `Nx.Defn.jit/2`)
  or compiled (via `Nx.Defn.compile/3`) one-arity function as argument.
  The JITted/compiled function will be called with the arguments
  returned by the `client_preprocessing` callback.

  A second argument called `process_options`, which is optional,
  can be given to customize the options when starting the serving
  under a process.
  """
  def new(function, process_options \\ [])

  def new(function, process_options) when is_function(function, 0) and is_list(process_options) do
    new(Nx.Serving.Default, function, process_options)
  end

  def new(module, arg) when is_atom(module) do
    new(module, arg, [])
  end

  @doc """
  Creates a new module-based serving.

  It expects a module and an argument that is given to its `init` callback.

  A third argument called `process_options`, which is optional,
  can be given to customize the options when starting the serving
  under a process.
  """
  def new(module, arg, process_options) when is_atom(module) and is_list(process_options) do
    %Nx.Serving{module: module, arg: arg, process_options: process_options}
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
  Runs `serving` with the given `input` inline with the current process.
  """
  def run(%Nx.Serving{} = serving, input) do
    %{
      module: module,
      arg: arg,
      client_preprocessing: preprocessing,
      client_postprocessing: postprocessing
    } = serving

    {:ok, state} = handle_init(module, :inline, arg)
    {%{size: size} = batch, info} = handle_preprocessing(preprocessing, input)
    {:execute, function, _} = handle_batch(module, batch, state)
    {output, metadata} = handle_executed(module, function.())
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

    * `:batch_size` - the maximum size to batch for. A value is first read
      from the `Nx.Serving` struct and then it falls back to this option
      (which defaults to `1`)

    * `:batch_timeout` - the maximum time to wait, in milliseconds,
      before executing the batch. A value is first read from the `Nx.Serving`
      struct and then it falls back to this option (which defaults to `100`ms)

  """
  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    {%Nx.Serving{process_options: serving_opts} = serving, opts} = Keyword.pop!(opts, :serving)
    {batch_size, opts} = process_option(:batch_size, serving_opts, opts, 1)
    {batch_timeout, opts} = process_option(:batch_timeout, serving_opts, opts, 100)
    arg = {name, serving, batch_size, batch_timeout}

    children = [
      %{
        id: __MODULE__,
        start: {GenServer, :start_link, [__MODULE__, arg, opts]}
      }
    ]

    Supervisor.start_link(children, strategy: :one_for_all, max_restarts: 0)
  end

  defp process_option(key, serving_opts, opts, default) do
    serving_value = serving_opts[key]
    {value, opts} = Keyword.pop(opts, key)

    if serving_value && value && serving_value != value do
      raise ArgumentError,
            "#{inspect(key)} has been set when starting an Nx.Serving process (#{inspect(value)}) " <>
              "but a conflicting value was already set on the Nx.Serving struct (#{inspect(serving_value)}). " <>
              "Please remove the option given to the Nx.Serving process"
    end

    {serving_value || value || default, opts}
  end

  @doc """
  Runs the given `input` on the process given by `name`.

  The process `name` will batch requests and send a response
  either when the batch is full or on timeout. See the module
  documentation for more information.

  Note you cannot batch an `input` larger than the configured
  `:batch_size` in the server.
  """
  def batched_run(name, input) when is_atom(name) do
    pid = GenServer.whereis(name) || exit({:noproc, {__MODULE__, :batched_run, [name, input]}})

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
  def init({name, serving, batch_size, batch_timeout}) do
    [parent | _] = Process.get(:"$ancestors")
    {:ok, module_state} = handle_init(serving.module, :process, serving.arg)

    :persistent_term.put(
      persistent_key(name),
      %{
        limit: batch_size,
        preprocessing: serving.client_preprocessing,
        postprocessing: serving.client_postprocessing
      }
    )

    # We keep batches in a stack. Once the stack is full
    # or it times out, we either execute or enqueue it.
    state = %{
      name: name,
      module: serving.module,
      module_state: module_state,
      stack: @empty_stack,
      limit: batch_size,
      timeout: batch_timeout,
      timer: :none,
      queue: @empty_queue,
      task: :none,
      task_supervisor: nil
    }

    {:ok, state, {:continue, {:task_supervisor, parent}}}
  end

  @impl true
  def handle_continue({:task_supervisor, parent}, state) do
    {:ok, task_supervisor} = Supervisor.start_child(parent, Task.Supervisor)
    {:noreply, %{state | task_supervisor: task_supervisor}}
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

  def handle_info({ref, reply}, %{task: {task, ref_sizes}} = state) when task.ref == ref do
    Process.demonitor(ref, [:flush])
    {output, metadata} = handle_executed(state.module, reply)

    for {ref, start, size} <- ref_sizes do
      send(ref, {ref, {start, size, output, metadata}})
    end

    {:noreply, server_task_done(state)}
  end

  def handle_info({:DOWN, ref, type, _process, reason}, %{task: {task, ref_sizes}} = state)
      when task.ref == ref do
    for {ref, _start, _size} <- ref_sizes do
      send(ref, {:DOWN, ref, type, self(), reason})
    end

    {:noreply, server_task_done(state)}
  end

  def handle_info(msg, state) do
    Logger.warning("Unknown message in Nx.Serving: #{inspect(msg)}")
    {:noreply, state}
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

    state = %{state | timer: :none, stack: @empty_stack, module_state: module_state}
    server_task_or_enqueue(state, function, ref_sizes)
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

  defp handle_init(module, type, arg) do
    case module.init(type, arg) do
      {:ok, _} = pair ->
        pair

      other ->
        raise "#{inspect(module)}.init/2 must return {:ok, state}. Got: #{inspect(other)}"
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
    case preprocessing.(input) do
      {%Nx.Batch{} = batch, info} ->
        {no_empty_batch!(batch), info}

      other ->
        raise "client_preprocessing function #{inspect(preprocessing)} must return a two element tuple " <>
                "where the first element is a Nx.Batch and the second is any value. Got: #{inspect(other)}"
    end
  end

  defp no_empty_batch!(%{size: 0}), do: raise(ArgumentError, "cannot run with empty Nx.Batch")
  defp no_empty_batch!(%{size: _} = batch), do: batch

  defp handle_postprocessing(nil, output, _metadata, _info),
    do: output

  defp handle_postprocessing(postprocessing, output, metadata, info),
    do: postprocessing.(output, metadata, info)
end

defmodule Nx.Serving.Default do
  @moduledoc false
  @behaviour Nx.Serving

  @impl true
  def init(_type, fun) do
    case fun.() do
      batch_fun when is_function(batch_fun, 1) ->
        {:ok, batch_fun}

      other ->
        raise "anonymous function given to Nx.Serving.new/2 should return a compiled or " <>
                "JITted function that expects one argument, got: #{inspect(other)}"
    end
  end

  @impl true
  def handle_batch(batch, batch_fun) do
    {:execute, fn -> {batch_fun.(batch), :server_info} end, batch_fun}
  end
end

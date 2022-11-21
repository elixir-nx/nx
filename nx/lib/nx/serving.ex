defmodule Nx.Serving do
  @moduledoc """
  Serving encapsulates client and server work to perform batched requests.

  Serving can be executed on the fly, without starting a server, but most
  often it is used to run servers that batch requests until a given size
  or timeout is reached.

  ## Inline/serverless workflow

  First, let's create a simple serving module:

      defmodule MyServing do
        @behaviour Nx.Serving
        import Nx.Defn

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
          {:execute, fn -> {function.(batch), :metadata} end, function}
        end
      end

  It has two functions: `c:init/2`, which receives some metadata
  about the type of serving (`:inline` or `:process`) and the
  serving argument. In this step, we capture `print_and_multiply/1`
  as a jitted function.

  The second function is called `handle_batch/2`. This function
  receives a `Nx.Batch` and it must return a function to execute.
  The function itself must return a two element-tuple: the batched
  results and some metadata. The metadata can be any value and we
  set it to the atom `:metadata`.

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

  You should see two values printed. The first is the result of
  `Nx.Defn.Kernel.print_value/1`, which shows the tensor that was
  actually part of the computation and how it was batched.
  Then we see the result of the computation.

  When defining a `Nx.Serving`, we can also customize how the data is
  batched by using the `client_preprocessing` as well as the result by
  using `client_postprocessing` hooks. Let's give it another try:

      iex> serving = (
      ...>   Nx.Serving.new(MyServing, :unused_arg)
      ...>   |> Nx.Serving.client_preprocessing(&Nx.Batch.stack/1)
      ...>   |> Nx.Serving.client_postprocessing(&{:postprocessing, &1, &2})
      ...> )
      iex> Nx.Serving.run(serving, [Nx.tensor([1, 2]), Nx.tensor([3, 4])])
      {:debug, #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >}
      {:post_processing, #Nx.Tensor<
         s64[2][2]
         [
           [2, 4],
           [6, 8]
         ]
       >, :metadata}

  You can see the results are a bit different now. First of all, notice we
  were able to run the serving passing a list of tensors. Our custom
  `client_preprocessing` function stacks those tensors into a batch of two
  entries. With the `client_preprocessing` function, you can transform the
  input in any way you desire before batching. It must return a `Nx.Batch`
  struct. The default client preprocessing simply enforces a batch was given.

  Then the result is a `{:postprocessing, ..., ...}` tuple containing the
  result and the execution metadata as second and third elements respectively.
  From this, we can infer the default implementation of `client_postprocessing`
  simply returns the result, discarding the metadata.

  Why these functions have a `client_` prefix in their name will become clearer
  in the next section.

  ## Stateful/process workflow

  TODO.
  """

  @doc false
  defstruct [:module, :arg, :client_preprocessing, :client_postprocessing]

  @type metadata() :: term()
  @type client_preprocessing() :: (term() -> Nx.Batch.t())
  @type client_postprocessing() :: (Nx.Container.t(), metadata() -> Nx.Batch.t())

  @type t :: %__MODULE__{
          module: atom(),
          arg: term(),
          client_preprocessing: client_preprocessing(),
          client_postprocessing: client_postprocessing()
        }

  @doc """
  The callback used to initialize the serving.

  The first argument reveals if the serving is executed inline,
  such as by calling `run/2`, by started with the process.
  The second argument is the serving argument given to `new/2`.

  It must return `{:ok, state}`, where the `state` can be any term.
  """
  @callback init(:inline | :process, arg :: term()) :: {:ok, state :: term()}

  @doc """
  Receives a batch and returns a function to execute the batch.

  In case of serving processes, the function is executed is an
  separate process.
  """
  @callback handle_batch(Nx.Batch.t(), state) ::
              {:execute, (-> {Nx.Container.t(), metadata()}), state}
            when state: term()

  @doc """
  Creates a new serving.

  It expects a module and an argument that is given to its `init` callback.
  """
  def new(module, arg) when is_atom(module) do
    %Nx.Serving{module: module, arg: arg}
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
      when is_function(function, 2) or is_nil(function) do
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
    batch = handle_preprocessing(preprocessing, input)
    {:execute, function, _} = handle_batch(module, batch, state)
    {output, metadata} = handle_executed(module, function.())
    handle_postprocessing(postprocessing, output, metadata)
  end

  ## Process API

  @doc """
  TODO
  """
  def start_link(opts) do
    {serving, opts} = Keyword.pop!(opts, :serving)
    {batch_size, opts} = Keyword.pop(opts, :batch_size, 1)
    {batch_timeout, opts} = Keyword.pop(opts, :batch_timeout, 100)
    name = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, {name, serving, batch_size, batch_timeout}, opts)
  end

  @doc """
  TODO
  """
  def batched_run(name, input) when is_atom(name) do
    pid = GenServer.whereis(name) || exit({:noproc, {__MODULE__, :batched_run, [name, input]}})

    %{
      preprocessing: preprocessing,
      postprocessing: postprocessing,
      limit: limit
    } = :persistent_term.get(persistent_key(name))

    batch = handle_preprocessing(preprocessing, input)

    if batch.size > limit do
      raise ArgumentError,
            "batch size (#{batch.size}) cannot exceed Nx.Serving server batch size of #{limit}"
    end

    # Use Process.monitor/2 on Elixir v1.15+
    ref = :erlang.monitor(:process, pid, alias: :demonitor)
    Process.send(pid, {__MODULE__, :batched_run, ref, batch}, [:noconnect])

    receive do
      {^ref, {size, output, metadata}} ->
        Process.demonitor(ref, [:flush])
        # TODO: handle size and postprocessing

      {:DOWN, ^ref, _, _, reason} ->
        # We fake monitor messages, so still demonitor and flush.
        Process.demonitor(ref, [:flush])
        exit({reason, {__MODULE__, :batched_run, [name, input]}})
    end
  end

  ## Process callbacks

  use GenServer
  require Logger

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
      start: {__MODULE__, :start_link, [opts]}
    }
  end

  @empty {[], 0}
  @timeout_message {__MODULE__, :timeout}

  @impl true
  def init({name, serving, batch_size, batch_timeout}) do
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
      stack: @empty,
      limit: batch_size,
      timeout: batch_timeout,
      timer: :none,
      queue: :queue.new(),
      task: :none
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

  def handle_info({ref, reply}, %{task: {task, ref_sizes}} = state) when task.ref == ref do
    {output, metadata} = handle_executed(state.module, reply)

    for {ref, size} <- ref_sizes do
      send(ref, {ref, {size, output, metadata}})
    end

    {:noreply, server_task_done(state)}
  end

  def handle_info({:DOWN, ref, type, _process, reason}, %{task: {task, ref_sizes}} = state)
      when task.ref == ref do
    for {ref, _size} <- ref_sizes do
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
    %{state | stack: {[{ref, batch} | stack]}, count: count + batch.size}
  end

  defp server_timer(%{timeout: timeout, timer: :none} = state),
    do: %{state | timer: Process.send_after(self(), @timeout_message, timeout)}

  defp server_execute(%{stack: @empty} = state), do: state

  defp server_execute(state) do
    %{
      module: module,
      module_state: module_state,
      stack: {stack, count},
      limit: limit,
      timer: timer
    } = state

    if is_reference(timer) and Process.cancel_timer(timer) == false do
      receive do
        @timeout_message -> :ok
      end
    end

    {ref_sizes, batches} =
      Enum.reduce(stack, {[], []}, fn {ref, batch}, {ref_sizes, batches} ->
        {[{ref, batch.size} | ref_sizes], [batch | batches]}
      end)

    batch =
      batches
      |> Nx.Batch.merge()
      |> Nx.Batch.pad(limit - count)

    {:execute, function, module_state} = handle_batch(module, batch, module_state)

    state = %{state | timer: :none, stack: @empty, module_state: module_state}
    server_task_or_enqueue(state, function, ref_sizes)
  end

  defp server_task_or_enqueue(%{task: :none} = state, function, ref_sizes) do
    # TODO: Use Task.Supervisor.async_nolink()
    task = Task.async(function)
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
      {{function, ref_sizes}, queue} ->
        server_task_or_enqueue(%{state | task: :none, queue: queue}, function, ref_sizes)
    end
  end

  # It timed out and there is no task, continue batching.
  defp server_timeout(%{task: {_, _}} = state), do: %{state | timer: :done}

  # It timed out and there is no task, execute it now.
  defp server_timeout(%{task: :none} = state), do: server_execute(%{state | timer: :done})

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
        input

      _ ->
        raise ArgumentError,
              "the default client_preprocessing expects a `Nx.Batch` as input. " <>
                "Give a batch or use a custom preprocessing"
    end
  end

  defp handle_preprocessing(preprocessing, input) do
    case preprocessing.(input) do
      %Nx.Batch{} = batch ->
        batch

      other ->
        raise "client_preprocessing function #{inspect(preprocessing)} must return a Nx.Batch. " <>
                "Got: #{inspect(other)}"
    end
  end

  defp handle_postprocessing(nil, output, _metadata),
    do: output

  defp handle_postprocessing(postprocessing, output, metadata),
    do: postprocessing.(output, metadata)
end

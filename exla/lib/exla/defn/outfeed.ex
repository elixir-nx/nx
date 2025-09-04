defmodule EXLA.Defn.Outfeed do
  @moduledoc false

  alias EXLA.Defn.Outfeed
  alias Nx.Defn.{Expr, Tree, Composite}
  alias Nx.Tensor, as: T

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value

  defstruct user_hooks: %{},
            default_hooks: %{},
            used_hooks: [],
            compiled_hooks: %{},
            token: nil,
            infeeds: []

  ## Functional API

  @doc """
  Computes used inputs by depth and used hooks.
  """
  def used_inputs_and_hooks(expr, force_inputs, lazy_transfers) do
    if lazy_transfers not in [:always, :never, :opt_in] do
      raise ArgumentError,
            ":lazy_transfers must be either :always or :never, got: #{inspect(lazy_transfers)}"
    end

    lazy? = lazy_transfers == :always
    inputs = Map.from_keys(force_inputs, nil)

    {_, used_inputs, used_hooks} =
      Composite.reduce(expr, {%{}, inputs, %{}}, &used_inputs_and_hooks(&1, &2, 0, lazy?))

    {used_inputs, used_hooks}
  end

  defp used_inputs_and_hooks(%T{data: %Expr{id: id} = expr} = t, acc, depth, lazy?) do
    {seen, inputs, hooks} = acc

    case seen do
      %{^id => true} ->
        acc

      %{} ->
        depth = depth + 1
        seen = Map.put(seen, id, true)
        inputs = used_inputs(expr, inputs, depth, lazy?)
        hooks = used_hooks(expr, hooks)
        acc = {seen, inputs, hooks}

        t
        |> Tree.apply_args(acc, &{&1, used_inputs_and_hooks(&1, &2, depth, lazy?)})
        |> elem(1)
    end
  end

  defp used_inputs(%Expr{op: :parameter, args: [i], context: :root}, inputs, depth, lazy?) do
    case inputs do
      %{^i => nil} when lazy? -> Map.put(inputs, i, depth)
      %{^i => nil} -> inputs
      %{^i => current} when current >= depth -> inputs
      %{^i => _current} -> Map.put(inputs, i, depth)
      %{} -> Map.put(inputs, i, if(lazy?, do: depth, else: nil))
    end
  end

  defp used_inputs(_, inputs, _depth, _lazy?),
    do: inputs

  defp used_hooks(%Expr{op: :token, args: [token]}, hooks),
    do: Enum.reduce(token.hooks, hooks, &Map.put(&2, &1.name, &1.callback))

  defp used_hooks(_, hooks),
    do: hooks

  ## Struct API

  defguard will_outfeed(outfeed) when outfeed.compiled_hooks != %{}

  @doc """
  An empty outfeed to be used when not outfeeding is supported.
  """
  def empty do
    %Outfeed{}
  end

  @doc """
  An outfeed struct to track the need for outfeeds during compilation.
  """
  def new(user_hooks, default_hooks) when is_map(user_hooks) and is_map(default_hooks) do
    # Hooks with default callbacks or user callbacks are part of the cache key
    used_hooks =
      Enum.sort(for {k, v} <- default_hooks, v != nil or Map.has_key?(user_hooks, k), do: k)

    # We don't store the user hooks yet, because we don't want them to be cached
    %Outfeed{
      default_hooks: default_hooks,
      used_hooks: used_hooks
    }
  end

  @doc """
  Sets the user hooks to outfeed.
  """
  def with_user_hooks(%Outfeed{} = outfeed, user_hooks), do: %{outfeed | user_hooks: user_hooks}

  @doc """
  Sets the token to outfeed.
  """
  def with_token(%Outfeed{} = outfeed, token), do: %{outfeed | token: token}

  @doc """
  Adds an infeed hook.
  """
  def add_infeeds(%Outfeed{} = outfeed, builder, entries) do
    %{compiled_hooks: compiled_hooks, token: token} = outfeed

    case entries do
      [] ->
        # No entries - return outfeed unchanged
        outfeed

      _ ->
        # Use custom infeed only for actual streaming scenarios
        if length(entries) == 1 do
          # Single entry
          [{pos, _, typespec}] = entries
          next_flag = next_hook(compiled_hooks)
          compiled_hooks = Map.put(compiled_hooks, next_flag, {:infeed, pos, typespec})

          # Send flag notification via outfeed
          Value.outfeed([Value.constant(builder, [next_flag], EXLA.Typespec.tensor({:u, 16}, {}))], builder)

          # Use the session tag argument (last function arg)
          [tag_arg] = EXLA.MLIR.Function.get_arguments(builder) |> Enum.reverse() |> Enum.take(1)
          {_next_tag, input} = Value.infeed_custom(tag_arg, typespec)

          %{outfeed | compiled_hooks: compiled_hooks, token: token, infeeds: [{pos, input}]}
        else
          # Multiple entries
          next_flag = next_hook(compiled_hooks)
          compiled_hooks = Map.put(compiled_hooks, next_flag, {:infeed_variadic, entries})

          # Send flag notification via outfeed
          Value.outfeed([Value.constant(builder, [next_flag], EXLA.Typespec.tensor({:u, 16}, {}))], builder)

          # Extract typespecs for variadic call
          typespecs = Enum.map(entries, fn {_pos, _depth, typespec} -> typespec end)
          [tag_arg] = EXLA.MLIR.Function.get_arguments(builder) |> Enum.reverse() |> Enum.take(1)
          {_next_tag, inputs} = Value.infeed_custom(tag_arg, typespecs)

          # Map inputs back to positions
          infeeds =
            entries
            |> Enum.zip(inputs)
            |> Enum.map(fn {{pos, _depth, _typespec}, input} -> {pos, input} end)

          %{outfeed | compiled_hooks: compiled_hooks, token: token, infeeds: infeeds}
        end
    end
  end



  @doc """
  Adds a function hook if it has a callback defined for it.
  """
  def maybe_add_function_hook(%Outfeed{} = outfeed, builder, tuple, name, expr) do
    cond do
      name in outfeed.used_hooks ->
        {outfeed, flag, typespecs} = outfeed_flat_tuple(outfeed, builder, tuple)
        put_in(outfeed.compiled_hooks[flag], {:function, typespecs, name, Nx.to_template(expr)})

      outfeed.token ->
        outfeed

      true ->
        raise "hooks are not supported inside #{builder.name}"
    end
  end

  @doc """
  Closes the outfeed at the end of a pipeline.

  Note the outfeed may be closed before the computation finishes.
  """
  def close(outfeed, builder)

  def close(%Outfeed{} = outfeed, %Function{} = builder) when will_outfeed(outfeed) do
    # Send close signal via outfeed
    Value.outfeed([Value.constant(builder, [0], EXLA.Typespec.tensor({:u, 16}, {}))], builder)
    outfeed
  end

  def close(%Outfeed{} = outfeed, _builder),
    do: outfeed

  defp outfeed_flat_tuple(%Outfeed{token: token, compiled_hooks: ch} = outfeed, builder, tuple) do
    flag = next_hook(ch)
    # Send flag notification via outfeed
    Value.outfeed([Value.constant(builder, [flag], EXLA.Typespec.tensor({:u, 16}, {}))], builder)
    typespecs = Enum.map(tuple, &Value.get_typespec/1)

    # Send individual tensor outfeeds using the main outfeed custom call.
    # We purposely avoid XLA's native outfeed queues and rely on our
    # custom-call implementation to deliver binaries to the Elixir process.
    Enum.each(tuple, fn elem ->
      Value.outfeed([elem], builder)
    end)

    {%{outfeed | token: token}, flag, typespecs}
  end

  # The index 0 is served for closing streams
  defp next_hook(compiled_hooks), do: map_size(compiled_hooks) + 1

  ## Process API

  @doc """
  Receives a client, device_id, and mappings of u16 to
  `{typespecs, {pid, ref} | {fun, template}}` pairs to
  deliver/execute the outputs. The computation must emit
  a 0 flag on exit.
  """
  def start_child(
        %EXLA.Executable{} = executable,
        %Outfeed{} = outfeed,
        group_leader,
        infeeds \\ %{}
      ) do
    %{client: client, device_id: device_id} = executable

    %{compiled_hooks: compiled_hooks, default_hooks: default_hooks, user_hooks: user_hooks} =
      outfeed

    hooks = Map.merge(default_hooks, user_hooks)

    # Ensure a single coordinator per device; if it exists, reuse and begin a new session.
    name = :"exla_feed_process_#{device_id}"
    pid =
      case Process.whereis(name) do
        nil ->
          {:ok, pid} =
            Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
              init(client, device_id, hooks, compiled_hooks, infeeds, group_leader)
            end)

          pid

        pid when is_pid(pid) ->
          pid
      end

    # Create and provide a fresh session tag for this run (closure binds device_id)
    fun = fn action -> infeed_callback(device_id, action) end
    tag = EXLA.NifCall.Runner.register(EXLA.NifCall.Runner, fun)
    send(pid, {:begin_session, hooks, compiled_hooks, infeeds, self(), tag})
    receive do
      {:session_ready, ^pid} -> {:ok, {pid, tag}}
    after
      5_000 -> {:error, :session_timeout}
    end
  end

  defp init(client, device_id, hooks, compiled_hooks, infeeds, group_leader) do
    Process.flag(:trap_exit, true)
    # Copy the group leader so we report to the proper device
    Process.group_leader(self(), group_leader)

    # Register this process for infeed/outfeed coordination (idempotent)
    name = :"exla_feed_process_#{device_id}"
    _ = Process.whereis(name) || Process.register(self(), name)

    # Enter the loop. We no longer use XLA's outfeed queues – the native
    # custom-calls send messages directly to this process.
    typespec = EXLA.Typespec.tensor({:u, 16}, {})
    loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, %{infeed_q: :queue.new(), session_tag: nil})
  end

  defp loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, state) do
    receive do
      {:begin_session, new_hooks, new_compiled_hooks, new_infeeds, caller, tag} ->
        send(caller, {:session_ready, self()})
        loop(client, device_id, typespec, new_hooks, new_compiled_hooks, new_infeeds, %{infeed_q: :queue.new(), session_tag: tag})

      # Handle infeed data requests
      {:infeed_data, data_and_typespecs} ->
        # Enqueue data into the infeed queue
        updated_q = :queue.in(data_and_typespecs, state.infeed_q)
        updated_state = %{state | infeed_q: updated_q}
        loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, updated_state)

      # Handle outfeed reader registrations
      {:register_outfeed_reader, reader_pid, reader_ref, reader_typespecs} ->
        # Store outfeed reader info
        updated_state = Map.put(state, :outfeed_reader, {reader_pid, reader_ref, reader_typespecs})
        loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, updated_state)

      # Close signal sent by native outfeed custom-call (flag 0)
      <<0::native-unsigned-16>> ->
        :ok

      # Flag sent by native outfeed custom-call
      <<flag::native-unsigned-16>> ->
        case Map.fetch!(compiled_hooks, flag) do
          {:infeed, index, data_typespec} ->
            data =
              case Map.fetch!(infeeds, index) do
                %EXLA.DeviceBuffer{} = buffer -> EXLA.DeviceBuffer.read(buffer)
                %EXLA.BinaryBuffer{data: data} -> data
              end

            # Enqueue into infeed queue for the custom infeed to consume
            updated_q = :queue.in([{data, data_typespec}], state.infeed_q)
            state = %{state | infeed_q: updated_q}
            loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, state)

          {:infeed_variadic, entries} ->
            # Handle variadic infeed: collect data for all entries
            data_list =
              Enum.map(entries, fn {pos, _depth, data_typespec} ->
                data = case Map.fetch!(infeeds, pos) do
                  %EXLA.DeviceBuffer{} = buffer -> EXLA.DeviceBuffer.read(buffer)
                  %EXLA.BinaryBuffer{data: data} -> data
                end
                {data, data_typespec}
              end)
            updated_q = :queue.in(data_list, state.infeed_q)
            state = %{state | infeed_q: updated_q}
            loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, state)

          {:function, typespecs, name, template} ->
            fun = Map.fetch!(hooks, name)
            # Expect the next N tensor messages to arrive via custom-call
            # Deliver them to the hook once collected.
            pending = %{fun: fun, template: template, remaining: length(typespecs), acc: []}
            loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, Map.put(state, :pending_hook, pending))
        end

      # Tensor payloads delivered by native outfeed custom-call.
      # They arrive as a list of binaries when multiple tensors are sent at once,
      # or a single binary for individual sends.
      list when is_list(list) ->
        state = Enum.reduce(list, state, &handle_tensor/2)
        loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, state)

      bin when is_binary(bin) ->
        state = handle_tensor(bin, state)
        loop(client, device_id, typespec, hooks, compiled_hooks, infeeds, state)
    end
  end

  # Called from native infeed custom call via NifCall with the current session tag
  defp infeed_callback(device_id, :next_variadic) do
    case Process.whereis(:"exla_feed_process_#{device_id}") do
      nil -> {[], :erlang.term_to_binary(nil)}
      pid ->
        send(pid, {:pop_infeed, self()})
        receive do
          {:infeed_data, list} ->
            binaries = Enum.map(list, &elem(&1, 0))
            {binaries, :erlang.term_to_binary(nil)}
        after
          0 -> {[], :erlang.term_to_binary(nil)}
        end
    end
  end

  defp handle_tensor(binary, %{pending_hook: %{remaining: n} = pending} = state) when n > 0 do
    new_pending = %{pending | remaining: n - 1, acc: [binary | pending.acc]}

    if new_pending.remaining == 0 do
      # Reverse to preserve original order
      buffers = Enum.reverse(new_pending.acc)
      new_pending.fun.(EXLA.Defn.Buffers.to_nx!(buffers, new_pending.template))
      Map.delete(state, :pending_hook)
    else
      %{state | pending_hook: new_pending}
    end
  end

  defp handle_tensor(binary, %{outfeed_reader: {pid, ref, [_ | rest]} = reader} = state) do
    send(pid, {ref, binary})

    if rest == [] do
      Map.delete(state, :outfeed_reader)
    else
      %{state | outfeed_reader: {pid, ref, rest}}
    end
  end

  defp handle_tensor(_binary, state), do: state

  defp apply_hook(parent, ref, length, fun, template) do
    buffers =
      for _ <- 1..length//1 do
        receive do
          {^ref, binary} -> binary
        end
      end

    send(parent, ref)
    fun.(EXLA.Defn.Buffers.to_nx!(buffers, template))
  end
end

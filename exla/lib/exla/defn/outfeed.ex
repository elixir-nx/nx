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
            infeeds: [],
            mesh: nil

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
  Sets the mesh for sharding support.
  """
  def with_mesh(%Outfeed{} = outfeed, mesh), do: %{outfeed | mesh: mesh}

  @doc """
  Adds an infeed hook.
  """
  def add_infeeds(%Outfeed{} = outfeed, builder, entries) do
    %{compiled_hooks: compiled_hooks, token: token} = outfeed

    # Reversed because higher depth comes first
    {infeeds, {compiled_hooks, token}} =
      entries
      |> List.keysort(1, :desc)
      |> Enum.map_reduce({compiled_hooks, token}, fn
        {pos, _, typespec}, {compiled_hooks, token} ->
          next_flag = next_hook(compiled_hooks)
          compiled_hooks = Map.put(compiled_hooks, next_flag, {:infeed, pos, typespec})

          token = Value.outfeed(Value.constant(builder, [next_flag], flag_typespec()), token)
          {token, [input]} = Value.infeed(token, [typespec])

          {{pos, input}, {compiled_hooks, token}}
      end)

    %{outfeed | compiled_hooks: compiled_hooks, token: token, infeeds: infeeds}
  end

  defp flag_typespec(), do: EXLA.Typespec.tensor({:u, 16}, {})

  @doc """
  Adds a function hook if it has a callback defined for it.
  """
  def maybe_add_function_hook(%Outfeed{} = outfeed, builder, tuple, name, expr, metadata \\ %{}) do
    cond do
      name in outfeed.used_hooks ->
        {outfeed, flag, typespecs} = outfeed_flat_tuple(outfeed, builder, tuple)

        put_in(
          outfeed.compiled_hooks[flag],
          {:function, typespecs, name, Nx.to_template(expr), metadata}
        )

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

  def close(%Outfeed{} = outfeed, %Function{} = builder) when will_outfeed(outfeed),
    do:
      update_in(
        outfeed.token,
        &Value.outfeed(Value.constant(builder, [0], flag_typespec()), &1, outfeed.mesh)
      )

  def close(%Outfeed{} = outfeed, _builder),
    do: outfeed

  defp outfeed_flat_tuple(%Outfeed{token: token, compiled_hooks: ch, mesh: mesh} = outfeed, builder, tuple) do
    flag = next_hook(ch)
    token = Value.outfeed(Value.constant(builder, [flag], flag_typespec()), token, mesh)
    typespecs = Enum.map(tuple, &Value.get_typespec/1)

    token =
      Enum.reduce(tuple, token, fn elem, token ->
        Value.outfeed(elem, token, mesh)
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

  For SPMD execution (num_partitions > 1), spawns a listener for each device.
  """
  def start_child(
        %EXLA.Executable{} = executable,
        %Outfeed{} = outfeed,
        group_leader,
        infeeds \\ %{}
      ) do
    %{
      client: client,
      device_id: device_id,
      outfeed_device_id: outfeed_device_id,
      num_partitions: num_partitions
    } = executable

    %{compiled_hooks: compiled_hooks, default_hooks: default_hooks, user_hooks: user_hooks} =
      outfeed

    hooks = Map.merge(default_hooks, user_hooks)

    # For SPMD execution, we need to spawn a listener for each device
    if device_id == -1 and num_partitions > 1 do
      # Start a coordinator task that manages multiple device listeners
      Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
        init_spmd(
          client,
          num_partitions,
          hooks,
          compiled_hooks,
          infeeds,
          group_leader
        )
      end)
    else
      # Single device execution
      Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
        init(client, outfeed_device_id, hooks, compiled_hooks, infeeds, group_leader)
      end)
    end
  end

  defp init(client, device_id, hooks, compiled_hooks, infeeds, group_leader) do
    Process.flag(:trap_exit, true)
    # Copy the group leader so we report to the proper device
    Process.group_leader(self(), group_leader)
    ref = make_ref()
    typespec = EXLA.Typespec.tensor({:u, 16}, {})
    loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds)
  end

  defp init_spmd(client, num_partitions, hooks, compiled_hooks, infeeds, group_leader) do
    Process.flag(:trap_exit, true)
    Process.group_leader(self(), group_leader)

    # Spawn a listener task for each device
    device_tasks =
      for device_id <- 0..(num_partitions - 1) do
        Task.async(fn ->
          ref = make_ref()
          typespec = EXLA.Typespec.tensor({:u, 16}, {})

          # Filter hooks based on partition metadata.
          # For SPMD, hooks execute on all partitions by default, unless
          # the hook has :partitions metadata specifying which partitions to run on.
          filtered_hooks = filter_hooks_for_partition(hooks, compiled_hooks, device_id)

          loop(client, device_id, ref, typespec, filtered_hooks, compiled_hooks, infeeds)
        end)
      end

    # Wait for all device listeners to complete
    Enum.each(device_tasks, fn task ->
      Task.await(task, :infinity)
    end)
  end

  defp filter_hooks_for_partition(hooks, compiled_hooks, device_id) do
    # Check each hook to see if it has partition restrictions
    Map.new(hooks, fn {name, fun} ->
      # Find the compiled hook to check its metadata
      should_run =
        Enum.any?(compiled_hooks, fn
          {_flag, {:function, _typespecs, ^name, _template, %{partitions: partitions}}} ->
            device_id in partitions

          {_flag, {:function, _typespecs, ^name, _template, _metadata}} ->
            true

          {_flag, {:function, _typespecs, ^name, _template}} ->
            true

          _ ->
            false
        end)

      filtered_fun = if should_run, do: fun, else: fn _tensor -> :ok end
      {name, filtered_fun}
    end)
  end

  defp loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds) do
    :ok = EXLA.Client.from_outfeed(client, device_id, [typespec], self(), ref)

    receive do
      {^ref, <<0::native-unsigned-16>>} ->
        :ok

      {^ref, <<flag::native-unsigned-16>>} ->
        case Map.fetch!(compiled_hooks, flag) do
          {:infeed, index, data_typespec} ->
            data =
              case Map.fetch!(infeeds, index) do
                %EXLA.DeviceBuffer{} = buffer -> EXLA.DeviceBuffer.read(buffer)
                %EXLA.BinaryBuffer{data: data} -> data
              end

            EXLA.Client.to_infeed(client, device_id, [{data, data_typespec}])
            loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds)

          {:function, typespecs, name, template, _metadata} ->
            fun = Map.fetch!(hooks, name)
            length = length(typespecs)
            parent = self()
            ref = make_ref()
            pid = spawn(fn -> apply_hook(parent, ref, length, fun, template, device_id) end)
            :ok = EXLA.Client.from_outfeed(client, device_id, typespecs, pid, ref)

            receive do
              ^ref -> loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds)
            end

          # Legacy format without metadata (for backward compatibility)
          {:function, typespecs, name, template} ->
            fun = Map.fetch!(hooks, name)
            length = length(typespecs)
            parent = self()
            ref = make_ref()
            pid = spawn(fn -> apply_hook(parent, ref, length, fun, template, device_id) end)
            :ok = EXLA.Client.from_outfeed(client, device_id, typespecs, pid, ref)

            receive do
              ^ref -> loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds)
            end
        end
    end
  end

  defp apply_hook(parent, ref, length, fun, template, device_id) do
    buffers =
      for _ <- 1..length//1 do
        receive do
          {^ref, binary} -> binary
        end
      end

    send(parent, ref)
    tensor = EXLA.Defn.Buffers.to_nx!(buffers, template)

    # Store device_id in process dictionary so hooks can access it via Process.get(:exla_hook_device_id)
    Process.put(:exla_hook_device_id, device_id)

    try do
      fun.(tensor)
    after
      Process.delete(:exla_hook_device_id)
    end
  end
end

defmodule EXLA.Defn.Outfeed do
  @moduledoc false

  alias EXLA.Defn.Outfeed
  alias Nx.Defn.{Expr, Tree, Composite}
  alias Nx.Tensor, as: T

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

    # Reversed because higher depth comes first
    {infeeds, {compiled_hooks, token}} =
      entries
      |> List.keysort(1, :desc)
      |> Enum.map_reduce({compiled_hooks, token}, fn {pos, _, shape}, {compiled_hooks, token} ->
        next_flag = next_hook(compiled_hooks)
        compiled_hooks = Map.put(compiled_hooks, next_flag, {:infeed, pos, shape})

        token = EXLA.Op.outfeed(EXLA.Op.constant_r0(builder, next_flag, {:u, 16}), token)
        infeed = EXLA.Op.infeed(token, shape)
        input = EXLA.Op.get_tuple_element(infeed, 0)
        token = EXLA.Op.get_tuple_element(infeed, 1)

        {{pos, input}, {compiled_hooks, token}}
      end)

    %{outfeed | compiled_hooks: compiled_hooks, token: token, infeeds: infeeds}
  end

  @doc """
  Adds a function hook if it has a callback defined for it.
  """
  def maybe_add_function_hook(%Outfeed{} = outfeed, builder, tuple, name, expr) do
    cond do
      name in outfeed.used_hooks ->
        {outfeed, flag, shapes} = outfeed_flat_tuple(outfeed, builder, tuple)
        put_in(outfeed.compiled_hooks[flag], {:function, shapes, name, Nx.to_template(expr)})

      outfeed.token ->
        outfeed

      true ->
        raise "hooks are not supported inside #{builder.name}"
    end
  end

  @doc """
  Adds a stream hook.

  Used by streams. Only one is allowed. Requires configuration.
  """
  def add_stream_hook(%Outfeed{} = outfeed, builder, tuple) do
    {outfeed, flag, shapes} = outfeed_flat_tuple(outfeed, builder, tuple)
    # We don't know the pid+ref pair for the stream, so we store it
    # under a special key called :stream and revert to the flag once configured
    put_in(outfeed.compiled_hooks[:stream], {flag, shapes})
  end

  def configure_stream_hook(%Outfeed{} = outfeed, pid, ref) when is_pid(pid) do
    {{flag, shapes}, outfeed} = pop_in(outfeed.compiled_hooks[:stream])
    {shapes, put_in(outfeed.compiled_hooks[flag], {:stream, shapes, pid, ref})}
  end

  @doc """
  Closes the outfeed at the end of a pipeline.

  Note the outfeed may be closed before the computation finishes.
  """
  def close(%Outfeed{} = outfeed, builder) when will_outfeed(outfeed),
    do: update_in(outfeed.token, &EXLA.Op.outfeed(EXLA.Op.constant_r0(builder, 0, {:u, 16}), &1))

  def close(%Outfeed{} = outfeed, _builder),
    do: outfeed

  defp outfeed_flat_tuple(%Outfeed{token: token, compiled_hooks: ch} = outfeed, builder, tuple) do
    flag = next_hook(ch)
    token = EXLA.Op.outfeed(EXLA.Op.constant_r0(builder, flag, {:u, 16}), token)
    %EXLA.Shape{dims: {size}, dtype: {:tuple, shapes}} = EXLA.Op.get_shape(tuple)

    token =
      Enum.reduce(1..size//1, token, fn pos, token ->
        EXLA.Op.outfeed(EXLA.Op.get_tuple_element(tuple, pos - 1), token)
      end)

    {%{outfeed | token: token}, flag, shapes}
  end

  # The index 0 is served for closing streams
  defp next_hook(compiled_hooks), do: map_size(compiled_hooks) + 1

  ## Process API

  @doc """
  Receives a client, device_id, and mappings of u16 to
  `{shapes, {pid, ref} | {fun, template}}` pairs to
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

    Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
      init(client, device_id, hooks, compiled_hooks, infeeds, group_leader)
    end)
  end

  defp init(client, device_id, hooks, compiled_hooks, infeeds, group_leader) do
    Process.flag(:trap_exit, true)
    # Copy the group leader so we report to the proper device
    Process.group_leader(self(), group_leader)
    ref = make_ref()
    shape = EXLA.Shape.make_shape({:u, 16}, {})
    loop(client, device_id, ref, shape, hooks, compiled_hooks, infeeds)
  end

  defp loop(client, device_id, ref, shape, hooks, compiled_hooks, infeeds) do
    :ok = EXLA.Client.from_outfeed(client, device_id, [shape], self(), ref)

    receive do
      {^ref, <<0::native-unsigned-16>>} ->
        :ok

      {^ref, <<flag::native-unsigned-16>>} ->
        case Map.fetch!(compiled_hooks, flag) do
          {:infeed, index, data_shape} ->
            data =
              case Map.fetch!(infeeds, index) do
                %EXLA.DeviceBuffer{} = buffer -> EXLA.DeviceBuffer.read(buffer)
                %EXLA.BinaryBuffer{data: data} -> data
              end

            EXLA.Client.to_infeed(client, device_id, [{data, data_shape}])
            loop(client, device_id, ref, shape, hooks, compiled_hooks, infeeds)

          {:stream, shapes, recv_pid, recv_ref} ->
            :ok = EXLA.Client.from_outfeed(client, device_id, shapes, recv_pid, recv_ref)
            loop(client, device_id, ref, shape, hooks, compiled_hooks, infeeds)

          {:function, shapes, name, template} ->
            fun = Map.fetch!(hooks, name)
            length = length(shapes)
            parent = self()
            ref = make_ref()
            pid = spawn(fn -> apply_hook(parent, ref, length, fun, template) end)
            :ok = EXLA.Client.from_outfeed(client, device_id, shapes, pid, ref)

            receive do
              ^ref -> loop(client, device_id, ref, shape, hooks, compiled_hooks, infeeds)
            end
        end
    end
  end

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

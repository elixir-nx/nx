defmodule EXLA.Defn.Outfeed do
  @moduledoc false
  require Logger

  alias EXLA.Defn.Outfeed
  alias Nx.Defn.{Expr, Tree, Composite}
  alias Nx.Tensor, as: T

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value

  defstruct user_hooks: %{},
            default_hooks: %{},
            used_hooks: [],
            compiled_hooks: %{},
            callbacks: %{},
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

  defguard will_outfeed(outfeed) when outfeed.compiled_hooks != %{} or outfeed.callbacks != %{}

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
  Registers a runtime_call callback in the outfeed struct.

  This stores the callback function, output template, and argument template
  so they can be passed to the outfeed task at runtime. The outfeed task
  handles `:exla_runtime_call` messages from the native bridge.
  """
  def register_callback(%Outfeed{} = outfeed, id, fun, out_template, arg_template)
      when is_function(fun) do
    put_in(outfeed.callbacks[id], {fun, out_template, arg_template})
  end

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

  def close(%Outfeed{compiled_hooks: ch} = outfeed, %Function{} = builder) when ch != %{},
    do:
      update_in(outfeed.token, &Value.outfeed(Value.constant(builder, [0], flag_typespec()), &1))

  def close(%Outfeed{} = outfeed, _builder),
    do: outfeed

  defp outfeed_flat_tuple(%Outfeed{token: token, compiled_hooks: ch} = outfeed, builder, tuple) do
    flag = next_hook(ch)
    token = Value.outfeed(Value.constant(builder, [flag], flag_typespec()), token)
    typespecs = Enum.map(tuple, &Value.get_typespec/1)

    token =
      Enum.reduce(tuple, token, fn elem, token ->
        Value.outfeed(elem, token)
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

  When callbacks are present, the task also registers itself
  as the runtime callback dispatcher via the NIF bridge, handling
  `:exla_runtime_call` messages from the native side.
  """
  def start_child(
        %EXLA.Executable{} = executable,
        %Outfeed{} = outfeed,
        group_leader,
        infeeds \\ %{}
      ) do
    %{client: client, device_id: device_id} = executable

    %{
      compiled_hooks: compiled_hooks,
      default_hooks: default_hooks,
      user_hooks: user_hooks,
      callbacks: callbacks
    } = outfeed

    hooks = Map.merge(default_hooks, user_hooks)

    Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
      init(client, device_id, hooks, compiled_hooks, callbacks, infeeds, group_leader)
    end)
  end

  defp init(client, device_id, hooks, compiled_hooks, callbacks, infeeds, group_leader) do
    Process.flag(:trap_exit, true)
    # Copy the group leader so we report to the proper device
    Process.group_leader(self(), group_leader)

    callback_ids = Map.keys(callbacks)

    if compiled_hooks != %{} do
      # When both hooks and callbacks are present, the outfeed task blocks
      # in from_outfeed (a dirty IO NIF) and can't receive runtime_call
      # messages. Spawn a separate helper process for callback handling.
      callback_helper =
        if callback_ids != [] do
          pid = spawn_link(fn -> callback_only_loop(callbacks) end)
          EXLA.Defn.CallbackDispatcher.register(callback_ids, pid)
          pid
        end

      try do
        ref = make_ref()
        typespec = EXLA.Typespec.tensor({:u, 16}, {})
        outfeed_loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds)
      after
        if callback_helper do
          send(callback_helper, :done)
          EXLA.Defn.CallbackDispatcher.unregister(callback_ids, callback_helper)
        end
      end
    else
      # No outfeed hooks — only runtime_call messages.
      # This task handles them directly.
      if callback_ids != [] do
        EXLA.Defn.CallbackDispatcher.register(callback_ids, self())
      end

      try do
        callback_only_loop(callbacks)
      after
        if callback_ids != [] do
          EXLA.Defn.CallbackDispatcher.unregister(callback_ids, self())
        end
      end
    end
  end

  # Loop for outfeed hooks. When callbacks are also present, they're
  # handled by a separate helper process (see init/7) because
  # from_outfeed is a blocking dirty IO NIF — this process can't
  # receive runtime_call messages while blocked in the NIF.
  defp outfeed_loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds) do
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
            outfeed_loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds)

          {:function, typespecs, name, template} ->
            fun = Map.fetch!(hooks, name)
            length = length(typespecs)
            parent = self()
            ref = make_ref()
            pid = spawn(fn -> apply_hook(parent, ref, length, fun, template) end)
            :ok = EXLA.Client.from_outfeed(client, device_id, typespecs, pid, ref)

            receive do
              ^ref ->
                outfeed_loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds)
            end
        end
    end
  end

  # Loop for when we only have runtime_call callbacks (no outfeed hooks).
  # The caller sends :done when execution completes. Also handles EXIT
  # signals so the process doesn't hang if the parent crashes.
  defp callback_only_loop(callbacks) do
    receive do
      {:exla_runtime_call, callback_id, args_spec, reply_tag} ->
        handle_runtime_call(callbacks, callback_id, args_spec, reply_tag)
        callback_only_loop(callbacks)

      :done ->
        :ok

      {:EXIT, _pid, _reason} ->
        :ok
    end
  end

  defp handle_runtime_call(callbacks, callback_id, args_spec, reply_tag) do
    {status, result} =
      try do
        case Map.fetch(callbacks, callback_id) do
          {:ok, {fun, out_template, arg_template}} ->
            args_spec
            |> decode_callback_args(arg_template)
            |> run_callback(fun, out_template)
            |> encode_callback_reply()

          :error ->
            Logger.error(
              "EXLA runtime_call received callback id #{inspect(callback_id)} that is not registered"
            )

            encode_callback_reply({:error, :unknown_callback})
        end
      catch
        kind, reason ->
          formatted = Exception.format(kind, reason, __STACKTRACE__)

          encode_callback_reply(
            {:error, {:runtime_error, "Elixir callback crashed: #{formatted}"}}
          )
      end

    try do
      EXLA.NIF.runtime_callback_reply(reply_tag, status, result)
    rescue
      _ ->
        Logger.error("EXLA runtime_call failed to send reply for tag #{inspect(reply_tag)}")
    end
  end

  defp run_callback({:error, reason}, _fun, _out_template), do: {:error, reason}

  defp run_callback({:ok, tensor_args}, fun, out_template) do
    result =
      try do
        fun.(tensor_args)
      rescue
        exception ->
          {:error, {:exception, exception, __STACKTRACE__}}
      catch
        kind, reason ->
          {:error, {kind, reason}}
      end

    case result do
      {:error, _} = error ->
        error

      value ->
        if Nx.compatible?(value, out_template) do
          {:ok, value}
        else
          {:error, {:shape_mismatch, value, out_template}}
        end
    end
  end

  defp decode_callback_args(args_spec, arg_template) when is_list(args_spec) do
    {container, remaining} =
      Nx.Defn.Composite.traverse(arg_template, args_spec, fn
        %Nx.Tensor{} = template, [{bin, {type, shape_list}} | rest] ->
          decoded =
            bin
            |> Nx.from_binary(type)
            |> Nx.reshape(List.to_tuple(shape_list))

          if Nx.compatible?(decoded, template) do
            {decoded, rest}
          else
            throw({:error, {:shape_mismatch, decoded, template}})
          end

        other, acc ->
          {other, acc}
      end)

    case remaining do
      [] -> {:ok, container}
      _ -> {:error, {:invalid_args_spec, :extra_values}}
    end
  catch
    {:error, reason} -> {:error, reason}
  end

  defp decode_callback_args(_other, _arg_template),
    do: {:error, {:invalid_args_spec, :bad_format}}

  defp encode_callback_reply({:ok, value}) do
    binaries =
      [value]
      |> Nx.Defn.Composite.flatten_list()
      |> Enum.map(&Nx.to_binary/1)

    {:ok, binaries}
  end

  defp encode_callback_reply({:error, {:shape_mismatch, left, right}}) do
    msg =
      "expected the runtime_call function to match the given output template " <>
        "#{inspect(right)}, got: #{inspect(left)}"

    {:error, {:argument_error, msg}}
  end

  defp encode_callback_reply({:error, {:exception, exception, _stack}}) do
    {:error, {:runtime_error, "Elixir callback raised: #{Exception.message(exception)}"}}
  end

  defp encode_callback_reply({:error, {:runtime_error, msg}}) do
    {:error, {:runtime_error, msg}}
  end

  defp encode_callback_reply({:error, :unknown_callback}) do
    {:error, {:runtime_error, "unknown EXLA runtime_call callback id"}}
  end

  defp encode_callback_reply({:error, {kind, reason}}) do
    {:error, {:runtime_error, "Elixir callback #{kind}: #{inspect(reason)}"}}
  end

  defp encode_callback_reply({:error, reason}) do
    {:error, {:runtime_error, "Elixir callback error: #{inspect(reason)}"}}
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

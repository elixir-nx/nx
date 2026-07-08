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
            infeed_flags: %{},
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

  defp used_hooks(%Expr{op: :hook, args: [_, _, spec, _, _]}, hooks) do
    case spec do
      {:named, name, callback} -> Map.put(hooks, name, callback)
      {:fn, _} -> hooks
    end
  end

  defp used_hooks(_, hooks),
    do: hooks

  ## Struct API

  defguard has_infeed_flags(outfeed) when outfeed.infeed_flags != %{}
  defguard has_callbacks(outfeed) when map_size(outfeed.callbacks) > 0

  @doc """
  An empty outfeed to be used when not outfeeding is supported.
  """
  def empty do
    %Outfeed{}
  end

  @doc """
  An outfeed struct to track the need for outfeeds during compilation.
  """
  def new(user_hooks, default_hooks)
      when is_map(user_hooks) and is_map(default_hooks) do
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
  Sets the user hooks for callbacks.
  """
  def with_user_hooks(%Outfeed{} = outfeed, user_hooks),
    do: %{outfeed | user_hooks: user_hooks}

  @doc """
  Sets the token to outfeed.
  """
  def with_token(%Outfeed{} = outfeed, token), do: %{outfeed | token: token}

  @doc """
  Registers a host callback to outfeed.

  `invoke` is either a callback spec (`{:fn, fun}` or `{:named, name, callback}`) for
  side-effect-only callbacks, or a function for callbacks that return tensors.
  When `out_template` is `nil`, the callback is invoked for side effects only and
  the native reply is an empty list.
  """
  def add_callback(%Outfeed{} = outfeed, {id, invoke, out_template, arg_template}) do
    add_callback(outfeed, {id, invoke, out_template, arg_template, nil})
  end

  def add_callback(
        %Outfeed{callbacks: callbacks} = outfeed,
        {id, invoke, out_template, arg_template, opts}
      ) do
    callback = {invoke, out_template, arg_template, opts}
    %{outfeed | callbacks: Map.put(callbacks, id, callback)}
  end

  @doc """
  Adds lazy-transfer infeed flags to the computation.
  """
  def add_infeeds(%Outfeed{} = outfeed, builder, entries) do
    %{infeed_flags: infeed_flags, token: token} = outfeed

    # Reversed because higher depth comes first
    {infeeds, {infeed_flags, token}} =
      entries
      |> List.keysort(1, :desc)
      |> Enum.map_reduce({infeed_flags, token}, fn
        {pos, _, typespec}, {infeed_flags, token} ->
          next_flag = next_infeed_flag(infeed_flags)
          infeed_flags = Map.put(infeed_flags, next_flag, {:infeed, pos, typespec})

          token = Value.outfeed(Value.constant(builder, [next_flag], flag_typespec()), token)
          {token, [input]} = Value.infeed(token, [typespec])

          {{pos, input}, {infeed_flags, token}}
      end)

    %{outfeed | infeed_flags: infeed_flags, token: token, infeeds: infeeds}
  end

  defp flag_typespec(), do: EXLA.Typespec.tensor({:u, 16}, {})

  @doc """
  Closes the outfeed at the end of a pipeline.

  Note the outfeed may be closed before the computation finishes.
  """
  def close(outfeed, builder)

  def close(
        %Outfeed{token: token, infeed_flags: infeed_flags} = outfeed,
        %Function{} = builder
      )
      when not is_nil(token) do
    if active_infeed_flags?(infeed_flags) do
      %{outfeed | token: Value.outfeed(Value.constant(builder, [0], flag_typespec()), token)}
    else
      outfeed
    end
  end

  def close(%Outfeed{} = outfeed, _builder),
    do: outfeed

  defp active_infeed_flags?(infeed_flags) do
    Enum.any?(infeed_flags, fn
      {_key, {:infeed, _, _}} -> true
      _ -> false
    end)
  end

  # The index 0 is reserved for closing the outfeed stream
  defp next_infeed_flag(infeed_flags) do
    infeed_flags
    |> Map.keys()
    |> Enum.filter(&is_integer/1)
    |> Enum.max(fn -> 0 end)
    |> Kernel.+(1)
  end

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
        infeeds
      ) do
    %{client: client, device_id: device_id} = executable

    %{
      infeed_flags: infeed_flags,
      default_hooks: default_hooks,
      user_hooks: user_hooks,
      callbacks: callbacks
    } =
      outfeed

    hooks = Map.merge(default_hooks, user_hooks)
    callbacks = resolve_callbacks(callbacks, hooks)

    Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
      init(client, device_id, infeed_flags, infeeds, callbacks, group_leader)
    end)
  end

  defp init(client, device_id, infeed_flags, infeeds, callbacks, group_leader) do
    Process.flag(:trap_exit, true)
    # Copy the group leader so we report to the proper device
    Process.group_leader(self(), group_leader)

    ref = make_ref()
    typespec = EXLA.Typespec.tensor({:u, 16}, {})

    loop(client, device_id, ref, typespec, infeed_flags, infeeds, callbacks)
  end

  defp loop(client, device_id, ref, typespec, infeed_flags, infeeds, callbacks) do
    if active_infeed_flags?(infeed_flags) do
      :ok = EXLA.Client.from_outfeed(client, device_id, [typespec], self(), ref)
    end

    receive do
      {^ref, <<0::native-unsigned-16>>} ->
        # Outfeed is done, now we wait for the computation to finish
        loop(
          client,
          device_id,
          ref,
          typespec,
          drop_infeed_flags(infeed_flags),
          infeeds,
          callbacks
        )

      {^ref, <<flag::native-unsigned-16>>} ->
        case Map.fetch!(infeed_flags, flag) do
          {:infeed, index, data_typespec} ->
            data =
              case Map.fetch!(infeeds, index) do
                %EXLA.DeviceBuffer{} = buffer -> EXLA.DeviceBuffer.read(buffer)
                %EXLA.BinaryBuffer{data: data} -> data
              end

            EXLA.Client.to_infeed(client, device_id, [{data, data_typespec}])

            loop(client, device_id, ref, typespec, infeed_flags, infeeds, callbacks)
        end

      {:exla_runtime_call, callback_id, args_spec, reply_tag} ->
        send_callback_reply(callbacks, callback_id, args_spec, reply_tag)
        loop(client, device_id, ref, typespec, infeed_flags, infeeds, callbacks)

      :stop ->
        :ok

      other ->
        Logger.debug("EXLA.Outfeed ignoring unexpected message: #{inspect(other)}")
        loop(client, device_id, ref, typespec, infeed_flags, infeeds, callbacks)
    end
  end

  defp drop_infeed_flags(infeed_flags) do
    keys = for {k, _} <- infeed_flags, is_integer(k), do: k
    Map.drop(infeed_flags, keys)
  end

  defp resolve_callbacks(callbacks, hooks) do
    Map.new(callbacks, fn {id, {invoke, out_template, arg_template, opts}} ->
      {id, {resolve_invoke(invoke, hooks), out_template, arg_template, opts}}
    end)
  end

  defp resolve_invoke(fun, _hooks) when is_function(fun), do: fun

  defp resolve_invoke({:fn, fun}, _hooks), do: {:fn, fun}

  defp resolve_invoke({:named, name, callback}, hooks) do
    case hooks[name] || callback do
      nil -> {:named, name, nil}
      fun -> {:fn, fun}
    end
  end

  defp send_callback_reply(callbacks, callback_id, args_spec, reply_tag) do
    reply =
      try do
        with {:ok, callback} <- Map.fetch(callbacks, callback_id),
             {:ok, tensor_args} <- materialize_callback_args(callback, args_spec) do
          callback
          |> invoke_callback(tensor_args)
          |> encode_callback_reply()
        else
          :error ->
            Logger.error(
              "EXLA.Outfeed received callback id #{inspect(callback_id)} that is not registered"
            )

            encode_callback_reply({:error, :unknown_callback})

          {:error, _} = error ->
            error
        end
      rescue
        exception ->
          send(self(), :stop)
          {:error, {:exception, Exception.format(:error, exception, __STACKTRACE__)}}
      catch
        kind, reason ->
          send(self(), :stop)
          {:error, {kind, Exception.format(kind, reason, __STACKTRACE__)}}
      end

    deliver_native_reply(reply_tag, reply)
  end

  defp deliver_native_reply(reply_tag, reply) do
    try do
      case reply do
        {:ok, payload} ->
          EXLA.NIF.runtime_callback_reply(reply_tag, :ok, payload)

        {:error, {tag, reason}} ->
          EXLA.NIF.runtime_callback_reply(reply_tag, :error, {tag, reason})
      end
    rescue
      _ ->
        Logger.error(
          "EXLA.Outfeed failed to send callback reply to native for tag #{inspect(reply_tag)}"
        )
    end
  end

  defp invoke_callback({invoke, nil, _arg_template, _opts}, tensor_args) do
    invoke_side_effect(invoke, tensor_args)
  end

  defp invoke_callback({fun, out_template, _arg_template, opts}, tensor_args)
       when is_function(fun) do
    run_runtime_callback({:ok, tensor_args}, fun, out_template, opts)
  end

  defp invoke_side_effect({:fn, fun}, tensor_args) when is_function(fun, 1) do
    run_side_effect(fun, tensor_args)
  end

  defp invoke_side_effect({:named, _, nil}, _tensor_args) do
    {:ok, []}
  end

  defp run_side_effect(fun, tensor_args) when is_function(fun, 1) do
    try do
      fun.(tensor_args)
      {:ok, []}
    rescue
      exception ->
        {:error, {:exception, Exception.format(:error, exception, __STACKTRACE__)}}
    catch
      kind, reason ->
        {:error, {kind, Exception.format(kind, reason, __STACKTRACE__)}}
    end
  end

  defp run_runtime_callback({:ok, tensor_args}, fun, out_template, opts) do
    result =
      try do
        if opts do
          fun.(tensor_args, opts)
        else
          fun.(tensor_args)
        end
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

  defp materialize_callback_args({_, _, arg_template, _}, args_spec) when is_list(args_spec) do
    materialize_callback_tensors(arg_template, args_spec)
  catch
    {:error, reason} ->
      {:error, {:decode_failed, ArgumentError.exception("invalid args_spec #{inspect(reason)}")}}
  end

  defp materialize_callback_args({_, _, _arg_template, _}, other) do
    {:error, {:invalid_args_spec, other}}
  end

  defp encode_callback_reply({:ok, []}), do: {:ok, []}
  defp encode_callback_reply({:ok, value}), do: {:ok, encode_callback_outputs(value)}

  defp encode_callback_reply({:error, {:shape_mismatch, left, right}}) do
    msg =
      "expected the runtime_call function to match the given output template " <>
        "#{inspect(right)}, got: #{inspect(left)}"

    raise ArgumentError.exception(msg)
  end

  defp encode_callback_reply({:error, {:invalid_result, left, right}}) do
    msg =
      "expected the runtime_call function to return a value compatible with the output " <>
        "template #{inspect(right)}, got: #{inspect(left)}"

    raise ArgumentError.exception(msg)
  end

  defp encode_callback_reply({:error, {:decode_failed, exception}}) do
    msg = Exception.message(exception)
    msg = "failed to decode Elixir callback arguments: #{msg}"
    raise ArgumentError.exception(msg)
  end

  defp encode_callback_reply({:error, {:invalid_args_spec, other}}) do
    msg = "invalid args_spec for Elixir callback: #{inspect(other)}"
    raise ArgumentError.exception(msg)
  end

  defp encode_callback_reply({:error, :unknown_callback}) do
    msg = "unknown EXLA runtime_call callback id"
    raise RuntimeError.exception(msg)
  end

  defp encode_callback_reply({:error, {:exception, exception, _stack}}) do
    raise exception
  end

  defp encode_callback_reply({:error, {kind, reason}}) when is_binary(reason) do
    msg = "Elixir callback #{kind}: #{reason}"
    raise RuntimeError.exception(msg)
  end

  defp encode_callback_reply({:error, {kind, reason}}) do
    msg = "Elixir callback #{kind}: #{inspect(reason)}"
    raise RuntimeError.exception(msg)
  end

  defp encode_callback_reply({:error, reason}) do
    msg = "Elixir callback error: #{inspect(reason)}"
    raise RuntimeError.exception(msg)
  end

  defp maybe_revectorize(decoded, %Nx.Tensor{vectorized_axes: axes}) when axes != [] do
    Nx.vectorize(decoded, axes)
  end

  defp maybe_revectorize(decoded, _template), do: decoded

  defp materialize_callback_tensors(arg_template, args_spec) do
    {container, remaining} =
      Composite.traverse(arg_template, args_spec, fn
        %Nx.Tensor{} = template, [{bin, {type, shape_list}} | rest] ->
          decoded =
            bin
            |> Nx.from_binary(type)
            |> Nx.reshape(List.to_tuple(shape_list))
            |> maybe_revectorize(template)

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
    :throw, {:error, _} = error -> error
  end

  defp encode_callback_outputs(container) do
    [container]
    |> Composite.flatten_list()
    |> Enum.map(&Nx.to_binary/1)
  end
end

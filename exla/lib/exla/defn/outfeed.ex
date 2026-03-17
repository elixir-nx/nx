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
            token: nil,
            infeeds: [],
            runtime_callbacks: %{}

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
  defguard has_runtime_calls(outfeed) when outfeed.runtime_callbacks != %{}

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
  Adds a runtime callback to outfeed.
  """
  def add_runtime_callback(
        %Outfeed{runtime_callbacks: runtime_callbacks} = outfeed,
        {id, fun, out_template, arg_template}
      ) do
    callback = {fun, out_template, arg_template, nil}
    %{outfeed | runtime_callbacks: Map.put(runtime_callbacks, id, callback)}
  end

  def add_runtime_callback(
        %Outfeed{runtime_callbacks: runtime_callbacks} = outfeed,
        {id, fun, out_template, arg_template, opts}
      ) do
    callback = {fun, out_template, arg_template, opts}
    %{outfeed | runtime_callbacks: Map.put(runtime_callbacks, id, callback)}
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

  def close(%Outfeed{} = outfeed, %Function{} = builder)
      when will_outfeed(outfeed),
      do:
        update_in(
          outfeed.token,
          &Value.outfeed(Value.constant(builder, [0], flag_typespec()), &1)
        )

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
  """
  def start_child(
        %EXLA.Executable{} = executable,
        %Outfeed{} = outfeed,
        group_leader,
        infeeds
      ) do
    %{client: client, device_id: device_id} = executable

    %{
      compiled_hooks: compiled_hooks,
      default_hooks: default_hooks,
      user_hooks: user_hooks,
      runtime_callbacks: runtime_callbacks
    } =
      outfeed

    hooks = Map.merge(default_hooks, user_hooks)

    Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
      init(client, device_id, hooks, compiled_hooks, infeeds, runtime_callbacks, group_leader)
    end)
  end

  defp init(client, device_id, hooks, compiled_hooks, infeeds, rt_callbacks, group_leader) do
    Process.flag(:trap_exit, true)
    # Copy the group leader so we report to the proper device
    Process.group_leader(self(), group_leader)

    ref = make_ref()
    typespec = EXLA.Typespec.tensor({:u, 16}, {})

    loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds, rt_callbacks)
  end

  defp loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds, rt_callbacks) do
    if compiled_hooks != %{} do
      # If we're not outfeeding, we only need to handle the runtime callback
      # and executable stop messaging
      :ok = EXLA.Client.from_outfeed(client, device_id, [typespec], self(), ref)
    end

    receive do
      {^ref, <<0::native-unsigned-16>>} ->
        # Outfeed is done, now we wait for the computation to finish
        loop(client, device_id, ref, typespec, hooks, %{}, infeeds, rt_callbacks)

      {^ref, <<flag::native-unsigned-16>>} ->
        case Map.fetch!(compiled_hooks, flag) do
          {:infeed, index, data_typespec} ->
            data =
              case Map.fetch!(infeeds, index) do
                %EXLA.DeviceBuffer{} = buffer -> EXLA.DeviceBuffer.read(buffer)
                %EXLA.BinaryBuffer{data: data} -> data
              end

            EXLA.Client.to_infeed(client, device_id, [{data, data_typespec}])
            loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds, rt_callbacks)

          {:function, typespecs, name, template} ->
            fun = Map.fetch!(hooks, name)
            length = length(typespecs)
            parent = self()
            ref = make_ref()
            pid = spawn(fn -> apply_hook(parent, ref, length, fun, template) end)
            :ok = EXLA.Client.from_outfeed(client, device_id, typespecs, pid, ref)

            receive do
              ^ref ->
                loop(
                  client,
                  device_id,
                  ref,
                  typespec,
                  hooks,
                  compiled_hooks,
                  infeeds,
                  rt_callbacks
                )
            end
        end

      {:exla_runtime_call, callback_id, args_spec, reply_tag} ->
        send_runtime_callback_reply(rt_callbacks, callback_id, args_spec, reply_tag)
        loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds, rt_callbacks)

      :stop ->
        :ok

      other ->
        Logger.debug("EXLA.Outfeed ignoring unexpected message: #{inspect(other)}")
        loop(client, device_id, ref, typespec, hooks, compiled_hooks, infeeds, rt_callbacks)
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

  defp send_runtime_callback_reply(runtime_callbacks, callback_id, args_spec, reply_tag) do
    reply =
      try do
        case Map.fetch(runtime_callbacks, callback_id) do
          {:ok, {fun, out_template, arg_template, opts}} ->
            args_spec
            |> decode_callback_args(arg_template)
            |> run_runtime_callback(fun, out_template, opts)
            |> encode_runtime_callback_reply()

          :error ->
            Logger.error(
              "EXLA.Outfeed received callback id #{inspect(callback_id)} that is not registered"
            )

            encode_runtime_callback_reply({:error, :unknown_callback})
        end
      rescue
        exception ->
          send(self(), :stop)
          {:error, {:exception, Exception.message(exception)}}
      catch
        kind, reason ->
          send(self(), :stop)
          {:error, {kind, format_runtime_callback_reason(reason)}}
      end

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

  defp format_runtime_callback_reason(reason) when is_binary(reason), do: reason
  defp format_runtime_callback_reason(reason), do: inspect(reason)

  defp run_runtime_callback({:error, reason}, _fun, _out_template, _opts), do: {:error, reason}

  defp run_runtime_callback({:ok, tensor_args}, fun, nil, opts) do
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

  defp decode_callback_args(args_spec, arg_template) when is_list(args_spec) do
    materialize_callback_args(arg_template, args_spec)
  catch
    {:error, reason} ->
      raise ArgumentError, "invalid args_spec #{inspect(reason)}"
  end

  defp decode_callback_args(other, _arg_template) do
    raise ArgumentError, "invalid args_spec #{inspect(other)}"
  end

  defp encode_runtime_callback_reply(:ok), do: {:ok, []}
  defp encode_runtime_callback_reply({:ok, value}), do: {:ok, encode_callback_outputs(value)}

  defp encode_runtime_callback_reply({:error, {:shape_mismatch, left, right}}) do
    msg =
      "expected the runtime_call function to match the given output template " <>
        "#{inspect(right)}, got: #{inspect(left)}"

    raise ArgumentError.exception(msg)
  end

  defp encode_runtime_callback_reply({:error, {:invalid_result, left, right}}) do
    msg =
      "expected the runtime_call function to return a value compatible with the output " <>
        "template #{inspect(right)}, got: #{inspect(left)}"

    raise ArgumentError.exception(msg)
  end

  defp encode_runtime_callback_reply({:error, {:decode_failed, exception}}) do
    msg = Exception.message(exception)
    msg = "failed to decode Elixir callback arguments: #{msg}"
    raise ArgumentError.exception(msg)
  end

  defp encode_runtime_callback_reply({:error, {:invalid_args_spec, other}}) do
    msg = "invalid args_spec for Elixir callback: #{inspect(other)}"
    raise ArgumentError.exception(msg)
  end

  defp encode_runtime_callback_reply({:error, :unknown_callback}) do
    msg = "unknown EXLA runtime_call callback id"
    raise RuntimeError.exception(msg)
  end

  defp encode_runtime_callback_reply({:error, {:exception, exception, _stack}}) do
    raise exception
  end

  defp encode_runtime_callback_reply({:error, {kind, reason}}) do
    msg = "Elixir callback #{kind}: #{inspect(reason)}"
    raise RuntimeError.exception(msg)
  end

  defp encode_runtime_callback_reply({:error, reason}) do
    msg = "Elixir callback error: #{inspect(reason)}"
    raise RuntimeError.exception(msg)
  end

  defp materialize_callback_args(arg_template, args_spec) do
    {container, remaining} =
      Composite.traverse(arg_template, args_spec, fn
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
  end

  defp encode_callback_outputs(container) do
    [container]
    |> Composite.flatten_list()
    |> Enum.map(&Nx.to_binary/1)
  end
end

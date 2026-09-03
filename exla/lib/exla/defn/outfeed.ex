defmodule EXLA.Defn.Outfeed do
  @moduledoc false

  require Logger

  alias EXLA.Defn.Outfeed
  alias Nx.Defn.{Expr, Tree, Composite}
  alias Nx.Tensor, as: T

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value

  defstruct user_io_calls: %{},
            default_io_calls: %{},
            used_io_call_names: [],
            infeed_flags: %{},
            callbacks: %{},
            token: nil,
            infeeds: []

  ## Functional API

  @doc """
  Computes used inputs by depth and used io_calls.
  """
  def used_inputs_and_io_calls(expr, force_inputs, lazy_transfers) do
    if lazy_transfers not in [:always, :never, :opt_in] do
      raise ArgumentError,
            ":lazy_transfers must be either :always or :never, got: #{inspect(lazy_transfers)}"
    end

    lazy? = lazy_transfers == :always
    inputs = Map.from_keys(force_inputs, nil)

    {_, used_inputs, used_io_calls} =
      Composite.reduce(expr, {%{}, inputs, %{}}, &used_inputs_and_io_calls(&1, &2, 0, lazy?))

    {used_inputs, used_io_calls}
  end

  defp used_inputs_and_io_calls(%T{data: %Expr{id: id} = expr} = t, acc, depth, lazy?) do
    {seen, inputs, io_calls} = acc

    case seen do
      %{^id => true} ->
        acc

      %{} ->
        depth = depth + 1
        seen = Map.put(seen, id, true)
        inputs = used_inputs(expr, inputs, depth, lazy?)
        io_calls = used_io_calls(expr, io_calls)
        acc = {seen, inputs, io_calls}

        t
        |> Tree.apply_args(acc, &{&1, used_inputs_and_io_calls(&1, &2, depth, lazy?)})
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

  defp used_io_calls(%Expr{op: :io_call, args: [_, _, spec, _, _]}, io_calls) do
    case spec do
      {:named, name, callback} -> Map.put(io_calls, name, callback)
      {:fn, _} -> io_calls
    end
  end

  defp used_io_calls(_, io_calls),
    do: io_calls

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
  def new(user_io_calls, default_io_calls)
      when is_map(user_io_calls) and is_map(default_io_calls) do
    # Io calls with default callbacks or user callbacks are part of the cache key
    used_io_call_names =
      Enum.sort(for {k, v} <- default_io_calls, v != nil or Map.has_key?(user_io_calls, k), do: k)

    # We don't store the user io_calls yet, because we don't want them to be cached
    %Outfeed{
      default_io_calls: default_io_calls,
      used_io_call_names: used_io_call_names
    }
  end

  @doc """
  Sets the user io_calls to outfeed.
  """
  def with_user_io_calls(%Outfeed{} = outfeed, user_io_calls),
    do: %{outfeed | user_io_calls: user_io_calls}

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
      default_io_calls: default_io_calls,
      user_io_calls: user_io_calls,
      callbacks: callbacks
    } =
      outfeed

    io_calls = Map.merge(default_io_calls, user_io_calls)
    callbacks = resolve_callbacks(callbacks, io_calls)

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

    loop(client, device_id, ref, typespec, infeed_flags, infeeds, callbacks, nil)
  end

  defp loop(
         client,
         device_id,
         ref,
         typespec,
         infeed_flags,
         infeeds,
         callbacks,
         callback_error
       ) do
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
          callbacks,
          callback_error
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

            loop(
              client,
              device_id,
              ref,
              typespec,
              infeed_flags,
              infeeds,
              callbacks,
              callback_error
            )
        end

      {:exla_runtime_call, callback_id, args_spec, reply_tag} ->
        error = send_callback_reply(callbacks, callback_id, args_spec, reply_tag)

        loop(
          client,
          device_id,
          ref,
          typespec,
          infeed_flags,
          infeeds,
          callbacks,
          callback_error || error
        )

      :stop ->
        stop_with_callback_error(callback_error)

      other ->
        Logger.debug("EXLA.Outfeed ignoring unexpected message: #{inspect(other)}")

        loop(
          client,
          device_id,
          ref,
          typespec,
          infeed_flags,
          infeeds,
          callbacks,
          callback_error
        )
    end
  end

  defp stop_with_callback_error({kind, reason, stacktrace}) do
    exit(
      {:shutdown, %EXLA.Defn.CallbackError{kind: kind, reason: reason, stacktrace: stacktrace}}
    )
  end

  defp stop_with_callback_error(nil), do: :ok

  defp drop_infeed_flags(infeed_flags) do
    keys = for {k, _} <- infeed_flags, is_integer(k), do: k
    Map.drop(infeed_flags, keys)
  end

  defp resolve_callbacks(callbacks, io_calls) do
    Map.new(callbacks, fn {id, {invoke, out_template, arg_template, opts}} ->
      {id, {resolve_invoke(invoke, io_calls), out_template, arg_template, opts}}
    end)
  end

  defp resolve_invoke(fun, _io_calls) when is_function(fun), do: fun

  defp resolve_invoke({:fn, fun}, _io_calls), do: {:fn, fun}

  defp resolve_invoke({:named, name, callback}, io_calls) do
    case io_calls[name] || callback do
      nil -> {:named, name, nil}
      fun -> {:fn, fun}
    end
  end

  defp send_callback_reply(callbacks, callback_id, args_spec, reply_tag) do
    {reply, callback_error} =
      try do
        with {:ok, callback} <- fetch_callback(callbacks, callback_id),
             {:ok, tensor_args} <- materialize_callback_args(callback, args_spec),
             {:ok, result} <- invoke_callback(callback, tensor_args) do
          {{:ok, encode_callback_outputs(result)}, nil}
        else
          {:error, exception} -> raise exception
        end
      catch
        kind, reason ->
          stacktrace = __STACKTRACE__

          exception =
            if kind == :error do
              Exception.normalize(kind, reason, stacktrace)
            else
              RuntimeError.exception(Exception.format(kind, reason, stacktrace))
            end

          Logger.error(Exception.message(exception))
          send(self(), :stop)
          {{:error, exception}, {kind, reason, stacktrace}}
      end

    deliver_native_reply(reply_tag, reply)
    callback_error
  end

  defp fetch_callback(callbacks, callback_id) do
    case Map.fetch(callbacks, callback_id) do
      {:ok, callback} ->
        {:ok, callback}

      :error ->
        {:error,
         RuntimeError.exception(
           "EXLA.Outfeed received callback id #{inspect(callback_id)} that is not registered"
         )}
    end
  end

  defp deliver_native_reply(reply_tag, reply) do
    try do
      case reply do
        {:ok, payload} ->
          EXLA.NIF.runtime_callback_reply(reply_tag, :ok, payload)

        {:error, exception} ->
          EXLA.NIF.runtime_callback_reply(
            reply_tag,
            :error,
            {:exception, Exception.message(exception)}
          )
      end
    rescue
      _ ->
        Logger.error(
          "EXLA.Outfeed failed to send callback reply to native for tag #{inspect(reply_tag)}"
        )
    end

    :ok
  end

  defp invoke_callback({{:fn, fun}, nil, _arg_template, _opts}, tensor_args)
       when is_function(fun, 1) do
    fun.(tensor_args)
    {:ok, []}
  end

  defp invoke_callback({{:named, _, nil}, nil, _arg_template, _opts}, _tensor_args) do
    {:ok, []}
  end

  defp invoke_callback({fun, out_template, _arg_template, opts}, tensor_args)
       when is_function(fun) do
    value = if opts, do: fun.(tensor_args, opts), else: fun.(tensor_args)

    case value do
      {:error, reason} -> raise RuntimeError, "Elixir callback error: #{inspect(reason)}"
      _ -> :ok
    end

    if Nx.compatible?(value, out_template) do
      {:ok, value}
    else
      msg =
        "expected the runtime_call function to match the given output template " <>
          "#{inspect(out_template)}, got: #{inspect(value)}"

      {:error, ArgumentError.exception(msg)}
    end
  end

  defp materialize_callback_args({_, _, arg_template, _}, args_spec) when is_list(args_spec) do
    materialize_callback_tensors(arg_template, args_spec)
  end

  defp materialize_callback_args({_, _, _arg_template, _}, other) do
    {:error, ArgumentError.exception("invalid args_spec for Elixir callback: #{inspect(other)}")}
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
            msg =
              "expected the runtime_call argument to match the given input template " <>
                "#{inspect(template)}, got: #{inspect(decoded)}"

            throw({:error, ArgumentError.exception(msg)})
          end

        other, acc ->
          {other, acc}
      end)

    case remaining do
      [] ->
        {:ok, container}

      _ ->
        {:error, ArgumentError.exception("invalid args_spec for Elixir callback: extra values")}
    end
  catch
    :throw, {:error, _} = error -> error
  end

  defp encode_callback_outputs([]), do: []

  defp encode_callback_outputs(container) do
    [container]
    |> Composite.flatten_list()
    |> Enum.map(&Nx.to_binary/1)
  end
end

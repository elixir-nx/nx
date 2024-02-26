defmodule EXLA.Defn do
  @moduledoc false

  require Logger
  require EXLA.Defn.Outfeed, as: Outfeed
  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  alias EXLA.MLIR.Value
  alias EXLA.MLIR.Function

  @doc false
  def __partitions_options__(options) do
    client_name = Keyword.get_lazy(options, :client, &EXLA.Client.default_name/0)
    device_count = EXLA.Client.fetch!(client_name).device_count

    Enum.map(1..device_count//1, &Keyword.put(options, :device_id, &1 - 1))
  end

  @doc false
  def __to_backend__(options) do
    client_name = Keyword.get_lazy(options, :client, &EXLA.Client.default_name/0)

    device_id =
      Keyword.get_lazy(options, :device_id, fn ->
        EXLA.Client.fetch!(client_name).default_device_id
      end)

    {EXLA.Backend, [client: client_name, device_id: device_id]}
  end

  @doc false
  def __stream__(key, input, acc, vars, fun, [args], options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])

    {client_name, compile_options} =
      Keyword.pop_lazy(compile_options, :client, &EXLA.Client.default_name/0)

    client = EXLA.Client.fetch!(client_name)
    compile_options = Keyword.put(compile_options, :lazy_transfers, :never)

    input_length = length(Nx.Defn.Composite.flatten_list([input]))
    acc_length = length(Nx.Defn.Composite.flatten_list([acc]))

    # The input vars should not be converted to buffers as they come from infeed
    # Accs are always considered as used
    used_buffers = input_length
    used_inputs = Enum.to_list(input_length..(input_length + acc_length - 1)//1)

    comp_fun =
      &to_stream_computation(client, input_length, acc_length, &1, &2, &3, &4, compile_options)

    {executable, used_inputs, {output, acc_output}, outfeed, extra, debug?} =
      compile(
        client,
        {:stream, key},
        vars,
        fun,
        compile_options,
        used_buffers,
        used_inputs,
        comp_fun
      )

    {input_shape, input_indexes} = extra

    # Also discard the stream inputs from used inputs, similar to how it is done to buffers
    # Note we discard all lazy transfers too, as they are not possible with streams
    used_inputs = Enum.sort(for {i, nil} <- used_inputs, i >= used_buffers, do: i)

    # Execution of streams requires the coordination of
    # multiple processes which is outlined below.

    # First, we get a lock on the executable, because we want
    # to avoid transfer to the device unless we know we are
    # ready to use the device.
    {time, lock} =
      :timer.tc(fn ->
        EXLA.Defn.Lock.lock(run_key(executable))
      end)

    if debug? do
      Logger.debug("EXLA device #{executable.device_id} lock in #{us_to_ms(time)}ms")
    end

    {time, streams} =
      :timer.tc(fn ->
        buffers =
          EXLA.Defn.Buffers.filter_by_indexes(args, used_inputs, fn arg, _ ->
            EXLA.Defn.Buffers.from_nx!(arg, executable)
          end)

        # Now that we have transferred to device, we spawn a runner process
        # to execute the stream. We use a runner instead of a task to avoid
        # leaking messages in the inbox. We also don't use a supervisor
        # to keep them linked, which is safe because the agent is not used
        # outside the scope of the current process.
        #
        # Finally, note the runner cannot start immediately, we need to
        # setup the outfeed reader and register the on_unlock callback
        # that cancels the stream atomically. This is done inside
        # EXLA.Defn.Stream.run.
        {:ok, runner} =
          EXLA.Defn.Runner.start_link(lock, fn ->
            EXLA.Executable.run(executable, [buffers], run_options)
          end)

        # The outfeed reader will redirect all outputs with flag 1 to the current
        # process. Once flag 0 is emitted, we know the stream is done.
        {output_shapes, outfeed} = Outfeed.configure_stream_hook(outfeed, self(), lock)
        {:ok, outfeed_pid} = Outfeed.start_child(executable, outfeed, Process.group_leader())

        stream =
          EXLA.Defn.Stream.run(
            executable,
            lock,
            runner,
            outfeed_pid,
            input,
            input_shape,
            input_indexes,
            output,
            output_shapes,
            acc_output
          )

        [stream]
      end)

    if debug? do
      Logger.debug("EXLA stream start on device #{executable.device_id} in #{us_to_ms(time)}ms")
    end

    streams
  end

  defp to_stream_computation(
         _client,
         input_length,
         acc_length,
         %Function{} = builder,
         expr,
         used_shapes,
         outfeed,
         options
       ) do
    %{token: root_token, infeeds: []} = outfeed

    {input_shapes, used_shapes} = Enum.split_while(used_shapes, fn {i, _} -> i < input_length end)

    # Get all input indexes and shape
    input_indexes = Enum.map(input_shapes, &elem(&1, 0))

    # Drop all accumulator entries from used_shapes as we will handle it separately.
    {acc_shapes, used_shapes} = Enum.split(used_shapes, acc_length)

    # The stream loop will be a three element tuple:
    #
    #   The result of calling infeed.
    #   The looping accumulator.
    #   The looping constants.
    #
    # The input will be read as part of the infeed.
    acc_shapes_l = Enum.map(acc_shapes, &elem(&1, 1))
    acc_shape = List.to_tuple(acc_shapes_l)

    constant_shapes_l = Enum.map(used_shapes, &elem(&1, 1))

    flag_shape = EXLA.Shape.make_shape({:pred, 8}, {})
    token_shape = EXLA.Shape.make_token_shape()

    arg_shapes = [flag_shape, token_shape] ++ acc_shapes_l ++ constant_shapes_l

    %{module: module, name: name} = subbuilder(builder, "while-pred")
    out_types = container_to_exla_shape(expr)

    pred_fun = EXLA.MLIR.Module.add_function(module, name, arg_shapes, out_types)

    [flag | _] = EXLA.MLIR.Function.get_arguments(pred_fun)

    r0 = Value.constant_r0(pred_fun, 1, {:pred, 8})

    pred_op = Value.equal(pred_fun, flag, r0)

    pred = EXLA.Builder.build(pred_op)

    %{module: module, name: name} = subbuilder(builder, "while-body")

    body_fun = EXLA.MLIR.Module.add_function(module, name, arg_shapes, out_types)

    [_flag, token | args] = EXLA.MLIR.Function.get_arguments(body_fun)

    {acc, constant} = Enum.split(args, acc_length)

    {indices, input_shape} = Enum.unzip(input_shapes)
    {token, input} = Value.infeed(token, input_shape)

    input_params = Enum.zip(indices, input)

    {%Outfeed{token: token} = outfeed, acc} =
      case expr do
        {output_expr, acc_expr} ->
          acc_params =
            Enum.map(acc_shapes, fn {pos, _shape} ->
              {pos, Enum.fetch!(acc, pos - input_length)}
            end)

          constant_params =
            Enum.with_index(used_shapes, fn {pos, _shape}, index ->
              {pos, Enum.fetch!(constant, index)}
            end)

          state = %{
            precision: Keyword.get(options, :precision, :default),
            builder: body_fun,
            params: Map.new(input_params ++ acc_params ++ constant_params),
            scope_ids: Tree.scope_ids(expr)
          }

          outfeed = Outfeed.with_token(outfeed, token)
          {output, cache} = recur_flatten(output_expr, state, new_cache(outfeed))
          {acc, cache} = recur_flatten(acc_expr, state, cache)
          outfeed = cache |> get_outfeed() |> Outfeed.add_stream_hook(body_fun, output)
          {outfeed, acc}

        _ ->
          raise "expected the function given to Nx.stream/3 to return a two-element tuple, got: " <>
                  inspect(expr)
      end

    # Emit the stream hook to signal loop output
    {token, [flag]} = Value.infeed(token, flag_shape)

    [%{function: body} | _] =
      Value.variadic_return(
        [
          flag,
          token,
          acc,
          constant
        ],
        true
      )

    args = EXLA.MLIR.Function.get_arguments(builder)

    {token, [flag]} = Value.infeed(root_token, flag_shape)

    init = Value.tuple(builder, [flag, token | args])

    [_flag, token | results] = Value.while(pred, body, init)

    acc = Enum.take(results, acc_length)

    acc = wrap_tuple_result(builder, acc, acc_shape)
    outfeed = outfeed |> Outfeed.with_token(token) |> Outfeed.close(builder)

    {EXLA.Builder.build(acc), {input_shape, input_indexes}, outfeed}
  end

  defp to_stream_computation(
         client,
         input_length,
         acc_length,
         %EXLA.Builder{} = builder,
         expr,
         used_shapes,
         outfeed,
         options
       ) do
    %{token: root_token, infeeds: []} = outfeed
    %{platform: platform} = client

    {input_shapes, used_shapes} = Enum.split_while(used_shapes, fn {i, _} -> i < input_length end)

    # Get all input indexes and shape
    input_indexes = Enum.map(input_shapes, &elem(&1, 0))
    input_shape = EXLA.Shape.make_tuple_shape(Enum.map(input_shapes, &elem(&1, 1)))

    # Drop all accumulator entries from used_shapes as we will handle it separately.
    {acc_shapes, used_shapes} = Enum.split(used_shapes, acc_length)

    # The stream loop will be a three element tuple:
    #
    #   The result of calling infeed.
    #   The looping accumulator.
    #   The looping constants.
    #
    # The input will be read as part of the infeed.
    acc_shape = EXLA.Shape.make_tuple_shape(Enum.map(acc_shapes, &elem(&1, 1)))
    constant_shape = EXLA.Shape.make_tuple_shape(Enum.map(used_shapes, &elem(&1, 1)))

    flag_shape = EXLA.Shape.make_shape({:pred, 8}, {})
    token_shape = EXLA.Shape.make_token_shape()
    infeed_shape = EXLA.Shape.make_tuple_shape([flag_shape, token_shape])
    arg_shape = EXLA.Shape.make_tuple_shape([infeed_shape, acc_shape, constant_shape])

    pred_b = EXLA.Builder.new(builder, "while-pred-" <> builder.name)
    param = EXLA.Op.parameter(pred_b, 0, arg_shape, "arg")
    infeed = EXLA.Op.get_tuple_element(param, 0)
    flag = EXLA.Op.get_tuple_element(infeed, 0)
    pred_op = EXLA.Op.equal(flag, EXLA.Op.constant_r0(pred_b, 1, {:pred, 8}))
    pred = EXLA.Builder.build(pred_op)

    body_b = EXLA.Builder.new(builder, "while-body-" <> builder.name)
    param = EXLA.Op.parameter(body_b, 0, arg_shape, "arg")
    infeed = EXLA.Op.get_tuple_element(param, 0)
    acc = EXLA.Op.get_tuple_element(param, 1)
    constant = EXLA.Op.get_tuple_element(param, 2)

    # The first infeed call is a flag.
    # Call infeed again to get the actual input.
    token = EXLA.Op.get_tuple_element(infeed, 1)

    # EXLA on host does not support tuples, so we emit multiple infeed operations.
    {input_params, token} =
      if platform == :host do
        Enum.map_reduce(input_shapes, token, fn {pos, shape}, token ->
          infeed = EXLA.Op.infeed(token, shape)
          {{pos, EXLA.Op.get_tuple_element(infeed, 0)}, EXLA.Op.get_tuple_element(infeed, 1)}
        end)
      else
        infeed = EXLA.Op.infeed(token, input_shape)
        input = EXLA.Op.get_tuple_element(infeed, 0)
        token = EXLA.Op.get_tuple_element(infeed, 1)

        {Enum.with_index(input_shapes, fn {pos, _shape}, i ->
           {pos, EXLA.Op.get_tuple_element(input, i)}
         end), token}
      end

    {%Outfeed{token: token} = outfeed, acc} =
      case expr do
        {output_expr, acc_expr} ->
          acc_params =
            Enum.map(acc_shapes, fn {pos, _shape} ->
              {pos, EXLA.Op.get_tuple_element(acc, pos - input_length)}
            end)

          constant_params =
            Enum.with_index(used_shapes, fn {pos, _shape}, index ->
              {pos, EXLA.Op.get_tuple_element(constant, index)}
            end)

          state = %{
            precision: Keyword.get(options, :precision, :default),
            builder: body_b,
            params: Map.new(input_params ++ acc_params ++ constant_params),
            scope_ids: Tree.scope_ids(expr)
          }

          outfeed = Outfeed.with_token(outfeed, token)
          {output, cache} = recur_flatten(output_expr, state, new_cache(outfeed))
          {acc, cache} = recur_flatten(acc_expr, state, cache)
          outfeed = cache |> get_outfeed() |> Outfeed.add_stream_hook(body_b, output)
          {outfeed, acc}

        _ ->
          raise "expected the function given to Nx.stream/3 to return a two-element tuple, got: " <>
                  inspect(expr)
      end

    # Emit the stream hook to signal loop output
    body_tuple = EXLA.Op.tuple(body_b, [EXLA.Op.infeed(token, flag_shape), acc, constant])
    body = EXLA.Builder.build(body_tuple)

    # Now we build the call to while, converting parameters to tuples.
    {acc_params, counter} =
      Enum.map_reduce(acc_shapes, 0, fn {_pos, shape}, i ->
        {EXLA.Op.parameter(builder, i, shape, "p#{i}"), i + 1}
      end)

    {constant_params, _} =
      Enum.map_reduce(used_shapes, counter, fn {_pos, shape}, i ->
        {EXLA.Op.parameter(builder, i, shape, "p#{i}"), i + 1}
      end)

    init =
      EXLA.Op.tuple(builder, [
        EXLA.Op.infeed(root_token, flag_shape),
        EXLA.Op.tuple(builder, acc_params),
        EXLA.Op.tuple(builder, constant_params)
      ])

    while = EXLA.Op.while(pred, body, init)
    infeed = EXLA.Op.get_tuple_element(while, 0)
    acc = EXLA.Op.get_tuple_element(while, 1)

    token = EXLA.Op.get_tuple_element(infeed, 1)
    outfeed = outfeed |> Outfeed.with_token(token) |> Outfeed.close(builder)
    {EXLA.Builder.build(acc), {input_shape, input_indexes}, outfeed}
  end

  @doc false
  def __jit__(key, vars, fun, args_list, options) do
    __compile__(key, vars, fun, options).(args_list)
  end

  @doc false
  def __compile__(key, vars, fun, options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])

    {client_name, compile_options} =
      Keyword.pop_lazy(compile_options, :client, &EXLA.Client.default_name/0)

    client = EXLA.Client.fetch!(client_name)

    callback = &to_root_computation(&1, &2, &3, &4, compile_options)

    {executable, used_inputs, outputs, outfeed, :ok, debug?} =
      compile(client, key, vars, fun, compile_options, 0, [], callback)

    fn [args] ->
      {time, lock} =
        :timer.tc(fn ->
          EXLA.Defn.Lock.lock(run_key(executable))
        end)

      if debug? do
        Logger.debug("EXLA device #{executable.device_id} lock in #{us_to_ms(time)}ms")
      end

      {time, res} =
        :timer.tc(fn ->
          maybe_outfeed(lock, executable, args, used_inputs, outputs, outfeed, run_options)
        end)

      if debug? do
        Logger.debug("EXLA execution on device #{executable.device_id} in #{us_to_ms(time)}ms")
      end

      res
    end
  end

  defp to_root_computation(builder, expr, used_shapes, outfeed, options) do
    params =
      case builder do
        %Function{} ->
          Enum.zip_with(used_shapes, Function.get_arguments(builder), fn {pos, _shape}, arg ->
            {pos, arg}
          end)

        _ ->
          Enum.with_index(used_shapes, fn {pos, shape}, i ->
            {pos, EXLA.Op.parameter(builder, i, shape, "p#{i}")}
          end)
      end

    state = %{
      precision: Keyword.get(options, :precision, :default),
      builder: builder,
      params: Map.new(params ++ outfeed.infeeds),
      scope_ids: Tree.scope_ids(expr)
    }

    {res, cache} = recur_flatten(expr, state, new_cache(outfeed))
    outfeed = cache |> get_outfeed() |> Outfeed.close(builder)

    {EXLA.Builder.build(res), :ok, outfeed}
  end

  defp maybe_outfeed(lock, executable, args, used_inputs, outputs, outfeed, run_options)
       when Outfeed.will_outfeed(outfeed) do
    {buffers, infeeds} =
      EXLA.Defn.Buffers.split_by_value(args, used_inputs, fn
        arg, _i, nil -> EXLA.Defn.Buffers.from_nx!(arg, executable, true)
        arg, i, _depth -> {i, EXLA.Defn.Buffers.from_nx!(arg, executable, false)}
      end)

    {:ok, runner} =
      EXLA.Defn.Runner.start_link(lock, fn ->
        EXLA.Executable.run(executable, [Enum.reverse(buffers)], run_options)
      end)

    {:ok, outfeed_pid} =
      Outfeed.start_child(executable, outfeed, Process.group_leader(), Map.new(infeeds))

    _ = EXLA.Defn.Lock.transfer(lock, fn -> send(runner, lock) end, outfeed_pid)
    ref = Process.monitor(outfeed_pid)

    receive do
      {:DOWN, ^ref, _, _, _} ->
        [result] = EXLA.Defn.Runner.read(runner)
        [EXLA.Defn.Buffers.to_nx!(result, outputs)]
    end
  end

  defp maybe_outfeed(lock, executable, args, used_inputs, outputs, _outfeed, run_options) do
    try do
      buffers =
        EXLA.Defn.Buffers.filter_by_indexes(args, used_inputs, fn arg, _i ->
          EXLA.Defn.Buffers.from_nx!(arg, executable)
        end)

      EXLA.Executable.run(executable, [buffers], run_options)
    else
      [result] ->
        [EXLA.Defn.Buffers.to_nx!(result, outputs)]
    after
      EXLA.Defn.Lock.unlock(lock)
    end
  end

  defp run_key(%{client: %{ref: ref}, device_id: device_id}), do: [ref | device_id]

  ## Compile

  defp compile(client, key, vars, fun, options, used_buffers, used_inputs, to_computation) do
    {{expr_cache_fun, comp_cache_fun}, options} =
      case Keyword.pop(options, :cache, true) do
        {true, options} ->
          Keyword.pop(options, EXLA, {&EXLA.Defn.LockedCache.run/2, &EXLA.Defn.LockedCache.run/2})

        {false, options} ->
          cache_fun = fn _key, fun -> fun.() end
          {{cache_fun, cache_fun}, options}
      end

    {debug?, options} = Keyword.pop(options, :debug, false)

    {args_key, reverse_args_identifiers} =
      Enum.map_reduce(vars, [], fn var, acc ->
        Nx.Defn.Composite.traverse(var, acc, fn
          %T{vectorized_axes: vectorized_axes} = t, acc ->
            %T{type: type, shape: shape, names: names} = Nx.devectorize(t)
            identifier = {type, shape, names}
            cache_key = {type, shape, names, vectorized_axes}
            {cache_key, [identifier | acc]}
        end)
      end)

    {lazy_transfers, options} = Keyword.pop(options, :lazy_transfers, :opt_in)

    {eval_time, {expr, {ref, outputs, {used_inputs, defined_hooks}}}} =
      :timer.tc(fn ->
        expr_cache_fun.({key, args_key}, fn ->
          expr = fun.(vars)
          inputs_and_hooks = Outfeed.used_inputs_and_hooks(expr, used_inputs, lazy_transfers)
          {expr, {make_ref(), Nx.to_template(expr), inputs_and_hooks}}
        end)
      end)

    if debug? do
      hit_or_miss = if expr, do: "miss", else: "hit"

      Logger.debug(
        "EXLA defn evaluation #{inspect(key)} cache #{hit_or_miss} in #{us_to_ms(eval_time)}ms"
      )
    end

    {hooks, options} = Keyword.pop(options, :hooks, %{})

    outfeed = Outfeed.new(hooks, defined_hooks)

    comp_key = {ref, client.name, outfeed.used_hooks, lazy_transfers, options}

    {comp_time, {evaled, {xla_time, executable, extra, outfeed}}} =
      :timer.tc(fn ->
        comp_cache_fun.(comp_key, fn ->
          {reverse_inputs_and_shapes, reverse_infeeds} =
            reverse_args_identifiers
            |> Enum.reverse()
            |> EXLA.Defn.Buffers.split_by_value(used_inputs, fn
              {type, shape, _names}, i, nil -> {i, EXLA.Shape.make_shape(type, shape)}
              {type, shape, _names}, i, depth -> {i, depth, EXLA.Shape.make_shape(type, shape)}
            end)

          inputs_and_shapes = Enum.reverse(reverse_inputs_and_shapes)

          mode = options[:compiler_mode] || Application.get_env(:exla, :compiler_mode, :mlir)

          {mod, compile_fn} =
            case mode do
              :xla -> {EXLA.Op, fn _, _, fun -> fun.(EXLA.Builder.new(inspect(key))) end}
              :mlir -> {Value, &EXLA.MLIR.Module.new/3}
            end

          comp_arg_shapes =
            for {i, shape} <- inputs_and_shapes, i >= used_buffers, do: shape

          out_types =
            [outputs]
            |> Nx.Defn.Composite.flatten_list()
            |> Enum.map(fn t ->
              t
              |> Nx.devectorize()
              |> then(&EXLA.Shape.make_shape(&1.type, &1.shape))
            end)

          compile_fn.(comp_arg_shapes, out_types, fn builder ->
            outfeed =
              outfeed
              |> Outfeed.with_token(mod.create_token(builder))
              |> Outfeed.add_infeeds(builder, reverse_infeeds)

            expr = Nx.Defn.Composite.traverse(expr || fun.(vars), &Nx.devectorize/1)

            {computation, extra, outfeed} =
              to_computation.(builder, expr, inputs_and_shapes, outfeed)

            {xla_time, executable} =
              :timer.tc(fn ->
                shapes = for {i, shape} <- inputs_and_shapes, i >= used_buffers, do: shape

                EXLA.Computation.compile(computation, client, shapes, options)
              end)

            {:ok, {xla_time, executable, extra, %{outfeed | infeeds: []}}}
          end)
        end)
      end)

    cond do
      not debug? ->
        :ok

      evaled ->
        Logger.debug(
          "EXLA compilation #{inspect(key)} cache miss in #{us_to_ms(comp_time)}ms (#{us_to_ms(xla_time)}ms in XLA)"
        )

      true ->
        Logger.debug("EXLA compilation #{inspect(key)} cache hit in #{us_to_ms(comp_time)}ms")
    end

    if expr || evaled do
      measurements = %{
        eval_time: eval_time,
        compile_time: comp_time,
        total_time: eval_time + comp_time
      }

      :telemetry.execute([:exla, :compilation], measurements, %{key: key})
    end

    outfeed = Outfeed.with_user_hooks(outfeed, hooks)
    {executable, used_inputs, outputs, outfeed, extra, debug?}
  end

  defp us_to_ms(time), do: Float.round(time / 1000, 1)

  ## Operator handling

  defp recur_flatten(composite, state, cache) do
    {acc, cache} =
      Composite.reduce(composite, {[], cache}, fn %T{} = expr, {acc, cache} ->
        {expr, cache} = recur_operator(expr, state, cache)
        {[expr | acc], cache}
      end)

    {EXLA.Op.tuple(state.builder, Enum.reverse(acc)), cache}
  end

  defp recur_operator(%T{data: %Expr{id: id, op: op}} = expr, state, cache) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {res, cache} = cached_recur_operator(op, Nx.devectorize(expr), state, cache)
        {res, Map.put(cache, id, res)}
    end
  end

  defp cached_recur_operator(:while, %T{data: %Expr{args: args}}, state, cache) do
    [initial_arg, arg, pred, body] = args

    initial_with_token = {get_token(cache), initial_arg}

    {initial, cache} =
      recur_composite(initial_with_token, &cast_pred_to_u8/1, state, cache)

    {pred, cache} = while_computation(:while_pred, arg, pred, {:pred, 8}, & &1, state, cache)

    {body, cache} =
      while_computation(:while_body, arg, body, :with_token, &cast_pred_to_u8/1, state, cache)

    {token, result} =
      case state.builder do
        %Function{} = function ->
          # for MLIR while, the return is variadic
          # like it would have come from Nx.Defn.Composite.flatten_list.
          # We need to collect the returned values into the nested tuples
          # that should have come from the while expr

          # TO-DO: This while can be build in a manner similar to if, where
          # we can write directly to the destination regions inside while_computation
          [token | results] = Value.while(pred, body, initial)
          result = wrap_tuple_result(function, results, initial_arg)
          {token, result}

        _ ->
          while = EXLA.Op.while(pred, body, initial)
          token = EXLA.Op.get_tuple_element(while, 0)
          result = EXLA.Op.get_tuple_element(while, 1)
          {token, result}
      end

    {result, update_token(cache, token)}
  end

  defp cached_recur_operator(:cond, %T{data: %Expr{args: args}} = t, state, cache) do
    [clauses, last] = args

    {cond_op, cache} =
      case clauses do
        [{pred, on_true}] ->
          to_if(pred, on_true, last, state, cache)

        _ ->
          # We convert cond into a nested tree of conds in order to compile it to ifs
          %T{data: %Expr{args: [[{pred, on_true}], on_false]}} =
            clauses
            |> Enum.reverse()
            |> Enum.reduce(last, fn {pred, on_true}, on_false ->
              update_in(t.data, fn data ->
                %{data | args: [[{pred, on_true}], on_false], id: make_ref()}
              end)
            end)

          to_if(pred, on_true, on_false, state, cache)
      end

    case state.builder do
      %Function{} ->
        {cond_op, cache}

      _ ->
        if get_token(cache) do
          token = EXLA.Op.get_tuple_element(cond_op, 0)
          {EXLA.Op.get_tuple_element(cond_op, 1), update_token(cache, token)}
        else
          {cond_op, cache}
        end
    end
  end

  defp cached_recur_operator(:parameter, %T{data: %Expr{args: [i]}}, state, cache) do
    {Map.fetch!(state.params, i), cache}
  end

  defp cached_recur_operator(:fun, %T{data: %Expr{args: args}, type: type}, state, cache) do
    [args, expr, {_, name, _}] = args
    {fun_computation(name, args, expr, type, state), cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{data: %Expr{args: [%{data: %{op: :top_k, args: [tensor, opts]}}, _expr, _callback]}} =
           _out,
         state,
         cache
       ) do
    {tensor, cache} = recur_operator(tensor, state, cache)

    result =
      case state.builder do
        %Function{} ->
          results = Value.top_k(tensor, opts[:k])
          Value.tuple(state.builder, results)

        %EXLA.Builder{} ->
          EXLA.Op.top_k(tensor, opts[:k])
      end

    {result, cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{data: %Expr{args: [%{data: %{op: :fft2, args: [tensor, opts]}}, _expr, _callback]}} =
           out,
         state,
         cache
       ) do
    {tensor, cache} = recur_operator(tensor, state, cache)

    fft_fn =
      case tensor do
        %Value{} -> &Value.fft(&1, :fft, &2)
        _ -> &EXLA.Op.fft/2
      end

    {fft2(fft_fn, [tensor, opts], out, state), cache}
  end

  defp cached_recur_operator(
         :optional,
         %T{data: %Expr{args: [%{data: %{op: :ifft2, args: [tensor, opts]}}, _expr, _callback]}} =
           out,
         state,
         cache
       ) do
    {tensor, cache} = recur_operator(tensor, state, cache)

    ifft_fn =
      case tensor do
        %Value{} -> &Value.fft(&1, :ifft, &2)
        _ -> &EXLA.Op.ifft/2
      end

    {fft2(ifft_fn, [tensor, opts], out, state), cache}
  end

  defp cached_recur_operator(:optional, %T{data: %Expr{args: args}}, state, cache) do
    [call, expr, _callback] = args
    %{data: %{args: in_args, op: op}} = call

    {args, opts} = Enum.split_while(in_args, &(not is_list(&1)))

    {call_args, cache} = Enum.map_reduce(args, cache, &recur_operator(&1, state, &2))
    key = computation_key(op, call_args ++ opts)

    {call_body, cache} =
      case cache do
        %{^key => computation} ->
          {computation, cache}

        %{} ->
          {computation, cache} = token_computation("optional", call_args, expr, state, cache)
          {computation, Map.put(cache, key, computation)}
      end

    case state.builder do
      %Function{} = function ->
        [token | result] = Value.call(state.builder, [get_token(cache) | call_args], call_body)
        {wrap_tuple_result(function, result, expr), update_token(cache, token)}

      _ ->
        result = EXLA.Op.call(state.builder, [get_token(cache) | call_args], call_body)
        token = EXLA.Op.get_tuple_element(result, 0)
        {EXLA.Op.get_tuple_element(result, 1), update_token(cache, token)}
    end
  end

  defp cached_recur_operator(:attach_token, %T{data: %Expr{args: [token, expr]}}, state, cache) do
    {op, cache} = recur_operator(expr, state, cache)
    {_, cache} = recur_operator(token, state, cache)
    {op, cache}
  end

  defp cached_recur_operator(:token, %T{data: %Expr{args: [token]}}, state, cache) do
    builder = state.builder

    cache =
      List.foldr(token.hooks, cache, fn %{name: name, expr: expr}, cache ->
        # First traverse the child because if it has hooks,
        # we need to handle them first
        {tuple, cache} = recur_flatten(expr, state, cache)

        cache
        |> get_outfeed()
        |> Outfeed.maybe_add_function_hook(builder, tuple, name, expr)
        |> then(&put_outfeed(cache, &1))
      end)

    {EXLA.Op.tuple(builder, []), cache}
  end

  defp cached_recur_operator(op, expr, state, cache) do
    {args, cache} = Tree.apply_args(expr, cache, &recur_operator(&1, state, &2))
    {to_operator(op, args, expr, state), cache}
  end

  ## to_operator creation

  defp to_operator(:constant, [constant], ans, state) do
    op = to_constant(state.builder, constant, ans.type)

    cond do
      ans.shape == {} ->
        op

      is_struct(op, EXLA.Op) ->
        EXLA.Op.broadcast_in_dim(op, ans.shape, {})

      is_struct(op, Value) ->
        Value.broadcast_in_dim(op, EXLA.Shape.make_shape(ans.type, ans.shape), {})
    end
  end

  defp to_operator(:tensor, [tensor], _ans, state) do
    tensor = Nx.devectorize(tensor)

    case tensor.shape do
      {} ->
        to_constant(state.builder, Nx.to_number(tensor), tensor.type)

      shape when is_struct(state.builder, EXLA.MLIR.Function) ->
        shape = EXLA.Shape.make_shape(tensor.type, shape)
        Value.constant_from_binary(state.builder, Nx.to_binary(tensor), shape)

      shape ->
        shape = EXLA.Shape.make_shape(tensor.type, shape)
        EXLA.Op.constant_from_binary(state.builder, Nx.to_binary(tensor), shape)
    end
  end

  defp to_operator(:iota, [axis], %{type: type, shape: shape}, state) do
    shape = EXLA.Shape.make_shape(type, shape)
    EXLA.Lib.iota(state.builder, shape, axis)
  end

  defp to_operator(:eye, [], %{type: type, shape: shape}, state) do
    iota_type = Nx.Type.merge_number({:u, 8}, Tuple.product(shape))
    iota_shape = EXLA.Shape.make_shape(iota_type, shape)
    rank = tuple_size(shape)

    {mod, equal_fn} =
      case state.builder do
        %Function{} -> {Value, &Value.equal(state.builder, &1, &2)}
        _ -> {EXLA.Op, &EXLA.Op.equal/2}
      end

    i0 = mod.iota(state.builder, iota_shape, rank - 2)
    i1 = mod.iota(state.builder, iota_shape, rank - 1)
    to_type(equal_fn.(i0, i1), type)
  end

  ## to_operator shape

  defp to_operator(:reshape, [%Value{} = op], %{shape: shape}, _state) do
    Value.reshape(op, shape)
  end

  defp to_operator(:reshape, [op], %{shape: shape}, _state) do
    EXLA.Op.reshape(op, shape)
  end

  defp to_operator(:pad, [%Value{} = op, %Value{} = value, padding_config], %{type: type}, _state) do
    Value.pad(to_type(op, type), to_type(value, type), padding_config)
  end

  defp to_operator(:pad, [op, value, padding_config], %{type: type}, _state) do
    EXLA.Op.pad(to_type(op, type), to_type(value, type), padding_config)
  end

  defp to_operator(:broadcast, [%Value{} = op, _shape, axes], ans, _state) do
    out_shape = EXLA.Shape.make_shape(ans.type, ans.shape)
    Value.broadcast_in_dim(to_type(op, ans.type), out_shape, List.to_tuple(axes))
  end

  defp to_operator(:broadcast, [op, _shape, axes], ans, _state) do
    EXLA.Op.broadcast_in_dim(op, ans.shape, List.to_tuple(axes))
  end

  defp to_operator(:transpose, [%Value{} = op, axes], _ans, _state) do
    Value.transpose(op, axes)
  end

  defp to_operator(:transpose, [op, axes], _ans, _state) do
    EXLA.Op.transpose(op, List.to_tuple(axes))
  end

  defp to_operator(:squeeze, [%mod{} = op, _axes], ans, _state) when mod in [EXLA.Op, Value] do
    mod.reshape(op, ans.shape)
  end

  ## to_operator others

  defp to_operator(:metadata, [op, _metadata], _ans, state) do
    %builder_mod{} = state.builder

    case op do
      %Value{} ->
        op

      %EXLA.Op{} ->
        op

      op when is_tuple(op) and builder_mod == EXLA.Builder ->
        EXLA.Op.tuple(state.builder, Tuple.to_list(op))

      op when is_tuple(op) ->
        Value.tuple(state.builder, Tuple.to_list(op))
    end
  end

  defp to_operator(:elem, [op, index], _ans, _state) do
    EXLA.Op.get_tuple_element(op, index)
  end

  defp to_operator(
         :dot,
         [
           %Value{} = left,
           contract_axes1,
           batch_axes1,
           %Value{} = right,
           contract_axes2,
           batch_axes2
         ],
         %{type: type, shape: shape},
         state
       ) do
    precision = state.precision
    output_shape = EXLA.Shape.make_shape(type, shape)

    Value.dot_general(
      output_shape,
      left,
      right,
      {contract_axes1, batch_axes1, contract_axes2, batch_axes2},
      precision
    )
  end

  defp to_operator(
         :dot,
         [left, contract_axes1, batch_axes1, right, contract_axes2, batch_axes2],
         %{type: type},
         state
       ) do
    precision = state.precision

    EXLA.Op.dot_general(
      to_type(left, type),
      to_type(right, type),
      {contract_axes1, batch_axes1, contract_axes2, batch_axes2},
      precision
    )
  end

  defp to_operator(
         :conv,
         [operand, kernel, opts],
         ans,
         state
       ) do
    padding = opts[:padding]
    strides = opts[:strides]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    feature_group_count = opts[:feature_group_size]
    batch_group_count = opts[:batch_group_size]

    %{type: output_type} = ans

    # Build general conv dims
    input_permutation = List.to_tuple(opts[:input_permutation])
    [out_features, in_features | spatial_features] = opts[:kernel_permutation]
    kernel_permutation = List.to_tuple([in_features, out_features | spatial_features])

    output_permutation =
      opts[:output_permutation]
      |> List.to_tuple()

    dimension_numbers = {input_permutation, kernel_permutation, output_permutation}

    # Ensure both types are floating
    operand = to_type(operand, output_type)
    kernel = to_type(kernel, output_type)

    case operand do
      %Value{} ->
        Value.convolution(
          operand,
          kernel,
          strides,
          padding,
          input_dilation,
          kernel_dilation,
          dimension_numbers,
          feature_group_count,
          batch_group_count,
          state.precision,
          ans.shape
        )

      _ ->
        EXLA.Op.conv_general_dilated(
          operand,
          kernel,
          strides,
          padding,
          input_dilation,
          kernel_dilation,
          dimension_numbers,
          feature_group_count,
          batch_group_count,
          state.precision
        )
    end
  end

  defp to_operator(
         :select,
         [%Value{} = pred, %Value{} = on_true, %Value{} = on_false],
         %{type: type, shape: shape},
         _state
       ) do
    pred = to_type(pred, {:pred, 8})

    out_shape = EXLA.Shape.make_shape(type, shape)

    on_true =
      on_true
      |> to_type(type)
      |> Value.broadcast_in_dim(out_shape, broadcast_axes(op_shape(on_true), shape))

    on_false =
      on_false
      |> to_type(type)
      |> Value.broadcast_in_dim(out_shape, broadcast_axes(op_shape(on_false), shape))

    Value.select(pred, on_true, on_false)
  end

  defp to_operator(:select, [pred, on_true, on_false], %{type: type, shape: shape}, _state) do
    pred = to_type(pred, {:pred, 8})

    on_true =
      on_true
      |> to_type(type)
      |> EXLA.Op.broadcast_in_dim(shape, broadcast_axes(op_shape(on_true), shape))

    on_false =
      on_false
      |> to_type(type)
      |> EXLA.Op.broadcast_in_dim(shape, broadcast_axes(op_shape(on_false), shape))

    EXLA.Op.select(pred, on_true, on_false)
  end

  defp to_operator(:triangular_solve, [%Value{} = a, b, opts], %{type: type}, _state) do
    left_side = Keyword.fetch!(opts, :left_side)
    lower = Keyword.fetch!(opts, :lower)
    transform = Keyword.fetch!(opts, :transform_a)

    case Value.get_shape(b).dims do
      {_} = b_shape ->
        b =
          b
          |> to_type(type)
          |> Value.reshape(Tuple.append(b_shape, 1))

        to_type(a, type)
        |> Value.triangular_solve(b, left_side, lower, transform)
        |> Value.reshape(b_shape)

      _ ->
        to_type(a, type)
        |> Value.triangular_solve(to_type(b, type), left_side, lower, transform)
    end
  end

  defp to_operator(:triangular_solve, [a, b, opts], %{type: type}, _state) do
    left_side = Keyword.fetch!(opts, :left_side)
    lower = Keyword.fetch!(opts, :lower)
    transform = Keyword.fetch!(opts, :transform_a)

    case EXLA.Op.get_shape(b).dims do
      {_} = b_shape ->
        b =
          b
          |> to_type(type)
          |> EXLA.Op.reshape(Tuple.append(b_shape, 1))

        to_type(a, type)
        |> EXLA.Op.triangular_solve(b, left_side, lower, false, transform)
        |> EXLA.Op.reshape(b_shape)

      _ ->
        to_type(a, type)
        |> EXLA.Op.triangular_solve(to_type(b, type), left_side, lower, false, transform)
    end
  end

  defp to_operator(:lu, [{_, _, _}, _tensor, _opts], _ans, _state) do
    raise ArgumentError, "XLA does not currently support the LU operation"
  end

  ## to_operator element-wise

  defp to_operator(:negate, [%Value{} = op], _ans, _state), do: Value.negate(op)
  defp to_operator(:negate, [op], _ans, _state), do: EXLA.Op.negate(op)

  defp to_operator(:abs, [%Value{} = op], _ans, _state), do: Value.abs(op)
  defp to_operator(:abs, [op], _ans, _state), do: EXLA.Op.abs(op)

  defp to_operator(:sign, [%Value{} = op], %{shape: shape, type: type}, state) do
    case type do
      {:u, _} ->
        ones_shape = Tuple.duplicate(1, tuple_size(shape))

        one = Enum.reduce(1..tuple_size(shape), 1, fn _, acc -> [acc] end)

        one =
          one
          |> Nx.tensor(type: type, backend: Nx.BinaryBackend)
          |> Nx.to_binary()
          |> then(
            &Value.constant_from_binary(state.builder, &1, %{dtype: type, dims: ones_shape})
          )

        one
        |> Value.broadcast_in_dim(Value.get_shape(op), List.to_tuple(Nx.axes(shape)))
        |> then(&Value.min(state.builder, &1, op))

      _ ->
        Value.sign(op)
    end
  end

  defp to_operator(:sign, [op], %{type: type}, state) do
    case type do
      {:u, _} -> EXLA.Op.min(op, EXLA.Op.constant_r0(state.builder, 1, type))
      _ -> EXLA.Op.sign(op)
    end
  end

  defp to_operator(:right_shift, [%Value{} = left, %Value{} = right], out, state) do
    op =
      if match?({:u, _}, out.type),
        do: :right_shift_logical,
        else: :right_shift_arithmetic

    apply_mlir_broadcasted_bin_op(state.builder, op, out, left, right)
  end

  defp to_operator(:right_shift, [left, right], %{type: type}, _state) do
    dims = broadcast_axes(op_shape(left), op_shape(right))

    op =
      if match?({:u, _}, type),
        do: :right_shift_logical,
        else: :right_shift_arithmetic

    apply(EXLA.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @bin_op [:add, :subtract, :multiply, :min, :max, :remainder, :pow, :divide, :atan2] ++
            [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift]

  defp to_operator(op, [%Value{} = left, %Value{} = right], out, state)
       when op in @bin_op do
    apply_mlir_broadcasted_bin_op(state.builder, op, out, left, right)
  end

  defp to_operator(op, [left, right], %{type: type}, _state) when op in @bin_op do
    dims = broadcast_axes(op_shape(left), op_shape(right))
    apply(EXLA.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  defp to_operator(:quotient, [left, right], ans, state) do
    to_operator(:divide, [to_type(left, ans.type), to_type(right, ans.type)], ans, state)
  end

  @bin_comp_op [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

  defp to_operator(op, [%Value{} = left, %Value{} = right], ans, state)
       when op in @bin_comp_op do
    apply_mlir_broadcasted_bin_op(state.builder, op, ans, left, right)
  end

  defp to_operator(op, [left, right], _ans, _state) when op in @bin_comp_op do
    # The answer type is always {:u, 8} but we need cast the inputs
    # to the same type which is not necessarily the answer type.
    left_shape = EXLA.Op.get_shape(left)
    right_shape = EXLA.Op.get_shape(right)
    type = merge_type(left_shape.dtype, right_shape.dtype)
    dims = broadcast_axes(left_shape.dims, right_shape.dims)
    apply(EXLA.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @bin_pred_op [logical_and: :bitwise_and, logical_or: :bitwise_or, logical_xor: :bitwise_xor]

  for {logical, bitwise} <- @bin_pred_op do
    defp to_operator(unquote(logical), [%Value{} = left, %Value{} = right], ans, state) do
      apply_mlir_broadcasted_bin_op(
        state.builder,
        unquote(bitwise),
        ans,
        to_mlir_logical(left),
        to_mlir_logical(right)
      )
    end

    defp to_operator(unquote(logical), [left, right], _ans, _state) do
      type = {:pred, 8}
      dims = broadcast_axes(op_shape(left), op_shape(right))
      apply(EXLA.Op, unquote(bitwise), [to_type(left, type), to_type(right, type), dims])
    end
  end

  @unary_op [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
              [:bitwise_not, :count_leading_zeros, :population_count, :cosh, :sinh, :acos] ++
              [:asin, :atan, :floor, :ceil, :round, :acosh, :asinh, :atanh, :erf] ++
              [:erfc, :erf_inv, :conjugate]

  defp to_operator(op, [%Value{} = arg], %{type: type}, _state) when op in @unary_op do
    apply(Value, op, [to_type(arg, type)])
  end

  defp to_operator(op, [arg], %{type: type}, _state) when op in @unary_op do
    apply(EXLA.Op, op, [to_type(arg, type)])
  end

  defp to_operator(:fft, [%Value{} | _] = args, out, state),
    do: fft(&Value.fft(&1, :fft, &2), args, out, state)

  defp to_operator(:fft, args, out, state), do: fft(&EXLA.Op.fft/2, args, out, state)

  defp to_operator(:ifft, [%Value{} | _] = args, out, state),
    do: fft(&Value.fft(&1, :ifft, &2), args, out, state)

  defp to_operator(:ifft, args, out, state), do: fft(&EXLA.Op.ifft/2, args, out, state)

  defp to_operator(:is_nan, [%Value{} = arg], _out, _state),
    do: Value.is_nan(arg)

  defp to_operator(:is_nan, [arg], out, state),
    do: EXLA.Op.is_nan(arg, op_type(arg), out.shape, state.builder)

  defp to_operator(:is_infinity, [%Value{} = arg], _out, _state),
    do: Value.is_infinity(arg)

  defp to_operator(:is_infinity, [arg], out, state),
    do: EXLA.Op.is_infinity(arg, op_type(arg), out.shape, state.builder)

  # These operations do the type conversion implicitly, and so
  # we cannot mess with the output type (e.g. the to_type conversion)
  # because it will throw an error
  @complex_op [:real, :imag]

  defp to_operator(op, [%Value{} = arg], %{type: type}, _state) when op in @complex_op do
    maybe_cast_arg =
      if Nx.Type.integer?(op_type(arg)) do
        to_type(arg, type)
      else
        arg
      end

    apply(Value, op, [maybe_cast_arg])
  end

  defp to_operator(op, [arg], %{type: type}, _state) when op in @complex_op do
    maybe_cast_arg =
      if Nx.Type.integer?(op_type(arg)) do
        to_type(arg, type)
      else
        arg
      end

    apply(EXLA.Op, op, [maybe_cast_arg])
  end

  @unary_lib_op [:tan]

  defp to_operator(op, [arg], %{type: type}, _state) when op in @unary_lib_op do
    apply(EXLA.Lib, op, [to_type(arg, type)])
  end

  defp to_operator(:as_type, [arg], %{type: type}, _state) do
    to_type(arg, type)
  end

  defp to_operator(:bitcast, [%Value{} = arg], %{type: type}, _state) do
    if op_type(arg) == type do
      arg
    else
      Value.bitcast_convert(arg, type)
    end
  end

  defp to_operator(:bitcast, [arg], %{type: type}, _state) do
    if op_type(arg) == type do
      arg
    else
      EXLA.Op.bitcast_convert_type(arg, type)
    end
  end

  ## to_operator reduction

  defp to_operator(:all, [arg, opts], %{shape: shape}, %{builder: %Function{}} = state) do
    to_aggregate(:bitwise_and, {:u, 8}, shape, to_mlir_logical(arg), 1, opts, state)
  end

  defp to_operator(:any, [arg, opts], %{shape: shape}, %{builder: %Function{}} = state) do
    to_aggregate(:bitwise_or, {:u, 8}, shape, to_mlir_logical(arg), 0, opts, state)
  end

  defp to_operator(:all, [arg, opts], %{shape: shape}, state) do
    to_aggregate(:bitwise_and, {:pred, 8}, shape, arg, 1, opts, state)
  end

  defp to_operator(:any, [arg, opts], %{shape: shape}, state) do
    to_aggregate(:bitwise_or, {:pred, 8}, shape, arg, 0, opts, state)
  end

  defp to_operator(:sum, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:add, type, shape, arg, 0, opts, state)
  end

  defp to_operator(:product, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:multiply, type, shape, arg, 1, opts, state)
  end

  defp to_operator(:reduce_max, [arg, opts], %{type: type, shape: shape}, state) do
    min_number = EXLA.Lib.min_number(state.builder, type)
    to_aggregate(:max, type, shape, arg, min_number, opts, state)
  end

  defp to_operator(:reduce_min, [arg, opts], %{type: type, shape: shape}, state) do
    max_number = EXLA.Lib.max_number(state.builder, type)
    to_aggregate(:min, type, shape, arg, max_number, opts, state)
  end

  defp to_operator(
         :reduce,
         [%Value{} = arg, %Value{} = acc, opts, fun],
         %{type: type, shape: shape},
         _state
       ) do
    arg = to_type(arg, type)
    keep_axes = opts[:keep_axes]
    [result] = Value.reduce(fun, [to_type(acc, type)], [arg], reduce_axes(arg, opts[:axes]))

    if keep_axes do
      Value.reshape(result, shape)
    else
      result
    end
  end

  defp to_operator(:reduce, [arg, acc, opts, fun], %{type: type, shape: shape}, _state) do
    arg = to_type(arg, type)
    keep_axes = opts[:keep_axes]
    result = EXLA.Op.reduce(arg, to_type(acc, type), fun, reduce_axes(arg, opts[:axes]))

    if keep_axes do
      EXLA.Op.reshape(result, shape)
    else
      result
    end
  end

  defp to_operator(:window_sum, [arg, window_dims, opts], %{type: type}, state) do
    to_window_aggregate(:add, type, arg, 0, window_dims, opts, state)
  end

  defp to_operator(:window_max, [arg, window_dims, opts], %{type: type}, state) do
    min_number = EXLA.Lib.min_number(state.builder, type)
    to_window_aggregate(:max, type, arg, min_number, window_dims, opts, state)
  end

  defp to_operator(:window_min, [arg, window_dims, opts], %{type: type}, state) do
    max_number = EXLA.Lib.max_number(state.builder, type)
    to_window_aggregate(:min, type, arg, max_number, window_dims, opts, state)
  end

  defp to_operator(:window_product, [arg, window_dims, opts], %{type: type}, state) do
    to_window_aggregate(:multiply, type, arg, 1, window_dims, opts, state)
  end

  defp to_operator(
         :window_reduce,
         [arg, acc, window_dimensions, opts, fun],
         %{type: type},
         %{builder: %Function{}}
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]
    window_dilations = opts[:window_dilations]
    arg = to_type(arg, type)
    acc = to_type(acc, type)

    [result] =
      Value.window_reduce(
        fun,
        [acc],
        [arg],
        window_dimensions,
        List.to_tuple(strides),
        Tuple.duplicate(1, tuple_size(op_shape(arg))),
        List.to_tuple(window_dilations),
        padding_config
      )

    result
  end

  defp to_operator(
         :window_reduce,
         [arg, acc, window_dimensions, opts, fun],
         %{type: type},
         _state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]
    window_dilations = opts[:window_dilations]
    arg = to_type(arg, type)

    EXLA.Op.window_reduce(
      arg,
      to_type(acc, type),
      fun,
      window_dimensions,
      strides,
      window_dilations,
      padding_config
    )
  end

  defp to_operator(
         :window_scatter_max,
         [%Value{} = arg, %Value{} = source, %Value{} = init_value, window_dimensions, opts],
         %{type: type},
         _state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    Value.select_and_scatter(
      arg,
      source,
      init_value,
      :gt,
      Tuple.to_list(window_dimensions),
      strides,
      padding_config
    )
  end

  defp to_operator(
         :window_scatter_max,
         [arg, source, init_value, window_dimensions, opts],
         %{type: type},
         state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    arg_shape = EXLA.Shape.make_shape(type, {})
    args = [arg_shape, arg_shape]
    select_fn = op_computation(:greater, args, :unused, state)
    scatter_fn = op_computation(:add, args, :unused, state)

    EXLA.Op.select_and_scatter(
      arg,
      select_fn,
      window_dimensions,
      strides,
      padding_config,
      source,
      init_value,
      scatter_fn
    )
  end

  defp to_operator(
         :window_scatter_min,
         [%Value{} = arg, %Value{} = source, %Value{} = init_value, window_dimensions, opts],
         %{type: type},
         _state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    Value.select_and_scatter(
      arg,
      source,
      init_value,
      :lt,
      Tuple.to_list(window_dimensions),
      strides,
      padding_config
    )
  end

  defp to_operator(
         :window_scatter_min,
         [arg, source, init_value, window_dimensions, opts],
         %{type: type},
         state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    arg_shape = EXLA.Shape.make_shape(type, {})
    args = [arg_shape, arg_shape]

    select_fn = op_computation(:less, args, :unused, state)
    scatter_fn = op_computation(:add, args, :unused, state)

    EXLA.Op.select_and_scatter(
      arg,
      select_fn,
      window_dimensions,
      strides,
      padding_config,
      source,
      init_value,
      scatter_fn
    )
  end

  defp to_operator(:indexed_add, [%Value{} | _] = tensors, out, _state) do
    mlir_scatter(tensors, out, :add)
  end

  defp to_operator(
         :indexed_add,
         tensors,
         %{type: type} = out,
         state
       ) do
    arg_shape = EXLA.Shape.make_shape(type, {})
    args = [arg_shape, arg_shape]
    scatter_fn = op_computation(:add, args, :unused, state)

    scatter(scatter_fn, tensors, out)
  end

  defp to_operator(:indexed_put, [%Value{} | _] = tensors, out, _state) do
    mlir_scatter(tensors, out, :put)
  end

  defp to_operator(:indexed_put, tensors, out, state) do
    # Build update computation

    subbuilder = subbuilder(state.builder, "scatter_reduction")

    param_shape = EXLA.Shape.make_shape(out.type, {})
    _left = EXLA.Op.parameter(subbuilder, 0, param_shape, "left")
    right = EXLA.Op.parameter(subbuilder, 1, param_shape, "right")

    scatter_fn = EXLA.Builder.build(right)

    scatter(scatter_fn, tensors, out)
  end

  defp to_operator(:map, [%Value{} = arg, _opts, fun], %{shape: shape, type: type}, _state) do
    arg = to_type(arg, type)

    Value.map(fun, [arg], Nx.axes(shape) |> List.to_tuple())
  end

  defp to_operator(:map, [arg, _opts, fun], %{shape: shape, type: type}, _state) do
    arg = to_type(arg, type)
    EXLA.Op.map(arg, fun, Nx.axes(shape))
  end

  defp to_operator(op, [arg, opts], ans, state) when op in [:argmax, :argmin] do
    apply(EXLA.Lib, op, [state.builder, arg, ans.type, opts])
  end

  defp to_operator(:clip, [%Value{} = operand, %Value{} = min, %Value{} = max], ans, _state) do
    min = to_type(min, ans.type)
    max = to_type(max, ans.type)
    operand = to_type(operand, ans.type)

    Value.clamp(operand, min, max)
  end

  defp to_operator(:clip, [operand, min, max], ans, _state) do
    min = to_type(min, ans.type)
    max = to_type(max, ans.type)
    operand = to_type(operand, ans.type)

    EXLA.Op.clamp(operand, min, max)
  end

  defp to_operator(:slice, [%Value{} = tensor, start_indices, lengths, strides], ans, _state) do
    all_static? = Enum.all?(start_indices, &is_integer/1)

    if all_static? do
      limit_indices = Enum.zip_with(start_indices, lengths, fn i, len -> i + len end)
      Value.slice(tensor, start_indices, limit_indices, strides)
    else
      zeros = List.duplicate(0, tuple_size(ans.shape))
      slice = Value.dynamic_slice(tensor, start_indices, lengths)

      if Enum.all?(strides, &(&1 == 1)) do
        slice
      else
        Value.slice(slice, zeros, lengths, strides)
      end
    end
  end

  defp to_operator(:slice, [tensor, start_indices, lengths, strides], ans, _state) do
    all_static? = Enum.all?(start_indices, &is_integer/1)

    if all_static? do
      limit_indices = Enum.zip_with(start_indices, lengths, fn i, len -> i + len end)
      EXLA.Op.slice(tensor, start_indices, limit_indices, strides)
    else
      zeros = List.duplicate(0, tuple_size(ans.shape))
      slice = EXLA.Op.dynamic_slice(tensor, start_indices, lengths)

      if Enum.all?(strides, &(&1 == 1)) do
        slice
      else
        EXLA.Op.slice(slice, zeros, lengths, strides)
      end
    end
  end

  defp to_operator(:put_slice, [%Value{} = tensor, start_indices, slice], ans, _state) do
    tensor = to_type(tensor, ans.type)
    slice = to_type(slice, ans.type)
    Value.dynamic_update_slice(tensor, slice, start_indices)
  end

  defp to_operator(:put_slice, [tensor, start_indices, slice], ans, _state) do
    tensor = to_type(tensor, ans.type)
    slice = to_type(slice, ans.type)
    EXLA.Op.dynamic_update_slice(tensor, slice, start_indices)
  end

  defp to_operator(:take, [%mod{} = tensor, indices, axis], _ans, _state) do
    tensor_rank = tensor |> op_shape() |> tuple_size()
    indices_rank = indices |> op_shape() |> tuple_size()
    result_rank = tensor_rank - 1 + indices_rank

    index_vector_dim = indices_rank
    slice_sizes = tensor |> op_shape() |> put_elem(axis, 1) |> Tuple.to_list()
    offset_dims = result_rank |> axes_for_rank() |> delete_slice(axis, indices_rank)
    collapsed_slice_dims = [axis]
    start_index_map = [axis]

    mod.gather(
      tensor,
      indices,
      index_vector_dim,
      slice_sizes,
      offset_dims,
      collapsed_slice_dims,
      start_index_map
    )
  end

  defp to_operator(:take_along_axis, [%mod{} = tensor, indices, axis], _ans, state) do
    indices_shape = op_shape(indices)
    indices_rank = tuple_size(indices_shape)

    axes_range = 0..(indices_rank - 1)//1

    index_vector_dim = indices_rank
    slice_sizes = List.duplicate(1, indices_rank)
    offset_dims = []
    collapsed_slice_dims = Enum.to_list(axes_range)
    start_index_map = Enum.to_list(axes_range)

    indices_exla_shape = mod.get_shape(indices)

    iotas =
      Enum.map(axes_range, fn axis ->
        mod.iota(state.builder, indices_exla_shape, axis)
      end)

    new_axis_shape = Tuple.append(indices_shape, 1)

    indices =
      iotas
      |> List.replace_at(axis, indices)
      |> Enum.map(&mod.reshape(&1, new_axis_shape))
      |> mod.concatenate(indices_rank)

    mod.gather(
      tensor,
      indices,
      index_vector_dim,
      slice_sizes,
      offset_dims,
      collapsed_slice_dims,
      start_index_map
    )
  end

  defp to_operator(:gather, [%mod{} = tensor, indices, opts], _ans, _state) do
    axes = Keyword.fetch!(opts, :axes)
    tensor_shape = op_shape(tensor)
    tensor_rank = tuple_size(tensor_shape)
    tensor_axes = axes_for_rank(tensor_rank)
    index_vector_dim = tuple_size(op_shape(indices)) - 1

    slice_sizes =
      for i <- tensor_axes do
        if i in axes, do: 1, else: elem(tensor_shape, i)
      end

    batch_size = tensor_rank - length(axes)
    offset_dims = count_up(batch_size, batch_size)
    mod.gather(tensor, indices, index_vector_dim, slice_sizes, offset_dims, axes, axes)
  end

  defp to_operator(:reverse, [%Value{} = tensor, axes], _ans, _state) do
    Value.reverse(tensor, axes)
  end

  defp to_operator(:reverse, [tensor, axes], _ans, _state) do
    EXLA.Op.reverse(tensor, axes)
  end

  defp to_operator(:concatenate, [[%Value{} | _rest] = tensors, axis], ans, _state) do
    tensors =
      tensors
      |> Enum.map(&to_type(&1, ans.type))

    Value.concatenate(tensors, axis)
  end

  defp to_operator(:concatenate, [tensors, axis], ans, _state) do
    tensors =
      tensors
      |> Enum.map(&to_type(&1, ans.type))

    EXLA.Op.concatenate(tensors, axis)
  end

  defp to_operator(:sort, [%mod{} = tensor, opts], ans, state) do
    dimension = opts[:axis]

    op =
      case opts[:direction] do
        :asc -> :less
        :desc -> :greater
      end

    args = [%{type: ans.type, shape: {}}, %{type: ans.type, shape: {}}]

    comp = sort_computation(op, ans.type, args, state)

    mod.sort(tensor, comp, dimension, opts[:stable] == true)
  end

  defp to_operator(:argsort, [tensor, opts], ans, state) do
    dimension = opts[:axis]
    stable = opts[:stable] == true

    op =
      case opts[:direction] do
        :asc -> :less
        :desc -> :greater
      end

    type = op_type(tensor)

    args = [
      %{type: type, shape: {}},
      %{type: type, shape: {}},
      %{type: ans.type, shape: {}},
      %{type: ans.type, shape: {}}
    ]

    comp = sort_computation(op, type, args, state)

    EXLA.Lib.argsort(state.builder, tensor, dimension, stable, comp, ans.type)
  end

  defp fft(exla_op, [%mod{} = tensor, opts], %{type: type}, state) do
    n = opts[:length]
    axis = opts[:axis]
    output_type = Nx.Type.to_complex(type)
    tensor = to_type(tensor, output_type)

    shape = op_shape(tensor)
    m = elem(shape, axis)

    tensor = fft_pad_or_slice(tensor, m, n, axis, shape, output_type, state)

    last_axis = tuple_size(shape) - 1

    if axis != last_axis do
      permutation =
        Enum.map(0..last_axis, fn
          ^axis -> last_axis
          ^last_axis -> axis
          ax -> ax
        end)
        |> List.to_tuple()

      tensor
      |> mod.transpose(permutation)
      |> exla_op.([n])
      |> mod.transpose(permutation)
    else
      exla_op.(tensor, [n])
    end
  end

  defp fft2(exla_op, [%mod{} = tensor, opts], %{type: type}, state) do
    [l1, l2] = lengths = opts[:lengths]
    [ax1, ax2] = axes = opts[:axes]
    output_type = Nx.Type.to_complex(type)
    tensor = to_type(tensor, output_type)

    shape = op_shape(tensor)
    m1 = elem(shape, ax1)
    m2 = elem(shape, ax2)

    tensor = fft_pad_or_slice(tensor, m1, l1, ax1, shape, output_type, state)
    tensor = fft_pad_or_slice(tensor, m2, l2, ax2, op_shape(tensor), output_type, state)

    last_axis = tuple_size(shape) - 1
    penultimate_axis = last_axis - 1
    last_axes = [penultimate_axis, last_axis]

    if axes != last_axes do
      permutation =
        Enum.map(0..last_axis, fn
          ^ax1 -> penultimate_axis
          ^penultimate_axis -> ax1
          ^ax2 -> last_axis
          ^last_axis -> ax2
          ax -> ax
        end)
        |> List.to_tuple()

      tensor
      |> mod.transpose(permutation)
      |> exla_op.(lengths)
      |> mod.transpose(permutation)
    else
      exla_op.(tensor, lengths)
    end
  end

  defp fft_pad_or_slice(tensor, m, n, axis, shape, output_type, state) do
    cond do
      m == n ->
        tensor

      m > n ->
        lengths =
          shape
          |> Tuple.insert_at(axis + 1, n)
          |> Tuple.delete_at(axis)
          |> Tuple.to_list()

        starts = List.duplicate(0, tuple_size(shape))
        strides = List.duplicate(1, tuple_size(shape))

        case tensor do
          %Value{} ->
            limit_indices = Enum.zip_with(starts, lengths, fn i, len -> i + len end)
            Value.slice(tensor, starts, limit_indices, strides)

          _ ->
            EXLA.Op.slice(tensor, starts, lengths, strides)
        end

      m < n ->
        zero =
          case tensor do
            %Value{function: func} ->
              Value.constant_r0(func, Complex.new(0), output_type)

            _ ->
              EXLA.Op.constant_r0(state.builder, Complex.new(0), output_type)
          end

        padding_config =
          {0, 0, 0}
          |> List.duplicate(tuple_size(shape))
          |> List.replace_at(axis, {0, n - m, 0})

        case tensor do
          %Value{} ->
            Value.pad(tensor, zero, padding_config)

          _ ->
            EXLA.Op.pad(tensor, zero, padding_config)
        end
    end
  end

  defp mlir_scatter([target, indices, updates, opts], %{type: type}, kind)
       when kind in [:add, :put] do
    target = to_type(target, type)
    updates = to_type(updates, type)
    update_rank = updates |> op_shape() |> tuple_size()
    update_axes = tl(axes_for_rank(update_rank))
    index_axes = Keyword.fetch!(opts, :axes)

    Value.scatter(target, indices, updates, kind, 1, update_axes, index_axes, index_axes)
  end

  defp scatter(scatter_fn, [target, indices, updates, opts], %{type: type}) do
    target = to_type(target, type)
    updates = to_type(updates, type)
    update_rank = updates |> op_shape() |> tuple_size()
    update_axes = tl(axes_for_rank(update_rank))
    index_axes = Keyword.fetch!(opts, :axes)
    EXLA.Op.scatter(target, indices, updates, scatter_fn, 1, update_axes, index_axes, index_axes)
  end

  ## Cache and hook helpers helpers

  defp no_token_cache(),
    do: %{__MODULE__ => Outfeed.empty()}

  defp new_cache(outfeed),
    do: %{__MODULE__ => outfeed}

  defp merge_outfeed(%{__MODULE__ => outfeed} = cache, %{__MODULE__ => new_outfeed}),
    do: %{cache | __MODULE__ => Outfeed.with_token(new_outfeed, outfeed.token)}

  defp reset_token(%{__MODULE__ => outfeed}, token),
    do: %{__MODULE__ => Outfeed.with_token(outfeed, token)}

  defp update_token(%{__MODULE__ => outfeed} = cache, token),
    do: %{cache | __MODULE__ => Outfeed.with_token(outfeed, token)}

  defp get_token(%{__MODULE__ => outfeed}), do: outfeed.token

  defp get_outfeed(%{__MODULE__ => value}), do: value

  defp put_outfeed(cache, outfeed), do: %{cache | __MODULE__ => outfeed}

  ## Computation helpers

  defp sort_computation(op, type, args, %{builder: %EXLA.MLIR.Function{} = builder}) do
    %{module: module, name: name} = subbuilder(builder, Atom.to_string(op))

    arg_shapes = Enum.map(args, &EXLA.Shape.make_shape(&1.type, &1.shape))

    function =
      EXLA.MLIR.Module.add_function(module, name, arg_shapes, [
        EXLA.Shape.make_shape({:pred, 8}, {})
      ])

    [lhs, rhs | _] = EXLA.MLIR.Function.get_arguments(function)

    op =
      cond do
        Nx.Type.integer?(type) ->
          apply(Value, op, [function, lhs, rhs])

        op == :less ->
          is_nan = Value.is_nan(rhs)
          Value.bitwise_or(function, is_nan, Value.less(function, lhs, rhs))

        op == :greater ->
          is_nan = Value.is_nan(lhs)
          Value.bitwise_or(function, is_nan, Value.greater(function, lhs, rhs))
      end

    EXLA.Builder.build(op)
  end

  defp sort_computation(op, type, args, state) do
    subbuilder = subbuilder(state.builder, Atom.to_string(op))

    [arg1, arg2 | _] =
      Enum.with_index(args, fn arg, i ->
        fun_shape = computation_arg_shape(arg)
        EXLA.Op.parameter(subbuilder, i, fun_shape, "p#{i}")
      end)

    op =
      cond do
        Nx.Type.integer?(type) ->
          apply(EXLA.Op, op, [arg1, arg2])

        op == :less ->
          is_nan = EXLA.Op.is_nan(arg2, type, {}, subbuilder)
          EXLA.Op.bitwise_or(is_nan, EXLA.Op.less(arg1, arg2))

        op == :greater ->
          is_nan = EXLA.Op.is_nan(arg1, type, {}, subbuilder)

          EXLA.Op.bitwise_or(is_nan, EXLA.Op.greater(arg1, arg2))
      end

    EXLA.Builder.build(op)
  end

  defp op_computation(op, args, out, state, prepare_args \\ & &1)

  defp op_computation(
         op,
         arg_shapes,
         out,
         %{builder: %EXLA.MLIR.Function{} = builder},
         prepare_args
       ) do
    %{module: module, name: name} = subbuilder(builder, Atom.to_string(op))

    function = EXLA.MLIR.Module.add_function(module, name, arg_shapes, out)

    args = EXLA.MLIR.Function.get_arguments(function)

    EXLA.Builder.build(apply(Value, op, [function | prepare_args.(args)]))
  end

  defp op_computation(op, args, _out, state, prepare_args) do
    subbuilder = subbuilder(state.builder, Atom.to_string(op))

    args =
      Enum.with_index(args, fn fun_shape, i ->
        EXLA.Op.parameter(subbuilder, i, fun_shape, "p#{i}")
      end)

    EXLA.Builder.build(apply(EXLA.Op, op, prepare_args.(args)))
  end

  defp fun_computation(
         name,
         args,
         expr,
         type,
         %{builder: %EXLA.MLIR.Function{module: module}} = state
       ) do
    arg_shapes =
      Enum.map(args, fn %{type: type, shape: shape} -> EXLA.Shape.make_shape(type, shape) end)

    out_type = container_to_exla_shape(expr)

    function = EXLA.MLIR.Module.add_function(module, Atom.to_string(name), arg_shapes, out_type)
    mlir_args = EXLA.MLIR.Function.get_arguments(function)

    arg_params = Enum.zip(args, mlir_args)

    params = Enum.flat_map(arg_params, &computation_arg_param/1)

    state = %{
      state
      | builder: function,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {res, _} = recur_composite(expr, state, no_token_cache())
    EXLA.Builder.build(to_type(res, type))
  end

  defp fun_computation(name, args, expr, type, state) do
    subbuilder = subbuilder(state.builder, Atom.to_string(name))

    arg_params =
      Enum.with_index(args, fn arg, i ->
        fun_shape = computation_arg_shape(arg)
        {arg, EXLA.Op.parameter(subbuilder, i, fun_shape, "p#{i}")}
      end)

    params = Enum.flat_map(arg_params, &computation_arg_param/1)

    state = %{
      state
      | builder: subbuilder,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {res, _} = recur_composite(expr, state, no_token_cache())
    EXLA.Builder.build(to_type(res, type))
  end

  defp while_computation(name, arg, expr, type, transform, %{builder: %Function{}} = state, cache) do
    arg_shapes = container_to_exla_shape(arg)

    arg_shapes = [EXLA.Shape.make_token_shape() | arg_shapes]

    %{module: module, name: name} = subbuilder(state.builder, Atom.to_string(name))

    out_types = container_to_exla_shape(expr)

    out_types =
      if type == :with_token do
        [EXLA.Shape.make_token_shape() | out_types]
      else
        out_types
      end

    function = EXLA.MLIR.Module.add_function(module, name, arg_shapes, out_types)

    [arg_token | arg_params] = EXLA.MLIR.Function.get_arguments(function)

    params = {arg, arg_params}

    params = computation_arg_param(params)

    state = %{
      state
      | builder: function,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {res, comp_cache} = recur_composite(expr, transform, state, reset_token(cache, arg_token))

    res =
      if type == :with_token do
        [get_token(comp_cache), res]
      else
        [to_type(res, type)]
      end

    Value.variadic_return(res, true)

    {function, merge_outfeed(cache, comp_cache)}
  end

  defp while_computation(name, arg, expr, type, transform, state, cache) do
    subbuilder = subbuilder(state.builder, Atom.to_string(name))
    arg_shape = computation_arg_shape(arg)

    tuple_shape = EXLA.Shape.make_tuple_shape([EXLA.Shape.make_token_shape(), arg_shape])
    param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p0")

    arg_token = EXLA.Op.get_tuple_element(param, 0)
    arg_param = EXLA.Op.get_tuple_element(param, 1)
    params = computation_arg_param({arg, arg_param})

    state = %{
      state
      | builder: subbuilder,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {res, comp_cache} = recur_composite(expr, transform, state, reset_token(cache, arg_token))

    res =
      if type == :with_token do
        EXLA.Op.tuple(subbuilder, [get_token(comp_cache), res])
      else
        to_type(res, type)
      end

    {EXLA.Builder.build(res), merge_outfeed(cache, comp_cache)}
  end

  defp token_computation(name, args, expr, %{builder: %Function{}} = state, cache) do
    %Function{module: module, name: name} = subbuilder(state.builder, name)

    token_shape = EXLA.Shape.make_token_shape()

    arg_shapes = Enum.map(args, &Value.get_shape/1)

    out_shapes = container_to_exla_shape(expr)

    function =
      EXLA.MLIR.Module.add_function(module, name, [token_shape | arg_shapes], [
        token_shape | out_shapes
      ])

    [arg_token | tail] = EXLA.MLIR.Function.get_arguments(function)

    params = Enum.with_index(tail, fn param, i -> {i, param} end)

    state = %{
      state
      | builder: function,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {%Value{} = res, comp_cache} = recur_composite(expr, state, reset_token(cache, arg_token))

    Value.variadic_return([get_token(comp_cache), res], true)

    {function, merge_outfeed(cache, comp_cache)}
  end

  defp token_computation(name, arg, expr, state, cache) do
    subbuilder = subbuilder(state.builder, name)

    arg_token = EXLA.Op.parameter(subbuilder, 0, EXLA.Shape.make_token_shape(), "p0")

    params =
      arg
      |> Enum.map(&EXLA.Op.get_shape/1)
      |> Enum.with_index(fn arg_shape, idx ->
        {idx, EXLA.Op.parameter(subbuilder, idx + 1, arg_shape, "p#{idx + 1}")}
      end)

    state = %{
      state
      | builder: subbuilder,
        params: Map.new(params),
        scope_ids: Tree.scope_ids(expr)
    }

    {res, comp_cache} = recur_composite(expr, state, reset_token(cache, arg_token))

    res = EXLA.Op.tuple(subbuilder, [arg_token, res])

    {EXLA.Builder.build(res), merge_outfeed(cache, comp_cache)}
  end

  # The cache is built on top of call args because we need to handle pred/u8.
  defp computation_key(op, args) do
    keys =
      Enum.map(args, fn
        %mod{} = op when mod in [EXLA.Op, Value] ->
          %EXLA.Shape{dims: dims, dtype: dtype} = mod.get_shape(op)
          {dims, dtype}

        opts ->
          opts
      end)

    {op, keys}
  end

  defp computation_arg_shape(%{type: type, shape: shape}) do
    EXLA.Shape.make_shape(type, shape)
  end

  defp computation_arg_shape(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&computation_arg_shape/1)
    |> EXLA.Shape.make_tuple_shape()
  end

  defp computation_arg_param({tuple, params}) when is_tuple(tuple) and is_list(params) do
    tuple
    |> Tuple.to_list()
    |> Enum.zip(params)
    |> Enum.flat_map(&computation_arg_param/1)
  end

  defp computation_arg_param({tuple, %mod{} = param}) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.with_index(fn arg, i -> {arg, mod.get_tuple_element(param, i)} end)
    |> Enum.flat_map(&computation_arg_param/1)
  end

  defp computation_arg_param({%T{data: %Expr{op: :parameter, args: [pos]}}, [param]}) do
    [{pos, param}]
  end

  defp computation_arg_param({%T{data: %Expr{op: :parameter, args: [pos]}}, param}) do
    [{pos, param}]
  end

  defp recur_composite(composite, state, cache) do
    recur_composite(composite, & &1, state, cache)
  end

  defp recur_composite(tuple, transform, state, cache) when is_tuple(tuple) do
    list = Tuple.to_list(tuple)

    if expr = full_tuple(list) do
      recur_composite(expr, transform, state, cache)
    else
      {elements, cache} = Enum.map_reduce(list, cache, &recur_composite(&1, transform, state, &2))

      tuple =
        case state.builder do
          %Function{} = function -> Value.tuple(function, elements)
          builder -> EXLA.Op.tuple(builder, elements)
        end

      {tuple, cache}
    end
  end

  defp recur_composite(%mod{} = op, transform, _state, cache) when mod in [Value, EXLA.Op] do
    {transform.(op), cache}
  end

  defp recur_composite(expr, transform, state, cache) do
    {op, cache} = recur_operator(expr, state, cache)
    {transform.(op), cache}
  end

  # If each element of the tuple is just a reference to the parent expression,
  # discard the tuple elements and return the parent expression.
  defp full_tuple(list) do
    with [%T{data: %Expr{op: :elem, args: [%T{data: %Expr{id: id} = expr}, 0]}} | _] <- list,
         true <- full_tuple?(list, 0, id) do
      expr
    else
      _ -> nil
    end
  end

  defp full_tuple?([arg | args], index, id) do
    case arg do
      %T{data: %Expr{op: :elem, args: [%T{data: %Expr{id: ^id}, shape: shape}, ^index]}} ->
        if shape == {:tuple, index} do
          true
        else
          full_tuple?(args, index + 1, id)
        end

      _ ->
        false
    end
  end

  # We got until the end without traversing the whole tuple
  defp full_tuple?([], _index, _id), do: false

  ## Aggregation

  defp to_aggregate(op, type, shape, %Value{} = arg, initial, opts, state) do
    arg = to_type(arg, type)

    acc =
      case initial do
        %Value{} = initial -> initial
        initial when is_number(initial) -> Value.constant_r0(state.builder, initial, type)
      end

    arg_shape = EXLA.Shape.make_shape(type, {})
    args = [arg_shape, arg_shape]
    comp = op_computation(op, args, [EXLA.Shape.make_shape(type, shape)], state, &Enum.reverse/1)

    keep_axes = opts[:keep_axes]
    [result] = Value.reduce(comp, [acc], [arg], reduce_axes(arg, opts[:axes]))

    if keep_axes do
      Value.reshape(result, shape)
    else
      result
    end
  end

  defp to_aggregate(op, type, shape, arg, initial, opts, state) do
    arg = to_type(arg, type)

    acc =
      case initial do
        %EXLA.Op{} = initial -> initial
        initial when is_number(initial) -> EXLA.Op.constant_r0(state.builder, initial, type)
      end

    arg_shape = EXLA.Shape.make_shape(type, {})
    args = [arg_shape, arg_shape]
    # We reverse the argument order because :nan + :infinity
    # returns :nan but :infinity + :nan returns :infinity.
    # So we want to keep the current value as first argument
    # to preserve such properties.
    comp = op_computation(op, args, :unused, state, &Enum.reverse/1)

    keep_axes = opts[:keep_axes]
    result = EXLA.Op.reduce(arg, acc, comp, reduce_axes(arg, opts[:axes]))

    if keep_axes do
      EXLA.Op.reshape(result, shape)
    else
      result
    end
  end

  defp to_window_aggregate(op, type, arg, initial, window_dimensions, opts, state) do
    arg = to_type(arg, type)

    mod =
      case state.builder do
        %Function{} -> Value
        _ -> EXLA.Op
      end

    acc =
      case initial do
        %^mod{} = initial ->
          initial

        initial when is_number(initial) ->
          mod.constant_r0(state.builder, initial, type)
      end

    arg_shape = EXLA.Shape.make_shape(type, {})
    args = [arg_shape, arg_shape]
    # We reverse the argument order because :nan + :infinity
    # returns :nan but :infinity + :nan returns :infinity.
    # So we want to keep the current value as first argument
    # to preserve such properties.
    comp =
      op_computation(
        op,
        args,
        [arg_shape],
        state,
        &Enum.reverse/1
      )

    strides = opts[:strides]
    padding = opts[:padding]
    window_dilations = opts[:window_dilations]

    case mod do
      Value ->
        [result] =
          Value.window_reduce(
            comp,
            [acc],
            [arg],
            window_dimensions,
            List.to_tuple(strides),
            Tuple.duplicate(1, tuple_size(op_shape(arg))),
            List.to_tuple(window_dilations),
            padding
          )

        result

      _ ->
        EXLA.Op.window_reduce(
          arg,
          acc,
          comp,
          window_dimensions,
          strides,
          window_dilations,
          padding
        )
    end
  end

  ## Cond

  defp to_if(pred, on_true, on_false, %{builder: %Function{} = function} = state, cache) do
    {pred_op, cache} = recur_operator(pred, state, cache)

    true_ids = Tree.scope_ids(on_true)
    false_ids = Tree.scope_ids(on_false)

    cache = recur_shared_ids(on_true, false_ids, state, cache)
    cache = recur_shared_ids(on_false, true_ids, state, cache)

    out_shape = container_to_exla_shape(on_true)

    in_token = get_token(cache)

    result_shapes =
      if in_token do
        [EXLA.Shape.make_token_shape() | out_shape]
      else
        out_shape
      end

    [node | _] = if_results = Value.if_op(pred_op, result_shapes)

    cache = to_mlir_if_branch(true, node, on_true, true_ids, state, cache)

    cache = to_mlir_if_branch(false, node, on_false, false_ids, state, cache)

    if in_token do
      {wrap_tuple_result(function, tl(if_results), on_true), update_token(cache, node)}
    else
      {wrap_tuple_result(function, if_results, on_true), cache}
    end
  end

  defp to_if(pred, on_true, on_false, state, cache) do
    {pred_op, cache} = recur_operator(pred, state, cache)

    pred_op = to_type(pred_op, {:pred, 8})

    true_ids = Tree.scope_ids(on_true)
    false_ids = Tree.scope_ids(on_false)

    {true_args, true_comp, cache} = to_if_branch(true, on_true, true_ids, false_ids, state, cache)

    {false_args, false_comp, cache} =
      to_if_branch(false, on_false, false_ids, true_ids, state, cache)

    {EXLA.Op.conditional(pred_op, true_args, true_comp, false_args, false_comp), cache}
  end

  defp collect_arg?(_id, :parameter, _args, _shared_ids),
    do: true

  # We never pass reference to tuples around, only through their elements,
  # so if a tuple is in a predicate, then it all must be in a predicate.
  defp collect_arg?(_id, :elem, [%T{data: %Expr{id: tuple_id}}, _pos], {parent_ids, sibling_ids})
       when is_map_key(parent_ids, tuple_id) or is_map_key(sibling_ids, tuple_id),
       do: true

  defp collect_arg?(id, _op, _args, {parent_ids, sibling_ids}),
    do: is_map_key(parent_ids, id) or is_map_key(sibling_ids, id)

  defp collect_args(%T{data: %Expr{id: id, op: op, args: args}} = expr, {cache, ids}, shared_ids) do
    cond do
      op == :constant or collect_arg?(id, op, args, shared_ids) ->
        case ids do
          %{^id => {_, _, new}} ->
            {new, {cache, ids}}

          %{} ->
            i = map_size(ids)
            param = Expr.parameter(expr, i)
            {param, {Map.put(cache, id, param), Map.put(ids, id, {i, expr, param})}}
        end

      expr = Map.get(cache, id) ->
        {expr, {cache, ids}}

      true ->
        {args, {cache, ids}} =
          Tree.apply_args(expr, :scope, {cache, ids}, &collect_args(&1, &2, shared_ids))

        expr = put_in(expr.data.args, args)
        {expr, {Map.put(cache, id, expr), ids}}
    end
  end

  defp recur_shared_ids(
         expr,
         other_ids,
         %{scope_ids: ids} = state,
         cache
       ) do
    {_, ids_args} =
      Composite.reduce(expr, {%{}, %{}}, fn node, acc ->
        {_, acc} = collect_args(node, acc, {ids, other_ids})
        acc
      end)

    Enum.reduce(ids_args, cache, fn {_, {_, old, _}}, cache ->
      {_, cache} = recur_operator(old, state, cache)
      cache
    end)
  end

  defp to_mlir_if_branch(bool, node, expr, current_ids, state, cache) do
    comp_state = %{state | scope_ids: current_ids}

    Value.set_if_block(node, bool)
    {res, res_cache} = recur_composite(expr, &cast_pred_to_u8/1, comp_state, cache)

    if token = get_token(cache) do
      Value.variadic_return([token, res], true)
    else
      Value.variadic_return([res], true)
    end

    Function.pop_region(state.builder)
    merge_outfeed(cache, res_cache)
  end

  defp to_if_branch(bool, expr, current_ids, other_ids, %{scope_ids: ids} = state, cache) do
    {expr, {_, ids_args}} =
      Composite.traverse(expr, {%{}, %{}}, &collect_args(&1, &2, {ids, other_ids}))

    sorted_ids_args = Enum.sort_by(ids_args, fn {_id, {i, _old, _new}} -> i end)

    {args, cache} =
      Enum.map_reduce(sorted_ids_args, cache, fn {_, {_, old, _}}, cache ->
        recur_operator(old, state, cache)
      end)

    subbuilder = subbuilder(state.builder, "if-#{Atom.to_string(bool)}")

    {args, comp, comp_cache} =
      if_branch_computation(subbuilder, expr, args, cache, fn subbuilder, params, comp_cache ->
        comp_state = %{
          state
          | builder: subbuilder,
            params: Map.new(params),
            scope_ids: current_ids
        }

        recur_composite(expr, &cast_pred_to_u8/1, comp_state, comp_cache)
      end)

    args = EXLA.Op.tuple(state.builder, args)

    {args, comp, merge_outfeed(cache, comp_cache)}
  end

  defp if_branch_computation(subbuilder, _out_expr, args, cache, fun) do
    shapes = Enum.map(args, &EXLA.Op.get_shape/1)

    if token = get_token(cache) do
      tuple_shape = EXLA.Shape.make_tuple_shape([EXLA.Shape.make_token_shape() | shapes])
      param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p")
      params = Enum.with_index(args, fn _, i -> {i, EXLA.Op.get_tuple_element(param, i + 1)} end)

      comp_token = EXLA.Op.get_tuple_element(param, 0)
      comp_cache = reset_token(cache, comp_token)
      {res, comp_cache} = fun.(subbuilder, params, comp_cache)
      comp = EXLA.Builder.build(EXLA.Op.tuple(subbuilder, [get_token(comp_cache), res]))
      {[token | args], comp, comp_cache}
    else
      tuple_shape = EXLA.Shape.make_tuple_shape(shapes)
      param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p")
      params = Enum.with_index(args, fn _, i -> {i, EXLA.Op.get_tuple_element(param, i)} end)
      {res, comp_cache} = fun.(subbuilder, params, cache)
      {args, EXLA.Builder.build(res), comp_cache}
    end
  end

  ## Axes helpers

  defp broadcast_axes(left, right) do
    {min, max} = if left <= right, do: {left, right}, else: {right, left}
    min_size = tuple_size(min)
    max_size = tuple_size(max)

    # To reproduce Nx broadcast, we simply match the lower dimensions to the highest ones.
    List.to_tuple(count_up(min_size, max_size - min_size))
  end

  defp reduce_axes(op, axes) do
    if axes do
      axes
      |> Enum.sort()
      |> List.to_tuple()
    else
      List.to_tuple(Nx.axes(op_shape(op)))
    end
  end

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  defp axes_for_rank(0), do: []

  defp axes_for_rank(rank) do
    Enum.to_list(0..(rank - 1))
  end

  ## Op Helpers

  defp op_type(%EXLA.Op{} = op), do: EXLA.Op.get_shape(op).dtype
  defp op_type(%Value{} = op), do: Value.get_shape(op).dtype

  defp op_shape(%EXLA.Op{} = op), do: EXLA.Op.get_shape(op).dims
  defp op_shape(%Value{} = op), do: Value.get_shape(op).dims

  defp to_type(%EXLA.Op{} = op, type) do
    if op_type(op) == type, do: op, else: EXLA.Op.convert_element_type(op, type)
  end

  defp to_type(%Value{} = op, type) do
    if op_type(op) == type do
      op
    else
      Value.convert(op, type)
    end
  end

  # Inside cond/while, we need to convert pred to u8.
  # We could do so lazily by comparing the versions of
  # the branches, but that gets tricky with cond/if,
  # so we always perform the operation.
  defp cast_pred_to_u8(%Value{} = op) do
    op
  end

  defp cast_pred_to_u8(op) do
    case EXLA.Op.get_shape(op).dtype do
      {:pred, 8} -> EXLA.Op.convert_element_type(op, {:u, 8})
      _ -> op
    end
  end

  defp merge_type({:pred, 8}, {:pred, 8}), do: {:pred, 8}
  defp merge_type(left, right), do: Nx.Type.merge(to_nx_type(left), to_nx_type(right))

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  defp to_constant(%EXLA.Builder{} = builder, constant, type) do
    EXLA.Op.constant_r0(builder, constant, type)
  end

  defp to_constant(%EXLA.MLIR.Function{} = function, constant, type) do
    Value.constant_r0(function, constant, type)
  end

  defp subbuilder(%EXLA.Builder{name: name} = builder, description) do
    suffix = System.unique_integer([:positive])
    EXLA.Builder.new(builder, name <> "-" <> description <> "-" <> Integer.to_string(suffix))
  end

  defp subbuilder(%EXLA.MLIR.Function{name: name} = function, description) do
    suffix = System.unique_integer([:positive])
    %{function | name: name <> "-" <> description <> "-" <> Integer.to_string(suffix)}
  end

  # Helpers

  defp delete_slice(enumerable, index, length) do
    {left, right} = Enum.split(enumerable, index)
    left ++ Enum.drop(right, length)
  end

  defp apply_mlir_broadcasted_bin_op(function, op, out, left, right) do
    left_shape = Value.get_shape(left)
    right_shape = Value.get_shape(right)
    out_shape = EXLA.Shape.make_shape(out.type, out.shape)
    left_dims = broadcast_axes(left_shape.dims, out_shape.dims)
    right_dims = broadcast_axes(right_shape.dims, out_shape.dims)

    type = merge_type(left_shape.dtype, right_shape.dtype)

    broadcast_shape = EXLA.Shape.make_shape(type, out_shape.dims)

    left =
      left
      |> to_type(type)
      |> Value.broadcast_in_dim(broadcast_shape, left_dims)

    right =
      right
      |> to_type(type)
      |> Value.broadcast_in_dim(broadcast_shape, right_dims)

    {left, right} =
      if not Nx.Type.float?(type) and Nx.Type.float?(out.type) do
        {to_type(left, out.type), to_type(right, out.type)}
      else
        {left, right}
      end

    Value
    |> apply(op, [function, left, right])
    |> to_type(out.type)
  end

  defp to_mlir_logical(%Value{} = value) do
    to_type(value, {:pred, 8})
  end

  defp container_to_exla_shape(container) do
    container
    |> List.wrap()
    |> Nx.Defn.Composite.flatten_list()
    |> Enum.flat_map(fn
      %Nx.Tensor{type: {:tuple, _}, data: %{args: values}} ->
        Enum.flat_map(values, &container_to_exla_shape/1)

      t ->
        [EXLA.Shape.make_shape(t.type, t.shape)]
    end)
  end

  defp wrap_tuple_result(function, list, template) when is_tuple(template) do
    Value.tuple(function, list)
  end

  defp wrap_tuple_result(function, list, %Nx.Tensor{type: {:tuple, _}}) do
    Value.tuple(function, list)
  end

  defp wrap_tuple_result(_, [value], _), do: value
end

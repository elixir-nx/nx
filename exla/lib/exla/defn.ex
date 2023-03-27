defmodule EXLA.Defn do
  @moduledoc false

  require Logger
  require EXLA.Defn.Outfeed, as: Outfeed
  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  @doc false
  def __partitions_options__(options) do
    client_name = Keyword.get_lazy(options, :client, &EXLA.Client.default_name/0)
    device_count = EXLA.Client.fetch!(client_name).device_count
    Enum.map(1..device_count//1, &Keyword.put(options, :device_id, &1 - 1))
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
         client,
         input_length,
         acc_length,
         builder,
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
            precision: Keyword.get(options, :precision, :highest),
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
      Enum.with_index(used_shapes, fn {pos, shape}, i ->
        {pos, EXLA.Op.parameter(builder, i, shape, "p#{i}")}
      end)

    state = %{
      precision: Keyword.get(options, :precision, :highest),
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
      [result] -> [EXLA.Defn.Buffers.to_nx!(result, outputs)]
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
          builder = EXLA.Builder.new(inspect(key))

          outfeed =
            outfeed
            |> Outfeed.with_token(EXLA.Op.create_token(builder))
            |> Outfeed.add_infeeds(builder, reverse_infeeds)

          {computation, extra, outfeed} =
            to_computation.(builder, expr || fun.(vars), inputs_and_shapes, outfeed)

          {xla_time, executable} =
            :timer.tc(fn ->
              shapes = for {i, shape} <- inputs_and_shapes, i >= used_buffers, do: shape
              EXLA.Computation.compile(computation, client, shapes, options)
            end)

          {:ok, {xla_time, executable, extra, %{outfeed | infeeds: []}}}
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
    [initial, arg, pred, body] = args

    {initial, cache} =
      recur_composite({get_token(cache), initial}, &cast_pred_to_u8/1, state, cache)

    {pred, cache} = while_computation(:while_pred, arg, pred, {:pred, 8}, & &1, state, cache)

    {body, cache} =
      while_computation(:while_body, arg, body, :with_token, &cast_pred_to_u8/1, state, cache)

    while = EXLA.Op.while(pred, body, initial)
    token = EXLA.Op.get_tuple_element(while, 0)
    {EXLA.Op.get_tuple_element(while, 1), update_token(cache, token)}
  end

  defp cached_recur_operator(:cond, %T{data: %Expr{args: args}} = t, state, cache) do
    [clauses, last] = args

    {cond, cache} =
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

    if get_token(cache) do
      token = EXLA.Op.get_tuple_element(cond, 0)
      {EXLA.Op.get_tuple_element(cond, 1), update_token(cache, token)}
    else
      {cond, cache}
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
         %T{data: %Expr{args: [%{data: %{op: :top_k, args: [tensor, opts]}}, _expr]}},
         state,
         cache
       ) do
    {tensor, cache} = recur_operator(tensor, state, cache)
    {EXLA.Op.top_k(tensor, Keyword.fetch!(opts, :k)), cache}
  end

  defp cached_recur_operator(:optional, %T{data: %Expr{args: args}}, state, cache) do
    [call, expr] = args
    %{data: %{args: args, op: op}} = call
    key = computation_key(op, args)

    {call_args, cache} = Enum.map_reduce(args, cache, &recur_operator(&1, state, &2))

    {call_body, cache} =
      case cache do
        %{^key => computation} ->
          {computation, cache}

        %{} ->
          {computation, cache} = token_computation("optional", call_args, expr, state, cache)
          {computation, Map.put(cache, key, computation)}
      end

    result = EXLA.Op.call(state.builder, [get_token(cache) | call_args], call_body)
    token = EXLA.Op.get_tuple_element(result, 0)
    {EXLA.Op.get_tuple_element(result, 1), update_token(cache, token)}
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

    if ans.shape == {} do
      op
    else
      EXLA.Op.broadcast_in_dim(op, ans.shape, {})
    end
  end

  defp to_operator(:tensor, [tensor], _ans, state) do
    tensor = Nx.devectorize(tensor)

    case tensor.shape do
      {} ->
        to_constant(state.builder, Nx.to_number(tensor), tensor.type)

      shape ->
        shape = EXLA.Shape.make_shape(tensor.type, shape)
        EXLA.Op.constant_from_binary(state.builder, Nx.to_binary(tensor), shape)
    end
  end

  defp to_operator(:random_uniform, [min, max], %{type: type, shape: shape}, _state) do
    if match?({int, size} when int in [:s, :u] and size < 32, type) do
      raise ArgumentError,
            "Nx.random_uniform/4 for EXLA requires signed and unsigned tensors to be " <>
              "at least of size 32, got: #{elem(type, 1)}"
    end

    min = to_type(min, type)
    max = to_type(max, type)
    shape = EXLA.Shape.make_shape(type, shape)
    EXLA.Op.rng_uniform(min, max, shape)
  end

  defp to_operator(:random_normal, [mu, sigma], %{type: type, shape: shape}, _state) do
    mu = to_type(mu, type)
    sigma = to_type(sigma, type)
    shape = EXLA.Shape.make_shape(type, shape)
    EXLA.Op.rng_normal(mu, sigma, shape)
  end

  defp to_operator(:iota, [axis], %{type: type, shape: shape}, state) do
    shape = EXLA.Shape.make_shape(type, shape)
    EXLA.Lib.iota(state.builder, shape, axis)
  end

  defp to_operator(:eye, [], %{type: type, shape: shape}, state) do
    iota_type = Nx.Type.merge_number({:u, 8}, Tuple.product(shape))
    iota_shape = EXLA.Shape.make_shape(iota_type, shape)
    rank = tuple_size(shape)

    i0 = EXLA.Op.iota(state.builder, iota_shape, rank - 2)
    i1 = EXLA.Op.iota(state.builder, iota_shape, rank - 1)
    to_type(EXLA.Op.equal(i0, i1), type)
  end

  ## to_operator shape

  defp to_operator(:reshape, [op], %{shape: shape}, _state) do
    EXLA.Op.reshape(op, shape)
  end

  defp to_operator(:pad, [op, value, padding_config], %{type: type}, _state) do
    EXLA.Op.pad(to_type(op, type), to_type(value, type), padding_config)
  end

  defp to_operator(:broadcast, [op, _shape, axes], ans, _state) do
    EXLA.Op.broadcast_in_dim(op, ans.shape, List.to_tuple(axes))
  end

  defp to_operator(:transpose, [op, axes], _ans, _state) do
    EXLA.Op.transpose(op, List.to_tuple(axes))
  end

  defp to_operator(:squeeze, [op, _axes], ans, _state) do
    EXLA.Op.reshape(op, ans.shape)
  end

  ## to_operator others

  defp to_operator(:metadata, [op, _metadata], _ans, state) do
    case op do
      %EXLA.Op{} -> op
      _ -> EXLA.Op.tuple(state.builder, Tuple.to_list(op))
    end
  end

  defp to_operator(:elem, [op, index], _ans, _state) do
    EXLA.Op.get_tuple_element(op, index)
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
    feature_groups = opts[:feature_group_size]
    batch_groups = opts[:batch_group_size]

    %{type: output_type} = ans

    # Build general conv dims
    input_permutation = List.to_tuple(opts[:input_permutation])
    [out_features, in_features | spatial_features] = opts[:kernel_permutation]
    kernel_permutation = List.to_tuple([in_features, out_features | spatial_features])

    output_permutation =
      opts[:output_permutation]
      |> List.to_tuple()

    conv_dim_nos = {input_permutation, kernel_permutation, output_permutation}

    # Ensure both types are floating
    operand = to_type(operand, output_type)
    kernel = to_type(kernel, output_type)

    EXLA.Op.conv_general_dilated(
      operand,
      kernel,
      strides,
      padding,
      input_dilation,
      kernel_dilation,
      conv_dim_nos,
      feature_groups,
      batch_groups,
      state.precision
    )
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

  defp to_operator(:qr, [{%{type: type}, %{type: type}}, tensor, opts], _ans, state) do
    {q, r} = EXLA.Op.qr(to_type(tensor, type), opts[:mode] != :reduced)
    EXLA.Op.tuple(state.builder, [q, r])
  end

  defp to_operator(:eigh, [{%{type: type}, %{type: type}}, tensor, opts], _ans, state) do
    {eigvec, eigval} =
      EXLA.Op.eigh(
        to_type(tensor, type),
        1,
        Keyword.fetch!(opts, :eps),
        Keyword.fetch!(opts, :max_iter)
      )

    EXLA.Op.tuple(state.builder, [eigval, eigvec])
  end

  ## to_operator element-wise

  defp to_operator(:negate, [op], _ans, _state), do: EXLA.Op.negate(op)

  defp to_operator(:abs, [op], _ans, _state), do: EXLA.Op.abs(op)

  defp to_operator(:sign, [op], %{type: type}, state) do
    case type do
      {:u, _} -> EXLA.Op.min(op, EXLA.Op.constant_r0(state.builder, 1, type))
      _ -> EXLA.Op.sign(op)
    end
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

  defp to_operator(op, [left, right], %{type: type}, _state) when op in @bin_op do
    dims = broadcast_axes(op_shape(left), op_shape(right))
    apply(EXLA.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  defp to_operator(:quotient, [left, right], %{type: type}, _state) do
    dims = broadcast_axes(op_shape(left), op_shape(right))
    apply(EXLA.Op, :divide, [to_type(left, type), to_type(right, type), dims])
  end

  @bin_comp_op [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

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

  defp to_operator(op, [arg], %{type: type}, _state) when op in @unary_op do
    apply(EXLA.Op, op, [to_type(arg, type)])
  end

  defp to_operator(:fft, args, out, state), do: fft(&EXLA.Op.fft/2, args, out, state)
  defp to_operator(:ifft, args, out, state), do: fft(&EXLA.Op.ifft/2, args, out, state)

  defp to_operator(:is_nan, [arg], out, state),
    do: EXLA.Op.is_nan(arg, op_type(arg), out.shape, Nx.axes(out), state)

  defp to_operator(:is_infinity, [arg], out, state),
    do: EXLA.Op.is_infinity(arg, op_type(arg), out.shape, Nx.axes(out), state)

  # These operations do the type conversion implicitly, and so
  # we cannot mess with the output type (e.g. the to_type conversion)
  # because it will throw an error
  @complex_op [:real, :imag]

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

  defp to_operator(:bitcast, [arg], %{type: type}, _state) do
    if op_type(arg) == type do
      arg
    else
      EXLA.Op.bitcast_convert_type(arg, type)
    end
  end

  ## to_operator reduction

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
         [arg, source, init_value, window_dimensions, opts],
         %{type: type},
         state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    select_fn = op_computation(:greater, args, state)
    scatter_fn = op_computation(:add, args, state)

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
         [arg, source, init_value, window_dimensions, opts],
         %{type: type},
         state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]

    select_fn = op_computation(:less, args, state)
    scatter_fn = op_computation(:add, args, state)

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
         :indexed_add,
         tensors,
         %{type: type} = out,
         state
       ) do
    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    scatter_fn = op_computation(:add, args, state)

    scatter(scatter_fn, tensors, out)
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

  defp to_operator(:map, [arg, _opts, fun], %{shape: shape, type: type}, _state) do
    arg = to_type(arg, type)
    EXLA.Op.map(arg, fun, Nx.axes(shape))
  end

  @reduction_op [:argmax, :argmin, :reduce_max, :reduce_min]

  defp to_operator(op, [arg, opts], ans, state)
       when op in @reduction_op do
    apply(EXLA.Lib, op, [state.builder, arg, [type: ans.type] ++ opts])
  end

  defp to_operator(:clip, [operand, min, max], ans, _state) do
    min = to_type(min, ans.type)
    max = to_type(max, ans.type)
    operand = to_type(operand, ans.type)

    EXLA.Op.clamp(operand, min, max)
  end

  defp to_operator(:slice, [tensor, start_indices, lengths, strides], ans, _state) do
    all_static? = Enum.all?(start_indices, &is_integer/1)

    if all_static? do
      limit_indices = Enum.zip_with(start_indices, lengths, fn i, len -> i + len end)
      EXLA.Op.slice(tensor, start_indices, limit_indices, strides)
    else
      zeros = List.duplicate(0, tuple_size(ans.shape))
      slice = EXLA.Op.dynamic_slice(tensor, start_indices, lengths)
      EXLA.Op.slice(slice, zeros, lengths, strides)
    end
  end

  defp to_operator(:put_slice, [tensor, start_indices, slice], ans, _state) do
    tensor = to_type(tensor, ans.type)
    slice = to_type(slice, ans.type)
    EXLA.Op.dynamic_update_slice(tensor, slice, start_indices)
  end

  defp to_operator(:take, [tensor, indices, axis], _ans, _state) do
    tensor_rank = tensor |> op_shape() |> tuple_size()
    indices_rank = indices |> op_shape() |> tuple_size()
    result_rank = tensor_rank - 1 + indices_rank

    index_vector_dim = indices_rank
    slice_sizes = tensor |> op_shape() |> put_elem(axis, 1) |> Tuple.to_list()
    offset_dims = result_rank |> axes_for_rank() |> delete_slice(axis, indices_rank)
    collapsed_slice_dims = [axis]
    start_index_map = [axis]

    EXLA.Op.gather(
      tensor,
      indices,
      index_vector_dim,
      slice_sizes,
      offset_dims,
      collapsed_slice_dims,
      start_index_map
    )
  end

  defp to_operator(:take_along_axis, [tensor, indices, axis], _ans, state) do
    indices_shape = op_shape(indices)
    indices_rank = tuple_size(indices_shape)

    axes_range = 0..(indices_rank - 1)//1

    index_vector_dim = indices_rank
    slice_sizes = List.duplicate(1, indices_rank)
    offset_dims = []
    collapsed_slice_dims = Enum.to_list(axes_range)
    start_index_map = Enum.to_list(axes_range)

    indices_exla_shape = EXLA.Op.get_shape(indices)

    iotas =
      Enum.map(axes_range, fn axis ->
        EXLA.Op.iota(state.builder, indices_exla_shape, axis)
      end)

    new_axis_shape = Tuple.append(indices_shape, 1)

    indices =
      iotas
      |> List.replace_at(axis, indices)
      |> Enum.map(&EXLA.Op.reshape(&1, new_axis_shape))
      |> EXLA.Op.concatenate(indices_rank)

    EXLA.Op.gather(
      tensor,
      indices,
      index_vector_dim,
      slice_sizes,
      offset_dims,
      collapsed_slice_dims,
      start_index_map
    )
  end

  defp to_operator(:gather, [tensor, indices], _ans, _state) do
    tensor_rank = tensor |> op_shape() |> tuple_size()
    indices_rank = indices |> op_shape() |> tuple_size()

    index_vector_dim = indices_rank - 1
    slice_sizes = List.duplicate(1, tensor_rank)
    offset_dims = []
    collapsed_slice_dims = axes_for_rank(tensor_rank)
    start_index_map = axes_for_rank(tensor_rank)

    EXLA.Op.gather(
      tensor,
      indices,
      index_vector_dim,
      slice_sizes,
      offset_dims,
      collapsed_slice_dims,
      start_index_map
    )
  end

  defp to_operator(:reverse, [tensor, axes], _ans, _state) do
    EXLA.Op.reverse(tensor, axes)
  end

  defp to_operator(:concatenate, [tensors, axis], ans, _state) do
    tensors =
      tensors
      |> Enum.map(&to_type(&1, ans.type))

    EXLA.Op.concatenate(tensors, axis)
  end

  defp to_operator(:cholesky, [tensor], ans, state) do
    tensor = to_type(tensor, ans.type)
    cholesky = EXLA.Op.cholesky(tensor)

    zeros =
      state.builder
      |> to_constant(0.0, ans.type)
      |> EXLA.Op.broadcast_in_dim(ans.shape, broadcast_axes({}, ans.shape))

    iota_shape = EXLA.Shape.make_shape({:s, 64}, ans.shape)
    iota_one = EXLA.Op.iota(state.builder, iota_shape, 1)
    iota_zero = EXLA.Op.iota(state.builder, iota_shape, 0)

    EXLA.Op.select(EXLA.Op.less_equal(iota_one, iota_zero), cholesky, zeros)
  end

  defp to_operator(:sort, [tensor, opts], ans, state) do
    dimension = opts[:axis]

    op =
      case opts[:direction] do
        :asc -> :less
        :desc -> :greater
      end

    args = [%{type: ans.type, shape: {}}, %{type: ans.type, shape: {}}]
    comp = op_computation(op, args, state)
    EXLA.Op.sort(tensor, comp, dimension)
  end

  defp to_operator(:argsort, [tensor, opts], ans, state) do
    dimension = opts[:axis]

    op =
      case opts[:direction] do
        :asc -> :less
        :desc -> :greater
      end

    args = [
      %{type: op_type(tensor), shape: {}},
      %{type: op_type(tensor), shape: {}},
      %{type: ans.type, shape: {}},
      %{type: ans.type, shape: {}}
    ]

    comp = op_computation(op, args, state, fn [arg1, arg2 | _] -> [arg1, arg2] end)
    EXLA.Lib.argsort(state.builder, tensor, dimension, comp, ans.type)
  end

  defp fft(exla_op, [tensor, opts], %{type: type}, state) do
    n = opts[:length]
    output_type = Nx.Type.to_complex(type)
    tensor = to_type(tensor, output_type)

    shape = op_shape(tensor)
    m = elem(shape, tuple_size(shape) - 1)

    tensor =
      cond do
        m == n ->
          tensor

        m > n ->
          lengths =
            shape
            |> Tuple.insert_at(tuple_size(shape), n)
            |> Tuple.delete_at(tuple_size(shape) - 1)
            |> Tuple.to_list()

          starts = List.duplicate(0, tuple_size(shape))
          strides = List.duplicate(1, tuple_size(shape))

          EXLA.Op.slice(tensor, starts, lengths, strides)

        m < n ->
          zero = EXLA.Op.constant_r0(state.builder, Complex.new(0), output_type)

          padding_config =
            {0, 0, 0}
            |> List.duplicate(tuple_size(shape))
            |> List.replace_at(tuple_size(shape) - 1, {0, n - m, 0})

          EXLA.Op.pad(tensor, zero, padding_config)
      end

    apply(exla_op, [tensor, n])
  end

  defp scatter(scatter_fn, [target, indices, updates], %{type: type}) do
    target = to_type(target, type)
    updates = to_type(updates, type)

    rank = target |> op_shape() |> tuple_size()
    # indices_rank is guaranteed to be 2 by Nx.Shape
    indices_rank = 2
    rank_diff = rank - indices_rank + 1

    indices_shape = op_shape(indices)

    indices_shape =
      [List.duplicate(1, rank_diff) | Tuple.to_list(indices_shape)]
      |> List.flatten()
      |> List.to_tuple()

    indices = EXLA.Op.reshape(indices, indices_shape)

    # If indices has shape {x, y}, updates is guaranteed by Nx.Shape to
    # have shape {x}, so if we reshaped indices to {..., x, y}, we need to
    # reshape updates to {..., x}

    updates_shape = Tuple.delete_at(indices_shape, tuple_size(indices_shape) - 1)

    updates = EXLA.Op.reshape(updates, updates_shape)

    axes = axes_for_rank(rank)

    EXLA.Op.scatter(
      target,
      indices,
      updates,
      scatter_fn,
      rank,
      [],
      axes,
      axes
    )
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

  defp op_computation(op, args, state, prepare_args \\ & &1) do
    subbuilder = subbuilder(state.builder, Atom.to_string(op))

    args =
      Enum.with_index(args, fn arg, i ->
        fun_shape = computation_arg_shape(arg)
        EXLA.Op.parameter(subbuilder, i, fun_shape, "p#{i}")
      end)

    EXLA.Builder.build(apply(EXLA.Op, op, prepare_args.(args)))
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
        EXLA.Op.tuple(subbuilder, [arg_token, res])
      else
        to_type(res, type)
      end

    {EXLA.Builder.build(res), merge_outfeed(cache, comp_cache)}
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

  defp computation_key(op, args) do
    keys =
      Enum.map(args, fn
        %Nx.Tensor{shape: shape, names: names, type: type, vectorized_axes: vectorized_axes} ->
          {type, shape, names, vectorized_axes}

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

  defp computation_arg_param({tuple, param}) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.with_index(fn arg, i -> {arg, EXLA.Op.get_tuple_element(param, i)} end)
    |> Enum.flat_map(&computation_arg_param/1)
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
      {EXLA.Op.tuple(state.builder, elements), cache}
    end
  end

  defp recur_composite(%EXLA.Op{} = op, transform, _state, cache) do
    {transform.(op), cache}
  end

  defp recur_composite(expr, transform, state, cache) do
    {op, cache} = recur_operator(expr, state, cache)
    {transform.(op), cache}
  end

  # If each element of the tuple is just a reference to the parent expression,
  # discard the tuple elements and return the parent expression.
  defp full_tuple(list) do
    with [%T{data: %Expr{op: :elem, args: args}} | rest] <- list,
         [%T{data: %Expr{id: id}} = expr, 0] <- args,
         true <- rest |> Enum.with_index(1) |> Enum.all?(&full_tuple?(&1, id)) do
      expr
    else
      _ -> nil
    end
  end

  defp full_tuple?({arg, index}, id) do
    match?(%T{data: %Expr{op: :elem, args: [%T{data: %Expr{id: ^id}}, ^index]}}, arg)
  end

  ## Aggregation

  defp to_aggregate(op, type, shape, arg, initial, opts, state) do
    arg = to_type(arg, type)

    acc =
      case initial do
        %EXLA.Op{} = initial -> initial
        initial when is_number(initial) -> EXLA.Op.constant_r0(state.builder, initial, type)
      end

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    # We reverse the argument order because :nan + :infinity
    # returns :nan but :infinity + :nan returns :infinity.
    # So we want to keep the current value as first argument
    # to preserve such properties.
    comp = op_computation(op, args, state, &Enum.reverse/1)

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

    acc =
      case initial do
        %EXLA.Op{} = initial ->
          initial

        initial when is_number(initial) ->
          EXLA.Op.constant_r0(state.builder, initial, type)
      end

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    # We reverse the argument order because :nan + :infinity
    # returns :nan but :infinity + :nan returns :infinity.
    # So we want to keep the current value as first argument
    # to preserve such properties.
    comp = op_computation(op, args, state, &Enum.reverse/1)

    strides = opts[:strides]
    padding = opts[:padding]
    window_dilations = opts[:window_dilations]

    EXLA.Op.window_reduce(arg, acc, comp, window_dimensions, strides, window_dilations, padding)
  end

  ## Cond

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
      op == :constant ->
        {expr, {cache, ids}}

      collect_arg?(id, op, args, shared_ids) ->
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
      if_branch_computation(subbuilder, args, cache, fn params, comp_cache ->
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

  defp if_branch_computation(subbuilder, args, cache, fun) do
    shapes = Enum.map(args, &EXLA.Op.get_shape/1)

    if token = get_token(cache) do
      tuple_shape = EXLA.Shape.make_tuple_shape([EXLA.Shape.make_token_shape() | shapes])
      param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p")
      params = Enum.with_index(args, fn _, i -> {i, EXLA.Op.get_tuple_element(param, i + 1)} end)

      comp_token = EXLA.Op.get_tuple_element(param, 0)
      comp_cache = reset_token(cache, comp_token)
      {res, comp_cache} = fun.(params, comp_cache)
      comp = EXLA.Builder.build(EXLA.Op.tuple(subbuilder, [get_token(comp_cache), res]))
      {[token | args], comp, comp_cache}
    else
      tuple_shape = EXLA.Shape.make_tuple_shape(shapes)
      param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p")
      params = Enum.with_index(args, fn _, i -> {i, EXLA.Op.get_tuple_element(param, i)} end)
      {res, comp_cache} = fun.(params, cache)
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

  defp op_type(op), do: EXLA.Op.get_shape(op).dtype
  defp op_shape(op), do: EXLA.Op.get_shape(op).dims

  defp to_type(op, type) do
    if op_type(op) == type, do: op, else: EXLA.Op.convert_element_type(op, type)
  end

  # Inside cond/while, we need to convert pred to u8.
  # We could do so lazily by comparing the versions of
  # the branches, but that gets tricky with cond/if,
  # so we always perform the operation.
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

  defp to_constant(builder, constant, type) do
    EXLA.Op.constant_r0(builder, constant, type)
  end

  defp subbuilder(%EXLA.Builder{name: name} = builder, desc) do
    suffix = System.unique_integer([:positive])
    EXLA.Builder.new(builder, name <> "-" <> desc <> "-" <> Integer.to_string(suffix))
  end

  # Helpers

  defp delete_slice(enumerable, index, length) do
    {left, right} = Enum.split(enumerable, index)
    left ++ Enum.drop(right, length)
  end
end

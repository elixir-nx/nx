defmodule EXLA.Defn do
  @moduledoc false

  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  @doc false
  def __stream__(key, input, acc, vars, fun, [args], options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])

    {client_name, compile_options} = Keyword.pop(compile_options, :client, :host)
    client = EXLA.Client.fetch!(client_name)

    # The input vars should not be converted to buffers as they come from infeed
    input_vars = Nx.Defn.Composite.flatten_list([input])
    input_shape = EXLA.Shape.make_tuple_shape(Enum.map(input_vars, &nx_to_shape!/1))
    acc_vars = Nx.Defn.Composite.flatten_list([acc])
    used_fun = &stream_used_inputs(&1, length(input_vars), length(acc_vars))

    comp_fun =
      &to_stream_computation(client, key, input_shape, acc_vars, &1, &2, &3, compile_options)

    {executable, used_inputs, {output, acc_output}, hooks, output_shapes} =
      compile(client, {:stream, key}, vars, fun, compile_options, used_fun, comp_fun)

    # Execution of streams requires the coordination of
    # multiple processes which is outlined below.

    # First, we get a lock on the executable, because we want
    # to avoid transfer to the device unless we know we are
    # ready to use the device.
    lock = EXLA.Defn.Lock.lock(run_key(executable))

    buffers =
      args
      |> filter_inputs(used_inputs)
      |> EXLA.Defn.Buffers.from_nx!()

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
    hooks = Map.put(hooks, 1, {output_shapes, {self(), lock}})
    {:ok, outfeed} = EXLA.Defn.Outfeed.start_child(executable, hooks)

    stream =
      EXLA.Defn.Stream.run(
        executable,
        lock,
        runner,
        outfeed,
        input,
        input_shape,
        output,
        output_shapes,
        acc_output
      )

    [stream]
  end

  defp stream_used_inputs(used, input_length, acc_length) do
    # Remove inputs from used buffers and include all accumulator entries.
    total = input_length + acc_length
    Enum.to_list(input_length..(total - 1)//1) ++ Enum.drop_while(used, &(&1 < total))
  end

  defp to_stream_computation(
         client,
         key,
         input_shape,
         acc_vars,
         expr,
         used_shapes,
         used_hooks,
         options
       ) do
    %{platform: platform} = client
    inspected_key = inspect(key)
    builder = EXLA.Builder.new(inspected_key)

    # Drop all accumulator entries from used_shapes as we will handle it separately.
    used_shapes = Enum.drop(used_shapes, length(acc_vars))

    # The stream loop will be a three element tuple:
    #
    #   The result of calling infeed.
    #   The looping accumulator.
    #   The looping constants.
    #
    # The input will be read as part of the infeed.
    acc_shapes = Enum.map(acc_vars, &nx_to_shape!/1)
    acc_shape = EXLA.Shape.make_tuple_shape(acc_shapes)
    constant_shape = EXLA.Shape.make_tuple_shape(Enum.map(used_shapes, &elem(&1, 1)))

    flag_shape = EXLA.Shape.make_shape({:pred, 8}, {})
    token_shape = EXLA.Shape.make_token_shape()
    infeed_shape = EXLA.Shape.make_tuple_shape([flag_shape, token_shape])

    arg_shape = EXLA.Shape.make_tuple_shape([infeed_shape, acc_shape, constant_shape])

    pred_b = EXLA.Builder.new(builder, "while-pred-" <> inspected_key)
    param = EXLA.Op.parameter(pred_b, 0, arg_shape, "arg")
    infeed = EXLA.Op.get_tuple_element(param, 0)
    flag = EXLA.Op.get_tuple_element(infeed, 0)
    pred_op = EXLA.Op.equal(flag, EXLA.Op.constant_r0(pred_b, 1, {:pred, 8}))
    pred = EXLA.Builder.build(pred_op)

    body_b = EXLA.Builder.new(builder, "while-body-" <> inspected_key)
    param = EXLA.Op.parameter(body_b, 0, arg_shape, "arg")
    infeed = EXLA.Op.get_tuple_element(param, 0)
    acc = EXLA.Op.get_tuple_element(param, 1)
    constant = EXLA.Op.get_tuple_element(param, 2)

    # The first infeed call is a flag.
    # Call infeed again to get the actual input.
    token = EXLA.Op.get_tuple_element(infeed, 1)
    %EXLA.Shape{dtype: {:tuple, shapes}} = input_shape

    # EXLA on host does not support tuples, so we emit multiple infeed operations.
    {infeeds, token} =
      if platform == :host do
        Enum.map_reduce(shapes, token, fn shape, token ->
          infeed = EXLA.Op.infeed(token, shape)
          {EXLA.Op.get_tuple_element(infeed, 0), EXLA.Op.get_tuple_element(infeed, 1)}
        end)
      else
        infeed = EXLA.Op.infeed(token, input_shape)
        input = EXLA.Op.get_tuple_element(infeed, 0)
        token = EXLA.Op.get_tuple_element(infeed, 1)
        {Enum.with_index(shapes, fn _shape, i -> EXLA.Op.get_tuple_element(input, i) end), token}
      end

    {output, acc, cache} =
      case expr do
        {output_expr, acc_expr} ->
          {input_params, counter} =
            Enum.map_reduce(infeeds, 0, fn infeed, i ->
              {{i, infeed}, i + 1}
            end)

          {acc_params, _counter} =
            Enum.map_reduce(acc_vars, counter, fn _shape, i ->
              {{i, EXLA.Op.get_tuple_element(acc, i - counter)}, i + 1}
            end)

          constant_params =
            Enum.with_index(used_shapes, fn {pos, _shape}, index ->
              {pos, EXLA.Op.get_tuple_element(constant, index)}
            end)

          state = %{
            precision: Keyword.get(options, :precision, :highest),
            builder: body_b,
            params: Map.new(input_params ++ acc_params ++ constant_params)
          }

          {output, cache} = recur_flatten(output_expr, state, new_cache(token, used_hooks))
          {acc, cache} = recur_flatten(acc_expr, state, cache)
          {output, acc, cache}

        _ ->
          raise "expected the function given to Nx.stream/3 to return a two-element tuple, got: " <>
                  inspect(expr)
      end

    # Emit the output flag of 1 to signal loop output
    {token, _, outfeed_hooks} = get_hooks(cache)
    {token, output_shapes} = outfeed_flat_tuple(body_b, 1, output, token)

    body_tuple = EXLA.Op.tuple(body_b, [EXLA.Op.infeed(token, flag_shape), acc, constant])
    body = EXLA.Builder.build(body_tuple)

    # Now we build the call to while, converting parameters to tuples.
    {acc_params, counter} =
      Enum.map_reduce(acc_shapes, 0, fn shape, i ->
        {EXLA.Op.parameter(builder, i, shape, "p#{i}"), i + 1}
      end)

    {constant_params, _} =
      Enum.map_reduce(used_shapes, counter, fn {_pos, shape}, i ->
        {EXLA.Op.parameter(builder, i, shape, "p#{i}"), i + 1}
      end)

    token = EXLA.Op.create_token(builder)

    init =
      EXLA.Op.tuple(builder, [
        EXLA.Op.infeed(token, flag_shape),
        EXLA.Op.tuple(builder, acc_params),
        EXLA.Op.tuple(builder, constant_params)
      ])

    while = EXLA.Op.while(pred, body, init)
    infeed = EXLA.Op.get_tuple_element(while, 0)
    acc = EXLA.Op.get_tuple_element(while, 1)

    token = EXLA.Op.get_tuple_element(infeed, 1)
    close_outfeed(builder, token)
    {EXLA.Builder.build(acc), output_shapes, outfeed_hooks}
  end

  @doc false
  def __jit__(key, vars, fun, [args], options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])
    {client_name, compile_options} = Keyword.pop(compile_options, :client, :host)
    client = EXLA.Client.fetch!(client_name)
    callback = &to_root_computation(key, &1, &2, &3, compile_options)

    {executable, used_inputs, outputs, hooks, :ok} =
      compile(client, key, vars, fun, compile_options, & &1, callback)

    maybe_outfeed(executable, args, used_inputs, outputs, hooks, run_options)
  end

  defp to_root_computation(key, expr, used_shapes, used_hooks, options) do
    builder = EXLA.Builder.new(inspect(key))

    params =
      Enum.with_index(used_shapes, fn {pos, shape}, i ->
        {pos, EXLA.Op.parameter(builder, i, shape, "p#{i}")}
      end)

    state = %{
      precision: Keyword.get(options, :precision, :highest),
      builder: builder,
      params: Map.new(params)
    }

    token = EXLA.Op.create_token(builder)
    {res, cache} = recur_flatten(expr, state, new_cache(token, used_hooks))
    {token, used_hooks, outfeed_hooks} = get_hooks(cache)
    close_outfeed(builder, used_hooks, token)
    {EXLA.Builder.build(res), :ok, outfeed_hooks}
  end

  defp maybe_outfeed(executable, args, used_inputs, outputs, hooks, run_options)
       when hooks == %{} do
    lock = EXLA.Defn.Lock.lock(run_key(executable))

    try do
      buffers =
        args
        |> filter_inputs(used_inputs)
        |> EXLA.Defn.Buffers.from_nx!()

      EXLA.Executable.run(executable, [buffers], run_options)
    else
      [result] -> [EXLA.Defn.Buffers.to_nx!(result, outputs)]
    after
      EXLA.Defn.Lock.unlock(lock)
    end
  end

  defp maybe_outfeed(executable, args, used_inputs, outputs, hooks, run_options) do
    lock = EXLA.Defn.Lock.lock(run_key(executable))

    buffers =
      args
      |> filter_inputs(used_inputs)
      |> EXLA.Defn.Buffers.from_nx!()

    {:ok, runner} =
      EXLA.Defn.Runner.start_link(lock, fn ->
        EXLA.Executable.run(executable, [buffers], run_options)
      end)

    {:ok, outfeed} = EXLA.Defn.Outfeed.start_child(executable, hooks)
    _ = EXLA.Defn.Lock.transfer(lock, fn -> send(runner, lock) end, outfeed)

    ref = Process.monitor(outfeed)

    receive do
      {:DOWN, ^ref, _, _, _} ->
        [result] = EXLA.Defn.Runner.read(runner)
        [EXLA.Defn.Buffers.to_nx!(result, outputs)]
    end
  end

  defp run_key(%{client: %{ref: ref}, device_id: device_id}), do: [ref | device_id]

  ## Compile

  defp compile(client, key, vars, fun, options, to_used, to_computation) do
    {{expr_cache_fun, comp_cache_fun}, options} =
      Keyword.pop(options, EXLA, {&EXLA.Defn.LockedCache.run/2, &EXLA.Defn.LockedCache.run/2})

    expr_args = for var <- vars, do: nx_to_expr_key!(var)
    expr_key = {key, expr_args}

    {expr, {used_inputs, defined_hooks, outputs}} =
      expr_cache_fun.(expr_key, fn ->
        expr = fun.(vars)
        {expr, used_inputs_and_hooks(expr)}
      end)

    # Hooks with default callbacks or user callbacks are part of the cache key
    {hooks, options} = Keyword.pop(options, :hooks, %{})
    used_hooks = Enum.sort(for {k, v} <- defined_hooks, v != nil or Map.has_key?(hooks, k), do: k)

    used_inputs = to_used.(used_inputs)
    cache_args = Enum.map(vars, &nx_to_cache_key!/1)
    cache_key = {key, cache_args, client.name, used_hooks, options}

    {_, {executable, extra, outfeed_hooks}} =
      comp_cache_fun.(cache_key, fn ->
        shapes = vars |> filter_inputs(used_inputs) |> Enum.map(&nx_to_shape!/1)
        inputs_and_shapes = Enum.zip(used_inputs, shapes)

        {computation, extra, hooks} =
          to_computation.(expr || fun.(vars), inputs_and_shapes, used_hooks)

        executable = EXLA.Computation.compile(computation, client, shapes, options)
        {nil, {executable, extra, hooks}}
      end)

    # Now finally compute the hooks to give to outfeed
    hooks =
      for {flag, {key, template, shapes}} <- outfeed_hooks,
          do: {flag, {shapes, compile_hook(key, hooks, defined_hooks, template)}},
          into: %{}

    {executable, used_inputs, outputs, hooks, extra}
  end

  defp compile_hook(key, hooks, defined_hooks, template) do
    {hooks[key] || Map.fetch!(defined_hooks, key), template}
  end

  defp used_inputs_and_hooks(expr) do
    {_, used_inputs, used_hooks} =
      Composite.reduce(expr, {%{}, %{}, %{}}, &used_inputs_and_hooks/2)

    {used_inputs |> Map.keys() |> Enum.sort(), used_hooks, Nx.to_template(expr)}
  end

  defp used_inputs_and_hooks(%T{data: %Expr{id: id} = expr} = t, {seen, inputs, hooks}) do
    case seen do
      %{^id => true} ->
        {seen, inputs, hooks}

      %{} ->
        acc = {Map.put(seen, id, true), used_inputs(expr, inputs), used_hooks(expr, hooks)}

        t
        |> Tree.apply_args(acc, &{&1, used_inputs_and_hooks(&1, &2)})
        |> elem(1)
    end
  end

  defp used_inputs(%Expr{op: :parameter, args: [i], context: :root}, inputs),
    do: Map.put(inputs, i, true)

  defp used_inputs(_, inputs),
    do: inputs

  defp used_hooks(%Expr{op: :token, args: [token]}, hooks),
    do: Enum.reduce(token.hooks, hooks, &Map.put(&2, &1.name, &1.callback))

  defp used_hooks(_, hooks),
    do: hooks

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
        {res, cache} = cached_recur_operator(op, expr, state, cache)
        {res, Map.put(cache, id, res)}
    end
  end

  defp cached_recur_operator(:while, %T{data: %Expr{args: args}}, state, cache) do
    [initial, arg, pred, body] = args

    {initial, cache} = recur_composite({get_token(cache), initial}, state, cache)
    {pred, cache} = token_computation(:while_pred, arg, pred, {:pred, 8}, state, cache)
    {body, cache} = token_computation(:while_body, arg, body, :with_token, state, cache)

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
              put_in(t.data.args, [[{pred, on_true}], on_false])
            end)

          to_if(pred, on_true, on_false, state, cache)
      end

    token = EXLA.Op.get_tuple_element(cond, 0)
    {EXLA.Op.get_tuple_element(cond, 1), update_token(cache, token)}
  end

  defp cached_recur_operator(:parameter, %T{data: %Expr{args: [i]}}, state, cache) do
    {Map.fetch!(state.params, i), cache}
  end

  defp cached_recur_operator(:fun, %T{data: %Expr{args: args}, type: type}, state, cache) do
    [args, expr, {_, name, _}] = args
    {no_token_computation(name, args, expr, type, state), cache}
  end

  defp cached_recur_operator(:optional, %T{data: %Expr{args: [_, default]}}, state, cache) do
    recur_operator(default, state, cache)
  end

  defp cached_recur_operator(:attach_token, %T{data: %Expr{args: [token, expr]}}, state, cache) do
    {op, cache} = recur_operator(expr, state, cache)
    {_, cache} = recur_operator(token, state, cache)
    {op, cache}
  end

  defp cached_recur_operator(:token, %T{data: %Expr{args: [token]}}, state, cache) do
    cache =
      List.foldr(token.hooks, cache, fn %{name: name, expr: expr}, cache ->
        # First traverse the child because if it has hooks,
        # we need to handle them first
        {tuple, cache} = recur_flatten(expr, state, cache)
        {token, used_hooks, outfeed_hooks} = get_hooks(cache)

        # Now, if we have a callback for this function, generate the outfeed code
        cond do
          name in used_hooks ->
            # The hook at position 0 is used to shutdown the outfeed.
            # The hook at position 1 is used to control streams.
            # We may need to introduce other defaults, which require bumping this.
            flag = map_size(outfeed_hooks) + 2
            {token, shapes} = outfeed_flat_tuple(state.builder, flag, tuple, token)
            outfeed_hooks = Map.put(outfeed_hooks, flag, {name, Nx.to_template(expr), shapes})
            put_hooks(cache, token, used_hooks, outfeed_hooks)

          token ->
            cache

          true ->
            raise "hooks are not supported inside #{state.builder.name}"
        end
      end)

    {EXLA.Op.tuple(state.builder, []), cache}
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

  defp to_operator(:eye, [], %{type: type, shape: {n, n}}, state) do
    iota_type = Nx.Type.merge_number({:u, 8}, n)
    iota_shape = EXLA.Shape.make_shape(iota_type, {n, n})

    i0 = EXLA.Op.iota(state.builder, iota_shape, 0)
    i1 = EXLA.Op.iota(state.builder, iota_shape, 1)
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

  defp to_operator(:metadata, [op, _metadata], _ans, _state) do
    op
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

  defp to_operator(
         :svd,
         [{%{type: type}, %{type: type}, %{type: type}}, tensor, _opts],
         _ans,
         state
       ) do
    {u, s, vt} = EXLA.Op.svd(to_type(tensor, type), state.precision)
    EXLA.Op.tuple(state.builder, [u, s, vt])
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

  @bin_op [:add, :subtract, :multiply, :min, :max, :remainder, :power, :divide, :atan2] ++
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

  defp to_operator(:fft, [tensor, opts], %{type: type}, state) do
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

    EXLA.Op.fft(tensor, n)
  end

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

  defp to_operator(:all, [arg, opts], _ans, state) do
    to_aggregate(:bitwise_and, {:pred, 8}, {}, arg, 1, opts, state)
  end

  defp to_operator(:any, [arg, opts], _ans, state) do
    to_aggregate(:bitwise_or, {:pred, 8}, {}, arg, 0, opts, state)
  end

  defp to_operator(:sum, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:add, type, shape, arg, 0, opts, state)
  end

  defp to_operator(:product, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:multiply, type, shape, arg, 1, opts, state)
  end

  defp to_operator(:reduce_max, [arg, opts], %{type: type, shape: shape}, state) do
    min_finite = EXLA.Lib.min_finite(state.builder, type)
    to_aggregate(:max, type, shape, arg, min_finite, opts, state)
  end

  defp to_operator(:reduce_min, [arg, opts], %{type: type, shape: shape}, state) do
    max_finite = EXLA.Lib.max_finite(state.builder, type)
    to_aggregate(:min, type, shape, arg, max_finite, opts, state)
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
    min_finite = EXLA.Lib.min_finite(state.builder, type)
    to_window_aggregate(:max, type, arg, min_finite, window_dims, opts, state)
  end

  defp to_operator(:window_min, [arg, window_dims, opts], %{type: type}, state) do
    max_finite = EXLA.Lib.max_finite(state.builder, type)
    to_window_aggregate(:min, type, arg, max_finite, window_dims, opts, state)
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
         [target, indices, updates],
         %{type: type},
         state
       ) do
    target = to_type(target, type)
    updates = to_type(updates, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    scatter_fn = op_computation(:add, args, state)

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

  ## Cache and hook helpers helpers

  defp no_token_cache(),
    do: %{__MODULE__ => {nil, [], %{}}}

  defp new_cache(token, used),
    do: %{__MODULE__ => {token, used, %{}}}

  defp reset_cache(%{__MODULE__ => {_, used, outfeed}}, token),
    do: %{__MODULE__ => {token, used, outfeed}}

  defp update_outfeed(%{__MODULE__ => {token, used, _}} = cache, %{__MODULE__ => {_, _, outfeed}}),
    do: %{cache | __MODULE__ => {token, used, outfeed}}

  defp update_token(%{__MODULE__ => {_token, used, outfeed}} = cache, token),
    do: %{cache | __MODULE__ => {token, used, outfeed}}

  defp get_token(%{__MODULE__ => {token, _, _}}),
    do: token

  defp get_hooks(%{__MODULE__ => value}),
    do: value

  defp put_hooks(cache, token, used_hooks, outfeed_hooks),
    do: %{cache | __MODULE__ => {token, used_hooks, outfeed_hooks}}

  ## Outfeed

  defp outfeed_flat_tuple(builder, flag, tuple, token) do
    token = EXLA.Op.outfeed(EXLA.Op.constant_r0(builder, flag, {:u, 16}), token)
    %EXLA.Shape{dims: {size}, dtype: {:tuple, shapes}} = EXLA.Op.get_shape(tuple)

    token =
      Enum.reduce(1..size//1, token, fn pos, token ->
        EXLA.Op.outfeed(EXLA.Op.get_tuple_element(tuple, pos - 1), token)
      end)

    {token, shapes}
  end

  defp close_outfeed(_builder, [], _token), do: :ok
  defp close_outfeed(builder, _, token), do: close_outfeed(builder, token)

  defp close_outfeed(builder, token) do
    EXLA.Op.outfeed(EXLA.Op.constant_r0(builder, 0, {:u, 16}), token)
  end

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

  defp no_token_computation(name, args, expr, type, state) do
    subbuilder = subbuilder(state.builder, Atom.to_string(name))

    arg_params =
      Enum.with_index(args, fn arg, i ->
        fun_shape = computation_arg_shape(arg)
        {arg, EXLA.Op.parameter(subbuilder, i, fun_shape, "p#{i}")}
      end)

    params = Enum.flat_map(arg_params, &computation_arg_param/1)
    state = %{state | builder: subbuilder, params: Map.new(params)}

    {res, _} = recur_composite(expr, state, no_token_cache())
    EXLA.Builder.build(to_type(res, type))
  end

  defp token_computation(name, arg, expr, type, state, cache) do
    subbuilder = subbuilder(state.builder, Atom.to_string(name))
    arg_shape = computation_arg_shape(arg)

    tuple_shape = EXLA.Shape.make_tuple_shape([EXLA.Shape.make_token_shape(), arg_shape])
    param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p0")

    arg_token = EXLA.Op.get_tuple_element(param, 0)
    arg_param = EXLA.Op.get_tuple_element(param, 1)
    params = computation_arg_param({arg, arg_param})
    state = %{state | builder: subbuilder, params: Map.new(params)}
    {res, comp_cache} = recur_composite(expr, state, reset_cache(cache, arg_token))

    res =
      if type == :with_token do
        EXLA.Op.tuple(subbuilder, [arg_token, res])
      else
        to_type(res, type)
      end

    {EXLA.Builder.build(res), update_outfeed(cache, comp_cache)}
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

  defp recur_composite(tuple, state, cache) when is_tuple(tuple) do
    list = Tuple.to_list(tuple)

    if expr = full_tuple(list) do
      recur_composite(expr, state, cache)
    else
      {elements, cache} = Enum.map_reduce(list, cache, &recur_composite(&1, state, &2))
      {EXLA.Op.tuple(state.builder, elements), cache}
    end
  end

  defp recur_composite(%EXLA.Op{} = op, _state, cache) do
    {op, cache}
  end

  defp recur_composite(expr, state, cache) do
    recur_operator(expr, state, cache)
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
        %EXLA.Op{} = initial ->
          initial

        initial when is_number(initial) ->
          EXLA.Op.constant_r0(state.builder, initial, type)
      end

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    comp = op_computation(op, args, state)
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
    comp = op_computation(op, args, state)

    strides = opts[:strides]
    padding = opts[:padding]
    window_dilations = opts[:window_dilations]

    EXLA.Op.window_reduce(arg, acc, comp, window_dimensions, strides, window_dilations, padding)
  end

  ## Cond

  defp to_if(pred, on_true, on_false, state, cache) do
    # Collect all predicate parameters, as those are evaluated
    # outside of the conditional. All other graphs are evaluated
    # only if necessary inside the conditional.
    {_, pred_ids} = collect_ids(pred, %{})

    {pred_op, cache} = recur_operator(pred, state, cache)
    pred_op = to_type(pred_op, {:pred, 8})

    {true_args, true_comp, cache} = to_if_branch(true, on_true, pred_ids, state, cache)
    {false_args, false_comp, cache} = to_if_branch(false, on_false, pred_ids, state, cache)
    {EXLA.Op.conditional(pred_op, true_args, true_comp, false_args, false_comp), cache}
  end

  defp collect_ids(%T{data: %Expr{id: id}} = t, ids) do
    case ids do
      %{^id => true} -> {t, ids}
      %{} -> Tree.apply_args(t, Map.put(ids, id, true), &collect_ids/2)
    end
  end

  defp collect_args(%T{data: %Expr{id: id, op: op}} = expr, {cache, ids}, pred_ids) do
    cond do
      op == :constant ->
        {expr, {cache, ids}}

      Map.has_key?(pred_ids, id) or op == :parameter ->
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
          Tree.apply_args(expr, {cache, ids}, &collect_args(&1, &2, pred_ids))

        expr = put_in(expr.data.args, args)
        {expr, {Map.put(cache, id, expr), ids}}
    end
  end

  defp to_if_branch(bool, expr, ids, state, cache) do
    {expr, {_, ids_args}} = Composite.traverse(expr, {%{}, %{}}, &collect_args(&1, &2, ids))
    sorted_ids_args = Enum.sort_by(ids_args, fn {_id, {i, _old, _new}} -> i end)

    {args, cache} =
      Enum.map_reduce(sorted_ids_args, cache, fn {_, {_, old, _}}, cache ->
        recur_operator(old, state, cache)
      end)

    subbuilder = subbuilder(state.builder, "if-#{Atom.to_string(bool)}")

    shapes =
      for {_, {_, _, %{type: type, shape: shape}}} <- sorted_ids_args do
        EXLA.Shape.make_shape(type, shape)
      end

    tuple_shape = EXLA.Shape.make_tuple_shape([EXLA.Shape.make_token_shape() | shapes])
    param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p")

    params =
      for {_, {i, _, _}} <- sorted_ids_args do
        {i, EXLA.Op.get_tuple_element(param, i + 1)}
      end

    comp_token = EXLA.Op.get_tuple_element(param, 0)
    comp_state = %{state | builder: subbuilder, params: Map.new(params)}
    {res, comp_cache} = recur_composite(expr, comp_state, reset_cache(cache, comp_token))

    args = EXLA.Op.tuple(state.builder, [get_token(cache) | args])
    comp = EXLA.Builder.build(EXLA.Op.tuple(subbuilder, [get_token(comp_cache), res]))
    {args, comp, update_outfeed(cache, comp_cache)}
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

  defp filter_inputs(args, inputs), do: filter_inputs(args, 0, inputs)

  defp filter_inputs([var | vars], i, [i | inputs]),
    do: [var | filter_inputs(vars, i + 1, inputs)]

  defp filter_inputs([_var | vars], i, inputs),
    do: filter_inputs(vars, i + 1, inputs)

  defp filter_inputs([], _i, []),
    do: []

  defp nx_to_shape!(%T{type: type, shape: shape}), do: EXLA.Shape.make_shape(type, shape)
  defp nx_to_cache_key!(%T{type: type, shape: shape}), do: {type, shape}
  defp nx_to_expr_key!(%T{type: type, shape: shape, names: names}), do: {type, shape, names}

  defp delete_slice(enumerable, index, length) do
    {left, right} = Enum.split(enumerable, index)
    left ++ Enum.drop(right, length)
  end
end

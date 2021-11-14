defmodule EXLA.Defn do
  @moduledoc false

  alias Nx.Defn.{Expr, Tree}
  alias Nx.Tensor, as: T

  @doc false
  def __stream__(key, input, acc, vars, fun, options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])
    keep_on_device? = Keyword.get(run_options, :keep_on_device, false)
    run_options = Keyword.put(run_options, :keep_on_device, true)

    {client_name, compile_options} = Keyword.pop(compile_options, :client, :default)
    client = EXLA.Client.fetch!(client_name)

    # The input vars should not be converted to buffers as they come from infeed
    input_vars = Nx.Defn.Tree.flatten_list([input])
    input_shape = EXLA.Shape.make_tuple_shape(Enum.map(input_vars, &nx_to_shape!/1))
    acc_vars = Nx.Defn.Tree.flatten_list([acc])
    split_fun = &split_stream(&1, &2, length(input_vars), length(acc_vars))
    comp_fun = &to_stream_computation(client, key, input_shape, acc_vars, &1, &2, compile_options)

    {inputs, {:tuple, [_output, acc_outputs]}, {executable, output_shape}} =
      compile(client, {:stream, key}, vars, fun, compile_options, split_fun, comp_fun)

    # Execution of streams requires the coordination of
    # multiple processes which is outlined below.

    # First, we get a lock on the executable, because we want
    # to avoid transfer to the device unless we know we are
    # ready to use the device.
    ref = EXLA.Defn.Lock.lock(run_key(executable))
    buffers = EXLA.Defn.Buffer.from_nx!(inputs)
    device_id = executable.device_id

    # Now that we have transferred to device, we spawn a task to
    # execute the stream. We keep this with regular async/await
    # because this task will never really exist beyond the scope
    # of the current process.
    #
    # Finally, note the task cannot start immediately, we need to
    # setup the outfeed reader and the relock the client/device_id.
    %{pid: task_pid, ref: task_ref} =
      Task.async(fn ->
        receive do
          ^ref -> EXLA.Executable.run(executable, buffers, run_options)
        end
      end)

    # The outfeed reader will redirect all outputs with flag 1 to the current
    # process. Once flag 0 is emitted, we know the stream is done.
    %EXLA.Shape{dtype: {:tuple, recv_shapes}} = output_shape
    mappings = %{1 => {recv_shapes, {self(), ref}}}
    {:ok, outfeed} = EXLA.Defn.Outfeed.start_child(client, device_id, mappings)

    # With the task and outfeed in place, we now relock the client/device_id.
    # If the current process shuts down, we send a terminate message
    ^ref =
      EXLA.Defn.Lock.relock(
        ref,
        fn -> send(task_pid, ref) end,
        fn -> halt_stream(client, device_id, outfeed) end
      )

    %EXLA.Defn.Stream{
      pid: self(),
      ref: task_ref,
      outfeed: outfeed,
      lock: ref,
      send: input,
      send_shape: input_shape,
      recv_shapes: recv_shapes,
      client: client,
      device_id: executable.device_id,
      done: acc_outputs,
      keep_on_device: keep_on_device?
    }
  end

  # It is time to halt the stream, we do it by sending 0 for the loop infeed.
  # Then we wait for the outfeed process to read all.
  defp halt_stream(client, device_id, outfeed) do
    pred = EXLA.Shape.make_shape({:pred, 8}, {})
    :ok = EXLA.Client.to_infeed(client, device_id, [{<<0::8-native>>, pred}])
    {:lock, outfeed, fn -> :unlocked end}
  end

  defp split_stream(vars, used, input_length, acc_length) do
    # Remove inputs from used buffers and include all accumulator entries.
    total = input_length + acc_length

    used =
      Enum.to_list(input_length..(input_length + acc_length - 1)) ++
        Enum.drop_while(used, &(&1 < total))

    {Enum.take(vars, input_length), used}
  end

  defp to_stream_computation(client, key, input_shape, acc_vars, expr, used_shapes, options) do
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

    {output, acc} =
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

          {output, cache} = recur_flatten(output_expr, state, %{})
          {acc, _cache} = recur_flatten(acc_expr, state, cache)
          {output, acc}

        _ ->
          raise "expected the function given to Nx.stream/3 to return a two-element tuple, got: " <>
                  inspect(expr)
      end

    # Emit the output flag of 1
    token = EXLA.Op.outfeed(EXLA.Op.constant_r0(body_b, 1, {:u, 16}), token)

    output_shape = EXLA.Op.get_shape(output)
    %EXLA.Shape{dims: {size}, dtype: {:tuple, _}} = output_shape

    token =
      Enum.reduce(1..size//1, token, fn pos, token ->
        EXLA.Op.outfeed(EXLA.Op.get_tuple_element(output, pos - 1), token)
      end)

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

    # Emit the output flag of 0
    token = EXLA.Op.get_tuple_element(infeed, 1)
    _ = EXLA.Op.outfeed(EXLA.Op.constant_r0(builder, 0, {:u, 16}), token)

    {EXLA.Builder.build(acc), output_shape}
  end

  @doc false
  def __jit__(key, vars, fun, options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])
    {client_name, compile_options} = Keyword.pop(compile_options, :client, :default)
    client = EXLA.Client.fetch!(client_name)
    callback = &to_root_computation(key, &1, &2, compile_options)

    {inputs, outputs, {executable, :ok}} =
      compile(client, key, vars, fun, compile_options, fn _, used -> {[], used} end, callback)

    ref = EXLA.Defn.Lock.lock(run_key(executable))

    try do
      EXLA.Executable.run(executable, EXLA.Defn.Buffer.from_nx!(inputs), run_options)
    else
      result -> EXLA.Defn.Buffer.to_nx!(result, outputs)
    after
      EXLA.Defn.Lock.unlock(ref)
    end
  end

  defp run_key(%{client: %{ref: ref}, device_id: device_id}), do: [ref | device_id]

  defp compile(client, key, vars, fun, options, to_split, to_computation) do
    expr_args = for var <- vars, do: nx_to_expr_key!(var)
    expr_key = {key, expr_args}

    {expr, {used_inputs, outputs}} =
      EXLA.Defn.LockedCache.run(expr_key, fn ->
        expr = fun.(vars)
        {expr, {used_inputs(expr), outputs(expr)}}
      end)

    {non_buffers, used_inputs} = to_split.(vars, used_inputs)
    {inputs, cache_args} = filter_inputs(vars, used_inputs)
    cache_args = Enum.map(non_buffers, &nx_to_cache_key!/1) ++ cache_args
    cache_key = {key, cache_args, client.name, options}

    {_, executable_extra} =
      EXLA.Defn.LockedCache.run(cache_key, fn ->
        shapes = Enum.map(inputs, &nx_to_shape!/1)
        {computation, extra} = to_computation.(expr || fun.(vars), Enum.zip(used_inputs, shapes))
        executable = EXLA.Computation.compile(computation, client, shapes)
        {nil, {executable, extra}}
      end)

    {inputs, outputs, executable_extra}
  end

  defp used_inputs(expr) do
    {_, {_, used_inputs}} = Tree.composite(expr, {%{}, %{}}, &used_inputs/2)
    used_inputs |> Map.keys() |> Enum.sort()
  end

  defp used_inputs(%T{data: %Expr{op: :parameter, args: [i], context: :root}} = t, {seen, used}),
    do: {t, {seen, Map.put(used, i, true)}}

  defp used_inputs(%T{data: %Expr{id: id}} = t, {seen, used}) do
    case seen do
      %{^id => true} -> {t, {seen, used}}
      %{} -> Tree.traverse_args(t, {Map.put(seen, id, true), used}, &used_inputs/2)
    end
  end

  defp outputs(%T{} = t),
    do: Nx.to_template(t)

  defp outputs(tuple) when is_tuple(tuple),
    do: {:tuple, tuple |> Tuple.to_list() |> Enum.map(&outputs/1)}

  defp outputs(map) when is_struct(map) do
    out =
      map
      |> Map.from_struct()
      |> Enum.sort()
      |> Enum.map(fn {k, v} -> {k, outputs(v)} end)

    {:struct, out, map.__struct__}
  end

  defp outputs(map) when is_map(map),
    do: {:map, map |> Enum.sort() |> Enum.map(fn {k, v} -> {k, outputs(v)} end)}

  defp to_root_computation(key, expr, used_shapes, options) do
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

    computation =
      expr
      |> recur_flatten(state, %{})
      |> elem(0)
      |> EXLA.Builder.build()

    {computation, :ok}
  end

  defp recur_flatten(composite, state, cache) do
    {acc, cache} = recur_flatten(composite, [], state, cache)
    {EXLA.Op.tuple(state.builder, Enum.reverse(acc)), cache}
  end

  defp recur_flatten(%T{} = expr, acc, state, cache) do
    {expr, cache} = recur_operator(expr, state, cache)
    {[expr | acc], cache}
  end

  defp recur_flatten(tuple, acc, state, cache) when is_tuple(tuple) do
    list = Tuple.to_list(tuple)

    Enum.reduce(list, {acc, cache}, fn expr, {acc, cache} ->
      recur_flatten(expr, acc, state, cache)
    end)
  end

  defp recur_flatten(map, acc, state, cache) when is_struct(map) do
    map
    |> Map.from_struct()
    |> Enum.sort()
    |> Enum.reduce({acc, cache}, fn {_, expr}, {acc, cache} ->
      recur_flatten(expr, acc, state, cache)
    end)
  end

  defp recur_flatten(map, acc, state, cache) when is_map(map) do
    map
    |> Enum.sort()
    |> Enum.reduce({acc, cache}, fn {_, expr}, {acc, cache} ->
      recur_flatten(expr, acc, state, cache)
    end)
  end

  ## Operator handling

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
    {initial, cache} = recur_composite(initial, state, cache)
    pred = recur_computation(:while_pred, [arg], pred, {:pred, 8}, state)
    body = recur_computation(:while_body, [arg], body, :any, state)
    {EXLA.Op.while(pred, body, initial), cache}
  end

  defp cached_recur_operator(:cond, %T{data: %Expr{args: args}} = t, state, cache) do
    [clauses, last] = args

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
  end

  defp cached_recur_operator(:parameter, %T{data: %Expr{args: [i]}}, state, cache) do
    {Map.fetch!(state.params, i), cache}
  end

  defp cached_recur_operator(:fun, expr, _state, cache) do
    {expr, cache}
  end

  defp cached_recur_operator(op, expr, state, cache) do
    {args, cache} = Tree.traverse_args(expr, cache, &recur_operator(&1, state, &2))
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
        to_constant(state.builder, Nx.to_scalar(tensor), tensor.type)

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
    iota_type = Nx.Type.merge_scalar({:u, 8}, n)
    iota_shape = EXLA.Shape.make_shape(iota_type, {n, n})

    i0 = EXLA.Op.iota(state.builder, iota_shape, 0)
    i1 = EXLA.Op.iota(state.builder, iota_shape, 1)
    to_type(EXLA.Op.equal(i0, i1), type)
  end

  ## to_operator shape

  defp to_operator(:reshape, [op, shape], _ans, _state) do
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

  defp to_operator(:elem, [op, index, _size], _ans, _state) do
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

  defp to_operator(:outer, [left, right], %{type: type, shape: shape}, _state) do
    left =
      left
      |> to_type(type)
      |> EXLA.Op.reshape({Nx.size(op_shape(left))})
      |> EXLA.Op.broadcast_in_dim(shape, {0})

    right =
      right
      |> to_type(type)
      |> EXLA.Op.reshape({Nx.size(op_shape(right))})
      |> EXLA.Op.broadcast_in_dim(shape, {1})

    EXLA.Op.multiply(left, right)
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

  @unary_op [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
              [:bitwise_not, :count_leading_zeros, :population_count, :cosh, :sinh, :acos] ++
              [:asin, :atan, :floor, :ceil, :round, :acosh, :asinh, :atanh, :erf] ++
              [:erfc, :erf_inv]

  defp to_operator(op, [arg], %{type: type}, _state) when op in @unary_op do
    apply(EXLA.Op, op, [to_type(arg, type)])
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

  defp to_operator(:all?, [arg, opts], _ans, state) do
    to_aggregate(:all?, {:pred, 8}, {}, arg, 1, opts, state, binary_op_fun(:bitwise_and))
  end

  defp to_operator(:any?, [arg, opts], _ans, state) do
    to_aggregate(:any?, {:pred, 8}, {}, arg, 0, opts, state, binary_op_fun(:bitwise_or))
  end

  defp to_operator(:sum, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:sum, type, shape, arg, 0, opts, state, binary_op_fun(:add))
  end

  defp to_operator(:product, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:product, type, shape, arg, 1, opts, state, binary_op_fun(:multiply))
  end

  defp to_operator(:reduce_max, [arg, opts], %{type: type, shape: shape}, state) do
    min_value = EXLA.Lib.min_value(state.builder, type)
    to_aggregate(:reduce_max, type, shape, arg, min_value, opts, state, binary_op_fun(:max))
  end

  defp to_operator(:reduce_min, [arg, opts], %{type: type, shape: shape}, state) do
    max_value = EXLA.Lib.max_value(state.builder, type)
    to_aggregate(:reduce_max, type, shape, arg, max_value, opts, state, binary_op_fun(:min))
  end

  defp to_operator(:reduce, [arg, acc, opts, fun], %{type: type, shape: shape}, state) do
    arg = to_type(arg, type)
    comp = recur_computation(fun, type, state)
    keep_axes = opts[:keep_axes]
    result = EXLA.Op.reduce(arg, to_type(acc, type), comp, reduce_axes(arg, opts[:axes]))

    if keep_axes do
      EXLA.Op.reshape(result, shape)
    else
      result
    end
  end

  defp to_operator(:window_sum, [arg, window_dims, opts], %{type: type}, state) do
    to_window_aggregate(:window_sum, type, arg, 0, window_dims, opts, state, binary_op_fun(:add))
  end

  defp to_operator(:window_max, [arg, window_dims, opts], %{type: type}, state) do
    min_value = EXLA.Lib.min_value(state.builder, type)
    fun = binary_op_fun(:max)
    to_window_aggregate(:window_max, type, arg, min_value, window_dims, opts, state, fun)
  end

  defp to_operator(:window_min, [arg, window_dims, opts], %{type: type}, state) do
    max_value = EXLA.Lib.max_value(state.builder, type)
    fun = binary_op_fun(:min)
    to_window_aggregate(:window_min, type, arg, max_value, window_dims, opts, state, fun)
  end

  defp to_operator(:window_product, [arg, window_dims, opts], %{type: type}, state) do
    fun = binary_op_fun(:multiply)
    to_window_aggregate(:window_product, type, arg, 1, window_dims, opts, state, fun)
  end

  defp to_operator(
         :reduce_window,
         [arg, acc, window_dimensions, opts, fun],
         %{type: type},
         state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]
    window_dilations = opts[:window_dilations]

    arg = to_type(arg, type)
    comp = recur_computation(fun, type, state)

    EXLA.Op.reduce_window(
      arg,
      to_type(acc, type),
      comp,
      window_dimensions,
      strides,
      window_dilations,
      padding_config
    )
  end

  defp to_operator(
         :scatter_window_max,
         [arg, source, window_dimensions, opts, init_value],
         %{type: type},
         state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]

    select_fn = to_computation(:scatter_window_max_select, args, state, binary_op_fun(:greater))
    scatter_fn = to_computation(:scatter_window_max_scatter, args, state, binary_op_fun(:add))

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
         :scatter_window_min,
         [arg, source, window_dimensions, opts, init_value],
         %{type: type},
         state
       ) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]

    select_fn = to_computation(:scatter_window_min_select, args, state, binary_op_fun(:less))
    scatter_fn = to_computation(:scatter_window_min_scatter, args, state, binary_op_fun(:add))

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
         :scatter_add,
         [target, indices, updates],
         %{type: type},
         state
       ) do
    target = to_type(target, type)
    updates = to_type(updates, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    scatter_fn = to_computation(:scatter_add_addition, args, state, binary_op_fun(:add))

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

  defp to_operator(:map, [arg, _opts, fun], %{shape: shape, type: type}, state) do
    arg = to_type(arg, type)
    comp = recur_computation(fun, type, state)
    EXLA.Op.map(arg, comp, Nx.axes(shape))
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

    fun =
      case opts[:direction] do
        :asc -> binary_op_fun(:less)
        :desc -> binary_op_fun(:greater)
      end

    args = [%{type: ans.type, shape: {}}, %{type: ans.type, shape: {}}]
    comp = to_computation(:comparator, args, state, fun)

    EXLA.Op.sort(tensor, comp, dimension)
  end

  defp to_operator(:argsort, [tensor, opts], ans, state) do
    dimension = opts[:axis]

    fun =
      case opts[:direction] do
        :asc -> binary_op_fun(:less)
        :desc -> binary_op_fun(:greater)
      end

    args = [
      %{type: op_type(tensor), shape: {}},
      %{type: op_type(tensor), shape: {}},
      %{type: ans.type, shape: {}},
      %{type: ans.type, shape: {}}
    ]

    comp = to_computation(:comparator, args, state, fun)
    EXLA.Lib.argsort(state.builder, tensor, dimension, comp, ans.type)
  end

  ## Computation helpers

  defp recur_computation(name, args, expr, type, state) do
    to_computation(name, args, state, fn state ->
      expr
      |> recur_composite(state, %{})
      |> elem(0)
      |> to_type(type)
    end)
  end

  defp recur_computation(%T{data: %Expr{op: :fun, args: args}}, type, state) do
    [args, expr, {_, name, _}] = args
    recur_computation(name, args, expr, type, state)
  end

  defp to_computation(name, args, state, fun) do
    subbuilder = subbuilder(state.builder, Atom.to_string(name))

    arg_params =
      Enum.with_index(args, fn arg, i ->
        fun_shape = computation_arg_shape(arg)
        {arg, EXLA.Op.parameter(subbuilder, i, fun_shape, "p#{i}")}
      end)

    {_, params} = Enum.reduce(arg_params, {0, []}, &computation_arg_param(&1, &2))
    state = %{state | builder: subbuilder, params: Map.new(params)}
    EXLA.Builder.build(fun.(state))
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

  defp computation_arg_param({%{}, param}, {counter, acc}) do
    {counter + 1, [{counter, param} | acc]}
  end

  defp computation_arg_param({tuple, param}, counter_acc) do
    tuple
    |> Tuple.to_list()
    |> Enum.with_index(fn arg, i -> {arg, EXLA.Op.get_tuple_element(param, i)} end)
    |> Enum.reduce(counter_acc, &computation_arg_param/2)
  end

  defp recur_composite(tuple, state, cache) when is_tuple(tuple) do
    list = Tuple.to_list(tuple)

    if expr = full_tuple(list, tuple_size(tuple)) do
      recur_composite(expr, state, cache)
    else
      {elements, cache} = Enum.map_reduce(list, cache, &recur_composite(&1, state, &2))
      {EXLA.Op.tuple(state.builder, elements), cache}
    end
  end

  defp recur_composite(expr, state, cache) do
    recur_operator(expr, state, cache)
  end

  # If each element of the tuple is just a reference to the parent expression,
  # discard the tuple elements and return the parent expression.
  defp full_tuple(list, size) do
    with [%T{data: %Expr{op: :elem, args: args}} | rest] <- list,
         [%T{data: %Expr{id: id}} = expr, 0, ^size] <- args,
         true <- rest |> Enum.with_index(1) |> Enum.all?(&full_tuple?(&1, id, size)) do
      expr
    else
      _ -> nil
    end
  end

  defp full_tuple?({arg, index}, id, size) do
    match?(%T{data: %Expr{op: :elem, args: [%T{data: %Expr{id: ^id}}, ^index, ^size]}}, arg)
  end

  ## Aggregation

  defp binary_op_fun(op) do
    fn %{params: %{0 => arg0, 1 => arg1}} -> apply(EXLA.Op, op, [arg0, arg1]) end
  end

  defp to_aggregate(name, type, shape, arg, initial, opts, state, fun) do
    arg = to_type(arg, type)

    acc =
      case initial do
        %EXLA.Op{} = initial ->
          initial

        initial when is_number(initial) ->
          EXLA.Op.constant_r0(state.builder, initial, type)
      end

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    comp = to_computation(name, args, state, fun)
    keep_axes = opts[:keep_axes]
    result = EXLA.Op.reduce(arg, acc, comp, reduce_axes(arg, opts[:axes]))

    if keep_axes do
      EXLA.Op.reshape(result, shape)
    else
      result
    end
  end

  defp to_window_aggregate(name, type, arg, initial, window_dimensions, opts, state, fun) do
    arg = to_type(arg, type)

    acc =
      case initial do
        %EXLA.Op{} = initial ->
          initial

        initial when is_number(initial) ->
          EXLA.Op.constant_r0(state.builder, initial, type)
      end

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    comp = to_computation(name, args, state, fun)

    strides = opts[:strides]
    padding = opts[:padding]
    window_dilations = opts[:window_dilations]

    EXLA.Op.reduce_window(arg, acc, comp, window_dimensions, strides, window_dilations, padding)
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
      %{} -> Tree.traverse_args(t, Map.put(ids, id, true), &collect_ids/2)
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
          Tree.traverse_args(expr, {cache, ids}, &collect_args(&1, &2, pred_ids))

        expr = put_in(expr.data.args, args)
        {expr, {Map.put(cache, id, expr), ids}}
    end
  end

  defp to_if_branch(bool, expr, ids, state, cache) do
    {expr, {_cache, ids_args}} = Tree.composite(expr, {%{}, %{}}, &collect_args(&1, &2, ids))
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

    tuple_shape = EXLA.Shape.make_tuple_shape(shapes)
    param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p")

    params =
      for {_, {i, _, _}} <- sorted_ids_args do
        {i, EXLA.Op.get_tuple_element(param, i)}
      end

    comp =
      expr
      |> recur_composite(%{state | builder: subbuilder, params: Map.new(params)}, %{})
      |> elem(0)
      |> EXLA.Builder.build()

    {EXLA.Op.tuple(state.builder, args), comp, cache}
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

  defp to_type(op, :any), do: op

  defp to_type(op, type) do
    if op_type(op) == type, do: op, else: EXLA.Op.convert_element_type(op, type)
  end

  defp merge_type({:pred, 8}, {:pred, 8}), do: {:pred, 8}
  defp merge_type(left, right), do: Nx.Type.merge(to_nx_type(left), to_nx_type(right))

  defp to_constant(builder, constant, type) do
    EXLA.Op.constant_r0(builder, constant, type)
  end

  defp subbuilder(%EXLA.Builder{name: name} = builder, desc) do
    suffix = System.unique_integer([:positive])
    EXLA.Builder.new(builder, name <> "-" <> desc <> "-" <> Integer.to_string(suffix))
  end

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  # Helpers

  defp filter_inputs(vars, inputs), do: filter_inputs(vars, 0, inputs, [], [])

  defp filter_inputs([var | vars], i, [i | inputs], buffers, cache) do
    i = i + 1
    filter_inputs(vars, i, inputs, [var | buffers], [nx_to_cache_key!(var) | cache])
  end

  defp filter_inputs([var | vars], i, inputs, buffers, cache) do
    filter_inputs(vars, i + 1, inputs, buffers, [nx_to_cache_key!(var) | cache])
  end

  defp filter_inputs([], _i, [], buffers, cache) do
    {Enum.reverse(buffers), cache}
  end

  defp nx_to_shape!(%T{type: type, shape: shape}), do: EXLA.Shape.make_shape(type, shape)
  defp nx_to_cache_key!(%T{type: type, shape: shape}), do: {type, shape}
  defp nx_to_expr_key!(%T{type: type, shape: shape, names: names}), do: {type, shape, names}

  defp delete_slice(enumerable, index, length) do
    {left, right} = Enum.split(enumerable, index)
    left ++ Enum.drop(right, length)
  end
end

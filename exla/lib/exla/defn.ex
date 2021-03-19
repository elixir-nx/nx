defmodule EXLA.Defn do
  @moduledoc false

  alias Nx.Defn.{Expr, Tree}
  alias Nx.Tensor, as: T

  @aot_runtimes %{
    dot: :matmul,
    conv: :conv2d
  }

  @doc false
  def __aot__(output_dir, module, tuples, aot_options) do
    comps_and_exprs =
      for {name, fun, vars, options} <- tuples do
        expr = fun.(vars)

        shapes =
          for %{shape: shape, type: type} <- vars do
            EXLA.Shape.make_shape(type, shape)
          end

        computation = to_root_computation(name, expr, shapes, options)
        {{computation, name, shapes}, expr}
      end

    {comps, exprs} = Enum.unzip(comps_and_exprs)
    {_, runtimes} = exprs |> List.to_tuple() |> Tree.composite(%{}, &aot_runtimes/2)
    aot_options = Keyword.put_new(aot_options, :runtimes, Map.keys(runtimes))

    case EXLA.AOT.compile(output_dir, module, comps, aot_options) do
      {:ok, nif} -> {:ok, exprs, nif}
      {:error, exception} -> {:error, exception}
    end
  end

  defp aot_runtimes(%T{data: %Expr{op: :fun, args: args}}, acc) do
    [_, expr, _] = args
    Tree.composite(expr, acc, &aot_runtimes/2)
  end

  defp aot_runtimes(%T{data: %Expr{op: op}} = expr, acc) do
    acc = if runtime = @aot_runtimes[op], do: Map.put(acc, runtime, true), else: acc
    aot_runtimes_args(expr, acc)
  end

  defp aot_runtimes_args(expr, acc) do
    {_, acc} = Tree.traverse_args(expr, acc, &aot_runtimes/2)
    {expr, acc}
  end

  @doc false
  def __async__(key, vars, fun, options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])
    {buffers, holes, executable} = compile(key, vars, fun, compile_options)

    async_exec = EXLA.Executable.async_run(executable, buffers, run_options)
    %EXLA.Async{executable: async_exec, holes: holes}
  end

  @doc false
  def __await__(%EXLA.Async{executable: async_exec, holes: holes}) do
    async_exec
    |> EXLA.Executable.await_run()
    |> buffer_to_nx(holes)
  end

  @doc false
  def __jit__(key, vars, fun, options) do
    {run_options, compile_options} = Keyword.pop(options, :run_options, [])
    {buffers, holes, executable} = compile(key, vars, fun, compile_options)

    executable
    |> EXLA.Executable.run(buffers, run_options)
    |> buffer_to_nx(holes)
  end

  defp compile(key, vars, fun, options) do
    expr_args = for var <- vars, do: nx_to_expr_key!(var)
    expr_key = {key, expr_args}

    {expr, holes} =
      EXLA.LockedCache.run(expr_key, fn ->
        expr = fun.(vars)
        {expr, holes(expr)}
      end)

    # TODO: We should extract the client and device ordinal from buffers first
    # TODO: Rename :client to :default_client
    {client_name, options} = Keyword.pop(options, :client, :default)
    buffers = for var <- vars, do: nx_to_buffer(var)
    cache_args = for var <- vars, do: nx_to_cache_key!(var)
    cache_key = {key, cache_args, client_name, options}

    {_, executable} =
      EXLA.LockedCache.run(cache_key, fn ->
        shapes = Enum.map(buffers, & &1.shape)
        computation = to_root_computation(key, expr || fun.(vars), shapes, options)
        client = EXLA.Client.fetch!(client_name)
        executable = EXLA.Computation.compile(computation, client, Enum.map(buffers, & &1.shape))
        :persistent_term.put(cache_key, executable)
        {nil, executable}
      end)

    {buffers, holes, executable}
  end

  defp holes(tuple) when is_tuple(tuple),
    do: {:tuple, tuple |> Tuple.to_list() |> Enum.map(&holes/1)}

  defp holes(%T{} = t),
    do: %{t | data: nil}

  defp to_root_computation(key, expr, shapes, options) do
    builder = EXLA.Builder.new(inspect(key))

    # TODO: Use Enum.with_index on Elixir v1.12
    params =
      for {shape, i} <- Enum.with_index(shapes) do
        EXLA.Op.parameter(builder, i, shape, "p#{i}")
      end

    state = %{
      precision: Keyword.get(options, :precision, :default),
      builder: builder,
      params: params
    }

    expr
    |> to_root_result(state, %{})
    |> EXLA.Builder.build()
  end

  defp to_root_result(tuple_or_expr, state, cache) do
    {acc, _cache} = to_root_result(tuple_or_expr, [], state, cache)
    EXLA.Op.tuple(state.builder, Enum.reverse(acc))
  end

  defp to_root_result(tuple, acc, state, cache) when is_tuple(tuple) do
    list = Tuple.to_list(tuple)

    Enum.reduce(list, {acc, cache}, fn expr, {acc, cache} ->
      to_root_result(expr, acc, state, cache)
    end)
  end

  defp to_root_result(expr, acc, state, cache) do
    {expr, cache} = recur_operator(expr, state, cache)
    {[expr | acc], cache}
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

  defp cached_recur_operator(:cond, %T{data: %Expr{args: [clauses, last]}} = t, state, cache) do
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
    {Enum.fetch!(state.params, i), cache}
  end

  defp cached_recur_operator(:fun, expr, _state, cache) do
    {expr, cache}
  end

  defp cached_recur_operator(op, expr, state, cache) do
    {args, cache} = Tree.traverse_args(expr, cache, &recur_operator(&1, state, &2))
    {to_operator(op, args, expr, state), cache}
  end

  ## to_operator creation

  defp to_operator(:scalar, [scalar], ans, state) do
    op = to_constant(state.builder, scalar, ans.type)

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

  defp to_operator(:pad, [op, value, padding_config], _ans, _state) do
    EXLA.Op.pad(op, value, padding_config)
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

  defp to_operator(:dot, [left, axes1, right, axes2], %{type: type}, state) do
    precision = state.precision
    EXLA.Op.dot_general(to_type(left, type), to_type(right, type), {axes1, axes2}, precision)
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

  defp to_operator(:triangular_solve, [a, b, _opts], %{type: type}, _state) do
    b_shape = EXLA.Op.get_shape(b).dims

    b =
      b
      |> to_type(type)
      |> EXLA.Op.reshape(Tuple.append(b_shape, 1))

    to_type(a, type)
    |> EXLA.Op.triangular_solve(b, true, true, false, :none)
    |> EXLA.Op.reshape(b_shape)
  end

  defp to_operator(:lu, [{_, _, _}, _tensor, _opts], _ans, _state) do
    raise ArgumentError, "XLA does not currently support the LU operation"
  end

  defp to_operator(:qr, [{%{type: type}, %{type: type}}, tensor, opts], _ans, state) do
    {q, r} = EXLA.Op.qr(to_type(tensor, type), opts[:mode] != :reduced, state.precision)
    EXLA.Op.tuple(state.builder, [q, r])
  end

  defp to_operator(:svd, [{%{type: type}, %{type: type}, %{type: type}}, tensor, _opts], _ans, state) do
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
    type = Nx.Type.merge(left_shape.dtype, right_shape.dtype)
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
    to_type(arg, type, :bitcast)
  end

  ## to_operator reduction

  defp to_operator(:all?, [arg, opts], _ans, state) do
    to_aggregate(
      :all?,
      {:pred, 8},
      {},
      arg,
      1,
      opts,
      state,
      &apply(EXLA.Op, :bitwise_and, &1.params)
    )
  end

  defp to_operator(:any?, [arg, opts], _ans, state) do
    to_aggregate(
      :any?,
      {:pred, 8},
      {},
      arg,
      0,
      opts,
      state,
      &apply(EXLA.Op, :bitwise_or, &1.params)
    )
  end

  defp to_operator(:sum, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(:sum, type, shape, arg, 0, opts, state, &apply(EXLA.Op, :add, &1.params))
  end

  defp to_operator(:product, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(
      :product,
      type,
      shape,
      arg,
      1,
      opts,
      state,
      &apply(EXLA.Op, :multiply, &1.params)
    )
  end

  defp to_operator(:reduce_max, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(
      :reduce_max,
      type,
      shape,
      arg,
      EXLA.Lib.min_value(state.builder, type),
      opts,
      state,
      &apply(EXLA.Op, :max, &1.params)
    )
  end

  defp to_operator(:reduce_min, [arg, opts], %{type: type, shape: shape}, state) do
    to_aggregate(
      :reduce_max,
      type,
      shape,
      arg,
      EXLA.Lib.max_value(state.builder, type),
      opts,
      state,
      &apply(EXLA.Op, :min, &1.params)
    )
  end

  defp to_operator(:reduce, [arg, acc, opts, fun], %{type: type, shape: shape}, state) do
    arg = to_type(arg, type)
    comp = to_computation(fun, type, state)
    keep_axes = opts[:keep_axes]
    result = EXLA.Op.reduce(arg, to_type(acc, type), comp, reduce_axes(arg, opts[:axes]))

    if keep_axes do
      EXLA.Op.reshape(result, shape)
    else
      result
    end
  end

  defp to_operator(:window_sum, [arg, window_dimensions, opts], %{type: type}, state) do
    to_window_aggregate(
      :window_sum,
      type,
      arg,
      0,
      window_dimensions,
      opts,
      state,
      &apply(EXLA.Op, :add, &1.params)
    )
  end

  defp to_operator(:window_max, [arg, window_dimensions, opts], %{type: type}, state) do
    to_window_aggregate(
      :window_max,
      type,
      arg,
      EXLA.Lib.min_value(state.builder, type),
      window_dimensions,
      opts,
      state,
      &apply(EXLA.Op, :max, &1.params)
    )
  end

  defp to_operator(:window_min, [arg, window_dimensions, opts], %{type: type}, state) do
    to_window_aggregate(
      :window_min,
      type,
      arg,
      EXLA.Lib.max_value(state.builder, type),
      window_dimensions,
      opts,
      state,
      &apply(EXLA.Op, :min, &1.params)
    )
  end

  defp to_operator(:window_product, [arg, window_dimensions, opts], %{type: type}, state) do
    to_window_aggregate(
      :window_product,
      type,
      arg,
      1,
      window_dimensions,
      opts,
      state,
      &apply(EXLA.Op, :multiply, &1.params)
    )
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
    comp = to_computation(fun, type, state)

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

  defp to_operator(:scatter_window_max, [arg, source, window_dimensions, opts, init_value], %{type: type}, state) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    select_fn = to_computation(:scatter_window_max_select, args, state, &apply(EXLA.Op, :greater, &1.params))
    scatter_fn = to_computation(:scatter_window_max_scatter, args, state, &apply(EXLA.Op, :add, &1.params))

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

  defp to_operator(:scatter_window_min, [arg, source, window_dimensions, opts, init_value], %{type: type}, state) do
    padding_config = opts[:padding]
    strides = opts[:strides]

    arg = to_type(arg, type)
    source = to_type(source, type)
    init_value = to_type(init_value, type)

    args = [%{type: type, shape: {}}, %{type: type, shape: {}}]
    select_fn = to_computation(:scatter_window_min_select, args, state, &apply(EXLA.Op, :less, &1.params))
    scatter_fn = to_computation(:scatter_window_min_scatter, args, state, &apply(EXLA.Op, :add, &1.params))

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

  defp to_operator(:map, [arg, fun], %{shape: shape, type: type}, state) do
    arg = to_type(arg, type)
    comp = to_computation(fun, type, state)
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

  defp to_operator(:slice, [tensor, start_indices, lengths, strides], _ans, _state) do
    # TODO: Use Enum.zip_with on Elixir v1.12
    limit_indices =
      start_indices
      |> Enum.zip(lengths)
      |> Enum.map(fn {i, len} -> i + len end)

    EXLA.Op.slice(tensor, start_indices, limit_indices, strides)
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

    EXLA.Op.select(EXLA.Op.equal(cholesky, tensor), zeros, cholesky)
  end

  defp to_operator(:sort, [tensor, opts, comparator], _ans, state) do
    dimension = opts[:axis]
    comp = to_computation(comparator, {:pred, 8}, state)
    EXLA.Op.sort(tensor, comp, dimension)
  end

  ## Computation helpers

  defp to_computation(%T{data: %Expr{op: :fun, args: [args, expr, fun]}}, type, state) do
    {:name, name} = Function.info(fun, :name)

    to_computation(name, args, state, fn state ->
      expr
      |> to_computation_result(state, %{})
      |> elem(0)
      |> to_type(type)
    end)
  end

  defp to_computation(name, args, state, fun) do
    subbuilder = subbuilder(state.builder, Atom.to_string(name))

    # TODO: Use Enum.with_index on Elixir v1.12
    params =
      for {%{type: type, shape: shape}, i} <- Enum.with_index(args) do
        fun_shape = EXLA.Shape.make_shape(type, shape)
        EXLA.Op.parameter(subbuilder, i, fun_shape, "p#{i}")
      end

    state = %{state | builder: subbuilder, params: params}
    EXLA.Builder.build(fun.(state))
  end

  defp to_computation_result(tuple, state, cache) when is_tuple(tuple) do
    list = Tuple.to_list(tuple)

    if expr = full_tuple(list, tuple_size(tuple)) do
      to_computation_result(expr, state, cache)
    else
      {elements, cache} = Enum.map_reduce(list, cache, &to_computation_result(&1, state, &2))
      {EXLA.Op.tuple(state.builder, elements), cache}
    end
  end

  defp to_computation_result(expr, state, cache) do
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
    {_, pred_ids} = collect_ids(pred, %{})

    {pred_op, cache} = recur_operator(pred, state, cache)
    pred_op = to_type(pred_op, {:pred, 8})

    {true_args, true_comp} = to_if_branch(true, on_true, pred_ids, state, cache)
    {false_args, false_comp} = to_if_branch(false, on_false, pred_ids, state, cache)
    {EXLA.Op.conditional(pred_op, true_args, true_comp, false_args, false_comp), cache}
  end

  defp collect_ids(%T{data: %Expr{id: id}} = t, ids) do
    case ids do
      %{^id => true} -> {t, ids}
      %{} -> Tree.traverse_args(t, Map.put(ids, id, true), &collect_ids/2)
    end
  end

  defp collect_args(%T{data: %Expr{id: id, op: op}} = expr, ids, pred_ids) do
    if Map.has_key?(pred_ids, id) or op == :parameter do
      case ids do
        %{^id => {_, _, new}} ->
          {new, ids}

        %{} ->
          i = map_size(ids)
          param = Expr.parameter(expr, i)
          {param, Map.put(ids, id, {i, expr, param})}
      end
    else
      {args, ids} = Tree.traverse_args(expr, ids, &collect_args(&1, &2, pred_ids))
      {put_in(expr.data.args, args), ids}
    end
  end

  defp to_if_branch(bool, expr, ids, state, cache) do
    {expr, ids_args} = Tree.composite(expr, %{}, &collect_args(&1, &2, ids))
    sorted_ids_args = Enum.sort_by(ids_args, fn {_id, {i, _old, _new}} -> i end)
    subbuilder = subbuilder(state.builder, "if-#{Atom.to_string(bool)}")

    shapes =
      for {_, {_, _, %{type: type, shape: shape}}} <- sorted_ids_args do
        EXLA.Shape.make_shape(type, shape)
      end

    tuple_shape = EXLA.Shape.make_tuple_shape(shapes)
    param = EXLA.Op.parameter(subbuilder, 0, tuple_shape, "p")

    params =
      for {_, {i, _, _}} <- sorted_ids_args do
        EXLA.Op.get_tuple_element(param, i)
      end

    comp =
      expr
      |> to_computation_result(%{state | builder: subbuilder, params: params}, %{})
      |> elem(0)
      |> EXLA.Builder.build()

    args =
      Enum.map(sorted_ids_args, fn
        {_, {_, %T{data: %Expr{op: :parameter, args: [i]}}, _}} -> Enum.fetch!(state.params, i)
        {id, {_, _, _}} -> Map.fetch!(cache, id)
      end)

    {EXLA.Op.tuple(state.builder, args), comp}
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

  ## Op Helpers

  defp op_type(op), do: EXLA.Op.get_shape(op).dtype
  defp op_shape(op), do: EXLA.Op.get_shape(op).dims

  defp to_type(op, type, cast \\ :convert) do
    fun =
      case cast do
        :convert -> &EXLA.Op.convert_element_type(&1, type)
        :bitcast -> &EXLA.Op.bitcast_convert_type(&1, type)
      end

    if op_type(op) == type, do: op, else: fun.(op)
  end

  defp to_constant(builder, constant, type) do
    EXLA.Op.constant_r0(builder, constant, type)
  end

  defp subbuilder(%EXLA.Builder{name: name} = builder, desc) do
    suffix = System.unique_integer([:positive])
    EXLA.Builder.new(builder, name <> "-" <> desc <> "-" <> Integer.to_string(suffix))
  end

  ## Nx <-> EXLA.Buffer

  defp buffer_to_nx(buffers, holes) do
    {result, []} = each_buffer_to_nx(holes, buffers)
    result
  end

  defp each_buffer_to_nx({:tuple, holes}, acc) when is_list(holes) do
    {exprs, acc} = Enum.map_reduce(holes, acc, &each_buffer_to_nx/2)
    {List.to_tuple(exprs), acc}
  end

  defp each_buffer_to_nx(hole, [%EXLA.Buffer{shape: shape} = buffer | acc]) do
    nx_type = to_nx_type(shape.dtype)
    nx_shape = shape.dims

    if hole.type != nx_type do
      raise "internal bug! Nx.Defn expected a tensor with type #{inspect(hole.type)} " <>
              "but got #{inspect(nx_type)}"
    end

    if hole.shape != nx_shape do
      raise "internal bug! Nx.Defn expected a tensor with shape #{inspect(hole.shape)} " <>
              "but got #{inspect(nx_shape)}"
    end

    {%{hole | data: buffer_to_data(buffer)}, acc}
  end

  defp buffer_to_data(%EXLA.Buffer{ref: ref, data: nil}),
    do: %EXLA.DeviceBackend{state: ref}

  defp buffer_to_data(%EXLA.Buffer{ref: nil, data: data}),
    do: %Nx.BinaryBackend{state: data}

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  defp nx_to_buffer(%T{data: data, type: type, shape: shape} = tensor) do
    case data do
      %EXLA.DeviceBackend{state: ref} ->
        EXLA.Buffer.buffer(ref, EXLA.Shape.make_shape(type, shape))

      _ ->
        # TODO: Call Nx.backend_transfer on the tensor instead
        EXLA.Buffer.buffer(Nx.to_binary(tensor), EXLA.Shape.make_shape(type, shape))
    end
  end

  defp nx_to_cache_key!(%T{type: type, shape: shape}), do: {type, shape}
  defp nx_to_expr_key!(%T{type: type, shape: shape, names: names}), do: {type, shape, names}
end

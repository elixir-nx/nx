defmodule Nx.Defn.Grad do
  @moduledoc false

  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  def transform(to_grad, fun, transform) do
    {to_grad, ids} =
      Composite.traverse(to_grad, %{}, fn to_grad, ids ->
        to_grad = Expr.metadata(to_grad, %{__MODULE__ => :to_grad})
        {to_grad, Map.put(ids, to_grad.data.id, :stop)}
      end)

    # Collect all IDs in the function environment and mark
    # them as stop grads. This is an optimization to avoid
    # traversing trees when not necessary.
    {:env, env} = Function.info(fun, :env)
    ids = stop_grads(env, ids)

    expr = to_grad |> fun.()
    transformed_expr = transform.(expr) |> validate_expr!()
    {parents, nodes} = parents_tree(transformed_expr, ids)
    grads = %{transformed_expr.data.id => [constant(1.0, transformed_expr)]}

    {graded, _} =
      Composite.traverse(to_grad, {%{}, grads}, fn to_grad, acc ->
        to_grad(to_grad, parents, nodes, acc)
      end)

    {expr, graded}
  end

  defp constant(float, shape) do
    shape = Nx.shape(shape)
    names = List.duplicate(nil, tuple_size(shape))
    Expr.constant(%T{shape: shape, type: {:f, 32}, names: names}, float, [])
  end

  defp validate_expr!(%T{data: %Expr{}} = expr) do
    expr
  end

  defp validate_expr!(%T{} = t) do
    raise ArgumentError,
          "can only compute gradients of tensor expressions, got: #{inspect(t)}"
  end

  defp validate_expr!(other) do
    validate_expr!(Expr.tensor(other))
  end

  defp stop_grads(list, ids) when is_list(list),
    do: Enum.reduce(list, ids, &stop_grads/2)

  defp stop_grads(tuple, ids) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.reduce(ids, &stop_grads/2)

  defp stop_grads(%T{data: %Expr{id: id}}, ids),
    do: Map.put(ids, id, :stop)

  defp stop_grads(%T{}, ids),
    do: ids

  defp stop_grads(map, ids) when is_map(map),
    do: map |> Map.values() |> Enum.reduce(ids, &stop_grads/2)

  defp stop_grads(_, ids),
    do: ids

  ## Build the parents tree

  @constants [:constant, :tensor, :parameter, :eye, :iota, :random_uniform, :random_normal] ++
               [:all?, :any?, :argmax, :argmin] ++
               [:bitwise_and, :bitwise_or, :bitwise_xor, :bitwise_not] ++
               [:logical_and, :logical_or, :logical_xor, :logical_not] ++
               [:left_shift, :right_shift, :count_leading_zeros, :population_count] ++
               [:floor, :round, :ceil, :sign, :token] ++
               [:equal, :greater, :greater_equal, :less, :less_equal, :not_equal, :argsort]

  defp parents_tree(expr, nodes) do
    Composite.reduce(expr, {%{}, nodes}, &recur_parents_tree/2)
  end

  defp recur_parents_tree(%T{data: %Expr{id: id, op: op}} = t, {parents, nodes}) do
    case nodes do
      _ when op in @constants ->
        {parents, nodes}

      %{^id => _} ->
        {parents, nodes}

      %{} ->
        nodes = Map.put(nodes, id, t)

        {_, acc} =
          Tree.apply_args(t, {parents, nodes}, fn arg, {parents, nodes} ->
            parents = Map.update(parents, arg.data.id, [id], &[id | &1])
            {arg, recur_parents_tree(arg, {parents, nodes})}
          end)

        acc
    end
  end

  ## Recursion

  defp to_grad(arg, parents, nodes, acc) do
    id = arg.data.id
    {seen, grads} = traverse_parents(id, parents, nodes, acc)

    res =
      case Map.get(grads, id, []) do
        [] -> Expr.tensor(0.0)
        gs -> Enum.reduce(gs, &Nx.add/2)
      end

    {Nx.broadcast(res, arg), {seen, grads}}
  end

  defp traverse_parents(id, parents, nodes, acc) do
    parents
    |> Map.get(id, [])
    |> Enum.reduce(acc, &recur_to_grad(&1, parents, nodes, &2))
  end

  defp recur_to_grad(id, parents, nodes, {seen, grads}) do
    case seen do
      %{^id => _} ->
        {seen, grads}

      %{} ->
        {seen, grads} = traverse_parents(id, parents, nodes, {seen, grads})

        grads =
          case Map.get(grads, id, []) do
            [] ->
              grads

            gs ->
              g = Enum.reduce(gs, &Nx.add/2)
              %T{data: %Expr{op: op, args: args}} = ans = Map.fetch!(nodes, id)
              pairs = grad(op, args, ans, g)

              Enum.reduce(pairs, grads, fn {child, g}, grads ->
                Map.update(grads, child.data.id, [g], &[g | &1])
              end)
          end

        {Map.put(seen, id, true), grads}
    end
  end

  ## Gradients

  defp grad(:metadata, [expr, %{custom_grad: fun}], _ans, g) do
    args = fun.(expr, g)

    unless is_list(args) and Enum.all?(args, &match?({_, _}, &1)) do
      raise "custom_grad/2 must return a list of tuples, " <>
              "where the first element is the expression to continue computing grad " <>
              "and the second element is the updated g"
    end

    args
  end

  defp grad(:metadata, [_, %{stop_grad: true}], _ans, _g) do
    []
  end

  defp grad(:metadata, [expr, _], _ans, g) do
    [{expr, g}]
  end

  defp grad(:select, [pred, on_true, on_false], ans, g) do
    gs = Nx.broadcast(g, ans)
    zeros = Nx.broadcast(Expr.tensor(0.0), ans)

    d_on_true = Nx.select(pred, gs, zeros)
    d_on_false = Nx.select(pred, zeros, gs)

    [{on_true, d_on_true}, {on_false, d_on_false}]
  end

  defp grad(:broadcast, [x, shape, axes], _ans, g) do
    implicit_axes =
      for {a, i} <- Enum.with_index(axes),
          elem(shape, a) != 1 and elem(x.shape, i) == 1,
          do: {a, i}

    {implicit_axes, broadcast_axes} = Enum.unzip(implicit_axes)
    explicit_axes = Nx.axes(shape) -- axes

    g =
      case explicit_axes ++ implicit_axes do
        [] -> g
        sum_axes -> Nx.sum(g, axes: sum_axes)
      end

    g =
      case broadcast_axes do
        [] -> g
        _ -> Nx.broadcast(g, x.shape, axes: Nx.axes(x.shape) -- broadcast_axes)
      end

    [{x, g}]
  end

  defp grad(:clip, [operand, min, max], _ans, g) do
    # w.r.t min
    w_min =
      Nx.select(
        Nx.bitwise_and(Nx.greater(min, operand), Nx.less(min, max)),
        Nx.broadcast(g, operand),
        0.0
      )

    # w.r.t operand
    w_operand =
      Nx.select(
        Nx.bitwise_and(Nx.greater(operand, min), Nx.less(operand, max)),
        g,
        0.0
      )

    # w.r.t max
    w_max = Nx.select(Nx.less(max, operand), Nx.broadcast(g, operand), 0.0)

    [
      {operand, Nx.multiply(g, w_operand)},
      {min, Nx.multiply(g, w_min)},
      {max, Nx.multiply(g, w_max)}
    ]
  end

  defp grad(:squeeze, [x, axes], _ans, g) do
    [{x, Nx.broadcast(g, x.shape, axes: Nx.axes(x.shape) -- axes)}]
  end

  defp grad(:reshape, [x], _ans, g) do
    [{x, Nx.reshape(g, x)}]
  end

  defp grad(:transpose, [x, axes], _ans, g) do
    [{x, Nx.transpose(g, axes: argsort(axes))}]
  end

  defp grad(:pad, [x, value, padding_config], _ans, g) do
    inverse_padding_config = Enum.map(padding_config, fn {lo, hi, _} -> {-lo, -hi, 0} end)
    unpadded = Nx.pad(g, 0.0, inverse_padding_config)

    start_indices = List.duplicate(0, Nx.rank(unpadded))
    lengths = Tuple.to_list(unpadded.shape)
    strides = padding_config |> Enum.map(fn {_, _, interior} -> interior + 1 end)

    g_operand = Nx.slice(unpadded, start_indices, lengths, strides: strides)
    g_value = Nx.subtract(Nx.sum(g), Nx.sum(g_operand))

    [{x, g_operand}, {value, g_value}]
  end

  defp grad(:slice, [x, start_indices, _lengths, strides], _ans, g) do
    padding_config = Enum.map(strides, &{0, 0, &1 - 1})
    pad_value = 0.0
    g = Nx.pad(g, pad_value, padding_config)

    zeros = Nx.broadcast(Expr.tensor(0.0), x)
    [{x, Nx.put_slice(zeros, start_indices, g)}]
  end

  defp grad(:put_slice, [x, start_indices, update], _ans, g) do
    zeros = Nx.broadcast(Expr.tensor(0.0), update)

    operand_t = Nx.put_slice(g, start_indices, zeros)
    update_t = Nx.slice(g, start_indices, Tuple.to_list(Nx.shape(update)))

    [{x, operand_t}, {update, update_t}]
  end

  defp grad(:reverse, [x, axes], _ans, g) do
    [{x, Nx.reverse(g, axes: axes)}]
  end

  defp grad(:sum, [x, opts], _ans, g) do
    [{x, reduce_g(x, opts, g)}]
  end

  defp grad(:product, [x, opts], _ans, g) do
    axes = opts[:axes] || Nx.axes(x)
    non_axes = Nx.axes(x) -- axes
    n = Enum.reduce(axes, 1, fn axis, size -> elem(x.shape, axis) * size end)

    non_axes_shape =
      non_axes
      |> Enum.map(&elem(x.shape, &1))
      |> List.to_tuple()

    permutation = axes ++ non_axes
    new_shape = Tuple.insert_at(non_axes_shape, 0, n)

    operand = Nx.reshape(Nx.transpose(x, axes: permutation), new_shape)
    x = reduce_prod_tree(operand, 0, n, non_axes_shape)
    [{x, g}]
  end

  @reduce_min_max_ops [:reduce_max, :reduce_min]

  defp grad(op, [x, opts], ans, g) when op in @reduce_min_max_ops do
    g = reduce_g(x, opts, g)
    axes = opts[:axes] || Nx.axes(x)

    shape =
      for {d, i} <- Enum.with_index(Tuple.to_list(x.shape)) do
        if i in axes, do: 1, else: d
      end

    locs = Nx.equal(x, Nx.reshape(ans, List.to_tuple(shape)))
    num = Nx.multiply(g, locs)
    den = Nx.sum(locs, axes: axes, keep_axes: true)
    [{x, Nx.divide(num, den)}]
  end

  defp grad(:dot, [x, axes_x, x_batch_axes, y, axes_y, y_batch_axes], ans, g) do
    g = Nx.broadcast(g, ans)

    batch_gx = up_to(0, length(x_batch_axes))
    batch_gy = up_to(0, length(y_batch_axes))

    contract_gx = up_to(Nx.rank(x.shape) - length(axes_x), Nx.rank(g.shape))
    contract_gy = up_to(length(y_batch_axes), Nx.rank(x.shape) - length(axes_x))

    contract_x = (Nx.axes(x.shape) -- axes_x) -- batch_gx
    contract_y = (Nx.axes(y.shape) -- axes_y) -- batch_gy

    transpose_x = Enum.map(argsort(axes_y), &Enum.fetch!(axes_x, &1))
    transpose_y = Enum.map(argsort(axes_x), &Enum.fetch!(axes_y, &1))

    gx =
      g
      |> Nx.dot(contract_gx, batch_gx, y, contract_y, y_batch_axes)
      |> Nx.transpose(axes: argsort(x_batch_axes ++ contract_x ++ transpose_x))

    gy =
      g
      |> Nx.dot(contract_gy, batch_gy, x, contract_x, x_batch_axes)
      |> Nx.transpose(axes: argsort(y_batch_axes ++ contract_y ++ transpose_y))

    [{x, gx}, {y, gy}]
  end

  defp grad(:conv, [x, y, opts], ans, g) do
    grad_conv(x, y, opts, ans, g)
  end

  @window_chooser_op [:window_min, :window_max]

  defp grad(op, [x, window_dimensions, opts], _ans, g) when op in @window_chooser_op do
    padding = opts[:padding]
    strides = opts[:strides]

    fun =
      if op == :window_min,
        do: &Nx.window_scatter_min/5,
        else: &Nx.window_scatter_max/5

    g = fun.(x, g, 0, window_dimensions, padding: padding, strides: strides)
    [{x, g}]
  end

  defp grad(:window_sum, [x, window_dimensions, opts], _, g) do
    strides = opts[:strides]
    window_dilation = opts[:window_dilations]
    base_dilation = List.duplicate(1, Nx.rank(x))
    padding = opts[:padding]

    padding_config =
      conv_lhs_padding(
        x.shape,
        window_dimensions,
        strides,
        g.shape,
        padding,
        base_dilation,
        window_dilation
      )

    padding_config =
      padding_config
      |> Enum.zip(strides)
      |> Enum.map(fn {{lo, hi}, s} -> {lo, hi, s - 1} end)

    g = Nx.pad(g, 0.0, padding_config)

    g =
      Nx.window_sum(
        g,
        window_dimensions,
        strides: base_dilation,
        padding: List.duplicate({0, 0}, Nx.rank(x)),
        window_dilations: window_dilation
      )

    [{x, g}]
  end

  defp grad(:concatenate, [tensors, axis], ans, g) do
    zero_axes = List.duplicate(0, Nx.rank(ans))
    ans_shape_list = Tuple.to_list(ans.shape)

    {pairs, _} =
      Enum.map_reduce(tensors, 0, fn t, limit ->
        t_len = elem(t.shape, axis)
        current_limit = t_len + limit
        start = List.replace_at(zero_axes, axis, limit)
        len = List.replace_at(ans_shape_list, axis, t_len)
        {{t, Nx.slice(g, start, len)}, current_limit}
      end)

    pairs
  end

  defp grad(:cholesky, [input], l, g) do
    num = g |> tril() |> Nx.dot([0], l, [0]) |> Nx.transpose()
    den = l |> Nx.eye() |> Nx.add(1)
    phi_tril = num |> Nx.divide(den) |> tril()

    bm = Nx.LinAlg.triangular_solve(l, phi_tril, transform_a: :transpose)
    dl = Nx.LinAlg.triangular_solve(l, bm, left_side: false)
    [{input, dl}]
  end

  defp grad(:sort, [t, opts], _ans, g) do
    idx = Nx.argsort(t, opts)
    take_along_opts = Keyword.take(opts, [:axis])
    g = Nx.take_along_axis(g, idx, take_along_opts)
    [{t, g}]
  end

  defp grad(:take_along_axis, [t, i, axis], _ans, g) do
    num_elements = i |> Nx.shape() |> Tuple.product()

    # Convert `i`, the take_along_axis indices, to a list of
    # fully qualified (i.e. [0, 2, 1] for a {_, _, _}-shaped tensor)
    # indices

    indices =
      0..(Nx.rank(g) - 1)//1
      |> Enum.map(fn
        # For the axis of interest, we'll use the actual take_along_axis indices
        ^axis ->
          Nx.reshape(i, {num_elements, 1})

        axis ->
          i
          |> Nx.iota(axis: axis)
          |> Nx.reshape({num_elements, 1})
      end)
      |> Nx.concatenate(axis: 1)

    # Since g is produced through the given indices,
    # we can reshape g to be a {num_elements} shaped tensor
    # which will directly correspond to each of the reshaped
    # indices above
    updates = Nx.reshape(g, {num_elements})

    # The intuition for this grad is that for each index taken, we'll
    # add the corresponding result grad to the original
    g =
      t
      |> Expr.broadcast(0, Nx.shape(t), Nx.axes(t))
      |> Nx.indexed_add(indices, updates)

    [{t, g}]
  end

  defp grad(:take, [t, i, axis], _ans, g) do
    axes_range = 0..(Nx.rank(t) - 1)//1

    indices_shape =
      axes_range
      |> Enum.flat_map(fn
        ^axis -> Tuple.to_list(i.shape)
        _ -> [1]
      end)
      |> List.to_tuple()

    idx_tiling =
      t.shape
      |> Tuple.to_list()
      |> Enum.with_index(fn
        _x, ^axis ->
          List.duplicate(1, Nx.rank(i))

        x, _ ->
          x
      end)
      |> List.flatten()

    num_elements = Tuple.product(g.shape)

    indices_for_axis =
      i
      |> Nx.reshape(indices_shape)
      |> Nx.tile(idx_tiling)

    axis_offset = Nx.rank(i) - 1

    indices =
      axes_range
      |> Enum.map(fn
        ^axis ->
          indices_for_axis
          |> Nx.reshape({num_elements, 1})

        current when current < axis ->
          indices_for_axis
          |> Nx.iota(axis: current)
          |> Nx.reshape({num_elements, 1})

        current when current > axis ->
          indices_for_axis
          |> Nx.iota(axis: current + axis_offset)
          |> Nx.reshape({num_elements, 1})
      end)
      |> Nx.concatenate(axis: 1)

    updates = Nx.reshape(g, {num_elements})

    g =
      t
      |> Expr.broadcast(0, Nx.shape(t), Nx.axes(t))
      |> Nx.indexed_add(indices, updates)

    [{t, g}]
  end

  defp grad(:gather, [t, i], _ans, g) do
    rank = Nx.rank(t)
    num_elements = i.shape |> Tuple.product() |> div(rank)

    indices = Nx.reshape(i, {num_elements, rank})
    updates = Nx.reshape(g, {num_elements})

    g = t |> Expr.broadcast(0, t.shape, Nx.axes(t)) |> Nx.indexed_add(indices, updates)
    [{t, g}]
  end

  defp grad(:add, [x, y], _ans, g) do
    [{x, g}, {y, g}]
  end

  defp grad(:subtract, [x, y], _ans, g) do
    [{x, g}, {y, Nx.negate(g)}]
  end

  defp grad(:multiply, [x, y], _ans, g) do
    # TODO: handle case x and y are the same?
    [{x, Nx.multiply(g, y)}, {y, Nx.multiply(g, x)}]
  end

  defp grad(:divide, [x, y], ans, g) do
    [{x, Nx.divide(g, y)}, {y, Nx.multiply(g, Nx.negate(Nx.divide(ans, y)))}]
  end

  defp grad(:remainder, [x, y], _ans, g) do
    [{x, g}, {y, Nx.multiply(g, Nx.negate(Nx.floor(Nx.divide(x, y))))}]
  end

  defp grad(:power, [x, y], ans, g) do
    # Since we do many operations against literals,
    # we try to surface any scalar number.
    sx = surface_nuldim_scalar(x)
    sy = surface_nuldim_scalar(y)

    exponent = Nx.select(Nx.equal(sy, 0.0), 1.0, Nx.subtract(sy, 1.0))
    base = Nx.select(Nx.equal(sx, 0.0), 1.0, sx)

    gx = Nx.multiply(sy, Nx.power(sx, exponent))
    gy = Nx.multiply(Nx.log(base), ans)
    [{x, Nx.multiply(g, gx)}, {y, Nx.multiply(g, gy)}]
  end

  defp grad(:atan2, [x, y], _ans, g) do
    den = Nx.add(Nx.multiply(x, x), Nx.multiply(y, y))
    [{x, Nx.multiply(g, Nx.divide(y, den))}, {y, Nx.multiply(g, Nx.negate(Nx.divide(x, den)))}]
  end

  defp grad(op, [x, y], ans, g) when op in [:min, :max] do
    lhs =
      Nx.divide(
        Nx.select(Nx.equal(x, ans), 1.0, 0.0),
        Nx.select(Nx.equal(y, ans), 2.0, 1.0)
      )

    rhs =
      Nx.divide(
        Nx.select(Nx.equal(y, ans), 1.0, 0.0),
        Nx.select(Nx.equal(x, ans), 2.0, 1.0)
      )

    [{x, Nx.multiply(g, lhs)}, {y, Nx.multiply(g, rhs)}]
  end

  defp grad(:outer, [x, y], _ans, g) do
    x = Nx.reshape(x, {Nx.size(x.shape), 1})
    y = Nx.reshape(y, {1, Nx.size(y.shape)})
    [{x, Nx.multiply(g, y)}, {y, Nx.multiply(g, x)}]
  end

  defp grad(:as_type, [x], _ans, g) do
    [{x, g}]
  end

  defp grad(:bitcast, [x], _ans, g) do
    [{x, g}]
  end

  defp grad(:abs, [x], _ans, g) do
    [{x, Nx.select(Nx.greater_equal(x, 0.0), g, Nx.negate(g))}]
  end

  defp grad(:sqrt, [x], ans, g) do
    [{x, Nx.divide(Nx.multiply(g, 0.5), ans)}]
  end

  defp grad(:cbrt, [x], ans, g) do
    [{x, Nx.divide(g, 3 |> Nx.multiply(ans) |> Nx.multiply(ans))}]
  end

  defp grad(:exp, [x], ans, g) do
    [{x, Nx.multiply(g, ans)}]
  end

  defp grad(:expm1, [x], ans, g) do
    [{x, Nx.multiply(g, Nx.add(ans, 1))}]
  end

  defp grad(:log, [x], _ans, g) do
    [{x, Nx.divide(g, x)}]
  end

  defp grad(:log1p, [x], _ans, g) do
    [{x, Nx.divide(g, Nx.add(x, 1))}]
  end

  defp grad(:logistic, [x], ans, g) do
    gs =
      x
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(ans)
      |> Nx.multiply(ans)

    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:negate, [x], _ans, g) do
    [{x, Nx.negate(g)}]
  end

  defp grad(:rsqrt, [x], _ans, g) do
    [{x, Nx.multiply(Nx.multiply(g, -0.5), Nx.power(x, -1.5))}]
  end

  defp grad(:sin, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.cos(x))}]
  end

  defp grad(:asin, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.rsqrt(Nx.subtract(1.0, Nx.multiply(x, x))))}]
  end

  defp grad(:sinh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.cosh(x))}]
  end

  defp grad(:asinh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.rsqrt(Nx.add(Nx.multiply(x, x), 1.0)))}]
  end

  defp grad(:acosh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.rsqrt(Nx.subtract(Nx.multiply(x, x), 1.0)))}]
  end

  defp grad(:atanh, [x], _ans, g) do
    [{x, Nx.divide(g, Nx.subtract(1.0, Nx.multiply(x, x)))}]
  end

  defp grad(:cos, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.negate(Nx.sin(x)))}]
  end

  defp grad(:acos, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.negate(Nx.rsqrt(Nx.subtract(1.0, Nx.multiply(x, x)))))}]
  end

  defp grad(:cosh, [x], _ans, g) do
    [{x, Nx.multiply(g, Nx.sinh(x))}]
  end

  defp grad(:tan, [x], _ans, g) do
    cos = Nx.cos(x)
    [{x, g |> Nx.divide(cos) |> Nx.divide(cos)}]
  end

  defp grad(:atan, [x], _ans, g) do
    [{x, Nx.divide(g, Nx.add(1.0, Nx.multiply(x, x)))}]
  end

  defp grad(:tanh, [x], ans, g) do
    [{x, Nx.multiply(g, Nx.subtract(1.0, Nx.multiply(ans, ans)))}]
  end

  @half_sqrt_pi :math.sqrt(:math.pi()) / 2
  @two_rsqrt_pi 2 / :math.sqrt(:math.pi())

  defp grad(:erf, [x], _ans, g) do
    gs =
      x
      |> Nx.multiply(x)
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(@two_rsqrt_pi)

    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:erfc, [x], _ans, g) do
    gs =
      x
      |> Nx.multiply(x)
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(-@two_rsqrt_pi)

    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:erf_inv, [x], ans, g) do
    gs = Nx.multiply(@half_sqrt_pi, Nx.exp(Nx.multiply(ans, ans)))
    [{x, Nx.multiply(g, gs)}]
  end

  defp grad(:attach_token, [_, x], _ans, g) do
    [{x, g}]
  end

  defp grad(:quotient, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.quotient/2.

    If a floating point computation is acceptable, consider \
    using an implementation of floor division. See the \
    documentation of `Nx.quotient` for more details.
    """
  end

  defp grad(:reduce, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.reduce/4.

    If you are computing the sum, product, or similar, use the \
    appropriate Nx functions instead. If you have a custom usage \
    of reduce, consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  defp grad(:window_reduce, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.window_reduce/5.

    If you are computing the sum, max, or similar of a window, use \
    the appropriate Nx functions instead. If you have a custom usage \
    of window_reduce, consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  @error [:map, :window_product]

  defp grad(op, args, _, _) when op in @error do
    raise ArgumentError, """
    cannot compute gradient for Nx.#{op}/#{length(args)}.

    Consider using stop_grad/1 to make the gradient equivalent to \
    the identify function or use custom_grad/2 to define a proper \
    gradient implementation
    """
  end

  defp grad(op, args, _, _) do
    raise ArgumentError, """
    gradient not yet implemented for Nx.#{op}/#{length(args)}.

    Please open up an issue so we can implement the missing gradient
    """
  end

  ## Windows

  defp reduce_prod_tree(_, _, 0, non_axes_shape),
    do: Nx.broadcast(Expr.tensor(1.0), non_axes_shape)

  defp reduce_prod_tree(x, axis, 1, _), do: Nx.squeeze(x, axes: [axis])

  defp reduce_prod_tree(x, axis, axis_value, non_axes_shape) do
    n1 = div(axis_value + 1, 2)
    n2 = axis_value - n1

    x1 = Nx.slice_axis(x, 0, n1, axis)
    x2 = Nx.slice_axis(x, n1, n2, axis)

    x2 =
      if n2 != n1 do
        paddings = List.duplicate({0, 0, 0}, Nx.rank(x.shape))
        paddings = List.update_at(paddings, axis, fn _ -> {0, 1, 0} end)
        Nx.pad(x2, 1, paddings)
      else
        x2
      end

    new_operand = Nx.multiply(x1, x2)
    new_axis_value = elem(new_operand.shape, 0)
    reduce_prod_tree(new_operand, axis, new_axis_value, non_axes_shape)
  end

  ## Conv

  defp grad_conv(x, y, opts, ans, g) do
    g = Nx.broadcast(g, ans)

    input_permutation = opts[:input_permutation]
    kernel_permutation = opts[:kernel_permutation]
    output_permutation = opts[:output_permutation]
    strides = opts[:strides]
    padding = opts[:padding]
    lhs_dilation = opts[:input_dilation]
    rhs_dilation = opts[:kernel_dilation]
    feature_group_size = opts[:feature_group_size]
    batch_group_size = opts[:batch_group_size]

    [lhs0, lhs1 | lhs_sdim_axes] = input_permutation
    [rhs0, rhs1 | rhs_sdim_axes] = kernel_permutation
    [_, _ | out_sdim_axes] = output_permutation

    t_lhs_permutation = conv_spec_transpose(input_permutation)
    t_rhs_permutation = conv_spec_transpose(kernel_permutation)
    t_out_permutation = conv_spec_transpose(output_permutation)

    lhs_sdims = conv_sdims(x.shape, lhs_sdim_axes)
    rhs_sdims = conv_sdims(y.shape, rhs_sdim_axes)
    out_sdims = conv_sdims(g.shape, out_sdim_axes)

    rhs =
      cond do
        feature_group_size > 1 ->
          y = reshape_axis_out_of(rhs0, feature_group_size, y)
          reshape_axis_into(rhs0, rhs1, y)

        batch_group_size > 1 ->
          y = reshape_axis_out_of(rhs0, batch_group_size, y)
          reshape_axis_into(rhs0, rhs1, y)

        true ->
          y
      end

    lhs_padding =
      conv_lhs_padding(
        lhs_sdims,
        rhs_sdims,
        strides,
        out_sdims,
        padding,
        lhs_dilation,
        rhs_dilation
      )

    rhs_padding =
      conv_rhs_padding(
        lhs_sdims,
        rhs_sdims,
        strides,
        out_sdims,
        padding,
        lhs_dilation,
        rhs_dilation
      )

    lhs_feature_group_size =
      if batch_group_size > 1, do: batch_group_size, else: feature_group_size

    {rhs_feature_group_size, rhs_batch_group_size} =
      cond do
        batch_group_size > 1 ->
          {batch_group_size, 1}

        feature_group_size > 1 ->
          {1, feature_group_size}

        true ->
          {1, 1}
      end

    revd_weights = Nx.reverse(rhs, axes: rhs_sdim_axes)

    gx =
      Nx.conv(g, revd_weights,
        strides: lhs_dilation,
        padding: lhs_padding,
        input_dilation: strides,
        kernel_dilation: rhs_dilation,
        input_permutation: output_permutation,
        kernel_permutation: t_rhs_permutation,
        output_permutation: input_permutation,
        feature_group_size: lhs_feature_group_size,
        batch_group_size: 1
      )

    gx =
      if batch_group_size > 1 do
        gx = reshape_axis_out_of(lhs1, batch_group_size, gx)
        reshape_axis_into(lhs1, lhs0, gx)
      else
        gx
      end

    gy =
      Nx.conv(x, g,
        strides: rhs_dilation,
        padding: rhs_padding,
        input_dilation: lhs_dilation,
        kernel_dilation: strides,
        input_permutation: t_lhs_permutation,
        kernel_permutation: t_out_permutation,
        output_permutation: t_rhs_permutation,
        feature_group_size: rhs_feature_group_size,
        batch_group_size: rhs_batch_group_size
      )

    [{x, gx}, {y, gy}]
  end

  defp conv_spec_transpose([dim0, dim1 | rest]), do: [dim1, dim0 | rest]

  defp conv_sdims(shape, axes) do
    axes
    |> Enum.map(&elem(shape, &1))
    |> List.to_tuple()
  end

  defp conv_lhs_padding(
         lhs_sdims,
         rhs_sdims,
         strides,
         out_sdims,
         padding,
         lhs_dilation,
         rhs_dilation
       ) do
    lhs_dilated_padding_config = Enum.map(lhs_dilation, &{0, 0, &1 - 1})
    rhs_dilated_padding_config = Enum.map(rhs_dilation, &{0, 0, &1 - 1})
    out_dilated_padding_config = Enum.map(strides, &{0, 0, &1 - 1})
    lhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(lhs_sdims, lhs_dilated_padding_config))
    rhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(rhs_sdims, rhs_dilated_padding_config))
    out_dilated_shape = Tuple.to_list(Nx.Shape.pad(out_sdims, out_dilated_padding_config))

    pad_before = Enum.zip_with(rhs_dilated_shape, padding, fn s, {lo, _} -> s - lo - 1 end)

    pad_after =
      [lhs_dilated_shape, rhs_dilated_shape, out_dilated_shape, pad_before]
      |> Enum.zip_with(fn [l, r, o, p] -> l + r - 1 - o - p end)

    Enum.zip(pad_before, pad_after)
  end

  defp conv_rhs_padding(
         lhs_sdims,
         rhs_sdims,
         strides,
         out_sdims,
         padding,
         lhs_dilation,
         rhs_dilation
       ) do
    lhs_dilated_padding_config = Enum.map(lhs_dilation, &{0, 0, &1 - 1})
    rhs_dilated_padding_config = Enum.map(rhs_dilation, &{0, 0, &1 - 1})
    out_dilated_padding_config = Enum.map(strides, &{0, 0, &1 - 1})
    lhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(lhs_sdims, lhs_dilated_padding_config))
    rhs_dilated_shape = Tuple.to_list(Nx.Shape.pad(rhs_sdims, rhs_dilated_padding_config))
    out_dilated_shape = Tuple.to_list(Nx.Shape.pad(out_sdims, out_dilated_padding_config))

    total_in_pad =
      [out_dilated_shape, rhs_dilated_shape, lhs_dilated_shape]
      |> Enum.zip_with(fn [o, r, l] -> o + r - l - 1 end)

    Enum.zip_with(padding, total_in_pad, fn {lo, _}, hi -> {lo, hi - lo} end)
  end

  defp reshape_axis_into(src, dst, x) do
    perm = for i <- 0..(Nx.rank(x.shape) - 1), i != src, do: i
    perm = List.insert_at(perm, dst, src)
    new_shape = Tuple.delete_at(x.shape, src)
    new_val = elem(new_shape, dst) * elem(x.shape, src)
    new_shape = put_elem(new_shape, dst, new_val)
    Nx.reshape(Nx.transpose(x, axes: perm), new_shape)
  end

  defp reshape_axis_out_of(src, size1, x) do
    size2 = div(elem(x.shape, src), size1)
    new_shape = x.shape
    new_shape = put_elem(new_shape, src, size1)
    new_shape = Tuple.insert_at(new_shape, src + 1, size2)
    Nx.reshape(x, new_shape)
  end

  ## General helpers

  defp reduce_g(x, opts, g) do
    axes = opts[:axes]
    keep_axes = opts[:keep_axes]

    if keep_axes || !axes do
      Nx.broadcast(g, x)
    else
      axes = Nx.axes(x.shape) -- axes
      Nx.broadcast(g, x, axes: axes)
    end
  end

  defp surface_nuldim_scalar(expr) do
    case expr do
      %T{data: %Expr{op: :constant, args: [scalar]}, shape: {}} -> scalar
      %T{} -> expr
    end
  end

  defp up_to(i, n) when i < n, do: [i | up_to(i + 1, n)]
  defp up_to(_, _), do: []

  defp argsort(list), do: list |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))

  defp tril(t) do
    lower_selector =
      t
      |> Nx.iota(axis: 0)
      |> Nx.greater_equal(Nx.iota(t, axis: 1))

    Nx.select(lower_selector, t, Nx.tensor(0, type: t.type))
  end
end

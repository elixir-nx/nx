defmodule Nx.Defn.Grad do
  @moduledoc false

  alias Nx.Defn.{Expr, Tree}
  alias Nx.Tensor, as: T

  def transform(to_grad, fun) do
    {to_grad, ids} =
      Tree.composite(to_grad, %{}, fn to_grad, ids ->
        validate_grad!(to_grad)
        to_grad = Expr.metadata(to_grad, %{__MODULE__ => :to_grad})
        {to_grad, Map.put(ids, to_grad.data.id, :to_grad)}
      end)

    expr = to_grad |> fun.() |> validate_expr!()

    # Collect all IDs in the function environment and mark
    # them as stop grads. This is an optimization to avoid
    # traversing trees when not necessary.
    {:env, env} = Function.info(fun, :env)
    ids = stop_grads(env, ids)

    # Grad all the parameters at the same time to share subtrees.
    {graded, _} = to_grad(expr, Expr.tensor(1.0), {ids, %{}})

    # Now traverse the expression again zerofying
    # the parts that comes from other variables.
    # We do so by encoding special nodes in the Expr
    # AST and unpack them as we verify.
    graded =
      Tree.composite(to_grad, fn to_grad ->
        id = to_grad.data.id
        {graded, _, _} = zerofy_ids(graded, %{}, Map.delete(ids, id))

        if graded.shape == to_grad.shape do
          graded
        else
          Nx.broadcast(graded, to_grad)
        end
      end)

    {expr, graded}
  end

  defp validate_grad!(%T{data: %Expr{}} = t), do: t

  defp validate_grad!(other) do
    raise ArgumentError,
          "the first argument of grad must be a tensor expression or a tuple of tensor expressions, " <>
            "got: #{inspect(other)}"
  end

  defp validate_expr!(%T{data: %Expr{}, shape: {}} = expr) do
    expr
  end

  defp validate_expr!(%T{data: %Expr{}, shape: shape}) do
    raise ArgumentError,
          "can only compute gradients of expressions that return scalars, " <>
            "got shape: #{inspect(shape)}"
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

  defp stop_grads(%_{}, ids),
    do: ids

  defp stop_grads(map, ids) when is_map(map),
    do: map |> Map.values() |> Enum.reduce(ids, &stop_grads/2)

  defp stop_grads(_, ids),
    do: ids

  ## Zerofy

  defp zerofy_ids(%T{data: %Expr{id: id}} = t, cache, ids) do
    case cache do
      %{^id => {res, tainted?}} ->
        {res, cache, tainted?}

      %{} ->
        {res, cache, tainted?} = zerofy_each(t, cache, ids)
        {res, Map.put(cache, id, {res, tainted?}), tainted?}
    end
  end

  defp zerofy_each(%T{data: %Expr{op: :metadata, args: [t, %{__MODULE__ => code}]}}, cache, ids) do
    case code do
      {:sum, exprs} ->
        {exprs, cache} =
          Enum.map_reduce(exprs, cache, fn expr, cache ->
            {expr, cache, _} = zerofy_ids(expr, cache, ids)
            {expr, cache}
          end)

        {Enum.reduce(exprs, &Nx.add/2), cache, true}

      {:tainted, id} ->
        if Map.has_key?(ids, id) do
          {Expr.tensor(0.0), cache, true}
        else
          {t, cache, false}
        end

      :to_grad ->
        {t, cache, false}
    end
  end

  defp zerofy_each(t, cache, ids) do
    {args, {cache, tainted?}} =
      Tree.traverse_args(t, {cache, false}, fn arg, {cache, acc_tainted?} ->
        {arg, cache, tainted?} = zerofy_ids(arg, cache, ids)
        {arg, {cache, tainted? or acc_tainted?}}
      end)

    if tainted? do
      {Tree.put_args(t, args), cache, true}
    else
      {t, cache, false}
    end
  end

  ## Recursion

  # The gradient recursion.
  #
  # We keep two caches. One is the result cache, which is used for
  # when visiting the same nodes in the AST.
  #
  # The other cache is the JVP cache, that shares parts of the JVP
  # computation. Both are important to reduce the amount of nodes
  # in the AST.
  defp to_grad(expr, res, cache) do
    Tree.composite(expr, cache, fn
      %T{data: %Expr{id: id, op: op, args: args}} = ans, {result_cache, jvp_cache} = cache ->
        key = [id | res.data.id]

        case result_cache do
          %{^id => :stop} ->
            {Expr.tensor(0.0), cache}

          %{^id => :to_grad} ->
            {Expr.metadata(res, %{__MODULE__ => {:tainted, id}}), cache}

          %{^key => res} ->
            {res, cache}

          %{} ->
            case grad(op, args, ans, res, cache) do
              {res, {result_cache, jvp_cache}} ->
                {res, {Map.put(result_cache, key, res), jvp_cache}}

              :none ->
                jvps =
                  case jvp_cache do
                    %{^id => jvps} -> jvps
                    %{} -> jvp(op, args, ans)
                  end

                {res, {result_cache, jvp_cache}} = grad_jvps(jvps, ans, res, cache)
                {res, {Map.put(result_cache, key, res), Map.put(jvp_cache, id, jvps)}}
            end
        end
    end)
  end

  defp grad_jvps([], _ans, _g, cache), do: {Expr.tensor(0.0), cache}

  defp grad_jvps(jvps, ans, g, cache) do
    {exprs, cache} =
      Enum.map_reduce(jvps, cache, fn {expr, subg}, cache ->
        to_grad(Nx.broadcast(expr, ans), Nx.multiply(g, subg), cache)
      end)

    template = Expr.tensor(Nx.template(g.shape, g.type))
    {Expr.metadata(template, %{__MODULE__ => {:sum, exprs}}), cache}
  end

  defp grad_pairs([], _g, cache), do: {Expr.tensor(0.0), cache}

  defp grad_pairs(parts, g, cache) do
    {exprs, cache} =
      Enum.map_reduce(parts, cache, fn {expr, g}, cache ->
        to_grad(expr, g, cache)
      end)

    template = Expr.tensor(Nx.template(g.shape, g.type))
    {Expr.metadata(template, %{__MODULE__ => {:sum, exprs}}), cache}
  end

  ## Syntax / linear grad

  defp grad(:metadata, [_, %{stop_grad: true}], _ans, _g, cache) do
    {Expr.tensor(1.0), cache}
  end

  defp grad(:metadata, [expr, %{custom_grad: fun}], _ans, g, cache) do
    args = fun.(expr, g)

    unless is_list(args) and Enum.all?(args, &match?({_, _}, &1)) do
      raise "custom_grad/2 must return a list of tuples, " <>
              "where the first element is the expression to continue computing grad " <>
              "and the second element is the updated g"
    end

    grad_pairs(args, g, cache)
  end

  defp grad(:cond, [clauses, last], _ans, g, cache) do
    {clauses, cache} =
      Enum.map_reduce(clauses, cache, fn {head, body}, cache ->
        {body, cache} = to_grad(body, g, cache)
        {{head, body}, cache}
      end)

    {last, cache} = to_grad(last, g, cache)
    {Expr.cond(clauses, last), cache}
  end

  defp grad(:elem, [tuple, index, _size], _ans, g, cache) do
    {tuple, cache} = to_grad(tuple, g, cache)
    {elem(tuple, index), cache}
  end

  defp grad(:select, [pred, on_true, on_false], _ans, g, cache) do
    {d_on_true, cache} = to_grad(on_true, g, cache)
    {d_on_false, cache} = to_grad(on_false, g, cache)
    result = Nx.select(pred, d_on_true, d_on_false)
    {result, cache}
  end

  defp grad(:broadcast, [x, shape, axes], _ans, g, cache) do
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

    to_grad(x, g, cache)
  end

  defp grad(:clip, [operand, min, max], _ans, g, cache) do
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

    parts = [
      {operand, Nx.multiply(g, w_operand)},
      {min, Nx.multiply(g, w_min)},
      {max, Nx.multiply(g, w_max)}
    ]

    grad_pairs(parts, g, cache)
  end

  defp grad(:squeeze, [x, axes], _ans, g, cache) do
    g = Nx.broadcast(g, x.shape, axes: Nx.axes(x.shape) -- axes)
    to_grad(x, g, cache)
  end

  defp grad(:reshape, [x, _new_shape], _ans, g, cache) do
    to_grad(x, Nx.reshape(g, x), cache)
  end

  defp grad(:transpose, [x, axes], _ans, g, cache) do
    to_grad(x, Nx.transpose(g, axes: argsort(axes)), cache)
  end

  defp grad(:pad, [x, value, padding_config], _ans, g, cache) do
    inverse_padding_config = Enum.map(padding_config, fn {lo, hi, _} -> {-lo, -hi, 0} end)
    unpadded = Nx.pad(g, 0.0, inverse_padding_config)

    start_indices = List.duplicate(0, Nx.rank(unpadded))
    lengths = Tuple.to_list(unpadded.shape)
    strides = padding_config |> Enum.map(fn {_, _, interior} -> interior + 1 end)

    g_operand = Nx.slice(unpadded, start_indices, lengths, strides: strides)
    g_value = Nx.subtract(Nx.sum(g), Nx.sum(g_operand))

    grad_pairs([{x, g_operand}, {value, g_value}], g, cache)
  end

  defp grad(:slice, [x, start_indices, _lengths, strides], _ans, g, cache) do
    lo_pads = start_indices
    hi_pads = hi_pads(0, g.shape, x.shape, start_indices, strides)
    interior_pads = Enum.map(strides, &(&1 - 1))

    padding_config = Enum.zip([lo_pads, hi_pads, interior_pads])
    pad_value = 0.0
    to_grad(x, Nx.pad(g, pad_value, padding_config), cache)
  end

  defp grad(:reverse, [x, axes], _ans, g, cache) do
    reversed = Nx.reverse(g, axes: axes)
    to_grad(x, reversed, cache)
  end

  defp grad(:sum, [x, opts], _ans, g, cache) do
    grad_reduce(x, opts, g, cache, & &1)
  end

  defp grad(:product, [x, opts], _ans, g, cache) do
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
    to_grad(x, g, cache)
  end

  @reduce_min_max_ops [:reduce_max, :reduce_min]

  defp grad(op, [x, opts], ans, g, cache) when op in @reduce_min_max_ops do
    grad_reduce(x, opts, g, cache, fn g ->
      axes = opts[:axes] || Nx.axes(x)

      shape =
        for {d, i} <- Enum.with_index(Tuple.to_list(x.shape)) do
          if i in axes, do: 1, else: d
        end

      locs = Nx.equal(x, Nx.reshape(ans, List.to_tuple(shape)))
      num = Nx.multiply(g, locs)
      den = Nx.sum(locs, axes: axes, keep_axes: true)
      Nx.divide(num, den)
    end)
  end

  defp grad(:dot, [x, axes_x, y, axes_y], ans, g, cache) do
    g = Nx.broadcast(g, ans)

    contract_gx = up_to(Nx.rank(x.shape) - length(axes_x), Nx.rank(g.shape))
    contract_gy = up_to(0, Nx.rank(x.shape) - length(axes_x))

    contract_x = Nx.axes(x.shape) -- axes_x
    contract_y = Nx.axes(y.shape) -- axes_y

    transpose_x = Enum.map(argsort(axes_y), &Enum.fetch!(axes_x, &1))
    transpose_y = Enum.map(argsort(axes_x), &Enum.fetch!(axes_y, &1))

    gx =
      g
      |> Nx.dot(contract_gx, y, contract_y)
      |> Nx.transpose(axes: argsort(contract_x ++ transpose_x))

    gy =
      g
      |> Nx.dot(contract_gy, x, contract_x)
      |> Nx.transpose(axes: argsort(contract_y ++ transpose_y))

    grad_pairs([{x, gx}, {y, gy}], g, cache)
  end

  defp grad(:conv, [x, y, opts], ans, g, cache) do
    grad_conv(x, y, opts, ans, g, cache)
  end

  @window_chooser_op [:window_min, :window_max]

  defp grad(op, [x, window_dimensions, opts], _ans, g, cache) when op in @window_chooser_op do
    padding = opts[:padding]
    strides = opts[:strides]

    fun =
      if op == :window_min,
        do: &Nx.scatter_window_min/5,
        else: &Nx.scatter_window_max/5

    g = fun.(x, g, window_dimensions, [padding: padding, strides: strides], 0)
    to_grad(x, g, cache)
  end

  defp grad(:window_sum, [x, window_dimensions, opts], _, ans, cache) do
    strides = opts[:strides]
    window_dilation = opts[:window_dilations]
    base_dilation = List.duplicate(1, Nx.rank(x))
    padding = opts[:padding]

    padding_config =
      conv_lhs_padding(
        x.shape,
        window_dimensions,
        strides,
        ans.shape,
        padding,
        base_dilation,
        window_dilation
      )

    padding_config =
      padding_config
      |> Enum.zip(strides)
      |> Enum.map(fn {{lo, hi}, s} -> {lo, hi, s - 1} end)

    g = Nx.pad(ans, 0.0, padding_config)

    g =
      Nx.window_sum(
        g,
        window_dimensions,
        strides: base_dilation,
        padding: List.duplicate({0, 0}, Nx.rank(x)),
        window_dilations: window_dilation
      )

    to_grad(x, g, cache)
  end

  defp grad(:concatenate, [tensors, axis], %{shape: ans_shape}, g, cache) do
    zero_axes = List.duplicate(0, tuple_size(ans_shape))
    ans_shape_list = Tuple.to_list(ans_shape)

    {pairs, _} =
      Enum.map_reduce(tensors, 0, fn t, limit ->
        t_len = elem(t.shape, axis)
        current_limit = t_len + limit
        start = List.replace_at(zero_axes, axis, limit)
        len = List.replace_at(ans_shape_list, axis, t_len)
        {{t, Nx.slice(g, start, len)}, current_limit}
      end)

    grad_pairs(pairs, g, cache)
  end

  defp grad(_op, _args, _ans, _g, _cache) do
    :none
  end

  ## JVP gradients

  defp jvp(:add, [x, y], _ans) do
    [{x, Expr.tensor(1.0)}, {y, Expr.tensor(1.0)}]
  end

  defp jvp(:subtract, [x, y], _ans) do
    [{x, Expr.tensor(1.0)}, {y, Expr.tensor(-1.0)}]
  end

  defp jvp(:multiply, [x, y], _ans) do
    [{x, y}, {y, x}]
  end

  defp jvp(:divide, [x, y], ans) do
    [{x, Nx.divide(1.0, y)}, {y, Nx.negate(Nx.divide(ans, y))}]
  end

  defp jvp(:quotient, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.quotient/2.

    If a floating point computation is acceptable, consider \
    using an implementation of floor division. See the \
    documentation of `Nx.quotient` for more details.
    """
  end

  defp jvp(:remainder, [x, y], _ans) do
    [{x, Expr.tensor(1.0)}, {y, Nx.negate(Nx.floor(Nx.divide(x, y)))}]
  end

  defp jvp(:power, [x, y], ans) do
    # Since we do many operations against literals,
    # we try to surface any scalar number.
    sx = surface_nuldim_scalar(x)
    sy = surface_nuldim_scalar(y)

    exponent = Nx.select(Nx.equal(sy, 0.0), 1.0, Nx.subtract(sy, 1.0))
    base = Nx.select(Nx.equal(sx, 0.0), 1.0, sx)

    gx = Nx.multiply(sy, Nx.power(sx, exponent))
    gy = Nx.multiply(Nx.log(base), ans)
    [{x, gx}, {y, gy}]
  end

  defp jvp(:atan2, [x, y], _ans) do
    den = Nx.add(Nx.multiply(x, x), Nx.multiply(y, y))
    [{x, Nx.divide(y, den)}, {y, Nx.negate(Nx.divide(x, den))}]
  end

  defp jvp(op, [x, y], ans) when op in [:min, :max] do
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

    [{x, lhs}, {y, rhs}]
  end

  defp jvp(:outer, [x, y], _ans) do
    x = Nx.reshape(x, {Nx.size(x.shape), 1})
    y = Nx.reshape(y, {1, Nx.size(y.shape)})
    [{x, y}, {y, x}]
  end

  defp jvp(:as_type, [x], _ans) do
    [{x, Expr.tensor(1.0)}]
  end

  defp jvp(:bitcast, [x], _ans) do
    [{x, Expr.tensor(1.0)}]
  end

  defp jvp(:metadata, [expr, _metadata], _ans) do
    [{expr, Expr.tensor(1.0)}]
  end

  defp jvp(:abs, [x], _ans) do
    [{x, Nx.select(Nx.greater_equal(x, 0.0), 1.0, -1.0)}]
  end

  defp jvp(:sqrt, [x], ans) do
    [{x, Nx.divide(0.5, ans)}]
  end

  defp jvp(:cbrt, [x], ans) do
    [{x, Nx.divide(1.0, 3 |> Nx.multiply(ans) |> Nx.multiply(ans))}]
  end

  defp jvp(:exp, [x], ans) do
    [{x, ans}]
  end

  defp jvp(:expm1, [x], ans) do
    [{x, Nx.add(ans, 1)}]
  end

  defp jvp(:log, [x], _ans) do
    [{x, Nx.divide(1.0, x)}]
  end

  defp jvp(:log1p, [x], _ans) do
    [{x, Nx.divide(1.0, Nx.add(x, 1))}]
  end

  defp jvp(:logistic, [x], ans) do
    g =
      x
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(ans)
      |> Nx.multiply(ans)

    [{x, g}]
  end

  defp jvp(:negate, [x], _ans) do
    [{x, Expr.tensor(-1.0)}]
  end

  defp jvp(:rsqrt, [x], _ans) do
    [{x, Nx.multiply(-0.5, Nx.power(x, -1.5))}]
  end

  defp jvp(:sin, [x], _ans) do
    [{x, Nx.cos(x)}]
  end

  defp jvp(:asin, [x], _ans) do
    [{x, Nx.rsqrt(Nx.subtract(1.0, Nx.multiply(x, x)))}]
  end

  defp jvp(:sinh, [x], _ans) do
    [{x, Nx.cosh(x)}]
  end

  defp jvp(:asinh, [x], _ans) do
    [{x, Nx.rsqrt(Nx.add(Nx.multiply(x, x), 1.0))}]
  end

  defp jvp(:acosh, [x], _ans) do
    [{x, Nx.rsqrt(Nx.subtract(Nx.multiply(x, x), 1.0))}]
  end

  defp jvp(:atanh, [x], _ans) do
    [{x, Nx.divide(1.0, Nx.subtract(1.0, Nx.multiply(x, x)))}]
  end

  defp jvp(:cos, [x], _ans) do
    [{x, Nx.negate(Nx.sin(x))}]
  end

  defp jvp(:acos, [x], _ans) do
    [{x, Nx.negate(Nx.rsqrt(Nx.subtract(1.0, Nx.multiply(x, x))))}]
  end

  defp jvp(:cosh, [x], _ans) do
    [{x, Nx.sinh(x)}]
  end

  defp jvp(:tan, [x], _ans) do
    cos = Nx.cos(x)
    [{x, 1 |> Nx.divide(cos) |> Nx.divide(cos)}]
  end

  defp jvp(:atan, [x], _ans) do
    [{x, Nx.divide(1.0, Nx.add(1.0, Nx.multiply(x, x)))}]
  end

  defp jvp(:tanh, [x], ans) do
    [{x, Nx.subtract(1.0, Nx.multiply(ans, ans))}]
  end

  @half_sqrt_pi :math.sqrt(:math.pi()) / 2
  @two_rsqrt_pi 2 / :math.sqrt(:math.pi())

  defp jvp(:erf, [x], _ans) do
    g =
      x
      |> Nx.multiply(x)
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(@two_rsqrt_pi)

    [{x, g}]
  end

  defp jvp(:erfc, [x], _ans) do
    g =
      x
      |> Nx.multiply(x)
      |> Nx.negate()
      |> Nx.exp()
      |> Nx.multiply(-@two_rsqrt_pi)

    [{x, g}]
  end

  defp jvp(:erf_inv, [x], ans) do
    g = Nx.multiply(@half_sqrt_pi, Nx.exp(Nx.multiply(ans, ans)))
    [{x, g}]
  end

  defp jvp(:reduce, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.reduce/4.

    If you are computing the sum, product, or similar, use the \
    appropriate Nx functions instead. If you have a custom usage \
    of reduce, consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  defp jvp(:window_product, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.window_product/3.

    Consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  defp jvp(:reduce_window, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.reduce_window/5.

    If you are computing the sum, max, or similar of a window, use \
    the appropriate Nx functions instead. If you have a custom usage \
    of reduce_window, consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  @error [:map]

  defp jvp(op, _, _) when op in @error do
    raise ArgumentError, """
    cannot compute gradient for Nx.#{op}.

    Consider using stop_grad/1 to make the gradient equivalent to \
    the identify function or use custom_grad/2 to define a proper \
    gradient implementation
    """
  end

  @constants [:scalar, :tensor, :parameter, :eye, :iota, :random_uniform, :random_normal] ++
               [:all?, :any?, :argmax, :argmin] ++
               [:bitwise_and, :bitwise_or, :bitwise_xor, :bitwise_not] ++
               [:logical_and, :logical_or, :logical_xor, :logical_not] ++
               [:left_shift, :right_shift, :count_leading_zeros, :population_count] ++
               [:floor, :round, :ceil, :sign] ++
               [:equal, :greater, :greater_equal, :less, :less_equal, :not_equal]

  defp jvp(op, _, _) when op in @constants do
    []
  end

  defp jvp(op, _, _) do
    raise ArgumentError, """
    gradient not yet implemented for Nx.#{op}.

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

  defp grad_conv(x, y, opts, ans, g, cache) do
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

    grad_pairs([{x, gx}, {y, gy}], g, cache)
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

    # TODO: Use Enum.zip_with on Elixir v1.12
    pad_before =
      rhs_dilated_shape
      |> Enum.zip(padding)
      |> Enum.map(fn {s, {lo, _}} -> s - lo - 1 end)

    pad_after =
      [lhs_dilated_shape, rhs_dilated_shape, out_dilated_shape, pad_before]
      |> Enum.zip()
      |> Enum.map(fn {l, r, o, p} -> l + r - 1 - o - p end)

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

    # TODO: Use Enum.zip_with on Elixir v1.12
    total_in_pad =
      [out_dilated_shape, rhs_dilated_shape, lhs_dilated_shape]
      |> Enum.zip()
      |> Enum.map(fn {o, r, l} -> o + r - l - 1 end)

    padding
    |> Enum.zip(total_in_pad)
    |> Enum.map(fn {{lo, _}, hi} -> {lo, hi - lo} end)
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

  ## Helpers

  defp grad_reduce(x, opts, g, cache, fun) do
    axes = opts[:axes]
    keep_axes = opts[:keep_axes]

    g =
      if keep_axes || !axes do
        Nx.broadcast(g, x)
      else
        axes = Nx.axes(x.shape) -- axes
        Nx.broadcast(g, x, axes: axes)
      end

    to_grad(x, fun.(g), cache)
  end

  defp hi_pads(pos, g_shape, x_shape, [start | starts], [stride | strides]) do
    g_dim = elem(g_shape, pos)
    x_dim = elem(x_shape, pos)

    val = x_dim - (start + (1 + stride * (g_dim - 1)))
    [val | hi_pads(pos + 1, g_shape, x_shape, starts, strides)]
  end

  defp hi_pads(_, _, _, [], []), do: []

  defp surface_nuldim_scalar(expr) do
    case expr do
      %T{data: %Expr{op: :scalar, args: [scalar]}, shape: {}} -> scalar
      %T{} -> expr
    end
  end

  defp up_to(i, n) when i < n, do: [i | up_to(i + 1, n)]
  defp up_to(_, _), do: []

  defp argsort(list), do: list |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
end

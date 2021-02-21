defmodule Nx.Defn.Grad do
  @moduledoc false

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  def transform({to_grad, expr}) do
    expr = validate_expr!(expr)

    Expr.traverse_exprs(to_grad, fn to_grad ->
      id = grad_id!(to_grad)
      {graded, _} = to_grad(expr, Expr.tensor(1.0), %{id => :stop})

      if graded.shape == to_grad.shape do
        graded
      else
        Nx.broadcast(graded, to_grad)
      end
    end)
  end

  defp grad_id!(%T{data: %Expr{id: id}}) do
    id
  end

  defp grad_id!(other) do
    raise ArgumentError,
          "the first argument of grad must be a variable or a tuple of defn expressions, " <>
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

  ## Recursion

  defp to_grad(tuple, res, cache) when is_tuple(tuple) do
    {res, cache} = tuple |> Tuple.to_list() |> Enum.map_reduce(cache, &to_grad(&1, res, &2))
    {List.to_tuple(res), cache}
  end

  defp to_grad(%T{data: %Expr{id: id, op: op, args: args}} = ans, res, cache) do
    key = [id | res.data.id]

    case cache do
      %{^id => :stop} ->
        {res, cache}

      %{^key => res} ->
        {res, cache}

      %{} ->
        {res, cache} = grad(op, args, ans, res, cache)
        {res, Map.put(cache, key, res)}
    end
  end

  ## Syntax nodes

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

  ## Binary broadcast gradients

  defp grad(:add, [x, y], ans, g, cache) do
    {x, y} = binary_broadcast(x, y, ans)
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    {maybe_add(dx, dy), cache}
  end

  defp grad(:subtract, [x, y], ans, g, cache) do
    {x, y} = binary_broadcast(x, y, ans)
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    {maybe_subtract(dx, dy), cache}
  end

  defp grad(:multiply, [x, y], ans, g, cache) do
    {x, y} = binary_broadcast(x, y, ans)
    {dx, cache} = to_grad(x, Nx.multiply(g, y), cache)
    {dy, cache} = to_grad(y, Nx.multiply(g, x), cache)

    {maybe_add(dx, dy), cache}
  end

  defp grad(:divide, [x, y], ans, g, cache) do
    {x, y} = binary_broadcast(x, y, ans)
    {dx, cache} = to_grad(x, Nx.divide(g, y), cache)
    {dy, cache} = to_grad(y, Nx.divide(Nx.multiply(g, ans), y), cache)

    {maybe_subtract(dx, dy), cache}
  end

  defp grad(:quotient, _, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.quotient/2.

    If a floating point computation is acceptable, consider \
    using an implementation of floor division. See the \
    documentation of `Nx.quotient` for more details.
    """
  end

  defp grad(:remainder, [x, y], ans, g, cache) do
    {x, y} = binary_broadcast(x, y, ans)
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, Nx.multiply(g, Nx.floor(Nx.divide(x, y))), cache)
    {maybe_subtract(dx, dy), cache}
  end

  defp grad(:power, [x, y], ans, g, cache) do
    {x, y} = binary_broadcast(x, y, ans)
    exponent = Nx.select(Nx.equal(y, 0.0), 1.0, Nx.subtract(y, 1.0))
    base = Nx.select(Nx.equal(x, 0.0), 1.0, x)

    {dx, cache} = to_grad(x, Nx.multiply(g, Nx.multiply(y, Nx.power(x, exponent))), cache)
    {dy, cache} = to_grad(y, Nx.multiply(g, Nx.multiply(Nx.log(base), ans)), cache)
    {maybe_add(dx, dy), cache}
  end

  defp grad(:arctan2, [x, y], ans, g, cache) do
    {x, y} = binary_broadcast(x, y, ans)
    den = Nx.add(Nx.power(x, 2), Nx.power(y, 2))
    {dx, cache} = to_grad(x, Nx.divide(Nx.multiply(g, y), den), cache)
    {dy, cache} = to_grad(y, Nx.divide(Nx.multiply(g, x), den), cache)

    {maybe_subtract(dx, dy), cache}
  end

  defp grad(op, [x, y], ans, g, cache) when op in [:min, :max] do
    {x, y} = binary_broadcast(x, y, ans)

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

    {dx, cache} = to_grad(x, Nx.multiply(g, lhs), cache)
    {dy, cache} = to_grad(y, Nx.multiply(g, rhs), cache)
    {maybe_add(dx, dy), cache}
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

    {d_operand, cache} = to_grad(operand, Nx.multiply(g, w_operand), cache)
    {d_min, cache} = to_grad(min, Nx.multiply(g, w_min), cache)
    {d_max, cache} = to_grad(max, Nx.multiply(g, w_max), cache)

    {maybe_add(maybe_add(d_operand, d_min), d_max), cache}
  end

  defp grad(:select, [pred, on_true, on_false], _ans, g, cache) do
    {d_on_true, cache} = to_grad(on_true, g, cache)
    {d_on_false, cache} = to_grad(on_false, g, cache)
    result = Nx.select(pred, d_on_true, d_on_false)
    {result, cache}
  end

  ## Linear gradients

  defp grad(:outer, [x, y], ans, g, cache) do
    x = Nx.reshape(x, {Nx.size(x.shape), 1})
    y = Nx.reshape(y, {1, Nx.size(y.shape)})
    grad(:multiply, [x, y], ans, g, cache)
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

    t_operand = Nx.slice(unpadded, start_indices, lengths, strides: strides)
    t_value = Nx.subtract(Nx.sum(g), Nx.sum(t_operand))

    {dx, cache} = to_grad(x, t_operand, cache)
    {dv, cache} = to_grad(value, t_value, cache)

    {maybe_add(dx, dv), cache}
  end

  defp grad(:slice, [x, start_indices, _lengths, strides], _ans, g, cache) do
    lo_pads = start_indices
    hi_pads = hi_pads(0, g.shape, x.shape, start_indices, strides)
    interior_pads = Enum.map(strides, &(&1 - 1))

    padding_config = Enum.zip([lo_pads, hi_pads, interior_pads])
    pad_value = 0.0

    t_op = Nx.pad(g, pad_value, padding_config)
    to_grad(x, t_op, cache)
  end

  defp grad(:reverse, [x, axes], _ans, g, cache) do
    reversed = Nx.reverse(g, axes: axes)
    to_grad(x, reversed, cache)
  end

  defp grad(:as_type, [x], _ans, g, cache) do
    to_grad(x, g, cache)
  end

  defp grad(:sum, [x, opts], _ans, g, cache) do
    grad_reduce(x, opts, g, cache, & &1)
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

    {dx, cache} = to_grad(x, gx, cache)
    {dy, cache} = to_grad(y, gy, cache)
    {maybe_add(dx, dy), cache}
  end

  ## Other gradients

  defp grad(:abs, [x], _ans, g, cache) do
    g = Nx.select(Nx.greater_equal(x, 0.0), g, Nx.negate(g))
    to_grad(x, g, cache)
  end

  defp grad(:sqrt, [x], ans, g, cache) do
    g = Nx.multiply(g, Nx.divide(0.5, ans))
    to_grad(x, g, cache)
  end

  defp grad(:cbrt, [x], ans, g, cache) do
    g = Nx.divide(g, 3 |> Nx.multiply(ans) |> Nx.multiply(ans))
    to_grad(x, g, cache)
  end

  defp grad(:exp, [x], ans, g, cache) do
    g = Nx.multiply(g, ans)
    to_grad(x, g, cache)
  end

  defp grad(:expm1, [x], ans, g, cache) do
    g = Nx.multiply(g, Nx.add(ans, 1))
    to_grad(x, g, cache)
  end

  defp grad(:log, [x], _ans, g, cache) do
    g = Nx.divide(g, x)
    to_grad(x, g, cache)
  end

  defp grad(:log1p, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.divide(1, Nx.add(x, 1)))
    to_grad(x, g, cache)
  end

  defp grad(:logistic, [x], ans, g, cache) do
    g =
      Nx.multiply(
        g,
        x
        |> Nx.negate()
        |> Nx.exp()
        |> Nx.multiply(ans)
        |> Nx.multiply(ans)
      )

    to_grad(x, g, cache)
  end

  defp grad(:negate, [x], _ans, g, cache) do
    g = Nx.negate(g)
    to_grad(x, g, cache)
  end

  defp grad(:rsqrt, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.multiply(-0.5, Nx.power(x, -1.5)))
    to_grad(x, g, cache)
  end

  defp grad(:sin, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.cos(x))
    to_grad(x, g, cache)
  end

  defp grad(:arcsin, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.rsqrt(Nx.subtract(1.0, Nx.power(x, 2.0))))
    to_grad(x, g, cache)
  end

  defp grad(:sinh, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.cosh(x))
    to_grad(x, g, cache)
  end

  defp grad(:cos, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.negate(Nx.sin(x)))
    to_grad(x, g, cache)
  end

  defp grad(:arccos, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.negate(Nx.rsqrt(Nx.subtract(1.0, Nx.power(x, 2.0)))))
    to_grad(x, g, cache)
  end

  defp grad(:cosh, [x], _ans, g, cache) do
    g = Nx.multiply(g, Nx.sinh(x))
    to_grad(x, g, cache)
  end

  defp grad(:tan, [x], _ans, g, cache) do
    g = Nx.divide(g, Nx.power(Nx.cos(x), 2))
    to_grad(x, g, cache)
  end

  defp grad(:arctan, [x], _ans, g, cache) do
    g = Nx.divide(g, Nx.add(1.0, Nx.power(x, 2.0)))
    to_grad(x, g, cache)
  end

  defp grad(:tanh, [x], ans, g, cache) do
    g = Nx.multiply(g, Nx.subtract(1.0, Nx.multiply(ans, ans)))
    to_grad(x, g, cache)
  end

  defp grad(:reduce, _, _, _, _) do
    raise ArgumentError, """
    cannot compute gradient for Nx.reduce/4.

    If you are computing the sum, product, or similar, use the \
    appropriate Nx functions instead. If you have a custom usage \
    of reduce, consider using stop_grad/1 (making it equivalent \
    to the identify function) or using custom_grad/2 (giving it \
    a proper gradient implementation).
    """
  end

  defp grad(:reduce_window, _, _, _, _) do
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

  defp grad(op, _, _, _, _) when op in @error do
    raise ArgumentError, """
    cannot compute gradient for Nx.#{op}.

    Consider using stop_grad/1 to make the gradient equivalent to \
    the identify function or use custom_grad/2 to define a proper \
    gradient implementation
    """
  end

  @constants [:tensor, :parameter, :iota, :random_uniform, :random_normal] ++
               [:all?, :any?, :argmax, :argmin] ++
               [:bitwise_and, :bitwise_or, :bitwise_xor, :bitwise_not] ++
               [:logical_and, :logical_or, :logical_xor, :logical_not] ++
               [:left_shift, :right_shift, :count_leading_zeros, :population_count] ++
               [:floor, :round, :ceil, :sign] ++
               [:equal, :greater, :greater_equal, :less, :less_equal, :not_equal]

  defp grad(op, _, _, _, cache) when op in @constants do
    {Expr.tensor(0.0), cache}
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

  defp binary_broadcast(x, y, ans) do
    {Nx.broadcast(x, ans), Nx.broadcast(y, ans)}
  end

  defp maybe_add(x, y) do
    cond do
      zero?(x) -> y
      zero?(y) -> x
      true -> Nx.add(x, y)
    end
  end

  defp maybe_subtract(x, y) do
    cond do
      zero?(y) -> x
      zero?(x) -> Nx.negate(y)
      true -> Nx.subtract(x, y)
    end
  end

  @zero Nx.tensor(0.0)
  defp zero?(expr), do: match?(%T{data: %Expr{op: :tensor, args: [@zero]}}, expr)

  defp up_to(i, n) when i < n, do: [i | up_to(i + 1, n)]
  defp up_to(_, _), do: []

  defp argsort(list), do: list |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
end

defmodule Nx.Defn.Grad do
  @moduledoc false

  alias Nx.Defn.Expr

  def transform({to_grad, expr}) do
    expr = validate_expr!(expr)
    to_result(to_grad, expr)
  end

  defp to_result(tuple, expr) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.map(&to_result(&1, expr))
    |> List.to_tuple()
  end

  defp to_result(to_grad, expr) do
    id = grad_id!(to_grad)
    initial = broadcast_constant(1.0, to_grad)
    {graded, _} = to_grad(expr, initial, %{id => :stop})
    graded
  end

  defp grad_id!(%Nx.Defn.Expr{id: id}) do
    id
  end

  defp grad_id!(other) do
    raise ArgumentError,
          "the first argument of grad must be a variable or a tuple of defn expressions, " <>
            "got: #{inspect(other)}"
  end

  defp validate_expr!(%Nx.Defn.Expr{shape: {}} = expr) do
    expr
  end

  defp validate_expr!(%Nx.Defn.Expr{shape: shape}) do
    raise ArgumentError,
          "can only compute gradients of expressions that return scalars, " <>
            "got shape: #{inspect(shape)}"
  end

  defp validate_expr!(other) do
    validate_expr!(Expr.to_expr(other))
  end

  ## Recursion

  defp to_grad(%Expr{id: id, op: op, args: args} = expr, res, cache) do
    case cache do
      %{^id => :stop} ->
        {res, cache}

      %{^id => res} ->
        {res, cache}

      %{} ->
        {res, cache} = grad(op, args, expr, res, cache)
        {res, Map.put(cache, id, res)}
    end
  end

  ## Rule-based gradients

  # Addition rule
  defp grad(op, [x, y], _ans, g, cache) when op in [:add, :subtract] do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    res =
      cond do
        zero?(dy) -> dx
        zero?(dx) and op == :add -> dy
        true -> apply(Expr, op, [dx, dy])
      end

    {multiply(g, res), cache}
  end

  # Product rule
  defp grad(:multiply, [x, y], _ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    res = Expr.add(multiply(dx, y), multiply(dy, x))
    {multiply(g, res), cache}
  end

  # Division rule
  defp grad(:divide, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    num = Expr.subtract(dx, multiply(ans, dy))
    {multiply(g, Expr.divide(num, y)), cache}
  end

  # Remainder rule
  defp grad(:remainder, [x, y], _, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    right = multiply(dy, Expr.floor(Expr.divide(x, y)))
    {multiply(g, Expr.subtract(dx, right)), cache}
  end

  # Power/Exponentiation rule
  defp grad(:power, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    res =
      if one?(dx) and zero?(dy) do
        multiply(y, Expr.power(x, Expr.subtract(y, 1)))
      else
        # g' * ln f
        left = multiply(dy, Expr.log(x))

        # f' * (g / f)
        right = multiply(dx, Expr.divide(y, x))

        # ans * (left + right)
        multiply(ans, Expr.add(left, right))
      end

    {multiply(g, res), cache}
  end

  # Arctan2 rule
  defp grad(:arctan2, [x, y], _, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    num = Expr.subtract(multiply(dx, y), multiply(x, dy))
    den = Expr.add(Expr.power(x, 2), Expr.power(y, 2))
    {multiply(g, Expr.divide(num, den)), cache}
  end

  ## Linear gradients

  defp grad(:broadcast, [x, shape, axes], _ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)

    implicit_axes =
      for {a, i} <- Enum.with_index(axes),
          elem(shape, a) != 1 and elem(x.shape, i) == 1,
          do: {a, i}

    {implicit_axes, broadcast_axes} = Enum.unzip(implicit_axes)
    explicit_axes = Nx.Shape.to_axes(shape) -- axes

    g =
      case explicit_axes ++ implicit_axes do
        [] -> g
        sum_axes -> Expr.sum(g, axes: sum_axes)
      end

    g =
      case implicit_axes do
        [] -> g
        _ -> Expr.broadcast(g, x.shape, Nx.Shape.to_axes(x.shape) -- broadcast_axes)
      end

    {multiply(g, dx), cache}
  end

  defp grad(:reshape, [x, _new_shape], _ans, _g, cache) do
    # Broadcast to shape before the reshape
    to_grad(x, to_one(x, g), cache)
  end

  defp grad(:transpose, [x, axes], _ans, g, cache) do
    # Broadcast to shape after transpose and undo the transpose
    g = Expr.transpose(to_one(x, g), axes)
    to_grad(x, g, cache)
  end

  defp grad(:pad, [x, _value, padding_config], _ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    inverse_padding_config = Enum.map(padding_config, fn {lo, hi} -> {-lo, -hi} end)
    g = Expr.pad(g, 0.0, inverse_padding_config)
    {multiply(g, dx), cache}
  end

  defp grad(:sum, [x, opts], _ans, g, cache) do
    grad_aggregate(x, opts, g, cache)
  end

  defp grad(:mean, [x, opts], ans, g, cache) do
    {res, cache} = grad_aggregate(x, opts, g, cache)
    factor = Expr.to_expr(Nx.Shape.size(ans.shape) / Nx.Shape.size(x.shape))
    {multiply(factor, res), cache}
  end

  ## Other gradients

  defp grad(:abs, [x], _ans, g, cache) do
    g = Expr.select(Expr.greater_equal(x, broadcast_constant(0.0, g)), g, Expr.negate(g))
    to_grad(x, g, cache)
  end

  defp grad(op, [x, y], ans, g, cache) when op in [:min, :max] do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    lhs =
      Expr.divide(
        Expr.select(Expr.equal(x, ans), broadcast(1.0, ans), broadcast(0.0, ans)),
        Expr.select(Expr.equal(y, ans), broadcast(2.0, ans), broadcast(1.0, ans))
      )

    rhs =
      Expr.divide(
        Expr.select(Expr.equal(y, ans), broadcast(1.0, ans), broadcast(0.0, ans)),
        Expr.select(Expr.equal(x, ans), broadcast(2.0, ans), broadcast(1.0, ans))
      )

    res = Expr.add(Expr.multiply(dx, lhs), Expr.multiply(dy, rhs))

    {multiply(g, res), cache}
  end

  defp grad(:cbrt, [x], ans, g, cache) do
    g = Expr.divide(g, 3 |> multiply(ans) |> multiply(ans))
    to_grad(x, g, cache)
  end

  defp grad(:cos, [x], _ans, g, cache) do
    g = multiply(g, Expr.negate(Expr.sin(x)))
    to_grad(x, g, cache)
  end

  defp grad(:exp, [x], ans, g, cache) do
    g = multiply(g, ans)
    to_grad(x, g, cache)
  end

  defp grad(:expm1, [x], ans, g, cache) do
    g = multiply(g, Expr.add(ans, 1))
    to_grad(x, g, cache)
  end

  defp grad(:log, [x], _ans, g, cache) do
    g = Expr.divide(g, x)
    to_grad(x, g, cache)
  end

  defp grad(:log1p, [x], _ans, g, cache) do
    g = multiply(g, Expr.divide(1, Expr.add(x, 1)))
    to_grad(x, g, cache)
  end

  defp grad(:logistic, [x], ans, g, cache) do
    g =
      multiply(
        g,
        x
        |> Expr.negate()
        |> Expr.exp()
        |> multiply(ans)
        |> multiply(ans)
      )

    to_grad(x, g, cache)
  end

  defp grad(:negate, [x], _ans, g, cache) do
    g = Expr.negate(g)
    to_grad(x, g, cache)
  end

  defp grad(:rsqrt, [x], _ans, g, cache) do
    g = multiply(g, multiply(-0.5, Expr.power(x, -1.5)))
    to_grad(x, g, cache)
  end

  defp grad(:sin, [x], _ans, g, cache) do
    g = multiply(g, Expr.cos(x))
    to_grad(x, g, cache)
  end

  defp grad(:sqrt, [x], ans, g, cache) do
    g = multiply(g, Expr.divide(0.5, ans))
    to_grad(x, g, cache)
  end

  defp grad(:tanh, [x], ans, g, cache) do
    g = multiply(g, Expr.subtract(1.0, multiply(ans, ans)))
    to_grad(x, g, cache)
  end

  @constants [:tensor, :parameter, :constant, :iota, :random_uniform, :random_normal] ++
               [:argmax, :argmin] ++
               [:bitwise_and, :bitwise_or, :bitwise_xor, :bitwise_not] ++
               [:left_shift, :right_shift, :count_leading_zeros, :population_count] ++
               [:floor, :round, :ceil, :sign]

  defp grad(op, _, _, g, cache) when op in @constants do
    {broadcast_constant(0.0, g), cache}
  end

  # TODO:
  # outer/2
  # dot_general
  # squeeze

  ## Grad helpers

  defp grad_aggregate(x, opts, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)

    g =
      if axes = opts[:axes] do
        axes = Nx.Shape.to_axes(x.shape) -- axes
        Expr.broadcast(g, dx, axes)
      else
        g
      end

    {multiply(g, dx), cache}
  end

  ## Optimizers

  # An optimized version of multiplication to reduce nodes.
  defp multiply(left, right) do
    cond do
      one?(left) and one?(right) and left.shape == right.shape ->
        left

      one?(left) and constant?(right) ->
        broadcast_constant(hd(right.args) * 1.0, left)

      one?(right) and constant?(left) ->
        broadcast_constant(hd(left.args) * 1.0, right)

      true ->
        Expr.multiply(left, right)
    end
  end

  # And optimized version of constant broadcast to reduce nodes.
  defp broadcast_constant(constant, expr) when is_number(constant) do
    if broadcast?(expr, constant), do: expr, else: Expr.broadcast(constant, expr.shape)
  end

  ## Helpers

  # Build a broadcast of ones with the given shape.
  # Also receives the current g as a possible optimization.
  defp to_one(expr, g) do
    if one?(g) and expr.shape == g.shape do
      g
    else
      broadcast_constant(1.0, expr)
    end
  end

  defp constant?(expr),
    do: match?(%Expr{op: :constant, args: [_]}, expr)

  defp broadcast?(expr, constant),
    do: match?(%Expr{op: :broadcast, args: [%Expr{op: :constant, args: [^constant]}, _]}, expr)

  defp zero?(expr), do: broadcast?(expr, 0.0)
  defp one?(expr), do: broadcast?(expr, 1.0)
end

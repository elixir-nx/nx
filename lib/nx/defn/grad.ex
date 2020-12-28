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
    {graded, _} = to_grad(expr, Expr.to_expr(1.0), %{id => :stop})

    if graded.shape == to_grad.shape do
      graded
    else
      Expr.broadcast(graded, to_grad)
    end
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
    {dx, cache} = to_grad(x, to_one(x), cache)
    {dy, cache} = to_grad(y, to_one(y), cache)

    res =
      cond do
        zero?(dy) -> dx
        zero?(dx) and op == :add -> dy
        true -> apply(Expr, op, [dx, dy])
      end

    {Expr.multiply(g, res), cache}
  end

  # Product rule
  defp grad(:multiply, [x, y], _ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x), cache)
    {dy, cache} = to_grad(y, to_one(y), cache)

    res = Expr.add(Expr.multiply(dx, y), Expr.multiply(dy, x))
    {Expr.multiply(g, res), cache}
  end

  # Division rule
  defp grad(:divide, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x), cache)
    {dy, cache} = to_grad(y, to_one(y), cache)

    num = Expr.subtract(dx, Expr.multiply(ans, dy))
    {Expr.multiply(g, Expr.divide(num, y)), cache}
  end

  # Remainder rule
  defp grad(:remainder, [x, y], _, g, cache) do
    {dx, cache} = to_grad(x, to_one(x), cache)
    {dy, cache} = to_grad(y, to_one(y), cache)

    right = Expr.multiply(dy, Expr.floor(Expr.divide(x, y)))
    {Expr.multiply(g, Expr.subtract(dx, right)), cache}
  end

  # Power/Exponentiation rule
  defp grad(:power, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x), cache)
    {dy, cache} = to_grad(y, to_one(y), cache)

    res =
      if one?(dx) and zero?(dy) do
        Expr.multiply(y, Expr.power(x, Expr.subtract(y, 1)))
      else
        # g' * ln f
        left = Expr.multiply(dy, Expr.log(x))

        # f' * (g / f)
        right = Expr.multiply(dx, Expr.divide(y, x))

        # ans * (left + right)
        Expr.multiply(ans, Expr.add(left, right))
      end

    {Expr.multiply(g, res), cache}
  end

  # Arctan2 rule
  defp grad(:arctan2, [x, y], _, g, cache) do
    {dx, cache} = to_grad(x, to_one(x), cache)
    {dy, cache} = to_grad(y, to_one(y), cache)

    num = Expr.subtract(Expr.multiply(dx, y), Expr.multiply(x, dy))
    den = Expr.add(Expr.power(x, 2), Expr.power(y, 2))
    {Expr.multiply(g, Expr.divide(num, den)), cache}
  end

  ## Linear gradients

  defp grad(:broadcast, [x, shape, axes], _ans, g, cache) do
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

    to_grad(x, g, cache)
  end

  defp grad(:reshape, [x, _new_shape], _ans, g, cache) do
    to_grad(x, Expr.reshape(g, x), cache)
  end

  defp grad(:transpose, [x, axes], _ans, g, cache) do
    to_grad(x, Expr.transpose(g, argsort(axes)), cache)
  end

  defp grad(:pad, [x, _value, padding_config], _ans, g, cache) do
    inverse_padding_config = Enum.map(padding_config, fn {lo, hi} -> {-lo, -hi} end)
    g = Expr.pad(g, 0.0, inverse_padding_config)
    to_grad(x, g, cache)
  end

  defp grad(:sum, [x, opts], _ans, g, cache) do
    grad_aggregate(x, opts, g, cache)
  end

  defp grad(:mean, [x, opts], ans, g, cache) do
    factor = Expr.to_expr(Nx.Shape.size(ans.shape) / Nx.Shape.size(x.shape))
    grad_aggregate(x, opts, Expr.multiply(factor, g), cache)
  end

  ## Other gradients

  defp grad(:abs, [x], _ans, g, cache) do
    g = Expr.select(Expr.greater_equal(x, broadcast_constant(0.0, g)), g, Expr.negate(g))
    to_grad(x, g, cache)
  end

  defp grad(op, [x, y], ans, g, cache) when op in [:min, :max] do
    {dx, cache} = to_grad(x, to_one(x), cache)
    {dy, cache} = to_grad(y, to_one(y), cache)

    lhs =
      Expr.divide(
        Expr.select(
          Expr.equal(x, ans),
          broadcast_constant(1.0, ans),
          broadcast_constant(0.0, ans)
        ),
        Expr.select(
          Expr.equal(y, ans),
          broadcast_constant(2.0, ans),
          broadcast_constant(1.0, ans)
        )
      )

    rhs =
      Expr.divide(
        Expr.select(
          Expr.equal(y, ans),
          broadcast_constant(1.0, ans),
          broadcast_constant(0.0, ans)
        ),
        Expr.select(
          Expr.equal(x, ans),
          broadcast_constant(2.0, ans),
          broadcast_constant(1.0, ans)
        )
      )

    res = Expr.add(Expr.multiply(dx, lhs), Expr.multiply(dy, rhs))

    {Expr.multiply(g, res), cache}
  end

  defp grad(:cbrt, [x], ans, g, cache) do
    g = Expr.divide(g, 3 |> Expr.multiply(ans) |> Expr.multiply(ans))
    to_grad(x, g, cache)
  end

  defp grad(:cos, [x], _ans, g, cache) do
    g = Expr.multiply(g, Expr.negate(Expr.sin(x)))
    to_grad(x, g, cache)
  end

  defp grad(:exp, [x], ans, g, cache) do
    g = Expr.multiply(g, ans)
    to_grad(x, g, cache)
  end

  defp grad(:expm1, [x], ans, g, cache) do
    g = Expr.multiply(g, Expr.add(ans, 1))
    to_grad(x, g, cache)
  end

  defp grad(:log, [x], _ans, g, cache) do
    g = Expr.divide(g, x)
    to_grad(x, g, cache)
  end

  defp grad(:log1p, [x], _ans, g, cache) do
    g = Expr.multiply(g, Expr.divide(1, Expr.add(x, 1)))
    to_grad(x, g, cache)
  end

  defp grad(:logistic, [x], ans, g, cache) do
    g =
      Expr.multiply(
        g,
        x
        |> Expr.negate()
        |> Expr.exp()
        |> Expr.multiply(ans)
        |> Expr.multiply(ans)
      )

    to_grad(x, g, cache)
  end

  defp grad(:negate, [x], _ans, g, cache) do
    g = Expr.negate(g)
    to_grad(x, g, cache)
  end

  defp grad(:rsqrt, [x], _ans, g, cache) do
    g = Expr.multiply(g, Expr.multiply(-0.5, Expr.power(x, -1.5)))
    to_grad(x, g, cache)
  end

  defp grad(:sin, [x], _ans, g, cache) do
    g = Expr.multiply(g, Expr.cos(x))
    to_grad(x, g, cache)
  end

  defp grad(:sqrt, [x], ans, g, cache) do
    g = Expr.multiply(g, Expr.divide(0.5, ans))
    to_grad(x, g, cache)
  end

  defp grad(:tanh, [x], ans, g, cache) do
    g = Expr.multiply(g, Expr.subtract(1.0, Expr.multiply(ans, ans)))
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
    g =
      if axes = opts[:axes] do
        axes = Nx.Shape.to_axes(x.shape) -- axes
        Expr.broadcast(g, x, axes)
      else
        Expr.broadcast(g, x)
      end

    to_grad(x, g, cache)
  end

  ## Helpers

  defp argsort(list), do: list |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))

  defp to_one(expr), do: broadcast_constant(1.0, expr)

  defp broadcast_constant?(expr, constant),
    do: match?(%Expr{op: :broadcast, args: [%Expr{op: :constant, args: [^constant]}, _]}, expr)

  defp zero?(expr), do: broadcast_constant?(expr, 0.0)
  defp one?(expr), do: broadcast_constant?(expr, 1.0)

  defp broadcast_constant(constant, expr) when is_number(constant) do
    if broadcast_constant?(expr, constant), do: expr, else: Expr.broadcast(constant, expr.shape)
  end
end

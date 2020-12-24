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
    {id, shape} = grad_id_and_shape!(to_grad)
    initial = Expr.broadcast(1.0, shape)
    {graded, _} = to_grad(expr, initial, %{id => :stop})
    graded
  end

  defp grad_id_and_shape!(%Nx.Defn.Expr{id: id, shape: shape}) do
    {id, shape}
  end

  defp grad_id_and_shape!(other) do
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

    {Expr.multiply(g, res), cache}
  end

  # Product rule
  defp grad(:multiply, [x, y], _ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    res =
      cond do
        zero?(dx) -> Expr.multiply(dy, x)
        zero?(dy) -> Expr.multiply(dx, y)
        true -> Expr.add(Expr.multiply(dx, y), Expr.multiply(dy, x))
      end

    {Expr.multiply(g, res), cache}
  end

  # Division rule
  defp grad(:divide, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    num = Expr.subtract(dx, Expr.multiply(ans, dy))
    {Expr.multiply(g, Expr.divide(num, y)), cache}
  end

  # Remainder rule
  defp grad(:remainder, [x, y], _, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    right = Expr.multiply(dy, Expr.floor(Expr.divide(x, y)))
    {Expr.multiply(g, Expr.subtract(dx, right)), cache}
  end

  # Power/Exponentiation rule
  defp grad(:power, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

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
    {dx, cache} = to_grad(x, to_one(x, g), cache)
    {dy, cache} = to_grad(y, to_one(y, g), cache)

    num = Expr.subtract(Expr.multiply(dx, y), Expr.multiply(x, dy))
    den = Expr.add(Expr.power(x, 2), Expr.power(y, 2))
    {Expr.multiply(g, Expr.divide(num, den)), cache}
  end

  ## Linear gradients

  defp grad(:broadcast, [x, _], _ans, g, cache) do
    to_grad(x, g, cache)
  end

  defp grad(op, [x, _], _ans, g, cache) when op in [:sum, :mean] do
    {dx, cache} = to_grad(x, Expr.broadcast(g, x), cache)

    case Nx.Shape.rank(dx.shape) - Nx.Shape.rank(g.shape) do
      0 -> {dx, cache}
      d -> {apply(Expr, op, [dx, [axes: Enum.to_list(0..(d - 1))]]), cache}
    end
  end

  ## Other gradients

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
    {Expr.broadcast(0.0, g.shape), cache}
  end

  # TODO:
  # abs/1 - requires select
  # max/2 - requires comparison
  # min/2 - requires comparison
  # outer/2
  # dot_general
  # reshape - deflinear
  # transpose - deflinear

  ## Helpers

  # Build a broadcast 1.0.
  #
  # If the expression or the current result are one
  # and match the desired shape, we reuse it.
  defp to_one(expr, g) do
    cond do
      one?(expr) ->
        expr

      one?(g) and expr.shape == g.shape ->
        g

      true ->
        Expr.broadcast(1.0, expr.shape)
    end
  end

  defp zero?(expr),
    do: match?(%Expr{op: :broadcast, args: [%Expr{op: :constant, args: [0.0]}, _]}, expr)

  defp one?(expr),
    do: match?(%Expr{op: :broadcast, args: [%Expr{op: :constant, args: [1.0]}, _]}, expr)
end

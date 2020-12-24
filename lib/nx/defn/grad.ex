defmodule Nx.Defn.Grad do
  @moduledoc false

  alias Nx.Defn.Expr

  def transform({to_grad, expr}) do
    expr = validate_expr!(expr)

    {id, shape} = grad_id_and_shape!(to_grad)
    initial = Expr.broadcast(1.0, shape)
    {expr, _cache} = to_grad(expr, initial, %{id => :stop})
    expr
  end

  defp grad_id_and_shape!(%Nx.Defn.Expr{id: id, shape: shape}) do
    {id, shape}
  end

  defp grad_id_and_shape!(other) do
    raise ArgumentError,
          "the first argument of grad must be a variable or a tuple of defn expressions, " <>
            "got: #{inspect(other)}"
  end

  defp validate_expr!(number) when is_number(number) do
    Expr.constant(number)
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
    raise ArgumentError,
          "the second argument of grad must be a defn expression, got: #{inspect(other)}"
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

  ## Gradient rules

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

  ## Other gradients

  defp grad(:broadcast, [arg, _], ans, g, cache) do
    val = Nx.Shape.size(ans.shape) / Nx.Shape.size(g.shape)
    to_grad(arg, Expr.multiply(g, val), cache)
  end

  defp grad(:sum, [arg, _], _, g, cache) do
    to_grad(arg, g, cache)
  end

  defp grad(:tanh, [arg], ans, g, cache) do
    g = Expr.multiply(g, Expr.subtract(1.0, Expr.multiply(ans, ans)))
    to_grad(arg, g, cache)
  end

  defp grad(:exp, [arg], ans, g, cache) do
    g = Expr.multiply(g, ans)
    to_grad(arg, g, cache)
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

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

  ## Other gradients

  defp grad(:tanh, [arg], ans, g, cache) do
    g = Expr.multiply(g, Expr.subtract(1.0, Expr.multiply(ans, ans)))
    to_grad(arg, g, cache)
  end

  defp grad(:exp, [arg], ans, g, cache) do
    g = Expr.multiply(g, ans)
    to_grad(arg, g, cache)
  end

  defp grad(ignore, _, _, g, cache) when ignore in [:tensor, :parameter, :constant] do
    {Nx.broadcast(0.0, g.shape), cache}
  end
end

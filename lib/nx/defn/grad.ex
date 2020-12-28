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

  defp grad(:add, [x, y], _ans, g, cache) do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    {maybe_add(dx, dy), cache}
  end

  defp grad(:subtract, [x, y], _ans, g, cache) do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    {maybe_subtract(dx, dy), cache}
  end

  defp grad(:multiply, [x, y], _ans, g, cache) do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    {maybe_add(maybe_multiply(dx, y), maybe_multiply(dy, x)), cache}
  end

  defp grad(:divide, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    num = maybe_subtract(dx, maybe_multiply(ans, dy))
    {maybe_divide(num, y), cache}
  end

  defp grad(:remainder, [x, y], _, g, cache) do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    right = maybe_multiply(dy, Expr.floor(Expr.divide(x, y)))
    {maybe_subtract(dx, right), cache}
  end

  defp grad(:power, [x, y], ans, g, cache) do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    exponent = Expr.select(Expr.equal(y, 0.0), 1.0, Expr.subtract(y, 1.0))
    left = maybe_multiply(dx, Expr.multiply(y, Expr.power(x, exponent)))

    base = Expr.select(Expr.equal(x, 0.0), 1.0, x)
    right = maybe_multiply(dy, Expr.multiply(Expr.log(base), ans))
    {maybe_add(left, right), cache}
  end

  defp grad(:arctan2, [x, y], _, g, cache) do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    num = maybe_subtract(maybe_multiply(dx, y), maybe_multiply(x, dy))
    den = Expr.add(Expr.power(x, 2), Expr.power(y, 2))
    {maybe_divide(num, den), cache}
  end

  defp grad(op, [x, y], ans, g, cache) when op in [:min, :max] do
    {dx, cache} = to_grad(x, g, cache)
    {dy, cache} = to_grad(y, g, cache)

    lhs =
      Expr.divide(
        Expr.select(Expr.equal(x, ans), 1.0, 0.0),
        Expr.select(Expr.equal(y, ans), 2.0, 1.0)
      )

    rhs =
      Expr.divide(
        Expr.select(Expr.equal(y, ans), 1.0, 0.0),
        Expr.select(Expr.equal(x, ans), 2.0, 1.0)
      )

    {maybe_add(maybe_multiply(dx, lhs), maybe_multiply(dy, rhs)), cache}
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
      case broadcast_axes do
        [] -> g
        _ -> Expr.broadcast(g, x.shape, Nx.Shape.to_axes(x.shape) -- broadcast_axes)
      end

    to_grad(x, g, cache)
  end

  defp grad(:squeeze, [x, axes], _ans, g, cache) do
    g =
      case axes do
        [] -> g
        _ -> Expr.broadcast(g, x.shape, Nx.Shape.to_axes(x.shape) -- axes)
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

  defp grad(:dot, [x, axes_x, y, axes_y], ans, g, cache) do
    g = Expr.broadcast(g, ans)

    contract_gx = up_to(Nx.Shape.rank(x.shape) - length(axes_x), Nx.Shape.rank(g.shape))
    contract_gy = up_to(0, Nx.Shape.rank(x.shape) - length(axes_x))

    contract_x = Nx.Shape.to_axes(x.shape) -- axes_x
    contract_y = Nx.Shape.to_axes(y.shape) -- axes_y

    transpose_x = Enum.map(argsort(axes_y), &Enum.fetch!(axes_x, &1))
    transpose_y = Enum.map(argsort(axes_x), &Enum.fetch!(axes_y, &1))

    gx =
      g
      |> Expr.dot(contract_gx, y, contract_y)
      |> Expr.transpose(argsort(contract_x ++ transpose_x))

    gy =
      g
      |> Expr.dot(contract_gy, x, contract_x)
      |> Expr.transpose(argsort(contract_y ++ transpose_y))

    {dx, cache} = to_grad(x, gx, cache)
    {dy, cache} = to_grad(y, gy, cache)
    {maybe_add(dx, dy), cache}
  end

  ## Other gradients

  defp grad(:abs, [x], _ans, g, cache) do
    g = Expr.select(Expr.greater_equal(x, 0.0), g, Expr.negate(g))
    to_grad(x, g, cache)
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

  defp grad(op, _, _, _, cache) when op in @constants do
    {Expr.to_expr(0.0), cache}
  end

  # TODO:
  # outer/2

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

  defp maybe_add(x, y) do
    cond do
      zero?(x) -> y
      zero?(y) -> x
      true -> Expr.add(x, y)
    end
  end

  defp maybe_subtract(x, y) do
    cond do
      zero?(y) -> x
      zero?(x) -> Expr.negate(y)
      true -> Expr.subtract(x, y)
    end
  end

  defp maybe_multiply(x, y) do
    cond do
      zero?(x) -> x
      zero?(y) -> y
      true -> Expr.multiply(x, y)
    end
  end

  defp maybe_divide(x, y) do
    cond do
      zero?(x) -> x
      true -> Expr.divide(x, y)
    end
  end

  defp up_to(i, n) when i < n, do: [i | up_to(i + 1, n)]
  defp up_to(_, _), do: []

  defp argsort(list), do: list |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
  defp zero?(expr), do: match?(%Expr{op: :constant, args: [0.0]}, expr)
end

defmodule Nx.Defn.Vectorize do
  @moduledoc false

  alias Nx.Defn.{Expr, Tree}
  alias Nx.Tensor, as: T

  def transform(fun, args, in_axes) do
    in_axes =
      case in_axes do
        nil ->
          List.duplicate(0, length(args))

        in_axes when is_list(in_axes) ->
          in_axes
      end

    validate_in_axes!(args, in_axes, args, in_axes, 0, [])

    expr = apply(fun, args)

    {vectorized, _} = to_vectorized(expr, in_axes, %{})

    vectorized
  end

  defp validate_in_axes!(all_args, all_axes, [_ | args], [nil | in_axes], i, batch_sizes) do
    validate_in_axes!(all_args, all_axes, args, in_axes, i + 1, batch_sizes)
  end

  defp validate_in_axes!(all_args, all_axes, [t | args], [axis | in_axes], i, batch_sizes) do
    unless axis < Nx.rank(t) do
      raise ArgumentError, "vmap input axes cannot exceed rank of input tensor"
                           <> " arg #{inspect(i)} has rank #{inspect(Nx.rank(t))}"
                           <> " and axis #{inspect(axis)}"
    end

    validate_in_axes!(all_args, all_axes, args, in_axes, i + 1, [elem(Nx.shape(t), axis) | batch_sizes])
  end

  defp validate_in_axes!(_, axes, [], [], _, batch_sizes) do
    case Enum.uniq(batch_sizes) do
      [_] ->
        :ok

      [] ->
        raise ArgumentError, "at least 1 input axis passed to vmap must be non-nil"

      _ ->
        raise ArgumentError, "input batch sizes must match, got sizes #{inspect(Enum.reverse(batch_sizes))}"
                             <> " for axes #{inspect(axes)}"
    end
  end

  defp validate_in_axes!(args, in_axes, _, _, _, _) do
    raise ArgumentError, "length of vmap input axes must match length of function"
                         " arguments, got #{inspect(args)} and #{inspect(in_axes)}"
  end

  defp to_vectorized(expr, in_axes, cache) do
    Tree.composite(expr, cache, fn
      %T{data: %Expr{id: id, op: op, args: args}} = expr, cache ->
        case cache do
          %{^id => res} ->
            {res, cache}

          %{} ->
            {args, cache} = Tree.traverse_args(expr, cache, &to_vectorized(&1, in_axes, &2))
            {res, cache} = vectorize(op, args, in_axes, expr, cache)
            {res, Map.put(cache, id, res)}
        end
    end)
  end

  ## Defvectorized

  ## These functions have no need for a batching rule.

  @defvectorized [:negate, :sign, :floor, :ceil, :round, :exp, :log] ++
                 [:expm1, :log1p, :tanh, :sin, :cos, :tan, :asin, :acos, :atan, :abs] ++
                 [:atan2, :sinh, :cosh, :asinh, :acosh, :atanh, :erf, :erfc, :erf_inv] ++
                 [:sqrt, :rsqrt, :power, :bitwise_not, :population_count, :count_leading_zeros]

  defp vectorize(op, [arg], _, _expr, cache) when op in @defvectorized do
    {apply(Nx, op, [arg]), cache}
  end

  @constants [:scalar, :tensor, :parameter]

  defp vectorize(op, _, _, expr, cache) when op in @constants do
    {expr, cache}
  end

  ## Special

  defp vectorize(:dot, [lhs, lhs_c_dims, lhs_b_dims, rhs, rhs_c_dims, rhs_b_dims], [lhs_in_axis, rhs_in_axis], expr, cache) do
    {lhs_contract, lhs_batch, rhs_contract, rhs_batch} =
      case {lhs_in_axis, rhs_in_axis} do
        {lbd, nil} ->
          lhs_batch = bump_dims(lhs_b_dims, lbd)
          lhs_contract = bump_dims(lhs_c_dims, lbd)
          {lhs_contract, lhs_batch, rhs_c_dims, rhs_b_dims}

        {nil, rbd} ->
          rhs_batch = bump_dims(rhs_b_dims, rbd)
          rhs_contract = bump_dims(rhs_c_dims, rbd)
          {lhs_c_dims, lhs_b_dims, rhs_contract, rhs_batch}

        {lbd, rbd} ->
          lhs_batch = [lbd] ++ bump_dims(lhs_b_dims, lbd)
          rhs_batch = [rbd] ++ bump_dims(rhs_b_dims, rbd)
          lhs_contract = bump_dims(lhs_c_dims, lbd)
          rhs_contract = bump_dims(rhs_c_dims, rbd)
          {lhs_contract, lhs_batch, rhs_contract, rhs_batch}
      end

    IO.inspect lhs_batch
    IO.inspect lhs_contract
    IO.inspect rhs_batch
    IO.inspect rhs_contract

    {Nx.dot(lhs, lhs_contract, lhs_batch, rhs, rhs_contract, rhs_batch), cache}
  end

  defp bump_dims(dims, b) do
    Enum.map(dims, fn d -> if d >= b, do: d + 1, else: d end)
  end

  ## Not implemented

  defp vectorize(op, _, _, _, _) do
    raise ArgumentError, "vmap not implemented for #{inspect(op)}"
  end

end
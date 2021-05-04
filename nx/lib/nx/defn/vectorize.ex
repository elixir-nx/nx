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
            {res, cache} = vectorize(op, args, in_axes, expr, cache)
            {res, Map.put(cache, id, res)}
        end
    end)
  end

  ## Defvectorized

  ## These functions have no need for a batching rule.

  @defvectorized [:parameter, :negate, :sign, :floor, :ceil, :round, :exp, :log] ++
                 [:expm1, :log1p, :tanh, :sin, :cos, :tan, :asin, :acos, :atan, :abs] ++
                 [:atan2, :sinh, :cosh, :asinh, :acosh, :atanh, :erf, :erfc, :erf_inv] ++
                 [:sqrt, :rsqrt, :power, :bitwise_not, :population_count, :count_leading_zeros] ++
                 [:scalar, :tensor, :parameter, :eye, :iota, :random_uniform, :random_normal]

  defp vectorize(op, _, _, expr, cache) when op in @defvectorized do
    {expr, cache}
  end
end
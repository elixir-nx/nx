defmodule Nx.Defn.Evaluator do
  @moduledoc """
  The default implementation of a `Nx.Defn.Compiler`
  that evaluates the expression tree against the
  tensor backend.
  """

  @behaviour Nx.Defn.Compiler
  alias Nx.Defn.{Expr, Tree}

  @creation_ops [:scalar, :eye, :iota, :from_binary]
  @random_ops [:random_uniform, :random_normal]

  @impl true
  def __stream__(key, input, acc, vars, fun, opts) do
    dynamic = Nx.Defn.Tree.flatten_list([input, acc])
    vars = Enum.drop(vars, length(dynamic))

    Nx.Defn.Stream.start_link(input, acc, fn input, acc ->
      vars = Nx.Defn.Tree.flatten_list([input, acc], vars)
      __jit__(key, vars, fun, opts)
    end)
  end

  @impl true
  def __jit__(_key, vars, fun, _opts) do
    fun.(vars)
    |> composite_eval(vars, %{})
    |> elem(0)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :fun, args: [args, expr, _mfa]}}, _vars, cache) do
    fun =
      case length(args) do
        1 -> fn arg1 -> expr |> composite_eval([arg1], %{}) |> elem(0) end
        2 -> fn arg1, arg2 -> expr |> composite_eval([arg1, arg2], %{}) |> elem(0) end
      end

    {fun, cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :parameter, args: [i]}}, vars, cache) do
    {Enum.fetch!(vars, i), cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _vars, cache) do
    {t, cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :cond, args: [clauses, last]}}, vars, cache) do
    {res, cache} = cond_clause(clauses, last, vars, cache)
    composite_eval(res, vars, cache)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :while, args: args}}, vars, cache) do
    [initial, _arg, condition, block] = args
    {initial, cache} = composite_eval(initial, vars, cache)
    {while(initial, condition, block, cache), cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :elem, args: args}}, vars, cache) do
    [tuple, i, _size] = args
    {tuple, cache} = composite_eval(tuple, vars, cache)
    {elem(tuple, i), cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :metadata, args: [expr, _meta]}}, vars, cache) do
    eval(expr, vars, cache)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}} = ans, vars, cache) do
    case cache do
      %{^id => res} -> {res, cache}
      %{} -> eval_apply(id, op, ans, vars, cache)
    end
  end

  defp eval(other, _vars, cache) do
    {other, cache}
  end

  defp eval_apply(id, op, ans, vars, cache) do
    {args, cache} = Tree.traverse_args(ans, cache, &eval(&1, vars, &2))

    {mod, args} =
      cond do
        op in @creation_ops ->
          {backend, backend_options} = Nx.default_backend()
          {backend, [ans | args] ++ [backend_options]}

        op in @random_ops ->
          {_, backend_options} = Nx.default_backend()
          {Nx.Shared.find_impl!(args), [ans | args] ++ [backend_options]}

        match?({:tuple, _}, ans.type) ->
          {Nx.Shared.find_impl!(args), args}

        true ->
          {Nx.Shared.find_impl!(args), [ans | args]}
      end

    res = apply(mod, op, args)
    {res, Map.put(cache, id, res)}
  end

  defp while(acc, condition, block, cache) do
    vars = composite_to_vars(acc)
    {pred, temp} = eval(condition, vars, cache)

    if Nx.to_scalar(pred) != 0 do
      {acc, _} = composite_eval(block, vars, temp)
      while(acc, condition, block, cache)
    else
      acc
    end
  end

  defp composite_eval(composite, vars, cache) do
    Tree.composite(composite, cache, &eval(&1, vars, &2))
  end

  defp composite_to_vars(composite) do
    composite |> composite_to_vars([]) |> Enum.reverse()
  end

  defp composite_to_vars(tuple, acc) when is_tuple(tuple) do
    Enum.reduce(Tuple.to_list(tuple), acc, &composite_to_vars/2)
  end

  defp composite_to_vars(other, acc) do
    [other | acc]
  end

  defp cond_clause([{pred, clause} | clauses], last, vars, cache) do
    {pred, cache} = eval(pred, vars, cache)
    if Nx.to_scalar(pred) != 0, do: {clause, cache}, else: cond_clause(clauses, last, vars, cache)
  end

  defp cond_clause([], last, _vars, cache) do
    {last, cache}
  end
end

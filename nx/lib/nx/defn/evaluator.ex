defmodule Nx.Defn.Evaluator do
  @moduledoc """
  The default implementation of a `Nx.Defn.Compiler`
  that evaluates the expression tree against the
  tensor backend.
  """

  @behaviour Nx.Defn.Compiler
  alias Nx.Defn.Expr

  @impl true
  def __async__(key, vars, fun, opts) do
    Nx.Defn.Async.async(fn -> __jit__(key, vars, fun, opts) end)
  end

  @impl true
  def __jit__(_key, vars, fun, opts) do
    fun.(vars)
    |> Expr.rewrite_types(opts)
    |> Expr.traverse_exprs(%{}, &eval(&1, vars, &2))
    |> elem(0)
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :fun, args: [_, _, fun]}}, _vars, cache) do
    {fun, cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :parameter, args: [i]}}, vars, cache) do
    {Enum.fetch!(vars, i), cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :tensor, args: [t]}}, _vars, cache) do
    {t, cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :cond, args: [clauses, last]}}, vars, cache) do
    {res, cache} = find_clause(clauses, last, vars, cache)
    Expr.traverse_exprs(res, cache, &eval(&1, vars, &2))
  end

  defp eval(%Nx.Tensor{data: %Expr{op: :elem, args: args}}, vars, cache) do
    [tuple, i, _size] = args
    {tuple, cache} = Expr.traverse_exprs(tuple, cache, &eval(&1, vars, &2))
    {elem(tuple, i), cache}
  end

  defp eval(%Nx.Tensor{data: %Expr{op: op, id: id}, type: type} = ans, vars, cache) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {args, cache} = Expr.traverse_args(ans, cache, &eval(&1, vars, &2))
        res = apply(Nx.Shared.find_impl!(args), op, eval_args(type, ans, args))
        {res, Map.put(cache, id, res)}
    end
  end

  defp eval(other, _vars, cache) do
    {other, cache}
  end

  defp eval_args({:tuple, _}, _, args), do: args
  defp eval_args(_, ans, args), do: [ans | args]

  defp find_clause([{pred, clause} | clauses], last, vars, cache) do
    {pred, cache} = eval(pred, vars, cache)
    if Nx.to_scalar(pred) != 0, do: {clause, cache}, else: find_clause(clauses, last, vars, cache)
  end

  defp find_clause([], last, _vars, cache) do
    {last, cache}
  end
end

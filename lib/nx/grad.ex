defmodule Nx.Grad do
  # @behaviour Nx.Defn.Transform

  defguardp is_var(var) when is_atom(elem(var, 0)) and is_atom(elem(var, 2))
  defguardp is_underscore(var) when elem(var, 0) == :_ and is_atom(elem(var, 2))

  # TODO: grad(grad())
  # TODO: Make grad pipeable
  # TODO: Allow to differentiate on multiple vars
  # TODO: Handle tuples
  # TODO: Add shape assertions

  # The first pass is to build a map where all expressions have been mapped to variables.
  # The key of those variables are their counters and the values is a tuple with the var
  # AST and their expression. Then we emit back the AST only with the nodes we need.
  def __transform__(_env, version, meta, {var, ast}, _opts) do
    state = %{
      version: version,
      counters: [var_counter(var)],
      vars: %{}
    }

    # SSA
    {ast, state} = ssa(ast, state)

    # Collect all variables by moving assigns and flattening blocks
    {result_var, state} = new_var(state)
    {ast, state} = collect_and_flatten({:=, meta, [result_var, ast]}, state)

    # Now compute the gradient
    grad_exprs =
      case unfold_var(result_var, [], state) do
        [] -> [0.0]
        grads -> [Enum.reduce(grads, &nx_call(meta, :dot, [&2, &1]))]
      end

    {state.version, append_to_block(ast, grad_exprs)}
  end

  ## SSA

  # Extract complex expressions out of nested calls.
  #
  # The goal is to convert this:
  #
  #     Nx.exp(Nx.tanh(x))
  #
  # Into this:
  #
  #     a = Nx.tanh(x)
  #     Nx.exp(a)
  #
  defp ssa(ast, state) do
    Macro.prewalk(ast, state, fn
      {{:., _, [Nx, _]} = call, meta, args}, state ->
        {args, {extra, state}} = Enum.map_reduce(args, {[], state}, &ssa_arg(&1, meta, &2))
        block = maybe_block(Enum.reverse([{call, meta, args} | extra]))
        {block, state}

      node, acc ->
        {node, acc}
    end)
  end

  defp ssa_arg({:=, _, [left, _]} = arg, _meta, {extra, state}) do
    {left, {[arg | extra], state}}
  end

  defp ssa_arg(arg, meta, {extra, state}) do
    if is_var(arg) or Macro.quoted_literal?(arg) do
      {arg, {extra, state}}
    else
      {var, state} = new_var(state)
      {var, {[{:=, meta, [var, arg]} | extra], state}}
    end
  end

  ## Collect and flatten

  defp collect_and_flatten(ast, state) do
    {ast, vars} = Macro.traverse(ast, %{}, &collect/2, &flatten/2)
    {ast, %{state | vars: vars}}
  end

  defp collect({:=, meta, [pattern, {:__block__, block_meta, exprs}]}, vars) do
    {prior, [last]} = Enum.split(exprs, -1)

    exprs =
      case last do
        {:=, _, [left, _]} -> prior ++ [last] ++ [{:=, meta, [pattern, left]}]
        _ -> prior ++ [{:=, meta, [pattern, last]}]
      end

    {{:__block__, block_meta, exprs}, vars}
  end

  defp collect({:=, _, [pattern, expr]} = assign, vars) do
    {_, vars} =
      Macro.prewalk(pattern, vars, fn
        underscore, vars when is_underscore(underscore) ->
          {underscore, vars}

        var, vars when is_var(var) ->
          {var, Map.put(vars, var_counter(var), expr)}

        expr, vars ->
          {expr, vars}
      end)

    {assign, vars}
  end

  defp collect(expr, vars) do
    {expr, vars}
  end

  defp flatten({:__block__, meta, exprs}, vars) do
    exprs =
      Enum.flat_map(exprs, fn
        {:__block__, _, exprs} -> exprs
        expr -> [expr]
      end)

    {{:__block__, meta, exprs}, vars}
  end

  defp flatten(expr, vars) do
    {expr, vars}
  end

  ## Unfold the gradient computation into a list

  defp unfold_var(var, exprs, state) when is_var(var) do
    counter = var_counter(var)

    case state.vars do
      %{^counter => expr} ->
        unfold_grad(expr, var, exprs, state)

      %{} ->
        if counter in state.counters, do: [1.0 | exprs], else: exprs
    end
  end

  defp unfold_var(_not_a_var, exprs, _state), do: exprs

  defp unfold_grad({{:., _, [Nx, name]}, meta, [x]}, y, exprs, state) do
    [grad_call(meta, name, [x, y]) | unfold_var(x, exprs, state)]
  end

  ## Helpers

  defp nx_call(meta, name, args), do: {{:., meta, [Nx, name]}, meta, args}
  defp grad_call(meta, name, args), do: {{:., meta, [Nx.Grad, name]}, meta, args}

  defp maybe_block([expr]), do: expr
  defp maybe_block(exprs), do: {:__block__, [], exprs}

  defp append_to_block({:__block__, meta, exprs}, extra), do: {:__block__, meta, exprs ++ extra}
  defp append_to_block(expr, extra), do: {:__block__, [], [expr] ++ extra}

  defp new_var(state) do
    counter = state.version + 1
    {{:nvar, [counter: counter], __MODULE__}, %{state | version: counter}}
  end

  defp var_counter({var, meta, ctx}) when is_atom(var) and is_atom(ctx) do
    Keyword.fetch!(meta, :counter)
  end

  import Nx.Defn

  @doc """
  The derivative of `Nx.tanh/2`.
  """
  defn tanh(_x, y), do: 1.0 - Nx.power(y, 2)

  @doc """
  The derivative of `Nx.exp/1`.
  """
  defn exp(_x, y), do: y
end

defmodule Nx.Defn.GradTransform do
  @moduledoc """
  A transform that returns the gradient of the given computation.
  """

  @behaviour Nx.Defn.Transform

  defguardp is_var(var)
            when is_tuple(var) and tuple_size(var) == 3 and is_atom(elem(var, 0)) and
                   is_atom(elem(var, 2))

  defguardp is_underscore(var)
            when is_tuple(var) and tuple_size(var) == 3 and elem(var, 0) == :_ and
                   is_atom(elem(var, 2))

  @hint "grad expects the numerical expression to return a scalar tensor"

  @impl true
  def __transform__(_env, version, meta, {vars, ast}, _opts) do
    state = %{version: version, vars: %{}}

    # SSA
    {ast, state} = ssa(ast, state)

    # Collect all variables by moving assigns and flattening blocks
    {result_var, state} = new_var(state)
    {ast, state} = collect_and_flatten({:=, meta, [result_var, ast]}, state)

    # Now compute the gradient
    grad = gradient_for(vars, meta, result_var, state)
    assertion = nx_call(meta, :assert_shape, [result_var, {:{}, meta, []}, @hint])
    {state.version, append_to_block(ast, [assertion, grad])}
  end

  defp gradient_for({left, right}, meta, result_var, state) do
    gradient_for({:{}, meta, [left, right]}, meta, result_var, state)
  end

  defp gradient_for({:{}, meta, args}, _meta, result_var, state) do
    {:{}, meta, Enum.map(args, &gradient_for(&1, meta, result_var, state))}
  end

  defp gradient_for(var, meta, result_var, state) when is_var(var) do
    unfold_grad = unfold_var(result_var, [], Map.put(state, :counter, var_counter(var)))
    to_dot(meta, unfold_grad)
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

  # Normalize tuples on the right...
  defp collect({:=, meta, [{:{}, _, _} = pattern, {left, right}]}, vars) do
    collect({:=, meta, [pattern, {:{}, meta, [left, right]}]}, vars)
  end

  # If the left side is a tuple, then the right side is either a matching tuple
  defp collect({:=, _, [{:{}, _, left}, {:{}, _, right}]} = assign, vars)
       when length(left) == length(right) do
    vars =
      for {left, right} <- Enum.zip(left, right),
          is_var(left) and not is_underscore(left),
          reduce: vars,
          do: (vars -> Map.put(vars, var_counter(left), right))

    {assign, vars}
  end

  # Otherwise it has to be a variable and an expression
  defp collect({:=, _, [var, expr]} = assign, vars) when is_var(var) do
    {assign, Map.put(vars, var_counter(var), expr)}
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
        if counter == state.counter, do: [1.0 | exprs], else: [0.0 | exprs]
    end
  end

  defp unfold_var(_not_a_var, exprs, _state), do: [0.0 | exprs]

  @passthroughs [:reshape, :broadcast, :assert_shape]

  defp unfold_grad({{:., _, [Nx, name]}, _meta, [x | _args]}, _y, exprs, state)
       when name in @passthroughs do
    unfold_var(x, exprs, state)
  end

  # Addition rule
  defp unfold_grad({{:., _, [Nx, name]}, meta, [x1, x2 | _args]}, _y, exprs, state)
       when name in [:add, :subtract] do
    dx1 = to_dot(meta, unfold_var(x1, [], state))
    dx2 = to_dot(meta, unfold_var(x2, [], state))
    [nx_call(meta, name, [dx1, dx2]) | exprs]
  end

  # Product rule
  defp unfold_grad({{:., _, [Nx, name]}, meta, [x1, x2 | _args]}, _y, exprs, state)
       when name in [:dot, :multiply] do
    dx1 = to_dot(meta, unfold_var(x1, [], state))
    dx2 = to_dot(meta, unfold_var(x2, [], state))

    expr =
      nx_call(meta, :add, [
        nx_call(meta, name, [dx1, x2]),
        nx_call(meta, name, [x1, dx2])
      ])

    [expr | exprs]
  end

  # Power/Exponentiation rule
  defp unfold_grad({{:., _, [Nx, :power]}, meta, [x1, x2 | _args]}, y, exprs, state) do
    dx1 = to_dot(meta, unfold_var(x1, [], state))
    dx2 = to_dot(meta, unfold_var(x2, [], state))

    if dx1 == 1.0 and dx2 == 0.0 do
      [grad_call(meta, :power, [x1, x2, y]) | exprs]
    else
      # g' ln f
      left = nx_call(meta, :dot, [dx2, nx_call(meta, :log, [x1])])

      # f' (g / f)
      right = nx_call(meta, :dot, [dx1, nx_call(meta, :divide, [x2, x1])])

      # y * (left + right)
      [nx_call(meta, :dot, [y, nx_call(meta, :add, [left, right])]) | exprs]
    end
  end

  defp unfold_grad({{:., _, [Nx, name]}, meta, [x]}, y, exprs, state) do
    [grad_call(meta, name, [x, y]) | unfold_var(x, exprs, state)]
  end

  defp unfold_grad(x, _y, exprs, state) when is_var(x) do
    unfold_var(x, exprs, state)
  end

  # Catch-alls. Explicitly list them to help bugs.
  defp unfold_grad({:%{}, _, _}, _y, exprs, _state), do: [0.0 | exprs]
  defp unfold_grad(x, _y, exprs, _state) when is_number(x) or is_atom(x), do: [0.0 | exprs]

  defp to_dot(meta, exprs) do
    if 0.0 in exprs do
      0.0
    else
      Enum.reduce(exprs, &nx_call(meta, :dot, [&2, &1]))
    end
  end

  ## Helpers

  defp nx_call(_meta, :add, [0.0, right]), do: right
  defp nx_call(_meta, :add, [left, 0.0]), do: left
  defp nx_call(_meta, :subtract, [left, 0.0]), do: left
  defp nx_call(_meta, :multiply, [0.0, _right]), do: 0.0
  defp nx_call(_meta, :multiply, [_left, 0.0]), do: 0.0
  defp nx_call(_meta, :multiply, [1.0, right]), do: right
  defp nx_call(_meta, :multiply, [left, 1.0]), do: left
  defp nx_call(_meta, :dot, [0.0, _right]), do: 0.0
  defp nx_call(_meta, :dot, [_left, 0.0]), do: 0.0
  defp nx_call(_meta, :dot, [1.0, right]), do: right
  defp nx_call(_meta, :dot, [left, 1.0]), do: left
  defp nx_call(meta, name, args), do: {{:., meta, [Nx, name]}, meta, args}

  defp grad_call(meta, name, args), do: {{:., meta, [__MODULE__, name]}, meta, args}

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
  defn tanh(_x, y), do: 1.0 - y * y

  @doc """
  The derivative of `Nx.exp/1`.
  """
  defn exp(_x, y), do: y

  @doc """
  The derivative of `Nx.power/2` (when x is the base).
  """
  defn power(base, exponent, _y), do: exponent * Nx.power(base, exponent - 1)
end

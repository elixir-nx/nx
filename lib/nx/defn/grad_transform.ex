defmodule Nx.Defn.GradTransform do
  @moduledoc """
  A transform that returns the gradient of the given computation.
  """

  @behaviour Nx.Defn.Transform
  @root __MODULE__

  defguardp is_var(var)
            when is_tuple(var) and tuple_size(var) == 3 and is_atom(elem(var, 0)) and
                   is_atom(elem(var, 2))

  defguardp is_underscore(var)
            when is_tuple(var) and tuple_size(var) == 3 and elem(var, 0) == :_ and
                   is_atom(elem(var, 2))

  @hint "grad expects the numerical expression to return a scalar tensor"

  @impl true
  def __transform__(_env, version, meta, {vars, ast}, _opts) do
    state = %{version: version, vars: %{}, meta: meta}

    # SSA
    {ast, state} = ssa(ast, state)

    # Collect all variables by moving assigns and flattening blocks
    {result_var, state} = new_var(state)
    {ast, state} = collect_and_flatten({:=, meta, [result_var, ast]}, state)

    # Now compute the gradient
    grad = gradient_for(vars, result_var, state)
    assertion = nx_call(meta, :assert_shape, [result_var, {:{}, meta, []}, @hint])
    {state.version, append_to_block(ast, [assertion, grad])}
  end

  defp gradient_for({left, right}, result_var, state) do
    gradient_for({:{}, [], [left, right]}, result_var, state)
  end

  defp gradient_for({:{}, meta, args}, result_var, state) do
    {:{}, meta, Enum.map(args, &gradient_for(&1, result_var, state))}
  end

  defp gradient_for(var, result_var, state) when is_var(var) do
    state = put_in(state.vars[var_counter(var)], @root)
    state = Map.put(state, :shape, var)

    result_var
    |> unfold_var([], state)
    |> to_dot(state)
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

  defp one(%{meta: meta, shape: shape}), do: nx_call(meta, :broadcast, [1.0, shape])
  defp zero(%{meta: meta, shape: shape}), do: nx_call(meta, :broadcast, [0.0, shape])

  defmacrop one_pattern(), do: quote(do: {{:., _, [Nx, :broadcast]}, _, [1.0, _]})
  defmacrop zero_pattern(), do: quote(do: {{:., _, [Nx, :broadcast]}, _, [0.0, _]})

  defp one?(var), do: match?(one_pattern(), var)
  defp zero?(var), do: match?(zero_pattern(), var)

  defp unfold_var(var, exprs, state) when is_var(var) do
    counter = var_counter(var)

    case state.vars do
      %{^counter => @root} -> [one(state) | exprs]
      %{^counter => expr} -> unfold_grad(expr, var, exprs, state)
      %{} -> [zero(state) | exprs]
    end
  end

  defp unfold_var(_not_a_var, exprs, state), do: [zero(state) | exprs]

  ## First we start with the per op rules

  # Reshape rule (changes the shape upstream and then fixes it)
  defp unfold_grad({{:., _, [Nx, :reshape]} = call, meta, [x, shape]}, _y, exprs, state) do
    [dx | exprs] = unfold_var(x, exprs, %{state | shape: shape})
    [{call, meta, [dx, state.shape]} | exprs]
  end

  # Addition rule
  defp unfold_grad({{:., _, [Nx, name]}, meta, [x1, x2 | _args]}, _y, exprs, state)
       when name in [:add, :subtract] do
    dx1 = x1 |> unfold_var([], state) |> to_dot(state)
    dx2 = x2 |> unfold_var([], state) |> to_dot(state)
    [nx_call(meta, name, [dx1, dx2]) | exprs]
  end

  # Product rule
  defp unfold_grad({{:., _, [Nx, name]}, meta, [x1, x2 | _args]}, _y, exprs, state)
       when name in [:dot, :multiply] do
    dx1 = x1 |> unfold_var([], state) |> to_dot(state)
    dx2 = x2 |> unfold_var([], state) |> to_dot(state)

    expr =
      nx_call(meta, :add, [
        nx_call(meta, name, [dx1, x2]),
        nx_call(meta, name, [x1, dx2])
      ])

    [expr | exprs]
  end

  # Power/Exponentiation rule
  defp unfold_grad({{:., _, [Nx, :power]}, meta, [x1, x2 | _args]}, y, exprs, state) do
    dx1 = x1 |> unfold_var([], state) |> to_dot(state)
    dx2 = x2 |> unfold_var([], state) |> to_dot(state)

    if one?(dx1) and zero?(dx2) do
      [grad_call(meta, :power, [state.shape, y, x1, x2]) | exprs]
    else
      # g' ln f
      left = nx_call(meta, :dot, [dx2, nx_call(meta, :log, [x1])])

      # f' (g / f)
      right = nx_call(meta, :dot, [dx1, nx_call(meta, :divide, [x2, x1])])

      # y * (left + right)
      [nx_call(meta, :dot, [y, nx_call(meta, :add, [left, right])]) | exprs]
    end
  end

  ## These are generalizations

  # Compute the grad based on the first argument but keep the computation.
  @keepthrough_first_arg [:assert_shape]

  defp unfold_grad({{:., _, [Nx, name]} = call, meta, [x | args]}, _y, exprs, state)
       when name in @keepthrough_first_arg do
    [dx | exprs] = unfold_var(x, exprs, state)
    [{call, meta, [dx | args]} | exprs]
  end

  # These operations are always treated as constants
  @constants [:size, :rank]

  defp unfold_grad({{:., _, [Nx, name]}, _meta, _args}, _y, exprs, state)
       when name in @constants do
    [zero(state) | exprs]
  end

  # Nx calls that depend exclusively on the first arg
  defp unfold_grad({{:., _, [Nx, name]}, meta, [x | _args]}, y, exprs, state) do
    [grad_call(meta, name, [state.shape, y, x]) | unfold_var(x, exprs, state)]
  end

  defp unfold_grad(x, _y, exprs, state) when is_var(x) do
    unfold_var(x, exprs, state)
  end

  # Catch-alls. Explicitly list them to help bugs.
  defp unfold_grad({:%{}, _, _}, _y, exprs, state) do
    [zero(state) | exprs]
  end

  defp unfold_grad(x, _y, exprs, state) when is_number(x) or is_atom(x) do
    [zero(state) | exprs]
  end

  defp unfold_grad(x, _y, _exprs, _state) do
    raise "cannot yet grad expression: #{Macro.to_string(x)}"
  end

  defp to_dot(exprs, state) do
    if Enum.any?(exprs, &zero?/1) do
      zero(state)
    else
      Enum.reduce(exprs, &nx_call(state.meta, :dot, [&2, &1]))
    end
  end

  ## Helpers

  defp nx_call(_meta, :add, [zero_pattern(), right]), do: right
  defp nx_call(_meta, :add, [left, zero_pattern()]), do: left
  defp nx_call(_meta, :subtract, [left, zero_pattern()]), do: left
  defp nx_call(_meta, :multiply, [zero_pattern() = zero, _right]), do: zero
  defp nx_call(_meta, :multiply, [_left, zero_pattern() = zero]), do: zero
  defp nx_call(_meta, :multiply, [one_pattern(), right]), do: right
  defp nx_call(_meta, :multiply, [left, one_pattern()]), do: left
  defp nx_call(_meta, :dot, [zero_pattern() = zero, _right]), do: zero
  defp nx_call(_meta, :dot, [_left, zero_pattern() = zero]), do: zero
  defp nx_call(_meta, :dot, [one_pattern(), right]), do: right
  defp nx_call(_meta, :dot, [left, one_pattern()]), do: left
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
  The derivative of broadcast.
  """
  defn broadcast(shape, y, _x), do: Nx.broadcast(Nx.size(y) / Nx.size(shape), shape)

  @doc """
  The derivative of sum.
  """
  defn sum(_shape, _y, _x), do: 1.0

  @doc """
  The derivative of `Nx.tanh/2`.
  """
  defn tanh(_shape, y, _x), do: 1.0 - y * y

  @doc """
  The derivative of `Nx.exp/1`.
  """
  defn exp(_shape, y, _x), do: y

  @doc """
  The derivative of `Nx.power/2` (when x is the base).
  """
  defn power(_shape, _y, base, exponent), do: exponent * Nx.power(base, exponent - 1)
end

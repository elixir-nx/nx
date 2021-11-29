defmodule Nx.Defn.Composite do
  @moduledoc """
  Functions to deal with composite data types according to `Nx.Container`.

  The functions in this module can be used both inside and outside `defn`.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @doc """
  Traverses two composite types to see if they are compatible.

  For non-composite types, the given `fun` will be called to
  compare numbers/tensors pairwise.
  """
  def compatible?(left, right, fun)
      when (is_number(left) or is_struct(left, T)) and (is_number(right) or is_struct(right, T)),
      do: fun.(left, right)

  def compatible?(left, right, fun) when tuple_size(left) == tuple_size(right) do
    Tuple.to_list(left)
    |> Enum.zip(Tuple.to_list(right))
    |> Enum.all?(fn {l, r} -> compatible?(l, r, fun) end)
  end

  def compatible?(%mod{} = left, %mod{} = right, fun) do
    left = Nx.Container.reduce(left, [], &[&1 | &2])
    right = Nx.Container.reduce(right, [], &[&1 | &2])
    Enum.zip(left, right) |> Enum.all?(fn {l, r} -> compatible?(l, r, fun) end)
  end

  def compatible?(%_{}, %_{}, _fun),
    do: false

  def compatible?(left, right, fun) when map_size(left) == map_size(right) do
    Enum.all?(left, fn {k, v1} ->
      case right do
        %{^k => v2} -> compatible?(v1, v2, fun)
        %{} -> false
      end
    end)
  end

  def compatible?(_, _, _),
    do: false

  @doc """
  Counts the number of non-composite types in the composite type.

  ## Examples

      iex> Nx.Defn.Composite.count(123)
      1
      iex> Nx.Defn.Composite.count({1, {2, 3}})
      3

  """
  def count(tree), do: count(tree, 0)
  defp count(%T{}, acc), do: acc + 1
  defp count(number, acc) when is_number(number), do: acc + 1
  defp count(container, acc), do: Nx.Container.reduce(container, acc, &count/2)

  @doc """
  Traverses the given composite types with `fun`.

  If composite tensor expressions are given, such as a tuple,
  the composite type is recursively traversed and returned.

  If a non-composite tensor expression is given, the function
  is invoked for it but not for its arguments.
  """
  def traverse(expr, fun) when is_function(fun, 1) do
    {result, []} = traverse(expr, [], fn expr, [] -> {fun.(expr), []} end)
    result
  end

  @doc """
  Traverses the given composite types with `acc` and `fun`.

  If composite tensor expressions are given, such as a tuple,
  the composite type is recursively traversed and returned.

  If a non-composite tensor expression is given, the function
  is invoked for it but not for its arguments.
  """
  def traverse(%T{} = expr, acc, fun) when is_function(fun, 2),
    do: fun.(expr, acc)

  def traverse(number, acc, fun) when is_number(number) and is_function(fun, 2),
    do: fun.(number, acc)

  def traverse(container, acc, fun),
    do: Nx.Container.traverse(container, acc, &traverse(&1, &2, fun))

  @doc """
  Reduces the given composite types with `acc` and `fun`.

  If composite tensor expressions are given, such as a tuple,
  the composite type is recursively traversed and returned.

  If a non-composite tensor expression is given, the function
  is invoked for it but not for its arguments.
  """
  def reduce(%T{} = expr, acc, fun) when is_function(fun, 2),
    do: fun.(expr, acc)

  def reduce(number, acc, fun) when is_number(number) and is_function(fun, 2),
    do: fun.(number, acc)

  def reduce(container, acc, fun),
    do: Nx.Container.reduce(container, acc, &reduce(&1, &2, fun))

  @doc """
  Flattens the given list of composite types.

  Elements that are not tensors (i.e. numbers) are kept as is
  unless a custom function is given.

  ## Examples

      iex> Nx.Defn.Composite.flatten_list([1, {2, 3}])
      [1, 2, 3]

      iex> Nx.Defn.Composite.flatten_list([1, {2, 3}], [Nx.tensor(4)])
      [1, 2, 3, Nx.tensor(4)]

      iex> Nx.Defn.Composite.flatten_list([1, {2, 3}], [Nx.tensor(4)], &Nx.tensor/1)
      [Nx.tensor(1), Nx.tensor(2), Nx.tensor(3), Nx.tensor(4)]

  """
  def flatten_list(args, tail \\ [], fun \\ & &1) when is_list(args) do
    args
    |> Enum.reduce([], &flatten_each(&1, &2, fun))
    |> Enum.reverse(tail)
  end

  defp flatten_each(%T{} = tensor, acc, _fun),
    do: [tensor | acc]

  defp flatten_each(number, acc, fun) when is_number(number),
    do: [fun.(number) | acc]

  defp flatten_each(container, acc, fun),
    do: Nx.Container.reduce(container, acc, &flatten_each(&1, &2, fun))

  ## Nx.Defn callbacks

  @doc false
  def from_compile_args(args, cache) do
    from_compile_args(args, cache, [])
  end

  defp from_compile_args([arg | args], cache, vars) when is_function(arg) do
    from_compile_args(args, [arg | cache], vars)
  end

  defp from_compile_args([arg | args], cache, vars) do
    from_compile_args(args, cache, [arg | vars])
  end

  defp from_compile_args([], cache, vars), do: {cache, Enum.reverse(vars)}

  @doc false
  def from_runtime_args(args, tail) do
    flatten_list(args, tail, &Nx.to_tensor/1)
  end

  @doc false
  def to_result(container) do
    traverse(container, &Expr.tensor/1)
  end

  @doc false
  def args_to_params(args, params) do
    {args, {[], _}} =
      Enum.map_reduce(args, {params, 0}, fn
        arg, acc
        when is_function(arg)
        when is_tuple(arg) and is_function(elem(arg, 0)) ->
          {arg, acc}

        arg, acc ->
          traverse(arg, acc, fn _, {[param | params], i} ->
            {Expr.parameter(param, :root, i), {params, i + 1}}
          end)
      end)

    args
  end
end

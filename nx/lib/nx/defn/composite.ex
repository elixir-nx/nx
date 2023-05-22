defmodule Nx.Defn.Composite do
  @moduledoc """
  Functions to deal with composite data types.

  Composite data-types are traversed according to `Nx.Container`.
  If a regular tensor is given, it is individually traversed. 
  Numerical values, such as integers, floats, and complex numbers
  are not normalized before hand. Use `Nx.to_tensor/1` to do so.

  The functions in this module can be used both inside and outside `defn`.
  Note that, when a value is given to `defn`, it is first converted to
  tensors and containers via `Nx.LazyContainer`. Inside `defn`, there are
  no lazy containers, only containers.
  """

  alias Nx.Tensor, as: T

  import Nx, only: [is_tensor: 1]

  @doc """
  Traverses two composite types to see if they are compatible.

  Non-tensor values are first compared using `Nx.LazyContainer`
  and then, if not available, as `Nx.Container`.

  For non-composite types, the given `fun` will be called to
  compare numbers/tensors pairwise.
  """
  def compatible?(left, right, fun)
      when is_tensor(left) and is_tensor(right),
      do: fun.(left, right)

  def compatible?(left, right, fun) when tuple_size(left) == tuple_size(right) do
    Tuple.to_list(left)
    |> Enum.zip(Tuple.to_list(right))
    |> Enum.all?(fn {l, r} -> compatible?(l, r, fun) end)
  end

  def compatible?(%mod{} = left, %mod{} = right, fun) do
    # LazyContainer is fully recursive but we don't want to go full recursive
    # unless we have to, so we can also compare structures along the way.
    {left, right} =
      case Nx.LazyContainer.impl_for(left) do
        Nx.LazyContainer.Any ->
          left = Nx.Container.reduce(left, [], &[&1 | &2])
          right = Nx.Container.reduce(right, [], &[&1 | &2])
          {left, right}

        impl ->
          {_, left} =
            impl.traverse(left, [], fn template, _fun, acc -> {template, [template | acc]} end)

          {_, right} =
            impl.traverse(right, [], fn template, _fun, acc -> {template, [template | acc]} end)

          {left, right}
      end

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
      iex> Nx.Defn.Composite.count({Complex.new(1), {Nx.tensor(2), 3}})
      3

  """
  def count(tree), do: count(tree, 0)
  defp count(tensor, acc) when is_tensor(tensor), do: acc + 1
  defp count(container, acc), do: Nx.Container.reduce(container, acc, &count/2)

  @doc """
  Traverses recursively the given composite types with `fun`.

  If a composite tensor is given, such as a tuple, the composite
  type is recursively traversed and returned.

  Otherwise the function is invoked with the tensor (be it a
  number, complex, or actual tensor).
  """
  def traverse(expr, fun) when is_function(fun, 1) do
    {result, []} = traverse(expr, [], fn expr, [] -> {fun.(expr), []} end)
    result
  end

  @doc """
  Traverses recursively the given composite types with `acc` and `fun`.

  If a composite tensor is given, such as a tuple, the composite
  type is recursively traversed and returned.

  Otherwise the function is invoked with the tensor (be it a
  number, complex, or actual tensor).
  """
  def traverse(expr, acc, fun) when is_tensor(expr) and is_function(fun, 2),
    do: fun.(expr, acc)

  def traverse(container, acc, fun),
    do: Nx.Container.traverse(container, acc, &traverse(&1, &2, fun))

  @doc """
  Reduces recursively the given composite types with `acc` and `fun`.

  If composite tensor expressions are given, such as a tuple,
  the composite type is recursively traversed and returned.

  If a non-composite tensor expression is given, the function
  is invoked for it but not for its arguments.
  """
  def reduce(expr, acc, fun) when is_tensor(expr) and is_function(fun, 2),
    do: fun.(expr, acc)

  def reduce(container, acc, fun),
    do: Nx.Container.reduce(container, acc, &reduce(&1, &2, fun))

  @doc """
  Flattens recursively the given list of composite types.

  Elements that are not tensors (i.e. numbers and `Complex` numbers) are kept as is
  unless a custom function is given.

  ## Examples

      iex> Nx.Defn.Composite.flatten_list([1, {2, 3}])
      [1, 2, 3]

      iex> Nx.Defn.Composite.flatten_list([1, {2, 3}], [Nx.tensor(4)])
      [1, 2, 3, Nx.tensor(4)]

  """
  def flatten_list(args, tail \\ []) when is_list(args) do
    args
    |> Enum.reduce([], &flatten_each/2)
    |> Enum.reverse(tail)
  end

  defp flatten_each(%T{} = tensor, acc),
    do: [tensor | acc]

  defp flatten_each(number, acc)
       when is_number(number) or is_struct(number, Complex),
       do: [number | acc]

  defp flatten_each(container, acc),
    do: Nx.Container.reduce(container, acc, &flatten_each/2)
end

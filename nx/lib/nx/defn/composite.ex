defmodule Nx.Defn.Composite do
  @moduledoc """
  Functions to deal with composite data types according to `Nx.Container`.

  The functions in this module can be used both inside and outside `defn`.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  import Nx, only: [is_tensor: 1]

  @doc """
  Traverses two composite types to see if they are compatible.

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
      iex> Nx.Defn.Composite.count({Complex.new(1), {Nx.tensor(2), 3}})
      3

  """
  def count(tree), do: count(tree, 0)
  defp count(tensor, acc) when is_tensor(tensor), do: acc + 1
  defp count(container, acc), do: Nx.Container.reduce(container, acc, &count/2)

  @doc """
  Traverses the given composite types with `fun`.

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
  Traverses the given composite types with `acc` and `fun`.

  If a composite tensor is given, such as a tuple, the composite
  type is recursively traversed and returned.

  Otherwise the function is invoked with the tensor (be it a
  number, complex, or actual tensor).
  """
  def traverse(expr, acc, fun) when (is_tensor(expr) or is_boolean(expr)) and is_function(fun, 2),
    do: fun.(expr, acc)

  def traverse(container, acc, fun),
    do: Nx.Container.traverse(container, acc, &traverse(&1, &2, fun))

  @doc """
  Reduces the given composite types with `acc` and `fun`.

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
  Flattens the given list of composite types.

  Elements that are not tensors (i.e. numbers and `Complex` numbers) are kept as is
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

  defp flatten_each(number, acc, fun)
       when is_number(number) or is_struct(number, Complex),
       do: [fun.(number) | acc]

  defp flatten_each(container, acc, fun),
    do: Nx.Container.reduce(container, acc, &flatten_each(&1, &2, fun))

  ## Nx.Defn callbacks

  @doc false
  def split_compile_args(args, cache) do
    split_compile_args(args, cache, [])
  end

  defp split_compile_args([arg | args], cache, vars)
       when is_function(arg)
       when is_tuple(arg) and is_function(elem(arg, 0)) do
    split_compile_args(args, [arg | cache], vars)
  end

  defp split_compile_args([arg | args], cache, vars) do
    split_compile_args(args, cache, [arg | vars])
  end

  defp split_compile_args([], cache, vars), do: {cache, Enum.reverse(vars)}

  @doc false
  def join_compile_args(partial_args, full_args) do
    {args, []} =
      Enum.map_reduce(full_args, partial_args, fn
        arg, acc
        when is_function(arg)
        when is_tuple(arg) and is_function(elem(arg, 0)) ->
          {arg, acc}

        _arg, [arg | acc] ->
          {arg, acc}
      end)

    args
  end

  @doc false
  def flatten_runtime_args(args, tail) do
    list = flatten_list(args, tail, &Nx.to_tensor/1)

    for %Nx.Tensor{data: %Nx.Defn.Expr{}} = tensor <- list do
      raise ArgumentError,
            "cannot pass a tensor expression as argument to defn, got: #{inspect(tensor)}"
    end

    list
  end

  @doc false
  def to_inputs(args) do
    {args, _} =
      Enum.map_reduce(args, 0, fn arg, i ->
        traverse(arg, i, fn
          arg, _i when is_boolean(arg) ->
            raise ArgumentError, "booleans are not supported as defn inputs"

          arg, i ->
            {Expr.parameter(Nx.to_tensor(arg), :root, i), i + 1}
        end)
      end)

    args
  end

  @doc false
  def to_output(container) do
    traverse(container, &Expr.tensor/1)
  end
end

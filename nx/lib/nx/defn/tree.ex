defmodule Nx.Defn.Tree do
  @moduledoc """
  Helper functions to traverse expressions.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @doc """
  Traverses two trees to see if they are compatible.
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
  Puts new args in the given expression and gives it a new id.
  """
  def put_args(%T{data: %Expr{} = expr} = t, args) do
    %{t | data: %{expr | id: Expr.id(), args: args}}
  end

  @doc """
  Counts the number of elements in the tree.

  ## Examples

      iex> Nx.Defn.Tree.count(123)
      1
      iex> Nx.Defn.Tree.count({1, {2, 3}})
      3

  """
  def count(tree), do: count(tree, 0)
  defp count(%T{}, acc), do: acc + 1
  defp count(number, acc) when is_number(number), do: acc + 1
  defp count(container, acc), do: Nx.Container.reduce(container, acc, &count/2)

  @doc """
  Traverses the arguments of a tensor expression.

  Warning: be very careful when using this function to traverse the expression
  recursively. If you plan to do so, you should consider also storing the visited
  nodes to avoid multiple traversals.
  """
  def traverse_args(expr, acc, fun)

  def traverse_args(%T{data: %Expr{op: :fun, args: [args, expr, mfa]}}, acc, fun) do
    {args, acc} = Enum.map_reduce(args, acc, &composite(&1, &2, fun))
    {expr, acc} = composite(expr, acc, fun)
    {[args, expr, mfa], acc}
  end

  def traverse_args(%T{data: %Expr{op: :cond, args: [clauses, last]}}, acc, fun) do
    {clauses, acc} =
      Enum.map_reduce(clauses, acc, fn {pred, expr}, acc ->
        {pred, acc} = fun.(pred, acc)
        {expr, acc} = composite(expr, acc, fun)
        {{pred, expr}, acc}
      end)

    {last, acc} = composite(last, acc, fun)
    {[clauses, last], acc}
  end

  def traverse_args(%T{data: %Expr{op: :while, args: args}}, acc, fun) do
    [initial, arg, pred, block] = args
    {initial, acc} = composite(initial, acc, fun)
    {arg, acc} = composite(arg, acc, fun)
    {pred, acc} = fun.(pred, acc)
    {block, acc} = composite(block, acc, fun)
    {[initial, arg, pred, block], acc}
  end

  def traverse_args(%T{data: %Expr{op: :concatenate, args: [list | args]}}, acc, fun) do
    {list, acc} = Enum.map_reduce(list, acc, fun)
    {[list | args], acc}
  end

  def traverse_args(%T{data: %Expr{op: :slice, args: [tensor, start_indices | args]}}, acc, fun) do
    {tensor, acc} = fun.(tensor, acc)

    {start_indices, acc} =
      Enum.map_reduce(start_indices, acc, fn
        x, acc when is_integer(x) -> {x, acc}
        x, acc -> fun.(x, acc)
      end)

    {[tensor, start_indices | args], acc}
  end

  def traverse_args(
        %T{data: %Expr{op: :put_slice, args: [tensor, start_indices, slice]}},
        acc,
        fun
      ) do
    {tensor, acc} = fun.(tensor, acc)
    {slice, acc} = fun.(slice, acc)

    {start_indices, acc} =
      Enum.map_reduce(start_indices, acc, fn
        x, acc when is_integer(x) -> {x, acc}
        x, acc -> fun.(x, acc)
      end)

    {[tensor, start_indices, slice], acc}
  end

  def traverse_args(%T{data: %Expr{args: args}}, acc, fun) do
    Enum.map_reduce(args, acc, fn
      %T{data: %Expr{}} = arg, acc -> fun.(arg, acc)
      arg, acc -> {arg, acc}
    end)
  end

  @doc """
  Traverses the given composite type of tensor expressions with `fun`.

  This function exists to handle composite types that may
  have multiple tensor expressions inside.

  If composite tensor expressions are given, such as a tuple,
  the composite type is recursively traversed and returned.

  If a non-composite tensor expression is given, the function
  is invoked for it but not for its arguments (see `traverse_args/3`
  for that).
  """
  def composite(expr, fun) when is_function(fun, 1) do
    {result, []} = composite(expr, [], fn expr, [] -> {fun.(expr), []} end)
    result
  end

  @doc """
  Traverses the given composite type of tensor expressions with `acc` and `fun`.

  This function exists to handle composite types that may
  have multiple tensor expressions inside.

  If composite tensor expressions are given, such as a tuple,
  the composite type is recursively traversed and returned.

  If a non-composite tensor expression is given, the function
  is invoked for it but not for its arguments (see `traverse_args/3`
  for that).
  """
  def composite(%T{} = expr, acc, fun) when is_function(fun, 2),
    do: fun.(expr, acc)

  def composite(number, acc, fun) when is_number(number) and is_function(fun, 2),
    do: fun.(number, acc)

  def composite(container, acc, fun),
    do: Nx.Container.traverse(container, acc, &composite(&1, &2, fun))

  ## Type helpers

  @doc """
  Rewrites the types of the given tensor expressions according to
  the given options.

  ## Options

    * `:max_float_type` - set the max float type
    * `:max_signed_type` - set the max signed integer type
    * `:max_unsigned_type` - set the max unsigned integer type

  """
  def rewrite_types(tensor_expr, opts \\ []) when is_list(opts) do
    {_, max_float_size} = max_float_type = opts[:max_float_type] || {:f, 64}
    {_, max_signed_size} = max_signed_type = opts[:max_signed_type] || {:s, 64}
    {_, max_unsigned_size} = max_unsigned_type = opts[:max_unsigned_type] || {:u, 64}

    if not Nx.Type.float?(max_float_type) do
      raise ArgumentError, ":max_float_type must be float type, got: #{inspect(max_float_type)}"
    end

    if max_float_type != {:f, 64} or max_signed_type != {:s, 64} or max_unsigned_type != {:u, 64} do
      rewrite_type(tensor_expr, fn
        {:u, size} when size >= max_unsigned_size -> max_unsigned_type
        {:s, size} when size >= max_signed_size -> max_signed_type
        {:f, size} when size >= max_float_size -> max_float_type
        {:bf, size} when size >= max_float_size -> max_float_type
        type -> type
      end)
    else
      tensor_expr
    end
  end

  defp rewrite_type(expr, fun) do
    {res, _} = rewrite_type(expr, %{}, fun)
    res
  end

  defp rewrite_type(expr, cache, fun) do
    composite(expr, cache, fn %T{data: %Expr{id: id, op: op}} = t, cache ->
      case cache do
        %{^id => res} ->
          {res, cache}

        %{} ->
          {args, cache} = traverse_args(t, cache, &rewrite_type(&1, &2, fun))
          res = rewrite_type(op, args, t, fun)
          {res, Map.put(cache, id, res)}
      end
    end)
  end

  defp rewrite_type(:parameter, _args, %{data: %{context: :root}} = t, type_fun) do
    Nx.as_type(t, type_fun.(t.type))
  end

  defp rewrite_type(:tensor, [arg], t, type_fun) do
    type = type_fun.(t.type)
    rewrite_type_args(t, type, [Nx.as_type(arg, type)])
  end

  defp rewrite_type(:constant, [arg], t, type_fun) do
    type = type_fun.(t.type)
    rewrite_type_args(t, type, [arg])
  end

  defp rewrite_type(_op, args, t, type_fun) do
    rewrite_type_args(t, type_fun.(t.type), args)
  end

  defp rewrite_type_args(%{data: data} = t, type, args) do
    %{t | data: %{data | id: Expr.id(), args: args}, type: type}
  end

  @doc """
  Flattens the given list of tensor expressions, flattening maps,
  tuples, containers, into a list.

  Elements that are not tensors (i.e. numbers) are kept as is
  unless a custom function is given.

  ## Examples

      iex> Nx.Defn.Tree.flatten_list([1, {2, 3}])
      [1, 2, 3]

      iex> Nx.Defn.Tree.flatten_list([1, {2, 3}], [Nx.tensor(4)])
      [1, 2, 3, Nx.tensor(4)]

      iex> Nx.Defn.Tree.flatten_list([1, {2, 3}], [Nx.tensor(4)], &Nx.tensor/1)
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
    # We want to allocate all inputs on the backend, so we use Nx.tensor/1
    flatten_list(args, tail, &Nx.tensor/1)
  end

  @doc false
  def to_result(%T{data: %Expr{}} = t),
    do: t

  def to_result(number) when is_number(number),
    do: Expr.tensor(number)

  def to_result(other),
    do: other |> Nx.Container.traverse(:ok, &{to_result(&1), &2}) |> elem(0)

  @doc false
  def args_to_params(args, params) do
    {args, {[], _}} =
      Enum.map_reduce(args, {params, 0}, fn
        arg, acc when is_function(arg) -> {arg, acc}
        arg, acc -> args_to_param(arg, acc)
      end)

    args
  end

  defp args_to_param(tensor, {[param | params], i})
       when is_struct(tensor, T) or is_number(tensor) do
    {Expr.parameter(param, :root, i), {params, i + 1}}
  end

  defp args_to_param(container, acc) do
    Nx.Container.traverse(container, acc, &args_to_param/2)
  end
end

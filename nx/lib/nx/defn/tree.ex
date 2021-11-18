defmodule Nx.Defn.Tree do
  @moduledoc """
  Helper functions to traverse expressions.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @doc """
  Puts new args in the given expression and gives it a new id.
  """
  def put_args(%T{data: %Expr{} = expr} = t, args) do
    %{t | data: %{expr | id: Expr.id(), args: args}}
  end

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

  Elements that are not tensors are converted to tensors via
  `Nx.tensor/1`.

  ## Examples

      iex> Nx.Defn.Tree.flatten_list([1, {2, 3}])
      [Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)]

      iex> Nx.Defn.Tree.flatten_list([1, {2, 3}], [Nx.tensor(4)])
      [Nx.tensor(1), Nx.tensor(2), Nx.tensor(3), Nx.tensor(4)]

  """
  def flatten_list(args, tail \\ []) when is_list(args) do
    args
    |> Enum.reduce([], &elem(flatten_each(&1, &2), 1))
    |> Enum.reverse(tail)
  end

  defp flatten_each(%T{} = tensor, acc),
    do: {tensor, [tensor | acc]}

  defp flatten_each(number, acc) when is_number(number),
    do: {number, [Nx.to_tensor(number) | acc]}

  defp flatten_each(container, acc),
    do: Nx.Container.traverse(container, acc, &flatten_each/2)

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
  def to_result(%T{data: %Expr{}} = t),
    do: t

  def to_result(number) when is_number(number),
    do: "defn must return a tensor expression or a tuple, got: #{inspect(number)}"

  def to_result(other),
    do: other |> Nx.Container.traverse(:ok, &{to_result(&1), &2}) |> elem(0)

  @doc false
  def args_to_params(args, params) do
    {args, {[], _}} =
      args_to(args, {params, 0}, fn _arg, {[param | params], i} ->
        {Expr.parameter(param, :root, i), {params, i + 1}}
      end)

    args
  end

  @doc false
  def args_to_templates(args, params) do
    {args, []} =
      args_to(args, params, fn _arg, [param | params] ->
        {Nx.to_template(param), params}
      end)

    args
  end

  defp args_to(args, acc, fun) when is_list(args) do
    Enum.map_reduce(args, acc, fn
      arg, acc when is_function(arg) -> {arg, acc}
      arg, acc -> Nx.Container.traverse(arg, acc, fun)
    end)
  end
end

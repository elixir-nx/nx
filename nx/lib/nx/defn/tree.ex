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
  """
  def traverse_args(expr, acc, fun)

  def traverse_args(%T{data: %Expr{op: :fun, args: [args, expr, mfa]}}, acc, fun) do
    {args, acc} = Enum.map_reduce(args, acc, &composite(&1, &2, fun))
    {expr, acc} = composite(expr, acc, fun)
    {[args, expr, mfa], acc}
  end

  def traverse_args(%T{data: %Expr{op: :cond, args: [clauses, last]}}, acc, fun) do
    {clauses, acc} =
      Enum.map_reduce(clauses, acc, fn {condition, expr}, acc ->
        {condition, acc} = fun.(condition, acc)
        {expr, acc} = composite(expr, acc, fun)
        {{condition, expr}, acc}
      end)

    {last, acc} = composite(last, acc, fun)
    {[clauses, last], acc}
  end

  def traverse_args(%T{data: %Expr{op: :while, args: args}}, acc, fun) do
    [initial, arg, conditional, block] = args
    {initial, acc} = composite(initial, acc, fun)
    {arg, acc} = composite(arg, acc, fun)
    {conditional, acc} = fun.(conditional, acc)
    {block, acc} = composite(block, acc, fun)
    {[initial, arg, conditional, block], acc}
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
  def composite(%T{} = expr, acc, fun) when is_function(fun, 2) do
    fun.(expr, acc)
  end

  def composite(tuple, acc, fun) when is_tuple(tuple) and is_function(fun, 2) do
    {list, acc} = Enum.map_reduce(Tuple.to_list(tuple), acc, &composite(&1, &2, fun))
    {List.to_tuple(list), acc}
  end

  def composite(map, acc, fun) when is_struct(map) and is_function(fun, 2) do
    {list, acc} =
      map
      |> Map.from_struct()
      |> Enum.map_reduce(acc, fn {k, v}, acc ->
        {v, acc} = composite(v, acc, fun)
        {{k, v}, acc}
      end)

    {struct(map.__struct__, list), acc}
  end

  def composite(map, acc, fun) when is_map(map) and is_function(fun, 2) do
    {list, acc} =
      map
      |> Enum.map_reduce(acc, fn {k, v}, acc ->
        {v, acc} = composite(v, acc, fun)
        {{k, v}, acc}
      end)

    {Map.new(list), acc}
  end

  def composite(expr, acc, fun) when is_function(fun, 2) do
    fun.(expr, acc)
  end

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

  defp rewrite_type(:scalar, [arg], t, type_fun) do
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
  tuples, into a list.

  Elements that are not tensors are converted to tensors via
  `Nx.tensor/1`.

  ## Examples

      iex> Nx.Defn.Tree.flatten_list([1, {2, 3}])
      [Nx.tensor(1), Nx.tensor(2), Nx.tensor(3)]

      iex> Nx.Defn.Tree.flatten_list([1, {2, 3}], [Nx.tensor(4)])
      [Nx.tensor(1), Nx.tensor(2), Nx.tensor(3), Nx.tensor(4)]

  """
  def flatten_list(args, tail \\ []) when is_list(args) do
    flatten_list(args, tail, &Nx.tensor/1)
  end

  defp flatten_list(args, tail, fun) do
    args
    |> Enum.reduce([], &flatten_each(&1, &2, fun))
    |> Enum.reverse(tail)
  end

  defp flatten_each(%T{} = tensor, acc, _fun),
    do: [tensor | acc]

  defp flatten_each(tuple, acc, fun) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.reduce(acc, &flatten_each(&1, &2, fun))

  defp flatten_each(map, acc, fun) when is_struct(map),
    do:
      map
      |> Map.from_struct()
      |> Enum.sort()
      |> Enum.reduce(acc, &flatten_each(elem(&1, 1), &2, fun))

  defp flatten_each(map, acc, fun) when is_map(map),
    do:
      map
      |> Enum.sort()
      |> Enum.reduce(acc, &flatten_each(elem(&1, 1), &2, fun))

  defp flatten_each(other, acc, fun),
    do: [fun.(other) | acc]

  ## Nx.Defn callbacks

  @doc false
  def from_compile_args(args, cache) do
    from_compile_args(args, cache, [])
  end

  defp from_compile_args([arg | args], cache, vars) when is_function(arg) do
    from_compile_args(args, [arg | cache], vars)
  end

  defp from_compile_args([arg | args], cache, vars) when is_tuple(arg) do
    if arg |> Tuple.to_list() |> Enum.all?(&is_function/1) do
      from_compile_args(args, [arg | cache], vars)
    else
      from_compile_args(args, cache, [arg | vars])
    end
  end

  defp from_compile_args([arg | args], cache, vars) do
    from_compile_args(args, cache, [arg | vars])
  end

  defp from_compile_args([], cache, vars), do: {cache, Enum.reverse(vars)}

  @doc false
  def from_runtime_args(args) do
    flatten_list(args, [], &from_arg/1)
  end

  @valid "defn arguments must be numbers, tensors, and functions. " <>
           "It may also be a maps with numbers/tensors as values, " <>
           "a tuple of numbers/tensors or a tuple of functions. "

  @doc false
  def from_arg(%T{} = tensor), do: tensor
  def from_arg(number) when is_number(number), do: Nx.tensor(number)

  def from_arg(other) when is_function(other) do
    raise ArgumentError,
          @valid <>
            "Anonymous functions are only allowed as direct arguments to defn: " <> inspect(other)
  end

  def from_arg(other) do
    raise ArgumentError, @valid <> "Got: #{inspect(other)}"
  end

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

  @doc false
  def to_result(%T{data: %Expr{}} = t),
    do: t

  def to_result(map) when is_struct(map),
    do:
      map
      |> Map.from_struct()
      |> Enum.map(fn {k, v} -> {k, to_result(v)} end)
      |> then(&struct(map.__struct__, &1))

  def to_result(map) when is_map(map),
    do: map |> Enum.map(fn {k, v} -> {k, to_result(v)} end) |> Map.new()

  def to_result(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&to_result/1) |> List.to_tuple()

  def to_result(other) do
    raise ArgumentError,
          "defn must return a tensor expression or a tuple, got: #{inspect(other)}"
  end

  defp args_to(args, acc, fun) when is_list(args) do
    Enum.map_reduce(args, acc, fn
      arg, acc
      when is_function(arg)
      when is_tuple(arg) and is_function(elem(arg, 0)) ->
        {arg, acc}

      arg, acc ->
        args_to_each(arg, acc, fun)
    end)
  end

  defp args_to_each(%T{} = arg, acc, fun) do
    fun.(arg, acc)
  end

  defp args_to_each(tuple, acc, fun) when is_tuple(tuple) do
    {list, acc} =
      tuple
      |> Tuple.to_list()
      |> Enum.map_reduce(acc, &args_to_each(&1, &2, fun))

    {List.to_tuple(list), acc}
  end

  defp args_to_each(map, acc, fun) when is_struct(map) do
    {list, acc} =
      map
      |> Map.from_struct()
      |> Enum.sort()
      |> Enum.map_reduce(acc, fn {k, v}, acc ->
        {v, acc} = args_to_each(v, acc, fun)
        {{k, v}, acc}
      end)

    {struct(map.__struct__, list), acc}
  end

  defp args_to_each(map, acc, fun) when is_map(map) do
    {list, acc} =
      map
      |> Enum.sort()
      |> Enum.map_reduce(acc, fn {k, v}, acc ->
        {v, acc} = args_to_each(v, acc, fun)
        {{k, v}, acc}
      end)

    {Map.new(list), acc}
  end

  defp args_to_each(arg, acc, fun) do
    fun.(arg, acc)
  end
end

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
  Helper to traverse the arguments of a tensor expression.

  Note the arguments of function nodes are never traversed, as it is
  not always desired to recursively modify them. If you want to modify
  a function, you will need to build a new function node by wrapping
  the function node `fun` with the new desired logic.
  """
  def traverse_args(expr, acc, fun)

  def traverse_args(%T{data: %Expr{op: :fun, args: args}}, acc, _fun) do
    {args, acc}
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

  def traverse_args(%T{data: %Expr{op: :concatenate, args: [list | args]}}, acc, fun) do
    {list, acc} = Enum.map_reduce(list, acc, fun)
    {[list | args], acc}
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
  def composite(tuple, acc, fun) when is_tuple(tuple) and is_function(fun, 2) do
    {list, acc} = Enum.map_reduce(Tuple.to_list(tuple), acc, &composite(&1, &2, fun))
    {List.to_tuple(list), acc}
  end

  def composite(%T{} = expr, acc, fun) when is_function(fun, 2) do
    fun.(expr, acc)
  end

  def composite(other, _acc, _fun) do
    raise ArgumentError,
          "expected a tensor expression or a tuple of tensor expressions, got: #{inspect(other)}"
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

  defp rewrite_type(:parameter, _args, t, type_fun) do
    Nx.as_type(t, type_fun.(t.type))
  end

  defp rewrite_type(:fun, [params, _expr, fun], _t, type_fun) do
    {:arity, arity} = Function.info(fun, :arity)
    params = Enum.map(params, &%{&1 | type: type_fun.(&1.type)})
    Expr.fun(params, rewrite_type_fun(arity, fun, type_fun))
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

  for arity <- 0..15 do
    args = Macro.generate_arguments(arity, __MODULE__)

    defp rewrite_type_fun(unquote(arity), op_fun, type_fun) do
      fn unquote_splicing(args) -> rewrite_type(op_fun.(unquote_splicing(args)), type_fun) end
    end
  end

  defp rewrite_type_args(%{data: data} = t, type, args) do
    %{t | data: %{data | id: Expr.id(), args: args}, type: type}
  end

  ## Nx.Defn callbacks

  @doc false
  # Returns tensors from flat args.
  def from_flat_args(vars) do
    for var <- vars do
      case var do
        %T{} = head ->
          head

        number when is_number(number) ->
          Nx.tensor(number)

        tuple when is_tuple(tuple) ->
          raise ArgumentError,
                "defn functions expects either numbers or tensors as arguments. " <>
                  "If you want to pass a tuple, you must explicitly pattern match on the tuple in the signature" <>
                  "Got: #{inspect(tuple)}"

        other ->
          raise ArgumentError,
                "defn functions expects either numbers or tensors as arguments. " <>
                  "If you want to pass Elixir values, they need to be sent as options and " <>
                  "tagged as default arguments. Got: #{inspect(other)}"
      end
    end
  end

  @doc false
  # Returns tensors from nested args.
  def from_nested_args(args) do
    args
    |> Enum.reduce([], &from_nested_args/2)
    |> Enum.reverse()
  end

  defp from_nested_args(tuple, acc) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.reduce(acc, &from_nested_args/2)

  defp from_nested_args(other, acc),
    do: [from_arg(other) | acc]

  @doc false
  # Returns tensor from a single arg.
  def from_arg(%T{} = t), do: t
  def from_arg(number) when is_number(number), do: Nx.tensor(number)

  def from_arg(other) do
    raise(
      ArgumentError,
      "arguments to defn functions must numbers, tensors, or tuples, got: #{inspect(other)}"
    )
  end

  @doc false
  # Converts nested args to nested params.
  def to_nested_params(args, params) do
    {args, {[], _}} =
      to_nested_args(args, {params, 0}, fn _arg, {[param | params], i} ->
        {Expr.parameter(param, :root, i), {params, i + 1}}
      end)

    args
  end

  @doc false
  # Converts flat args to flat params.
  # TODO: Use Enum.with_index/2 on Elixir v1.12+
  def to_flat_params(vars),
    do: to_flat_params(vars, 0)

  defp to_flat_params([head | tail], i),
    do: [Expr.parameter(head, :root, i) | to_flat_params(tail, i + 1)]

  defp to_flat_params([], _i),
    do: []

  @doc false
  # Converts nested args to nested templates.
  def to_nested_templates(args, params) do
    {args, []} =
      to_nested_args(args, params, fn _arg, [param | params] ->
        {Nx.template(param, param.type), params}
      end)

    args
  end

  @doc false
  def to_result(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&to_result/1) |> List.to_tuple()

  def to_result(%T{data: %Expr{}} = t),
    do: t

  def to_result(other) do
    raise ArgumentError,
          "defn must return a tensor expression or a tuple, got: #{inspect(other)}"
  end

  defp to_nested_args(args, acc, fun) when is_list(args) do
    Enum.map_reduce(args, acc, &to_nested_each(&1, &2, fun))
  end

  defp to_nested_each(arg, acc, fun) when is_tuple(arg) do
    {list, acc} =
      arg
      |> Tuple.to_list()
      |> Enum.map_reduce(acc, &to_nested_each(&1, &2, fun))

    {List.to_tuple(list), acc}
  end

  defp to_nested_each(arg, acc, fun) do
    fun.(arg, acc)
  end
end

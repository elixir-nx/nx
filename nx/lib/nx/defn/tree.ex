defmodule Nx.Defn.Tree do
  @moduledoc """
  Helper functions to traverse defn expressions,
  either as single nodes or recursively.
  """

  alias Nx.Defn.{Composite, Expr}
  alias Nx.Tensor, as: T

  @doc """
  Check if the given tree has any of the given hooks in it.
  """
  def has_hooks?(tree, hooks) do
    Composite.reduce(tree, %{}, &detect_hook(&1, &2, hooks))
    false
  catch
    :side_effect -> true
  end

  defp detect_hook(%T{data: %Expr{op: :token, args: [token]}} = t, cache, hooks) do
    if Enum.any?(token.hooks, &(hooks[&1.name] || &1.callback)) do
      throw(:side_effect)
    else
      fallback_detect_hook(t, cache, hooks)
    end
  end

  defp detect_hook(t, cache, hooks), do: fallback_detect_hook(t, cache, hooks)

  defp fallback_detect_hook(%T{data: %Expr{id: id}} = t, cache, hooks) do
    case cache do
      %{^id => _} ->
        cache

      %{} ->
        {_, cache} = apply_args(t, cache, &{&1, detect_hook(&1, &2, hooks)})
        Map.put(cache, id, true)
    end
  end

  @doc """
  Puts new args in the given tensor expression and gives it a new id.
  """
  def put_args(%T{data: %Expr{} = expr} = t, args) do
    %{t | data: %{expr | id: Expr.id(), args: args}}
  end

  @doc """
  Replaces args in the given tensor expression.

  Use this function with extreme care. Changing the args but keeping
  the same id may mean you have different versions of the same node.
  Do this change only if you guarante all nodes in the tree have been
  replaced equally.
  """
  def replace_args(%T{data: %Expr{} = expr} = t, args) do
    %{t | data: %{expr | args: args}}
  end

  @doc """
  Applies the given function to the arguments of the node,
  with the given accumulator as a starting value.

  Warning: be very careful when using this function to traverse the expression
  recursively. If you plan to do so, you should consider also storing the visited
  nodes to avoid multiple traversals.
  """
  def apply_args(expr, acc, fun)

  def apply_args(%T{data: %Expr{op: :token, args: [token]}}, acc, fun) do
    {hooks, acc} =
      Enum.map_reduce(token.hooks, acc, fn %{expr: expr} = token, acc ->
        {expr, acc} = Composite.traverse(expr, acc, fun)
        {%{token | expr: expr}, acc}
      end)

    {[%{token | hooks: hooks}], acc}
  end

  def apply_args(%T{data: %Expr{op: :fun, args: [args, expr, mfa]}}, acc, fun) do
    {args, acc} = Enum.map_reduce(args, acc, &Composite.traverse(&1, &2, fun))
    {expr, acc} = Composite.traverse(expr, acc, fun)
    {[args, expr, mfa], acc}
  end

  def apply_args(%T{data: %Expr{op: :cond, args: [clauses, last]}}, acc, fun) do
    {clauses, acc} =
      Enum.map_reduce(clauses, acc, fn {pred, expr}, acc ->
        {pred, acc} = fun.(pred, acc)
        {expr, acc} = Composite.traverse(expr, acc, fun)
        {{pred, expr}, acc}
      end)

    {last, acc} = Composite.traverse(last, acc, fun)
    {[clauses, last], acc}
  end

  def apply_args(%T{data: %Expr{op: :while, args: args}}, acc, fun) do
    [initial, arg, pred, block] = args
    {initial, acc} = Composite.traverse(initial, acc, fun)
    {arg, acc} = Composite.traverse(arg, acc, fun)
    {pred, acc} = fun.(pred, acc)
    {block, acc} = Composite.traverse(block, acc, fun)
    {[initial, arg, pred, block], acc}
  end

  def apply_args(%T{data: %Expr{op: :concatenate, args: [list | args]}}, acc, fun) do
    {list, acc} = Enum.map_reduce(list, acc, fun)
    {[list | args], acc}
  end

  def apply_args(%T{data: %Expr{op: :slice, args: [tensor, start_indices | args]}}, acc, fun) do
    {tensor, acc} = fun.(tensor, acc)

    {start_indices, acc} =
      Enum.map_reduce(start_indices, acc, fn
        x, acc when is_integer(x) -> {x, acc}
        x, acc -> fun.(x, acc)
      end)

    {[tensor, start_indices | args], acc}
  end

  def apply_args(
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

  def apply_args(%T{data: %Expr{op: :optional, args: [expr, default_impl_expr]}}, acc, fun) do
    {expr, acc} = fun.(expr, acc)
    {default_impl_expr, acc} = fun.(default_impl_expr, acc)
    {[expr, default_impl_expr], acc}
  end

  def apply_args(%T{data: %Expr{args: args}}, acc, fun) do
    Enum.map_reduce(args, acc, fn
      %T{data: %Expr{}} = arg, acc -> fun.(arg, acc)
      arg, acc -> {arg, acc}
    end)
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
    {res, _} = Composite.traverse(expr, %{}, &rewrite_type(&1, &2, fun))
    res
  end

  defp rewrite_type(%T{data: %Expr{id: id, op: op}} = t, cache, fun) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {args, cache} = apply_args(t, cache, &rewrite_type(&1, &2, fun))
        res = rewrite_type(op, args, t, fun)
        {res, Map.put(cache, id, res)}
    end
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
end

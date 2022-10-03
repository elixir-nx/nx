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
  Gets all IDs of all elements in the same scope.

  `while`'s condition and body, `fun`'s body and similar are
  considered different scopes. When it comes to `cond`, an ID will
  only be considered if it is used outside of the `cond` or used
  in several distinct conds. Constants are also ignored, as they
  have global IDs based on the constants themselves.

  An existing maps of `ids` can be given to accumulate on top of it.
  """
  def scope_ids(expr, ids \\ %{}) do
    Composite.reduce(expr, {ids, %{}}, &scope_ids_each(&1, nil, &2)) |> elem(0)
  end

  # Ignore constants
  defp scope_ids_each(%Nx.Tensor{data: %Expr{op: :constant}}, _scope, {ids, cond_ids}) do
    {ids, cond_ids}
  end

  # We are at the root.
  defp scope_ids_each(%Nx.Tensor{data: %Expr{id: id, op: op}} = t, nil, {ids, cond_ids}) do
    case ids do
      %{^id => _} ->
        {ids, cond_ids}

      %{} ->
        scope = if op == :cond, do: id, else: nil
        ids = Map.put(ids, id, op)

        t
        |> apply_args(:scope, {ids, cond_ids}, &{&1, scope_ids_each(&1, scope, &2)})
        |> elem(1)
    end
  end

  # If we are inside a cond, we want to collect all of the IDs inside that
  # cond separately and, in case it is present in more than one direct cond,
  # move it to the parent scope.
  defp scope_ids_each(%Nx.Tensor{data: %Expr{id: id}} = t, scope, {ids, cond_ids}) do
    case cond_ids do
      %{^id => ^scope} ->
        {ids, cond_ids}

      %{^id => _} ->
        scope_ids_each(t, nil, {ids, cond_ids})

      %{} ->
        cond_ids = Map.put(cond_ids, id, scope)

        t
        |> apply_args(:scope, {ids, cond_ids}, &{&1, scope_ids_each(&1, scope, &2)})
        |> elem(1)
    end
  end

  @doc """
  Puts new args in the given tensor expression and gives it a new id.
  """
  def put_args(%T{data: %Expr{} = expr} = t, args) do
    %{t | data: %{expr | id: Expr.id(), args: args}}
  end

  @doc """
  Applies the given function to the arguments of the node,
  with the given accumulator as a starting value.

  By default, `type` is `:all`, which means all arguments
  are traversed. If `type` is `:scope`, only expressions
  that are in the same scope are traversed. Therefore,
  expressions such as `while`'s condition and body, 
  `optional`'s default implementation, functions, and so forth
  are not traversed. Note `cond`s are always traversed because,
  while they introduce a new scope, they can also access its
  parents directly, so you must take `cond`s into account
  accordingly.

  Warning: be very careful when using this function to traverse
  the expression recursively. If you plan to do so, you should
  consider also storing the visited nodes to avoid multiple
  traversals by using `tensor.data.expr.id` as cache key.
  """
  def apply_args(expr, type \\ :all, acc, fun)

  def apply_args(%T{data: %Expr{op: :fun, args: [args, expr, mfa]}}, type, acc, fun) do
    {args, acc} = Enum.map_reduce(args, acc, &Composite.traverse(&1, &2, fun))

    {expr, acc} =
      case type do
        :all -> Composite.traverse(expr, acc, fun)
        :scope -> {expr, acc}
      end

    {[args, expr, mfa], acc}
  end

  def apply_args(%T{data: %Expr{op: :cond, args: [clauses, last]}}, _type, acc, fun) do
    {clauses, acc} =
      Enum.map_reduce(clauses, acc, fn {pred, expr}, acc ->
        {pred, acc} = fun.(pred, acc)
        {expr, acc} = Composite.traverse(expr, acc, fun)
        {{pred, expr}, acc}
      end)

    {last, acc} = Composite.traverse(last, acc, fun)
    {[clauses, last], acc}
  end

  def apply_args(%T{data: %Expr{op: :while, args: args}}, type, acc, fun) do
    [initial, arg, pred, block] = args
    {initial, acc} = Composite.traverse(initial, acc, fun)

    case type do
      :all ->
        {arg, acc} = Composite.traverse(arg, acc, fun)
        {pred, acc} = fun.(pred, acc)
        {block, acc} = Composite.traverse(block, acc, fun)
        {[initial, arg, pred, block], acc}

      :scope ->
        {[initial, arg, pred, block], acc}
    end
  end

  def apply_args(%T{data: %Expr{op: :optional, args: args}}, type, acc, fun) do
    [expr, default_impl_expr] = args
    {expr, acc} = fun.(expr, acc)

    {default_impl_expr, acc} =
      case type do
        :all -> fun.(default_impl_expr, acc)
        :scope -> {[expr, default_impl_expr], acc}
      end

    {[expr, default_impl_expr], acc}
  end

  def apply_args(%T{data: %Expr{op: :token, args: [token]}}, _type, acc, fun) do
    {hooks, acc} =
      Enum.map_reduce(token.hooks, acc, fn %{expr: expr} = token, acc ->
        {expr, acc} = Composite.traverse(expr, acc, fun)
        {%{token | expr: expr}, acc}
      end)

    {[%{token | hooks: hooks}], acc}
  end

  def apply_args(%T{data: %Expr{op: :concatenate, args: [list | args]}}, _type, acc, fun) do
    {list, acc} = Enum.map_reduce(list, acc, fun)
    {[list | args], acc}
  end

  def apply_args(%T{data: %Expr{op: :slice, args: args}}, _type, acc, fun) do
    [tensor, start_indices | args] = args
    {tensor, acc} = fun.(tensor, acc)

    {start_indices, acc} =
      Enum.map_reduce(start_indices, acc, fn
        x, acc when is_integer(x) -> {x, acc}
        x, acc -> fun.(x, acc)
      end)

    {[tensor, start_indices | args], acc}
  end

  def apply_args(%T{data: %Expr{op: :put_slice, args: args}}, _type, acc, fun) do
    [tensor, start_indices, slice] = args
    {tensor, acc} = fun.(tensor, acc)
    {slice, acc} = fun.(slice, acc)

    {start_indices, acc} =
      Enum.map_reduce(start_indices, acc, fn
        x, acc when is_integer(x) -> {x, acc}
        x, acc -> fun.(x, acc)
      end)

    {[tensor, start_indices, slice], acc}
  end

  def apply_args(%T{data: %Expr{args: args}}, _type, acc, fun) do
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

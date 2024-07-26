defmodule Nx.Defn.Tree do
  @moduledoc """
  Helper functions to traverse defn expressions,
  either as single nodes or in-depth.
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

  An existing map of `ids` can be given to accumulate on top of it.
  """
  def scope_ids(expr, ids \\ %{}) do
    Composite.reduce(expr, {ids, %{}}, &scope_ids_each(&1, &2, nil)) |> elem(0)
  end

  # Ignore constants
  defp scope_ids_each(%Nx.Tensor{data: %Expr{op: :constant}}, {ids, cond_ids}, _scope) do
    {ids, cond_ids}
  end

  # We are at the root.
  defp scope_ids_each(%Nx.Tensor{data: %Expr{id: id, op: op} = expr} = t, {ids, cond_ids}, nil) do
    case ids do
      %{^id => _} ->
        {ids, cond_ids}

      %{} when op == :cond ->
        ids = Map.put(ids, id, op)
        acc = {ids, cond_ids}

        # We will treat the predicate as part of the scope to avoid executing it more than once
        [[{pred, body} | clauses], last] = expr.args
        acc = scope_ids_each(pred, acc, nil)
        acc = Composite.reduce(body, acc, &scope_ids_each(&1, &2, id))

        # Now we traverse as in apply_args
        acc =
          Enum.reduce(clauses, acc, fn {pred, body}, acc ->
            acc = scope_ids_each(pred, acc, id)
            Composite.reduce(body, acc, &scope_ids_each(&1, &2, id))
          end)

        Composite.reduce(last, acc, &scope_ids_each(&1, &2, id))

      %{} ->
        ids = Map.put(ids, id, op)
        scope_ids_args(t, {ids, cond_ids}, nil)
    end
  end

  # If we are inside a cond, we want to collect all of the IDs inside that
  # cond separately and, in case it is present in more than one direct cond,
  # move it to the parent scope.
  defp scope_ids_each(%Nx.Tensor{data: %Expr{id: id}} = t, {ids, cond_ids}, scope) do
    case cond_ids do
      %{^id => ^scope} ->
        {ids, cond_ids}

      %{^id => _} ->
        scope_ids_each(t, {ids, cond_ids}, nil)

      %{} ->
        cond_ids = Map.put(cond_ids, id, scope)
        scope_ids_args(t, {ids, cond_ids}, scope)
    end
  end

  defp scope_ids_args(t, acc, scope) do
    t
    |> apply_args(:scope, acc, &{&1, scope_ids_each(&1, &2, scope)})
    |> elem(1)
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
    [call, expr, callback] = args
    {call, acc} = fun.(call, acc)

    {expr, acc} =
      case type do
        :all -> Composite.traverse(expr, acc, fun)
        :scope -> {expr, acc}
      end

    {[call, expr, callback], acc}
  end

  def apply_args(%T{data: %Expr{op: :token, args: [token]}}, _type, acc, fun) do
    {hooks, acc} =
      Enum.map_reduce(token.hooks, acc, fn %{expr: expr} = token, acc ->
        {expr, acc} = Composite.traverse(expr, acc, fun)
        {%{token | expr: expr}, acc}
      end)

    {[%{token | hooks: hooks}], acc}
  end

  def apply_args(%T{data: %Expr{op: op, args: [list | args]}}, _type, acc, fun)
      when op in [:concatenate, :stack] do
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

  def apply_args(%T{data: %Expr{op: :metadata, args: [expr, metadata]}}, _type, acc, fun) do
    {expr, acc} = Composite.traverse(expr, acc, fun)
    {[expr, metadata], acc}
  end

  def apply_args(%T{data: %Expr{args: args}}, _type, acc, fun) do
    Enum.map_reduce(args, acc, fn
      %T{data: %Expr{}} = arg, acc -> fun.(arg, acc)
      arg, acc -> {arg, acc}
    end)
  end
end

defmodule Nx.Defn.Expr do
  @doc """
  The expression used by `Nx.Defn.Compiler`.

  `Nx.Defn.Compiler` changes `Nx` default implementation from
  `Nx.BinaryTensor` to `Nx.Defn.Expr`. It is a struct with the
  following fields:

    * `:id` - a unique identifier
    * `:op` - the operation name
    * `:args` - the operation arguments

  ## Syntax nodes

  Most nodes are created directly via the `Nx` module and
  therefore map directly to `Nx.Tensor` callbacks. However
  the following syntax nodes exist:

    * `parameter(integer)`

    * `tensor(Nx.Tensor.t)`

    * `fun(parameters, t, fun)`

    * `cond(clauses, otherwise)` - note it may return tuples.

    * `elem(tuple, pos, size)` - created automatically from
      `cond`, `fun` and `loop` when they return tuples.
      Note it may return tuples too in case the expressions
      above return nested tuples.

  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  import Nx.Shared

  @enforce_keys [:id, :op, :args, :context]
  @type t :: %Expr{}
  defstruct [:id, :op, :args, :context]

  ## Public API

  @doc """
  Builds an expression from a tensor.

  This implements a superset of `c:Nx.Tensor.tensor/1` as
  it also handles numbers for convenience.
  """
  @impl true
  def tensor(t), do: to_expr(t)

  @doc """
  Creates a parameter based on the given tensor expression.
  """
  def parameter(tensor, pos) when is_integer(pos) and pos >= 0 do
    expr(tensor, tensor.data.context, :parameter, [pos])
  end

  @doc """
  Creates a parameter with the given `context`, `type`, `shape`, and `pos`.
  """
  def parameter(context, type, shape, pos) do
    names = List.duplicate(nil, tuple_size(shape))
    expr(%T{type: type, shape: shape, names: names}, context, :parameter, [pos])
  end

  @doc """
  Creates a function node with the given args and anonuymous function.
  """
  def fun(args, fun) when is_function(fun, length(args)) do
    out = to_expr(apply(fun, args))
    expr(out, out.data.context, :fun, [args, out, fun])
  end

  @doc """
  Creates a `cond` expression.
  """
  def cond(clauses, last) do
    {preds, exprs} = Enum.unzip(clauses)
    {preds, context} = to_exprs(preds)
    [last | exprs] = cond_clauses(last, exprs)
    clauses = Enum.zip(preds, exprs)
    cond_result(last, context, &expr(&1, context, :cond, [clauses, last]))
  end

  defp cond_result(tuple, context, fun) when is_tuple(tuple) do
    size = tuple_size(tuple)
    expr = fun.(%T{shape: {}, names: [], type: {:tuple, size}})

    # TODO: Use Enum.with_index on Elixir v1.12
    tuple
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.map(fn {tensor, i} ->
      fun = &expr(&1, context, :elem, [expr, i, size])
      cond_result(tensor, context, fun)
    end)
    |> List.to_tuple()
  end

  defp cond_result(tensor, _context, fun), do: fun.(tensor)

  defp cond_clauses(last, exprs) when is_tuple(last) do
    size = tuple_size(last)

    for expr <- exprs,
        not is_tuple(expr) or tuple_size(expr) != size,
        do: branch_mismatch!(expr, last)

    # TODO: Use Enum.with_index on Elixir v1.12
    list_of_lists =
      last
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.map(fn {last, index} ->
        exprs = Enum.map(exprs, &elem(&1, index))
        cond_clauses(last, exprs)
      end)

    {last_and_exprs, _} =
      Enum.map_reduce([last | exprs], list_of_lists, fn _, list_of_lists ->
        unzip_cons(list_of_lists, [], [])
      end)

    last_and_exprs
  end

  defp cond_clauses(type = last, exprs) do
    %{shape: shape, names: names} = last = to_expr(last)

    {exprs, {type, shape, names}} =
      Enum.map_reduce(exprs, {type, shape, names}, fn expr, {type, shape, names} ->
        if is_tuple(expr), do: branch_mismatch!(expr, last)
        type = binary_type(type, expr)
        expr = to_expr(expr)
        {shape, names} = Nx.Shape.binary_broadcast(shape, names, expr.shape, expr.names)
        {expr, {type, shape, names}}
      end)

    for expr <- [last | exprs] do
      expr
      |> Nx.as_type(type)
      |> Nx.broadcast(shape, names: names)
    end
  end

  defp unzip_cons([[head | tail] | rest], heads, tails),
    do: unzip_cons(rest, [head | heads], [tail | tails])

  defp unzip_cons([], heads, tails),
    do: {heads |> Enum.reverse() |> List.to_tuple(), Enum.reverse(tails)}

  defp branch_mismatch!(left, right) do
    raise ArgumentError,
          "cond/if expects all branches to return tensors or tuples of the same size, " <>
            "got #{inspect(left)} and #{inspect(right)}"
  end

  ## Traversal helpers

  @doc """
  Helper to traverse the expression arguments of an expression.

  Note function expressions are never traversed, as they shouldn't
  be modified as that would ultimately change the function itself.
  """
  def traverse_args(expr, acc, fun)

  def traverse_args(%T{data: %Expr{op: :fun, args: args}}, acc, _fun) do
    {args, acc}
  end

  def traverse_args(%T{data: %Expr{op: :cond, args: [clauses, last]}}, acc, fun) do
    {clauses, acc} =
      Enum.map_reduce(clauses, acc, fn {condition, expr}, acc ->
        {condition, acc} = fun.(condition, acc)
        {expr, acc} = traverse_tuple_or_expr(expr, acc, fun)
        {{condition, expr}, acc}
      end)

    {last, acc} = traverse_tuple_or_expr(last, acc, fun)
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

  defp traverse_tuple_or_expr(tuple, acc, fun) when is_tuple(tuple) do
    {list, acc} = Enum.map_reduce(Tuple.to_list(tuple), acc, &traverse_tuple_or_expr(&1, &2, fun))
    {List.to_tuple(list), acc}
  end

  defp traverse_tuple_or_expr(expr, acc, fun) do
    fun.(expr, acc)
  end

  ## Nx.Defn callbacks

  @doc false
  def validate_args(args) do
    args
    |> Enum.reduce([], &validate_args/2)
    |> Enum.reverse()
  end

  defp validate_args(%T{} = t, acc),
    do: [t | acc]

  defp validate_args(number, acc) when is_number(number),
    do: [Nx.tensor(number) | acc]

  defp validate_args(tuple, acc) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.reduce(acc, &validate_args/2)

  defp validate_args(other, _acc) do
    raise(
      ArgumentError,
      "arguments to compiled functions must numbers, tensors, or tuples, got: #{inspect(other)}"
    )
  end

  @doc false
  def validate_vars(vars) do
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
  def to_args(args, params) when is_list(args) do
    {args, []} = Enum.map_reduce(args, params, &to_args_each/2)
    args
  end

  defp to_args_each(arg, params) when is_tuple(arg) do
    {list, params} =
      arg
      |> Tuple.to_list()
      |> Enum.map_reduce(params, &to_args_each/2)

    {List.to_tuple(list), params}
  end

  defp to_args_each(_arg, [param | params]) do
    {param, params}
  end

  @doc false
  def to_params(vars),
    do: to_params(vars, 0)

  defp to_params([head | tail], i),
    do: [expr(head, :root, :parameter, [i]) | to_params(tail, i + 1)]

  defp to_params([], _i),
    do: []

  @doc false
  def to_result(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&to_result/1) |> List.to_tuple()

  def to_result(%T{data: %Expr{}} = t),
    do: t

  def to_result(other) do
    raise ArgumentError,
          "defn must return an expression tensor or a tuple, got: #{inspect(other)}"
  end

  ## Nx.Defn AST callbacks

  @doc false
  def cond(file, clauses, last) do
    clauses =
      for {meta, {pred, expr}} <- clauses do
        pred = to_expr(pred)

        if pred.shape != {} do
          raise CompileError,
            line: meta[:line],
            file: file,
            description: "condition must be a scalar tensor, got: #{inspect(pred.shape)}"
        end

        {pred, expr}
      end

    cond(clauses, last)
  end

  ## Nx.Tensor Callbacks

  @behaviour Nx.Tensor

  @impl true
  def from_binary(out, binary) do
    to_expr(Nx.BinaryTensor.from_binary(out, binary))
  end

  @impl true
  def iota(out, axis) do
    expr(out, nil, :iota, [axis])
  end

  @impl true
  def random_uniform(out, min, max) do
    expr(out, nil, :random_uniform, [min, max])
  end

  @impl true
  def random_normal(out, mu, sigma) do
    expr(out, nil, :random_normal, [mu, sigma])
  end

  unary_ops =
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
      [:negate, :sign, :abs, :bitwise_not, :population_count, :count_leading_zeros] ++
      [:floor, :ceil, :round, :as_type]

  for op <- unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor])
    end
  end

  binary_ops =
    [:add, :subtract, :multiply, :divide, :power, :remainder, :arctan2, :max, :min] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :less_equal, :greater_equal] ++
      [:logical_and, :logical_or, :logical_xor] ++
      [:outer]

  for op <- binary_ops do
    @impl true
    def unquote(op)(out, t1, t2) do
      {[t1, t2], context} = to_exprs([t1, t2])
      expr(out, context, unquote(op), [t1, t2])
    end
  end

  aggregate_ops = [:all?, :any?, :argmax, :argmin, :sum]

  for op <- aggregate_ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor, opts])
    end
  end

  @impl true
  def reduce(%{type: type} = out, tensor, acc, opts, fun) do
    args = [parameter(:reduce, type, {}, 0), parameter(:reduce, type, {}, 1)]
    {[tensor, acc], context} = to_exprs([tensor, acc])
    fun = fun(args, fun)

    if fun.shape != {} do
      raise "reduce function must return a scalar tensor, got: #{inspect(fun.shape)}"
    end

    expr(out, context, :reduce, [tensor, acc, opts, fun])
  end

  @impl true
  def reduce_window(
        %{type: type} = out,
        tensor,
        acc,
        window_dims,
        opts,
        fun
      ) do
    args = [parameter(:reduce_window, type, {}, 0), parameter(:reduce_window, type, {}, 1)]
    {[tensor, acc], context} = to_exprs([tensor, acc])
    fun = fun(args, fun)

    if fun.shape != {} do
      raise "reduce_window function must return a scalar tensor, got: #{inspect(fun.shape)}"
    end

    expr(out, context, :reduce_window, [tensor, acc, window_dims, opts, fun])
  end

  @impl true
  def map(%{type: type} = out, tensor, fun) do
    args = [parameter(:map, type, {}, 0)]
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :map, [tensor, fun(args, fun)])
  end

  @impl true
  def reshape(out, tensor, shape) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :reshape, [tensor, shape])
  end

  @impl true
  def squeeze(out, tensor, opts) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :squeeze, [tensor, opts])
  end

  @impl true
  def transpose(out, tensor, opts) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :transpose, [tensor, opts])
  end

  @impl true
  def broadcast(out, tensor, shape, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :broadcast, [tensor, shape, axes])
  end

  @impl true
  def dot(out, t1, a1, t2, a2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    expr(out, context, :dot, [t1, a1, t2, a2])
  end

  @impl true
  def conv(out, inp, kernel, opts) do
    {[inp, kernel], context} = to_exprs([inp, kernel])
    expr(out, context, :conv, [inp, kernel, opts])
  end

  @impl true
  def pad(out, expr, value, config) do
    {[expr, value], context} = to_exprs([expr, value])
    expr(out, context, :pad, [expr, value, config])
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    {[pred, on_true, on_false], context} = to_exprs([pred, on_true, on_false])
    expr(out, context, :select, [pred, on_true, on_false])
  end

  @impl true
  def clip(out, operand, min, max) do
    {[operand, min, max], context} = to_exprs([operand, min, max])
    expr(out, context, :clip, [operand, min, max])
  end

  @impl true
  def slice(out, tensor, start_indices, limit_indices, strides) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :slice, [tensor, start_indices, limit_indices, strides])
  end

  @impl true
  def reverse(out, tensor, opts) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :reverse, [tensor, opts])
  end

  @impl true
  def concatenate(out, tensors, axis) do
    {tensors, context} = to_exprs(tensors)
    expr(out, context, :concatenate, [tensors, axis])
  end

  @impl true
  def cholesky(out, tensor) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :cholesky, [tensor])
  end

  @impl true
  def sort(out, tensor, opts) do
    comparator = opts[:comparator]

    %{type: type} = out
    tensor = to_expr(tensor)

    args = [parameter(:sort, type, {}, 0), parameter(:sort, type, {}, 1)]
    comparator = to_nx_comparator(comparator)
    fun = fun(args, comparator)

    if fun.shape != {} do
      raise "sort comparator must return a scalar tensor, got: #{inspect(fun.shape)}"
    end

    if fun.type != {:u, 8} do
      raise "sort comparator must return a predicate type, got: #{inspect(fun.type)}"
    end

    expr(out, tensor.data.context, :sort, [tensor, opts, fun])
  end

  defp to_nx_comparator(:desc), do: &Nx.less/2
  defp to_nx_comparator(:asc), do: &Nx.greater/2
  defp to_nx_comparator(comp) when is_function(comp, 2), do: comp

  defp to_nx_comparator(_),
    do: "comparator must be either :desc or :asc or a function with arity 2"

  ## Undefined

  ops = [device_deallocate: 1, device_read: 1, device_transfer: 3, to_binary: 1]

  for {op, arity} <- ops do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(op)(unquote_splicing(args)) do
      raise ArgumentError, """
      cannot invoke #{unquote(op)}/#{unquote(arity)} on Nx.Defn.Expr.

      This typically means you are invoking an unsupported Nx function
      by code inside `defn` or JIT/AOT compiled code
      """
    end
  end

  ## Helpers

  defp expr(tensor, context, op, args) do
    %{tensor | data: %Expr{id: System.unique_integer(), op: op, args: args, context: context}}
  end

  defp to_expr(%T{data: %Expr{}} = t), do: t
  defp to_expr(%T{} = t), do: expr(t, nil, :tensor, [t])
  defp to_expr(number) when is_number(number), do: to_expr(Nx.tensor(number))

  defp to_expr(other) do
    raise ArgumentError,
          "unable to convert #{inspect(other)} into a Nx.Defn.Expr, expected a tensor or a number"
  end

  defp to_exprs(list) do
    Enum.map_reduce(list, nil, fn tensor, acc ->
      %{data: %{context: context}} = expr = to_expr(tensor)

      if context != acc and context != nil and acc != nil do
        raise """
        cannot build defn because expressions come from different contexts: \
        #{inspect(context)} and #{inspect(acc)}.

        This typically happens on anonymous functions, which do not behave \
        like closures inside defn. For example, this is not valid:

            defn example(t, amplifier) do
              Nx.reduce(t, 0, fn val, acc ->
                val * amplifier + acc
              end)
            end

        In the example above, "amplifier" is a variable defined outside of \
        the anonymous function, which is not allowed in defn.
        """
      end

      {expr, context || acc}
    end)
  end

  ## Inspect

  import Inspect.Algebra

  @impl true
  def inspect(tensor, opts) do
    {_, acc} = inspect_expr(tensor, {[], [], %{}})
    {_, {exprs, params, _var_map}} = traverse_args(tensor, acc, &inspect_expr/2)

    all =
      params
      |> Enum.reverse()
      |> Kernel.++(Enum.reverse(exprs))

    length = Enum.reduce(all, 0, fn {str, _tensor}, acc -> max(byte_size(str), acc) end)

    all
    |> Enum.map(fn {str, tensor} ->
      String.pad_trailing(str, length, " ") <> "  " <> to_type_shape(tensor)
    end)
    |> Enum.uniq()
    |> Enum.reduce(color("Nx.Defn.Expr", :map, opts), &concat(&2, concat(line(), &1)))
  end

  # Scalars and funs are shown as is
  defp inspect_expr(%T{data: %Expr{op: :tensor}, shape: {}} = t, acc), do: {t, acc}
  defp inspect_expr(%T{data: %Expr{op: :fun}} = t, acc), do: {t, acc}

  defp inspect_expr(%T{data: %Expr{op: op, id: id}} = t, {exprs, params, var_map})
       when op in [:tensor, :parameter] do
    {var, var_map} = var_for_id(var_map, id)
    param = Atom.to_string(op) <> " " <> var
    {t, {exprs, [{param, t} | params], var_map}}
  end

  defp inspect_expr(%T{} = t, acc) do
    %{data: %Expr{id: id, op: op, args: args}} = t
    {_, {exprs, params, var_map}} = traverse_args(t, acc, &inspect_expr/2)
    {var, var_map} = var_for_id(var_map, id)
    args_str = inspect_args(op, args, var_map)
    expr_str = var <> " = " <> Atom.to_string(op) <> " [ " <> args_str <> " ]"
    {t, {[{expr_str, t} | exprs], params, var_map}}
  end

  defp inspect_args(:cond, [clauses, last], var_map) do
    clauses =
      Enum.map(clauses, fn {pred, expr} ->
        [inspect_arg(pred, var_map), " -> ", inspect_arg(expr, var_map), ", "]
      end)

    IO.iodata_to_binary([clauses, ":otherwise -> ", inspect_arg(last, var_map)])
  end

  defp inspect_args(_op, args, var_map), do: inspect_args(args, var_map)

  defp inspect_args(args, var_map) do
    Enum.map_join(args, ", ", &inspect_arg(&1, var_map))
  end

  defp inspect_arg(arg, var_map) do
    case arg do
      %T{data: %Expr{op: :fun, args: [_, _, fun]}} ->
        inspect(fun)

      %T{data: %Expr{op: :tensor, args: [t]}, shape: {}} ->
        t |> Nx.to_scalar() |> to_string()

      %T{data: %Expr{id: id}} ->
        Map.fetch!(var_map, id)

      _ ->
        cond do
          Keyword.keyword?(arg) and arg != [] ->
            Enum.map_join(arg, ", ", fn {k, v} -> "#{k}: #{inspect(v)}" end)

          is_list(arg) ->
            [?[, inspect_args(arg, var_map), ?]]

          is_tuple(arg) ->
            [?{, inspect_args(Tuple.to_list(arg), var_map), ?}]

          true ->
            inspect(arg)
        end
    end
  end

  defp var_for_id(var_map, id) do
    case var_map do
      %{^id => var} ->
        {var, var_map}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, Map.put(var_map, id, var)}
    end
  end

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [Enum.at(?a..?z, counter)]

  defp to_type_shape(%{type: type, shape: shape}) do
    brackets =
      shape
      |> Tuple.to_list()
      |> Enum.map(&[?[, Integer.to_string(&1), ?]])

    IO.iodata_to_binary([Nx.Type.to_string(type) | brackets])
  end
end

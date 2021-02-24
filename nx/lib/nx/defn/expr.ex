defmodule Nx.Defn.Expr do
  @doc """
  The expression used by `Nx.Defn.Compiler`.

  `Nx.Defn.Compiler` changes `Nx` default implementation from
  `Nx.BinaryBackend` to `Nx.Defn.Expr`. It is a struct with the
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

    * `cond(clauses, otherwise)`

    * `metadata(expr, metadata)`

    * `elem(tuple, pos, size)` - created automatically from
      expression that return tuples. Note it may return tuples
      too, which means we have nested tuples

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
  """
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
  Creates a metadata node that around the given expression.
  """
  def metadata(expr, metadata) when is_map(metadata) do
    expr = to_expr(expr)
    expr(expr, expr.data.context, :metadata, [expr, metadata])
  end

  @doc """
  Creates a function node with the given args and anonymous function.
  """
  def fun(args, fun) when is_function(fun, length(args)) do
    out = to_expr(apply(fun, args))
    expr(out, out.data.context, :fun, [args, out, fun])
  end

  @doc """
  Creates a tuple, possibly recursively, by executing the
  given function for each element.
  """
  def tuple(tuple, context, fun) when is_tuple(tuple) do
    recur_tuple(tuple, context, fun)
  end

  defp recur_tuple(tuple, context, fun) when is_tuple(tuple) do
    size = tuple_size(tuple)
    expr = fun.(%T{shape: {}, names: [], type: {:tuple, size}})

    # TODO: Use Enum.with_index on Elixir v1.12
    tuple
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.map(fn {tensor, i} ->
      fun = &expr(&1, context, :elem, [expr, i, size])
      recur_tuple(tensor, context, fun)
    end)
    |> List.to_tuple()
  end

  defp recur_tuple(tensor, _context, fun), do: fun.(tensor)

  @doc """
  Creates a `cond` expression.
  """
  def cond(clauses, last) do
    {preds, exprs} = Enum.unzip(clauses)
    {preds, context} = to_exprs(preds)
    [last | exprs] = cond_clauses(last, exprs)
    clauses = Enum.zip(preds, exprs)
    fun = &expr(&1, context, :cond, [clauses, last])

    if is_tuple(last) do
      tuple(last, context, fun)
    else
      fun.(last)
    end
  end

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
  Helper to traverse the arguments of an expression.

  Note function expressions are never traversed, as they generally
  shouldn't be implicitly modified as that would ultimately change
  the function itself.
  """
  def traverse_args(expr, acc, fun)

  def traverse_args(%T{data: %Expr{op: :fun, args: args}}, acc, _fun) do
    {args, acc}
  end

  def traverse_args(%T{data: %Expr{op: :cond, args: [clauses, last]}}, acc, fun) do
    {clauses, acc} =
      Enum.map_reduce(clauses, acc, fn {condition, expr}, acc ->
        {condition, acc} = fun.(condition, acc)
        {expr, acc} = traverse_exprs(expr, acc, fun)
        {{condition, expr}, acc}
      end)

    {last, acc} = traverse_exprs(last, acc, fun)
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
  Traverses the given expressions.

  This function exists to handle composite types that may
  have multiple expressions.

  If an expression is given, the function is invoked for it
  but not for its arguments (see `traverse_args/3` for that).

  If a tuple of expressions are given, the tuple is recursively
  traversed for each expression and returned.
  """
  def traverse_exprs(expr, fun) when is_function(fun, 1) do
    {result, []} = traverse_exprs(expr, [], fn expr, [] -> {fun.(expr), []} end)
    result
  end

  @doc """
  Traverses the given expressions with the accumulator

  If an expression is given, the function is invoked for it
  but not for its arguments (see `traverse_args/3` for that).

  If a tuple of expressions are given, the tuple is recursively
  traversed for each expression and returned.
  """
  def traverse_exprs(tuple, acc, fun) when is_tuple(tuple) and is_function(fun, 2) do
    {list, acc} = Enum.map_reduce(Tuple.to_list(tuple), acc, &traverse_exprs(&1, &2, fun))
    {List.to_tuple(list), acc}
  end

  def traverse_exprs(%T{} = expr, acc, fun) when is_function(fun, 2) do
    fun.(expr, acc)
  end

  def traverse_exprs(other, _acc, _fun) do
    raise ArgumentError, "expected a tensor expression, got: #{inspect(other)}"
  end

  ## Type helpers

  @doc """
  Rewrites the types of the given expression according to the given options.

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
    {res, _} = cached_rewrite_type(expr, %{}, fun)
    res
  end

  defp cached_rewrite_type(tuple, cache, fun) when is_tuple(tuple) do
    {list, cache} =
      tuple |> Tuple.to_list() |> Enum.map_reduce(cache, &cached_rewrite_type(&1, &2, fun))

    {List.to_tuple(list), cache}
  end

  defp cached_rewrite_type(%T{data: %Expr{id: id, op: op}} = t, cache, fun) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {args, cache} = traverse_args(t, cache, &cached_rewrite_type(&1, &2, fun))
        res = rewrite_type(op, args, t, fun)
        {res, Map.put(cache, id, res)}
    end
  end

  defp rewrite_type(:parameter, _args, t, type_fun) do
    Nx.as_type(t, type_fun.(t.type))
  end

  defp rewrite_type(:fun, [params, _expr, fun], _t, type_fun) do
    {:arity, arity} = Function.info(fun, :arity)
    params = Enum.map(params, &%{&1 | type: type_fun.(&1.type)})
    fun(params, rewrite_type_fun(arity, fun, type_fun))
  end

  defp rewrite_type(:tensor, [arg], t, type_fun) do
    type = type_fun.(t.type)
    rewrite_type_args(t, type, [Nx.as_type(arg, type)])
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
    %{t | data: %{data | id: id(), args: args}, type: type}
  end

  ## Nx.Defn callbacks

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
  def to_vars(args) do
    args
    |> Enum.reduce([], &to_vars/2)
    |> Enum.reverse()
  end

  defp to_vars(%T{} = t, acc),
    do: [t | acc]

  defp to_vars(number, acc) when is_number(number),
    do: [Nx.tensor(number) | acc]

  defp to_vars(tuple, acc) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.reduce(acc, &to_vars/2)

  defp to_vars(other, _acc) do
    raise(
      ArgumentError,
      "arguments to compiled functions must numbers, tensors, or tuples, got: #{inspect(other)}"
    )
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

  ## Nx.Backend Callbacks

  @behaviour Nx.Backend

  @impl true
  def from_binary(out, binary, opts) do
    to_expr(Nx.BinaryBackend.from_binary(out, binary, opts))
  end

  @impl true
  def eye(out) do
    expr(out, nil, :eye, [])
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
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
      [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :cbrt, :negate, :sign, :abs, :bitwise_not] ++
      [:population_count, :count_leading_zeros, :floor, :ceil, :round, :as_type] ++
      [:erf, :erfc, :erf_inv, :acos, :asin, :atan]

  for op <- unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor])
    end
  end

  binary_ops =
    [:add, :subtract, :multiply, :divide, :power, :remainder, :atan2, :max, :min, :quotient] ++
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

  aggregate_ops = [:all?, :any?, :argmax, :argmin, :sum, :product, :reduce_min, :reduce_max]

  for op <- aggregate_ops do
    @impl true
    def unquote(op)(out, tensor, opts) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor, opts])
    end
  end

  window_aggregate_ops = [:window_sum, :window_product, :window_max, :window_min]

  for op <- window_aggregate_ops do
    @impl true
    def unquote(op)(out, tensor, window_dimensions, opts) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor, window_dimensions, opts])
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
  def squeeze(out, tensor, axes) do
    tensor = to_expr(tensor)

    # If we are in a sequence of squeezes, we collapse them.
    # This helps us fuse the access syntax.
    with %T{data: %Expr{op: :squeeze, args: [tensor, inner_axes]}} <- tensor do
      axes = merge_squeeze(Enum.sort(inner_axes), Enum.sort(axes), 0)
      expr(out, tensor.data.context, :squeeze, [tensor, axes])
    else
      _ -> expr(out, tensor.data.context, :squeeze, [tensor, axes])
    end
  end

  defp merge_squeeze([inner_axis | inner_axes], [axis | axes], extra)
       when inner_axis <= axis + extra,
       do: [inner_axis | merge_squeeze(inner_axes, [axis | axes], extra + 1)]

  defp merge_squeeze(inner_axes, [axis | axes], extra),
    do: [axis + extra | merge_squeeze(inner_axes, axes, extra)]

  defp merge_squeeze([], [], _extra),
    do: []

  @impl true
  def transpose(out, tensor, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :transpose, [tensor, axes])
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
  def slice(out, tensor, start, lengths, strides) do
    tensor = to_expr(tensor)

    # If we are in a sequence of slices, it is the access syntax,
    # so we compact them into a single slice.
    with true <- ones_stride?(strides),
         {slice, axes} <- maybe_squeeze(tensor),
         %T{data: %Expr{op: :slice, args: [tensor, inner_start, inner_lengths, strides]}} <-
           slice,
         true <- ones_stride?(strides) do
      {start, lengths} =
        0
        |> merge_slice(axes, inner_start, start, inner_lengths, lengths)
        |> Enum.unzip()

      tensor
      |> Nx.slice(start, lengths)
      |> Nx.squeeze(axes: axes)
    else
      _ -> expr(out, tensor.data.context, :slice, [tensor, start, lengths, strides])
    end
  end

  defp ones_stride?(strides), do: Enum.all?(strides, &(&1 == 1))

  defp maybe_squeeze(%T{data: %Expr{op: :squeeze, args: [slice, axes]}}), do: {slice, axes}
  defp maybe_squeeze(slice), do: {slice, []}

  defp merge_slice(_axis, _axes, [], [], [], []), do: []

  defp merge_slice(axis, axes, [is | inner_start], start, [il | inner_lengths], lengths) do
    # This is one of the erased axes, so we need to get coordinates from inner
    if axis in axes do
      [{is, il} | merge_slice(axis + 1, axes, inner_start, start, inner_lengths, lengths)]
    else
      [s | start] = start
      [l | lengths] = lengths
      [{is + s, l} | merge_slice(axis + 1, axes, inner_start, start, inner_lengths, lengths)]
    end
  end

  @impl true
  def reverse(out, tensor, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :reverse, [tensor, axes])
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
  def qr({q, r}, tensor, opts) do
    tensor = to_expr(tensor)
    context = tensor.data.context
    tuple({q, r}, context, &expr(&1, context, :qr, [{q, r}, tensor, opts]))
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

  ops = [backend_deallocate: 1, backend_transfer: 3, to_binary: 2, to_batched_list: 2]

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

  defp id(), do: System.unique_integer()

  defp expr(tensor, context, op, args) do
    %{tensor | data: %Expr{id: id(), op: op, args: args, context: context}}
  end

  defp to_expr(%T{data: %Expr{}} = t), do: t
  defp to_expr(%T{} = t), do: expr(t, nil, :tensor, [t])
  defp to_expr(number) when is_number(number), do: to_expr(Nx.tensor(number))

  defp to_expr(other) do
    raise ArgumentError,
          "unable to build tensor expression, expected a tensor or a number, got: #{inspect(other)}"
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

    header = concat(line(), color("Nx.Defn.Expr", :map, opts))
    length = Enum.reduce(all, 0, fn {str, _tensor}, acc -> max(byte_size(str), acc) end)

    all
    |> Enum.map(fn {str, tensor} ->
      String.pad_trailing(str, length, " ") <> "  " <> to_type_shape(tensor)
    end)
    |> Enum.uniq()
    |> Enum.reduce(header, &concat(&2, concat(line(), &1)))
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

  defp inspect_args(:metadata, [expr, metadata], var_map) do
    IO.iodata_to_binary([inspect_arg(expr, var_map), ", ", inspect(Map.keys(metadata))])
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

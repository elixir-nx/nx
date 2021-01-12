defmodule Nx.Defn.Expr do
  @doc """
  The expression used by `Nx.Defn.Compiler`.

  `Nx.Defn.Compiler` changes `Nx` default implementation from
  `Nx.BinaryTensor` to `Nx.Defn.Expr`. It is a struct with the
  following fields:

    * `:id` - a unique identifier
    * `:op` - the operation name
    * `:args` - the operation arguments

  All `:op` nodes translate to `Nx.Tensor` operations, except for:

    * `:parameter` - holds a parameter.
      `:args` is a one element index to the actual parameter.

    * `:tensor` - holds a tensor.
      `:args` is a one element list with the tensor.

    * `:fun` - holds a function.
      `:args` is a three element list with the parameters of
      the function, the function expression, and the anonymous
      function used to build the expression. The shape of the
      tensor holding the function is the shape of the output.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  import Nx.Shared

  @enforce_keys [:id, :op, :args, :context]
  @type t :: %Expr{}
  defstruct [:id, :op, :args, :context]

  @doc """
  Converts the given `arg` into an expression tensor.
  """
  def to_expr(%T{data: %Expr{}} = t), do: t
  def to_expr(%T{} = t), do: expr(t, nil, :tensor, [t])
  def to_expr(number) when is_number(number), do: to_expr(Nx.tensor(number))

  def to_expr(other) do
    raise ArgumentError,
          "unable to convert #{inspect(other)} into a Nx.Defn.Expr, expected a tensor or a number"
  end

  @doc """
  Creates parameters for defn anonymous functions.
  """
  def parameter(context, type, shape, pos) when is_integer(pos) and pos >= 0 do
    names = List.duplicate(nil, tuple_size(shape))
    expr(%T{type: type, shape: shape, names: names}, context, :parameter, [pos])
  end

  @doc """
  Creates a function expression.

  The `args` are used to precompute the expression of the fun.
  A handler of funs can choose to either work with the expressions
  directly or by invoking the underlying fun.
  """
  def fun(args, fun) when is_function(fun, length(args)) do
    out = to_expr(apply(fun, args))
    expr(out, out.data.context, :fun, [args, out, fun])
  end

  ## Nx.Defn callbacks

  @doc false
  def to_vars(vars) do
    for var <- vars do
      case var do
        %T{} = head ->
          head

        number when is_number(number) ->
          Nx.tensor(number)

        tuple when is_tuple(tuple) ->
          raise ArgumentError,
                "defn functions expects either numbers or %Nx.Tensor{} as arguments. " <>
                  "If you want to pass a tuple, you must explicitly pattern match on the tuple in the signature" <>
                  "Got: #{inspect(tuple)}"

        other ->
          raise ArgumentError,
                "defn functions expects either numbers or %Nx.Tensor{} as arguments. " <>
                  "Got: #{inspect(other)}"
      end
    end
  end

  @doc false
  def to_params(vars), do: to_params(vars, 0)

  defp to_params([head | tail], i), do: [expr(head, :root, :parameter, [i]) | to_params(tail, i + 1)]
  defp to_params([], _i), do: []

  @doc false
  def to_result(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&to_result/1) |> List.to_tuple()

  def to_result(%T{} = t),
    do: to_expr(t)

  def to_result(number) when is_number(number),
    do: to_expr(number)

  def to_result(other) do
    raise ArgumentError, "defn must return a tensor, a number or a tuple, got: #{inspect(other)}"
  end

  ## Nx.Tensor Callbacks

  unary_ops =
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
      [:negate, :sign, :abs, :bitwise_not, :population_count, :count_leading_zeros] ++
      [:floor, :ceil, :round]

  for op <- unary_ops do
    @doc false
    def unquote(op)(out, tensor) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor])
    end
  end

  binary_ops =
    [:add, :subtract, :multiply, :divide, :power, :remainder, :arctan2, :max, :min] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :less_equal, :greater_equal] ++
      [:outer]

  for op <- binary_ops do
    @doc false
    def unquote(op)(out, t1, t2) do
      {[t1, t2], context} = to_exprs([t1, t2])
      expr(out, context, unquote(op), [t1, t2])
    end
  end

  aggregate_ops = [:sum, :argmax, :argmin]

  for op <- aggregate_ops do
    @doc false
    def unquote(op)(out, tensor, opts) do
      tensor = to_expr(tensor)
      expr(out, tensor.data.context, unquote(op), [tensor, opts])
    end
  end

  @doc false
  def reduce(%{type: type} = out, tensor, acc, opts, fun) do
    args = [parameter(:reduce, type, {}, 0), parameter(:reduce, type, {}, 1)]
    {[tensor, acc], context} = to_exprs([tensor, acc])
    expr(out, context, :reduce, [tensor, acc, opts, fun(args, fun)])
  end

  ## Creation ops

  @doc false
  def iota(shape, opts \\ []) do
    {out, axis} = iota_out(shape, opts)
    expr(out, nil, :iota, [axis])
  end

  @doc false
  def random_uniform(shape, opts \\ []) do
    random_uniform(shape, 0.0, 1.0, opts)
  end

  @doc false
  def random_uniform(tensor_or_shape, min, max, opts \\ [])
      when is_number(min) and is_number(max) do
    out = random_uniform_out(tensor_or_shape, min, max, opts)
    expr(out, nil, :random_uniform, [min, max])
  end

  @doc false
  def random_normal(shape, opts \\ []) do
    random_normal(shape, 0.0, 1.0, opts)
  end

  @doc false
  def random_normal(tensor_or_shape, mu, sigma, opts \\ [])
      when is_float(mu) and is_float(sigma) do
    out = random_normal_out(tensor_or_shape, opts)
    expr(out, nil, :random_normal, [mu, sigma])
  end

  @doc false
  def reshape(out, tensor, shape) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :reshape, [tensor, shape])
  end

  @doc false
  def squeeze(out, tensor, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :squeeze, [tensor, axes])
  end

  @doc false
  def transpose(out, tensor, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :transpose, [tensor, axes])
  end

  @doc false
  def broadcast(out, tensor, shape, axes) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :broadcast, [tensor, shape, axes])
  end

  @doc false
  def dot(out, t1, a1, t2, a2) do
    {[t1, t2], context} = to_exprs([t1, t2])
    expr(out, context, :dot, [t1, a1, t2, a2])
  end

  @doc false
  def conv(out, inp, kernel, stride, padding, input_dilation, kernel_dilation) do
    {[inp, kernel], context} = to_exprs([inp, kernel])
    expr(out, context, :conv, [inp, kernel, stride, padding, input_dilation, kernel_dilation])
  end

  @doc false
  def pad(out, expr, value, config) do
    {[expr, value], context} = to_exprs([expr, value])
    expr(out, context, :pad, [expr, value, config])
  end

  @doc false
  def select(out, pred, on_true, on_false) do
    {[pred, on_true, on_false], context} = to_exprs([pred, on_true, on_false])
    expr(out, context, :select, [pred, on_true, on_false])
  end

  @doc false
  def clip(out, operand, min, max) do
    {[operand, min, max], context} = to_exprs([operand, min, max])
    expr(out, context, :clip, [operand, min, max])
  end

  def slice(out, tensor, start_indices, limit_indices, strides) do
    tensor = to_expr(tensor)
    expr(out, tensor.data.context, :slice, [tensor, start_indices, limit_indices, strides])
  end

  ## Helpers

  defp expr(tensor, context, op, args) do
    %{tensor | data: %Expr{id: System.unique_integer(), op: op, args: args, context: context}}
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

  @doc false
  def inspect(tensor, opts) do
    {exprs, params, _var_map} = inspect_expr_args([tensor], [], [], %{})

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

  # Post-order traversal of the Expr AST, but we pull all parameters to the front
  defp inspect_expr_args([], exprs, params, var_map), do: {exprs, params, var_map}

  defp inspect_expr_args([%T{data: %Expr{op: :tensor}, shape: {}} | tail], exprs, params, var_map) do
    inspect_expr_args(tail, exprs, params, var_map)
  end

  defp inspect_expr_args([%T{data: %Expr{op: :fun}} | tail], exprs, params, var_map) do
    inspect_expr_args(tail, exprs, params, var_map)
  end

  defp inspect_expr_args(
         [%T{data: %Expr{op: op, id: id}} = tensor | tail],
         exprs,
         params,
         var_map
       )
       when op in [:tensor, :parameter] do
    {var, var_map} = var_for_id(var_map, id)
    param = Atom.to_string(op) <> " " <> var
    inspect_expr_args(tail, exprs, [{param, tensor} | params], var_map)
  end

  defp inspect_expr_args([%T{} = tensor | tail], exprs, params, var_map) do
    %{data: %Expr{id: id, op: op, args: expr_args}} = tensor
    {exprs, params, var_map} = inspect_expr_args(expr_args, exprs, params, var_map)

    expr_args_strs = inspect_args(expr_args, var_map)
    {var, var_map} = var_for_id(var_map, id)

    expr_str =
      var <>
        " = " <>
        Atom.to_string(op) <>
        " [ " <> Enum.join(expr_args_strs, ", ") <> " ]"

    inspect_expr_args(tail, [{expr_str, tensor} | exprs], params, var_map)
  end

  defp inspect_expr_args([_ | tail], exprs, params, var_map) do
    inspect_expr_args(tail, exprs, params, var_map)
  end

  defp inspect_args([], _var_map), do: []

  defp inspect_args([arg | args], var_map) do
    case arg do
      %T{data: %Expr{op: :fun, args: [_, _, fun]}} ->
        [inspect(fun) | inspect_args(args, var_map)]

      %T{data: %Expr{op: :tensor, args: [t]}, shape: {}} ->
        [t |> Nx.Util.to_scalar() |> to_string() | inspect_args(args, var_map)]

      %T{data: %Expr{id: id}} ->
        [Map.fetch!(var_map, id) | inspect_args(args, var_map)]

      value ->
        if Keyword.keyword?(value) and value != [] do
          [
            Enum.map_join(value, ", ", fn {k, v} -> "#{k}: #{inspect(v)}" end)
            | inspect_args(args, var_map)
          ]
        else
          [inspect(value) | inspect_args(args, var_map)]
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

  defp to_type_shape(%{type: {kind, size}, shape: shape}) do
    brackets =
      shape
      |> Tuple.to_list()
      |> Enum.map(&[?[, Integer.to_string(&1), ?]])

    IO.iodata_to_binary([Atom.to_string(kind), Integer.to_string(size) | brackets])
  end
end

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
      `:args` is a four element list with the function name,
      the parameters of the function, the function output,
      and the anonymous function itself. The shape of the
      tensor holding the function is the shape of the output.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  import Nx.Shared

  @enforce_keys [:id, :op, :args]
  @type t :: %Expr{}
  defstruct [:id, :op, :args]

  @doc """
  Converts the given `arg` into an expression tensor.
  """
  def to_expr(%T{data: %Expr{}} = t), do: t
  def to_expr(%T{} = t), do: expr(t, :tensor, [t])
  def to_expr(number) when is_number(number), do: to_expr(Nx.tensor(number))

  def to_expr(other) do
    raise ArgumentError,
          "unable to convert #{inspect(other)} into a Nx.Defn.Expr, expected a tensor or a number"
  end

  @doc """
  Creates parameters for defn anonymous functions.
  """
  def parameter(type, shape, pos) when is_integer(pos) and pos >= 0 do
    names = List.duplicate(nil, tuple_size(shape))
    expr(%T{type: type, shape: shape, names: names}, :parameter, [pos])
  end

  @doc """
  Creates a function expression.

  The `args` are used to precompute the expression of the fun.
  A handler of funs can choose to either work with the expressions
  directly or by invoking the underlying fun.
  """
  def fun(name, args, fun) when is_atom(name) and is_function(fun, length(args)) do
    out = apply(fun, args)
    expr(out, :fun, [name, args, out, fun])
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

  defp to_params([head | tail], i), do: [expr(head, :parameter, [i]) | to_params(tail, i + 1)]
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
      expr(out, unquote(op), [to_expr(tensor)])
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
      expr(out, unquote(op), [to_expr(t1), to_expr(t2)])
    end
  end

  aggregate_ops = [:sum, :argmax, :argmin]

  for op <- aggregate_ops do
    @doc false
    def unquote(op)(out, tensor, opts) do
      expr(out, unquote(op), [to_expr(tensor), opts])
    end
  end

  @doc false
  def reduce(%{type: type} = out, tensor, acc, opts, fun) do
    args = [parameter(type, {}, 0), parameter(type, {}, 1)]
    expr(out, :reduce, [to_expr(tensor), to_expr(acc), opts, fun(:reduce, args, fun)])
  end

  ## Creation ops

  @doc false
  def iota(shape, opts \\ []) do
    {out, axis} = iota_out(shape, opts)
    expr(out, :iota, [axis])
  end

  @doc false
  def random_uniform(shape, opts \\ []) do
    random_uniform(shape, 0.0, 1.0, opts)
  end

  @doc false
  def random_uniform(tensor_or_shape, min, max, opts \\ [])
      when is_number(min) and is_number(max) do
    out = random_uniform_out(tensor_or_shape, min, max, opts)
    expr(out, :random_uniform, [min, max])
  end

  @doc false
  def random_normal(shape, opts \\ []) do
    random_normal(shape, 0.0, 1.0, opts)
  end

  @doc false
  def random_normal(tensor_or_shape, mu, sigma, opts \\ [])
      when is_float(mu) and is_float(sigma) do
    out = random_normal_out(tensor_or_shape, opts)
    expr(out, :random_normal, [mu, sigma])
  end

  @doc false
  def reshape(out, tensor, shape), do: expr(out, :reshape, [to_expr(tensor), shape])

  @doc false
  def squeeze(out, tensor, axes), do: expr(out, :squeeze, [to_expr(tensor), axes])

  @doc false
  def transpose(out, tensor, axes), do: expr(out, :transpose, [to_expr(tensor), axes])

  @doc false
  def dot(out, t1, a1, t2, a2), do: expr(out, :dot, [to_expr(t1), a1, to_expr(t2), a2])

  @doc false
  def conv(out, inp, kernel, stride, padding, input_dilation, kernel_dilation) do
    expr(out, :conv, [
      to_expr(inp),
      to_expr(kernel),
      stride,
      padding,
      input_dilation,
      kernel_dilation
    ])
  end

  @doc false
  def pad(out, expr, value, config), do: expr(out, :pad, [to_expr(expr), to_expr(value), config])

  @doc false
  def broadcast(out, expr, shape, axes), do: expr(out, :broadcast, [to_expr(expr), shape, axes])

  @doc false
  def select(out, pred, on_true, on_false) do
    expr(out, :select, [to_expr(pred), to_expr(on_true), to_expr(on_false)])
  end

  @doc false
  def clip(out, operand, min, max) do
    expr(out, :clip, [to_expr(operand), to_expr(min), to_expr(max)])
  end

  def slice(out, tensor, start_indices, limit_indices, strides) do
    expr(out, :slice, [to_expr(tensor), start_indices, limit_indices, strides])
  end

  ## Helpers

  defp expr(tensor, op, args) do
    %{tensor | data: %Expr{id: System.unique_integer(), op: op, args: args}}
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
      %T{data: %Expr{op: :fun, args: [name, fun_args, _, _]}} ->
        ["&#{name}/#{length(fun_args)}" | inspect_args(args, var_map)]

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

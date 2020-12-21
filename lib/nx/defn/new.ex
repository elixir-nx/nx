defmodule Nx.Defn.Expr do
  @moduledoc false
  # Represents a value at a point of execution in the Nx.Defn AST
  # The value of the Expr is equivalent to the main unit
  # of computation in whatever defn backend you are using
  @enforce_keys [:shape, :op, :args]
  defstruct [:shape, :op, :args]
end

defmodule Nx.Defn.Translation do
  @moduledoc false
  # The purpose of this module is to describe each Nx operation
  # in terms of how it translates the given tensor in space.
  #
  # A translation is simply just a change in shape. As an example,
  # consider calling `Nx.sum/2` on a 3x2x3 tensor along axis 1, the
  # "translation" condenses the shape of the output tensor to a
  # 3x3 tensor.
  #
  # These translation rules remain the same regardless of the defn
  # compiler used, therefore any defn backend needs to ensure their
  # operation semantics enforce these translations.

  ## Element-wise unary ops

  # The shapes of element-wise unary ops remain
  # unchanged regardless of the input.
  def unary_op_rule(shape), do: shape

  ## Element-wise binary ops

  # The shapes of element-wise binary ops are
  # broadcasted if broadcasting is compatible.
  def binary_op_rule(s1, s2), do: broadcast(s1, s2)

  ## Aggregate ops

  # The shapes of aggregate ops are contracted along
  # the specified dimension(s).
  def aggregate_op_rule(shape, axes), do: contract(shape, axes)

  ## Translation primitives

  def broadcast(shape, shape), do: shape

  def broadcast(s1, s2) when is_tuple(s1) and is_tuple(s2),
    do: List.to_tuple(do_broadcast(Tuple.to_list(s1), Tuple.to_list(s2)))

  defp do_broadcast(s1, s2) when length(s1) > length(s2) do
    [dim | s1] = s1
    [dim | do_broadcast(s1, s2)]
  end

  defp do_broadcast(s1, s2) when length(s2) > length(s1) do
    [dim | s2] = s2
    [dim | do_broadcast(s1, s2)]
  end

  defp do_broadcast([], s2), do: s2
  defp do_broadcast(s1, []), do: s1
  defp do_broadcast([1 | s1], [dim2 | s2]) do
    [dim2 | do_broadcast(s1, s2)]
  end
  defp do_broadcast([dim1 | s1], [1 | s2]) do
    [dim1 | do_broadcast(s1, s2)]
  end
  defp do_broadcast([dim | s1], [dim | s2]) do
    [dim | do_broadcast(s1, s2)]
  end
  defp do_broadcast([dim1 | _s1], [dim2 | _s2]) do
    raise ArgumentError, "could not broadcast shapes because dimensions are" <>
                         " incompatible, expected dimensions to be equal or" <>
                         " either dimension to be 1, got: #{dim1} and #{dim2}"
  end

  def contract(shape, []), do: shape

  def contract(shape, [axis | []]), do: Tuple.delete_at(shape, axis)

  def contract(shape, axes) when is_list(axes) do
    shape
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.filter(fn {_, i} -> i not in axes end)
    |> List.to_tuple()
  end

  def contract(shape, axis) when is_integer(axis), do: Tuple.delete_at(shape, axis)

end

defmodule Nx.Defn.New do
  @moduledoc false
  # Refactor of the original defn compiler
  alias Nx.Defn.Expr
  alias Nx.Defn.Translation
  alias Nx.Tensor, as: T

  def __compile__(_env, _kind, _meta, _vars, ast, _options) do
    {ast, _state} = traverse(ast, %{})

    quote do
      unquote(ast)
    end
  end

  ## Expr creation

  defp to_expr(%Expr{} = expr), do: expr

  defp to_expr(integer) when is_integer(integer),
    do: %Expr{args: integer, op: :tensor, shape: {}}

  defp to_expr(number) when is_number(number),
    do: %Expr{args: number, op: :tensor, shape: {}}

  defp to_expr(%T{shape: shape} = t), do: %Expr{args: t, op: :tensor, shape: shape}

  defp make_expr(shape, op, args) do
    %Expr{shape: shape, op: op, args: args}
  end

  ## Operations

  def make_unary_op_expr(op, expr) do
    %Expr{shape: shape} = expr = to_expr(expr)
    output_shape = Translation.unary_op_rule(shape)
    make_expr(output_shape, op, [expr])
  end

  def make_binary_op_expr(op, expr1, expr2) do
    %Expr{shape: s1} = expr1 = to_expr(expr1)
    %Expr{shape: s2} = expr2 = to_expr(expr2)
    output_shape = Translation.binary_op_rule(s1, s2)
    make_expr(output_shape, op, [expr1, expr2])
  end

  def make_aggregate_op_expr(op, expr, opts) do
    %Expr{shape: shape} = expr = to_expr(expr)
    axes = opts[:axes] || opts[:axis] || all_dimensions(shape)
    output_shape = Translation.aggregate_op_rule(shape, axes)
    make_expr(output_shape, op, [expr, opts])
  end

  @element_wise_unary_op [:exp, :expm1, :log,
                          :logp1, :logistic, :cos,
                          :sin, :tanh, :sqrt, :rsqrt,
                          :cbrt, :negate, :sign, :abs,
                          :bitwise_not, :population_count,
                          :count_leading_zeros, :floor,
                          :ceil, :round]

  @element_wise_binary_op [:add, :subtract, :multiply, :divide,
                           :power, :remainder, :arctan2, :max,
                           :min, :bitwise_and, :bitwise_or,
                           :bitwise_xor, :left_shift, :right_shift,
                           :equal, :not_equal, :greater, :less,
                           :less_equal, :greater_equal]

  @aggregate_op [:sum, :mean, :argmax, :argmin]

  ## Traversal

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @element_wise_unary_op do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :make_unary_op_expr, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @element_wise_binary_op do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :make_binary_op_expr, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @aggregate_op do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :make_aggregate_op_expr, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state) do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :"#{name}", args), state}
  end

  defp traverse({left, right}, state) do
    {left, state} = traverse(left, state)
    {right, state} = traverse(right, state)
    {{left, right}, state}
  end

  defp traverse({name, meta, args}, state) do
    {args, state} = traverse(args, state)
    {{name, meta, args}, state}
  end

  defp traverse(list, state) when is_list(list) do
    Enum.map_reduce(list, state, &traverse/2)
  end

  defp traverse(other, state) do
    {other, state}
  end

  defp to_expr_call(dot_meta, meta, fun, args) do
    {{:., dot_meta, [__MODULE__, fun]}, meta, args}
  end

  ## Helpers
  defp all_dimensions(shape) do
    for i <- 0..(tuple_size(shape) - 1), do: i
  end
end
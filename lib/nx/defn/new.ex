defmodule Nx.Defn.Expr do
  @moduledoc false
  # A defn AST which carries with it it's current shape and arguments
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
  def binary_op_rule(s1, s2), do: Nx.Shape.binary_broadcast(s1, s2)

  ## Aggregate ops

  # The shapes of aggregate ops are contracted along
  # the specified dimension(s).
  def aggregate_op_rule(shape, axes), do: Nx.Shape.contract(shape, axes)

  ## Creation ops

  # The output shape of a creation op is equal to the
  # desired input shape
  def creation_op_rule(shape), do: shape

  ## Matrix ops

  # See Nx.Shape.transpose
  def transpose_op_rule(shape, permutation), do: Nx.Shape.transpose(shape, permutation)

  # See Nx.Shape.outer
  def outer_op_rule(s1, s2), do: Nx.Shape.outer(s1, s2)

  # The output shape of a dot product is the outer product of s1 and s2
  # contracted the specified dimensions. The contraction dimensions depend
  # on the rank of each input shape. See Nx.dot for a full description
  # of dot product semantics
  def dot_op_rule(s1, s2) do
    case {tuple_size(s1), tuple_size(s2)} do
      {0, _} -> Nx.Shape.binary_broadcast(s1, s2)
      {_, 0} -> Nx.Shape.binary_broadcast(s1, s2)
      {n, 1} -> dot_op_rule(s1, [n - 1], s2, [0])
      {1, m} -> dot_op_rule(s1, [0], s2, [m - 2])
      {n, m} when n >= 2 and m >= 2 -> dot_op_rule(s1, [n - 1], s2, [m - 2])
    end
  end

  defp dot_op_rule(s1, axes1, s2, axes2), do: Nx.Shape.outer(Nx.Shape.contract(s1, axes1), Nx.Shape.contract(s2, axes2))

  # The output shape of a select is equal to the shape of pred_shape
  # achieved by broadcasting both on_true and on_false to match
  def select_op_rule(pred_shape, on_true_shape, on_false_shape) do
    # broadcast twice to validate both shapes are valid
    Nx.Shape.broadcast(on_true_shape, pred_shape)
    Nx.Shape.broadcast(on_false_shape, pred_shape)
  end

  # See Nx.Shape.reshape
  def reshape_op_rule(shape, new_shape), do: Nx.Shape.reshape(shape, new_shape)

  # See Nx.Shape.broadcast
  def broadcast_op_rule(shape, new_shape), do: Nx.Shape.broadcast(shape, new_shape)

end

defmodule Nx.Defn.New do
  @moduledoc false
  # Refactor of the original defn compiler
  import Nx.Shared
  alias Nx.Defn.Expr
  alias Nx.Defn.Translation
  alias Nx.Tensor, as: T

  def __compile__(_env, _kind, _meta, _vars, ast, _options) do
    {ast, _state} = traverse(ast, %{})

    quote do
      unquote(ast)
    end
  end

  ## Shape normalization

  defp to_shape(%T{shape: shape}), do: shape

  defp to_shape(%Expr{shape: shape}), do: shape

  defp to_shape(shape) when is_tuple(shape), do: shape

  defp to_shape(other) do
    raise "unable to interpret #{inspect(other)} as a valid shape"
  end

  ## Expr creation

  defp to_expr(%Expr{} = expr), do: expr

  defp to_expr(integer) when is_integer(integer),
    do: %Expr{args: integer, op: :tensor, shape: {}}

  defp to_expr(number) when is_number(number),
    do: %Expr{args: number, op: :tensor, shape: {}}

  defp to_expr(%T{shape: shape} = t), do: %Expr{args: t, op: :tensor, shape: shape}

  defp to_expr(other) do
    raise "unable to convert #{inspect(other)} into a valid Expr"
  end

  defp make_expr(shape, op, args) do
    %Expr{shape: shape, op: op, args: args}
  end

  ## Rank, Shape, Size

  # These functions are computed and returned at compile-time

  def rank(expr) do
    %Expr{shape: shape} = to_expr(expr)
    tuple_size(shape)
  end

  def shape(expr) do
    %Expr{shape: shape} = to_expr(expr)
    shape
  end

  def size(expr) do
    %Expr{shape: shape} = to_expr(expr)
    tuple_product(shape)
  end

  @rank_shape_size [:rank, :shape, :size]

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
    axes = Nx.Shape.normalize_axis(shape, axes)
    output_shape = Translation.aggregate_op_rule(shape, axes)
    make_expr(output_shape, op, [expr, opts])
  end

  def make_creation_op_expr(op, shape, args) do
    shape = to_shape(shape)
    output_shape = Translation.creation_op_rule(shape)
    make_expr(output_shape, op, [shape | args])
  end

  def make_transpose_op_expr(expr), do: make_transpose_op_expr(expr, [])
  def make_transpose_op_expr(expr, permutation) do
    %Expr{shape: shape} = expr = to_expr(expr)
    permutation = Nx.Shape.normalize_axis(shape, permutation)
    output_shape = Translation.transpose_op_rule(shape, permutation)
    make_expr(output_shape, :transpose, [expr, permutation])
  end

  def make_dot_op_expr(expr1, expr2) do
    %Expr{shape: s1} = expr1 = to_expr(expr1)
    %Expr{shape: s2} = expr2 = to_expr(expr2)
    output_shape = Translation.dot_op_rule(s1, s2)
    make_expr(output_shape, :dot, [expr1, expr2])
  end

  def make_outer_op_expr(expr1, expr2) do
    %Expr{shape: s1} = expr1 = to_expr(expr1)
    %Expr{shape: s2} = expr2 = to_expr(expr2)
    output_shape = Translation.outer_op_rule(s1, s2)
    make_expr(output_shape, :outer, [expr1, expr2])
  end

  def make_select_op_expr(pred_expr, true_expr, false_expr) do
    %Expr{shape: pred_shape} = pred_expr = to_expr(pred_expr)
    %Expr{shape: true_shape} = true_expr = to_expr(true_expr)
    %Expr{shape: false_shape} = false_expr = to_expr(false_expr)
    output_shape = Translation.select_op_rule(pred_shape, true_shape, false_shape)
    make_expr(output_shape, :select, [pred_expr, true_expr, false_expr])
  end

  def make_reshape_op_expr(expr, shape) do
    %Expr{shape: old_shape} = expr = to_expr(expr)
    shape = to_shape(shape)
    output_shape = Translation.reshape_op_rule(old_shape, shape)
    make_expr(output_shape, :reshape, [expr, shape])
  end

  def make_broadcast_op_expr(expr, shape) do
    %Expr{shape: old_shape} = expr = to_expr(expr)
    shape = to_shape(shape)
    output_shape = Translation.broadcast_op_rule(old_shape, shape)
    make_expr(output_shape, :broadcast, [expr, shape])
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

  @creation_op [:iota, :random_normal, :random_uniform]

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

  defp traverse({{:., dot_meta, [Nx, name]}, meta, [shape | args]}, state)
       when name in @creation_op do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :make_creation_op_expr, [name, shape | [args]]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @rank_shape_size do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :"#{name}", args), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state) do
    {args, state} = traverse(args, state)
    {to_expr_call(dot_meta, meta, :"make_#{name}_op_expr", args), state}
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
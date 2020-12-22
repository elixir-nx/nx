defmodule Nx.Defn.Expr do
  # A defn AST which carries with it it's current shape and arguments
  @moduledoc false

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @enforce_keys [:shape, :op, :args]
  defstruct [:shape, :op, :args]

  @doc """
  Expression equivalent to `Nx.rank/1`.
  """
  def rank(expr) do
    %Expr{shape: shape} = to_expr(expr)
    Nx.Shape.rank(shape)
  end

  @doc """
  Expression equivalent to `Nx.shape/1`.
  """
  def shape(expr) do
    %Expr{shape: shape} = to_expr(expr)
    shape
  end

  @doc """
  Expression equivalent to `Nx.size/1`.
  """
  def size(expr) do
    %Expr{shape: shape} = to_expr(expr)
    Nx.Shape.size(shape)
  end

  unary_ops =
    [:exp, :expm1, :log, :logp1, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
      [:negate, :sign, :abs, :bitwise_not, :population_count, :count_leading_zeros] ++
      [:floor, :ceil, :round]

  for op <- unary_ops do
    @doc """
    Expression equivalent to `Nx.#{op}/1`.
    """
    def unquote(op)(expr) do
      %Expr{shape: shape} = expr = to_expr(expr)
      make_expr(shape, unquote(op), [expr])
    end
  end

  binary_ops =
    [:add, :subtract, :multiply, :divide, :power, :remainder, :arctan2, :max, :min] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :less_equal, :greater_equal]

  for op <- binary_ops do
    @doc """
    Expression equivalent to `Nx.#{op}/2`.
    """
    def unquote(op)(expr1, expr2) do
      %Expr{shape: s1} = expr1 = to_expr(expr1)
      %Expr{shape: s2} = expr2 = to_expr(expr2)
      output_shape = Nx.Shape.binary_broadcast(s1, s2)
      make_expr(output_shape, unquote(op), [expr1, expr2])
    end
  end

  aggregate_axes_ops = [:sum, :mean]

  for op <- aggregate_axes_ops do
    @doc """
    Expression equivalent to `Nx.#{op}/2`.
    """
    def unquote(op)(expr, opts \\ []) do
      %Expr{shape: shape} = expr = to_expr(expr)

      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes)
        output_shape = Nx.Shape.contract(shape, axes)
        make_expr(output_shape, unquote(op), [expr, [axes: axes]])
      else
        make_expr({}, unquote(op), [expr, []])
      end
    end
  end

  aggregate_axis_ops = [:argmin, :argmax]

  for op <- aggregate_axis_ops do
    @doc """
    Expression equivalent to `Nx.#{op}/2`.
    """
    def unquote(op)(expr, opts \\ []) do
      %Expr{shape: shape} = expr = to_expr(expr)

      if axis = opts[:axis] do
        axis = Nx.Shape.normalize_axis(shape, axis)
        output_shape = Nx.Shape.contract(shape, [axis])
        make_expr(output_shape, unquote(op), [expr, [axis: axis]])
      else
        make_expr({}, unquote(op), [expr, []])
      end
    end
  end

  @doc """
  Expression equivalent to `Nx.iota/2`.
  """
  # TODO: Normalize axis
  def iota(shape, opts \\ []) do
    shape = to_shape(shape)
    make_expr(shape, :iota, [shape, opts])
  end

  random_ops = [:random_normal, :random_uniform]

  for op <- random_ops do
    @doc """
    Expression equivalent to `Nx.#{op}/2`.
    """
    def unquote(op)(shape, opts \\ []) do
      unquote(op)(shape, 0.0, 1.0, opts)
    end

    @doc """
    Expression equivalent to `Nx.#{op}/4`.
    """
    def unquote(op)(shape, min, max, opts \\ []) do
      shape = to_shape(shape)
      make_expr(shape, unquote(op), [shape, min, max, opts])
    end
  end

  @doc """
  Expression equivalent to `Nx.transpose/1`.
  """
  def transpose(expr) do
    %Expr{shape: shape} = expr = to_expr(expr)
    transpose(expr, Nx.Shape.transpose_axes(shape))
  end

  @doc """
  Expression equivalent to `Nx.transpose/2`.
  """
  def transpose(expr, axes) do
    %Expr{shape: shape} = expr = to_expr(expr)
    axes = Nx.Shape.normalize_axes(shape, axes)
    output_shape = Nx.Shape.transpose(shape, axes)
    make_expr(output_shape, :transpose, [expr, axes])
  end

  @doc """
  Expression equivalent to `Nx.outer/2`.
  """
  def outer(expr1, expr2) do
    %Expr{shape: s1} = expr1 = to_expr(expr1)
    %Expr{shape: s2} = expr2 = to_expr(expr2)
    output_shape = Nx.Shape.outer(s1, s2)
    make_expr(output_shape, :outer, [expr1, expr2])
  end

  @doc """
  Expression equivalent to `Nx.dot/2`.
  """
  def dot(expr1, expr2) do
    %Expr{shape: s1} = expr1 = to_expr(expr1)
    %Expr{shape: s2} = expr2 = to_expr(expr2)
    output_shape = Nx.Shape.dot(s1, s2)
    make_expr(output_shape, :dot, [expr1, expr2])
  end

  @doc """
  Expression equivalent to `Nx.reshape/2`.
  """
  def reshape(expr, shape) do
    %Expr{shape: old_shape} = expr = to_expr(expr)
    shape = to_shape(shape)
    output_shape = Nx.Shape.reshape(old_shape, shape)
    make_expr(output_shape, :reshape, [expr, shape])
  end

  @doc """
  Expression equivalent to `Nx.broadcast/2`.
  """
  def broadcast(expr, shape) do
    %Expr{shape: old_shape} = expr = to_expr(expr)
    shape = to_shape(shape)
    output_shape = Nx.Shape.broadcast(old_shape, shape)
    make_expr(output_shape, :broadcast, [expr, shape])
  end

  @doc """
  Expression equivalent to `Nx.select/3`.
  """
  def select(pred_expr, true_expr, false_expr) do
    %Expr{shape: pred_shape} = pred_expr = to_expr(pred_expr)
    %Expr{shape: true_shape} = true_expr = to_expr(true_expr)
    %Expr{shape: false_shape} = false_expr = to_expr(false_expr)
    Nx.Shape.broadcast(true_shape, pred_shape)
    Nx.Shape.broadcast(false_shape, pred_shape)
    make_expr(pred_shape, :select, [pred_expr, true_expr, false_expr])
  end

  ## Expr normalization

  defp to_expr(%Expr{} = expr), do: expr
  defp to_expr(number) when is_number(number), do: %Expr{args: number, op: :tensor, shape: {}}
  defp to_expr(%T{shape: shape} = t), do: %Expr{args: t, op: :tensor, shape: shape}

  defp to_expr(other) do
    raise ArgumentError,
          "unable to convert #{inspect(other)} into a Nx.Defn.Expr, expected a tensor or a number"
  end

  defp make_expr(shape, op, args) do
    %Expr{shape: shape, op: op, args: args}
  end

  ## Shape normalization

  defp to_shape(shape) when is_tuple(shape), do: shape
  defp to_shape(shape) when is_number(shape), do: {}
  defp to_shape(%T{shape: shape}), do: shape
  defp to_shape(%Expr{shape: shape}), do: shape

  defp to_shape(other) do
    raise ArgumentError,
          "expected a shape as argument. A shape is a n-element tuple with the size of each dimension. " <>
            "Alternatively you can pass a tensor (or a number) and the shape will be retrieved from the tensor. " <>
            "Got: #{inspect(other)}"
  end
end

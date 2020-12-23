defmodule Nx.Defn.Expr do
  @doc """
  The expression used by `Nx.Defn.Compiler`.

  It is a struct with the following fields:

    * `:id` - a unique identifier
    * `:op` - the operation name
    * `:args` - the operation arguments
    * `:shape` - the operation resulting shape

  All `:op` nodes translate to `Nx` operations, except for:

    * `:parameter` - holds a parameter constructed by `parameter/2`.
      `:args` is a one element list with the given arg.

    * `:constant` - holds a numeric constant.
      `:args` is a one element list with a number.
  """

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @enforce_keys [:id, :shape, :op, :args]
  @type t :: %Expr{}
  defstruct [:id, :shape, :op, :args]

  @doc """
  Builds a parameter must be passed to the evaluation function.
  """
  def parameter(shape, arg) do
    make_expr(shape, :parameter, [arg])
  end

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
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
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
        opts = Keyword.put(opts, :axes, axes)
        make_expr(output_shape, unquote(op), [expr, opts])
      else
        make_expr({}, unquote(op), [expr, opts])
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
        opts = Keyword.put(opts, :axis, axis)
        make_expr(output_shape, unquote(op), [expr, opts])
      else
        make_expr({}, unquote(op), [expr, opts])
      end
    end
  end

  @doc """
  Expression equivalent to `Nx.iota/2`.
  """
  def iota(shape, opts \\ []) do
    shape = to_shape(shape)

    opts =
      if axis = opts[:axis] do
        Keyword.put(opts, :axis, Nx.Shape.normalize_axis(shape, axis))
      else
        opts
      end

    make_expr(shape, :iota, [shape, opts])
  end

  @doc """
  Expression equivalent to `Nx.random_uniform/2`.
  """
  def random_uniform(shape, opts \\ []) do
    random_uniform(shape, 0.0, 1.0, opts)
  end

  @doc """
  Expression equivalent to `Nx.random_uniform/4`.
  """
  def random_uniform(shape, min, max, opts \\ []) when is_number(min) and is_number(max) do
    shape = to_shape(shape)
    make_expr(shape, :random_uniform, [shape, min, max, opts])
  end

  @doc """
  Expression equivalent to `Nx.random_normal/2`.
  """
  def random_normal(shape, opts \\ []) do
    random_normal(shape, 0.0, 1.0, opts)
  end

  @doc """
  Expression equivalent to `Nx.random_normal/4`.
  """
  def random_normal(shape, mu, sigma, opts \\ []) when is_float(mu) and is_float(sigma) do
    shape = to_shape(shape)
    make_expr(shape, :random_normal, [shape, mu, sigma, opts])
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
    output_shape =
      case pred_shape do
        {} ->
          if Nx.Shape.size(true_shape) > Nx.Shape.size(false_shape),
            do: true_shape,
            else: false_shape
        _ ->
          pred_shape
      end
    Nx.Shape.broadcast(true_shape, output_shape)
    Nx.Shape.broadcast(false_shape, output_shape)
    make_expr(output_shape, :select, [pred_expr, true_expr, false_expr])
  end

  ## Results normalization

  @doc false
  def to_result(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&to_result/1) |> List.to_tuple()

  def to_result(%Expr{} = expr),
    do: expr

  def to_result(%T{} = t),
    do: to_expr(t)

  def to_result(number) when is_number(number),
    do: to_expr(number)

  def to_result(other) do
    raise ArgumentError, "defn must return a tensor, a number or a tuple, got: #{inspect(other)}"
  end

  ## Expr normalization

  defp to_expr(%Expr{} = expr), do: expr
  defp to_expr(number) when is_number(number), do: make_expr({}, :constant, [number])

  defp to_expr(%T{shape: shape, data: data} = t) do
    case data do
      {Nx.BitStringDevice, bitstring} when is_bitstring(bitstring) ->
        make_expr(shape, :tensor, [t])

      _ ->
        raise ArgumentError, "tensors inside defn must be allocated on Nx.BitStringDevice"
    end
  end

  defp to_expr(other) do
    raise ArgumentError,
          "unable to convert #{inspect(other)} into a Nx.Defn.Expr, expected a tensor or a number"
  end

  defp make_expr(shape, op, args) when is_tuple(shape) and is_atom(op) and is_list(args) do
    id = System.unique_integer()
    %Expr{id: id, shape: shape, op: op, args: args}
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

  defimpl Inspect do
    import Inspect.Algebra

    @vars "abcdefghijklmnopqrstuvwxyz"

    def inspect(expr, opts) do
      expr_doc = inspect_expr(expr)
      color("#Nx.Defn.Expr<", :map, opts)
      |> concat(nest(expr_doc, 2))
      |> concat(color("\n>", :map, opts))
    end

    defp inspect_expr(%Expr{op: op, shape: shape, args: args}) do
      {exprs, params, var_map} = inspect_expr_args(args, [], %{})

      args_strs = inspect_args(args, var_map)
      var = counter_to_var(map_size(var_map))

      expr_str = var <> " = " <> Atom.to_string(op) <> " [ " <> Enum.join(args_strs, ", ") <> " ] " <> shape_to_string(shape)

      exprs = List.flatten([expr_str | [exprs | params]])

      exprs
      |> Enum.reverse()
      |> Enum.uniq()
      |> Enum.reduce("", fn str, acc -> concat(acc, concat(line(), str)) end)
    end

    # Post-order traversal of the Expr AST, but we pull all parameters to the front
    defp inspect_expr_args([], params, var_map), do: {[], params, var_map}

    defp inspect_expr_args([head | tail], params, var_map) do
      case head do
        %Expr{op: :constant} ->
          {[], params, var_map}

        %Expr{id: id, op: :parameter} ->
          var = counter_to_var(map_size(var_map))
          var_map = Map.update(var_map, id, var, fn _ -> var end)
          {[], ["parameter " <> var | params], var_map}

        %Expr{id: id, op: :tensor} ->
          var = counter_to_var(map_size(var_map))
          var_map = Map.update(var_map, id, var, fn _ -> var end)
          {["tensor " <> var], params, var_map}

        %Expr{id: id, op: op, args: expr_args, shape: shape} ->
          {expr_children, params, var_map} =
            inspect_expr_args(expr_args, params, var_map)

          {expr_siblings, params, var_map} =
            inspect_expr_args(tail, params, var_map)

          expr_args_strs = inspect_args(expr_args, var_map)
          var = counter_to_var(map_size(var_map))

          expr_str =
            var <> " = " <> Atom.to_string(op) <> " [ " <> Enum.join(expr_args_strs, ", ") <> " ] " <> shape_to_string(shape)

          {[expr_str | [expr_siblings | expr_children]], params, Map.update(var_map, id, var, fn _ -> var end)}

        _ ->
          inspect_expr_args(tail, params, var_map)
      end
    end

    defp inspect_args([], _var_map), do: []

    defp inspect_args([arg | args], var_map) do
      case arg do
        %Expr{op: :constant, args: [number]} ->
          [to_string(number) | inspect_args(args, var_map)]

        %Expr{id: id} ->
          [Map.get(var_map, id) | inspect_args(args, var_map)]

        [_ | _] = opts when args == [] ->
          [Enum.map_join(opts, ", ", fn {k, v} -> "#{k}: #{inspect(v)}" end) | inspect_args(args, var_map)]

        value ->
          ["#{inspect(value)}" | inspect_args(args, var_map)]
      end
    end

    defp counter_to_var(counter) do
      String.at(@vars, rem(counter, 26))
    end

    defp shape_to_string({}), do: "()"
    defp shape_to_string(shape) do
      shape_str =
        shape
        |> Tuple.to_list()
        |> Enum.join("x")
      "(" <> shape_str <> ")"
    end
  end
end

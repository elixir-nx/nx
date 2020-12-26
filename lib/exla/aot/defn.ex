defmodule Exla.Aot.Defn do
  @moduledoc false

  def __compile__(env, _kind, _vars, fun, options) do
    %{module: module, function: {name, arity}} = env

    shapes = options[:shapes]
    types = options[:types]

    builder = Exla.Builder.new("#{name}/#{arity}")

    params_and_vars =
      for {{shape, type}, i} <- Enum.with_index(Enum.zip(shapes, types)) do
        exla_shape = Exla.Shape.make_shape(type, shape)
        param = Exla.Op.parameter(builder, i, exla_shape, "p#{i}")
        {Nx.Defn.Expr.parameter(shape, param), %{id: i, name: "p#{i}", dims: shape}}
      end

    {params, vars} = Enum.unzip(params_and_vars)

    op =
      fun.(params)
      |> to_result(builder, %{})
      |> elem(0)

    op = Exla.Op.tuple(builder, [op])
    computation = Exla.Builder.build(op)

    %Exla.Shape{dtype: {:t, [output]}} = computation.output_shape

    output_size = tuple_product(output.dims, tuple_size(output.dims))

    IO.inspect output_size

    Exla.Aot.Compile.compile([computation], [{name, arity, vars, output_size}], module)
  end

  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  defp to_result(tuple, builder, cache) when is_tuple(tuple) do
    {elements, cache} =
      tuple
      |> Tuple.to_list()
      |> Enum.map_reduce(cache, &to_result(&1, builder, &2))

    {Exla.Op.tuple(builder, elements), cache}
  end

  defp to_result(expr, builder, cache) do
    {expr, cache} = recur_operator(expr, builder, cache)
    {to_operator(builder, expr), cache}
  end

  defp recur_operator(%Nx.Defn.Expr{op: :parameter, args: [param]}, _builder, cache) do
    {param, cache}
  end

  defp recur_operator(%Nx.Defn.Expr{op: :constant, args: [number]}, _builder, cache) do
    {number, cache}
  end

  defp recur_operator(%Nx.Defn.Expr{id: id, op: op, args: args, shape: shape}, builder, cache) do
    case cache do
      %{^id => res} ->
        res

      %{} ->
        {shape_ops, cache} =
          Enum.map_reduce(args, cache, fn
            %Nx.Defn.Expr{} = arg, cache ->
              {op, cache} = recur_operator(arg, builder, cache)
              {{arg, op}, cache}

            arg, cache ->
              {arg, cache}
          end)

        op = to_operator(op, shape_ops, shape, builder)
        {op, Map.put(cache, id, op)}
    end
  end

  ## to_operator creation

  defp to_operator(:tensor, [%Nx.Tensor{type: type, data: {_, data}}], shape, builder) do
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.constant_from_binary(builder, data, shape)
  end

  defp to_operator(:random_uniform, [shape, min, max, opts], _shape, builder) do
    type = opts[:type] || Exla.Type.infer(max - min)

    if match?({int, size} when int in [:s, :u] and size < 32, type) do
      raise ArgumentError,
            "Nx.random_uniform/4 for Exla requires signed and unsigned tensors to be " <>
              "at least of size 32, got: #{elem(type, 1)}"
    end

    min = to_typed_operator(builder, min, type, type)
    max = to_typed_operator(builder, max, type, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_uniform(min, max, shape)
  end

  defp to_operator(:random_normal, [shape, mu, sigma, opts], _shape, builder) do
    type = opts[:type] || {:f, 64}
    mu = to_typed_operator(builder, mu, type, type)
    sigma = to_typed_operator(builder, sigma, type, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_normal(mu, sigma, shape)
  end

  defp to_operator(:iota, [shape, opts], _shape, builder) do
    type = opts[:type] || {:s, 64}
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Lib.iota(builder, shape, opts)
  end

  ## to_operator shape

  defp to_operator(:reshape, [{_, arg}, shape], _shape, builder) do
    Exla.Op.reshape(to_operator(builder, arg), shape)
  end

  defp to_operator(:broadcast, [{expr, arg}, _shape], output_shape, builder) do
    arg = to_operator(builder, arg)
    Exla.Op.broadcast_in_dim(arg, output_shape, broadcast_dimensions(expr.shape, output_shape))
  end

  defp to_operator(:transpose, [{_expr, arg}, dims], _output_shape, builder) do
    arg = to_operator(builder, arg)
    Exla.Op.transpose(arg, List.to_tuple(dims))
  end

  ## to_operator others

  defp to_operator(:dot, [{_, left}, {_, right}], _output_shape, builder) do
    {left, right} = binary_op_type(builder, left, right, & &1)

    %Exla.Shape{dims: s1} = Exla.Op.get_shape(left)
    %Exla.Shape{dims: s2} = Exla.Op.get_shape(right)

    # To keep the semantics the same as Numpy, XLA will raise otherwise
    case {tuple_size(s1), tuple_size(s2)} do
      {0, _} -> Exla.Op.multiply(left, right)
      {_, 0} -> Exla.Op.multiply(left, right)
      {m, n} when m >= 2 and n > 2 -> Exla.Op.dot_general(left, right, {m - 1, n - 2})
      _ -> Exla.Op.dot(left, right)
    end
  end

  defp to_operator(
         :select,
         [{_, pred}, {expr_true, on_true}, {expr_false, on_false}],
         output_shape,
         builder
       ) do
    pred = to_typed_operator(builder, pred, op_type(pred), {:pred, 1})
    {on_true, on_false} = binary_op_type(builder, on_true, on_false, & &1)

    on_true =
      Exla.Op.broadcast_in_dim(
        on_true,
        output_shape,
        broadcast_dimensions(expr_true.shape, output_shape)
      )

    on_false =
      Exla.Op.broadcast_in_dim(
        on_false,
        output_shape,
        broadcast_dimensions(expr_false.shape, output_shape)
      )

    Exla.Op.select(pred, on_true, on_false)
  end

  ## to_operator element-wise

  defp to_operator(:negate, [{_, arg}], _shape, _builder) do
    case arg do
      %Exla.Op{} = op -> Exla.Op.negate(op)
      number when is_number(number) -> -number
    end
  end

  defp to_operator(:abs, [{_, arg}], _shape, builder) do
    arg = to_operator(builder, arg)

    case op_type(arg) do
      {:u, _} -> arg
      _ -> Exla.Op.abs(arg)
    end
  end

  defp to_operator(:sign, [{_, arg}], _shape, builder) do
    arg = to_operator(builder, arg)

    case op_type(arg) do
      {:u, _} = type -> Exla.Op.min(arg, Exla.Op.constant_r0(builder, 1, type))
      _ -> Exla.Op.sign(arg)
    end
  end

  defp to_operator(:right_shift, [{left_expr, left}, {right_expr, right}], _shape, builder) do
    {left, right} = binary_op_type(builder, left, right, &assert_integer_type!(&1, :right_shift))
    dims = broadcast_dimensions(left_expr.shape, right_expr.shape)

    op =
      if match?({:u, _}, constant_or_type(left)),
        do: :right_shift_logical,
        else: :right_shift_arithmetic

    apply(Exla.Op, op, [left, right, dims])
  end

  @bin_arith_op [:add, :subtract, :multiply, :min, :max, :remainder, :power] ++
                  [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

  defp to_operator(op, [{left_expr, left}, {right_expr, right}], _shape, builder)
       when op in @bin_arith_op do
    {left, right} = binary_op_type(builder, left, right, & &1)
    dims = broadcast_dimensions(left_expr.shape, right_expr.shape)
    apply(Exla.Op, op, [left, right, dims])
  end

  @bin_float_arith_op [:divide, :arctan2]

  defp to_operator(op, [{left_expr, left}, {right_expr, right}], _shape, builder)
       when op in @bin_float_arith_op do
    {left, right} = binary_op_type(builder, left, right, &Exla.Type.to_floating/1)
    dims = broadcast_dimensions(left_expr.shape, right_expr.shape)
    apply(Exla.Op, op, [left, right, dims])
  end

  @bin_bitwise_op [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift]

  defp to_operator(op, [{left_expr, left}, {right_expr, right}], _shape, builder)
       when op in @bin_bitwise_op do
    {left, right} = binary_op_type(builder, left, right, &assert_integer_type!(&1, op))
    dims = broadcast_dimensions(left_expr.shape, right_expr.shape)
    apply(Exla.Op, op, [left, right, dims])
  end

  @unary_bitwise_op [:bitwise_not, :count_leading_zeros, :population_count]

  defp to_operator(op, [{_expr, arg}], _shape, builder)
       when op in @unary_bitwise_op do
    arg = to_operator(builder, arg)
    assert_integer_type!(op_type(arg), op)
    apply(Exla.Op, op, [arg])
  end

  @unary_float_op [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt]

  defp to_operator(op, [{_expr, arg}], _shape, builder)
       when op in @unary_float_op do
    apply(Exla.Op, op, [to_float_operator(builder, arg)])
  end

  @unary_noop_integer_op [:floor, :ceil, :round]

  defp to_operator(op, [{_expr, arg}], _shape, builder)
       when op in @unary_noop_integer_op do
    arg = to_operator(builder, arg)

    case op_type(arg) do
      {:s, _} -> arg
      {:u, _} -> arg
      _ -> apply(Exla.Op, op, [arg])
    end
  end

  ## to_operator reduction

  @reduction_op [:sum, :mean, :argmax, :argmin]

  defp to_operator(op, [{_expr, arg}, opts], _shape, builder)
       when op in @reduction_op do
    apply(Exla.Lib, op, [builder, arg, opts])
  end

  ## constant/operator

  defp to_operator(_builder, %Exla.Op{} = op),
    do: op

  defp to_operator(builder, int) when is_integer(int),
    do: Exla.Op.constant_r0(builder, int, {:s, 64})

  defp to_operator(builder, float) when is_float(float),
    do: Exla.Op.constant_r0(builder, float, {:f, 64})

  defp to_float_operator(_builder, %Exla.Op{} = op) do
    current = op_type(op)
    type = Exla.Type.to_floating(current)
    if current != type, do: Exla.Op.convert_element_type(op, type), else: op
  end

  defp to_float_operator(builder, constant) when is_number(constant) do
    Exla.Op.constant_r0(builder, constant, {:f, 64})
  end

  defp to_typed_operator(_builder, %Exla.Op{} = op, type, type),
    do: op

  defp to_typed_operator(_builder, %Exla.Op{} = op, _type, type),
    do: Exla.Op.convert_element_type(op, type)

  defp to_typed_operator(builder, constant, _type, type) when is_number(constant),
    do: Exla.Op.constant_r0(builder, constant, type)

  ## Dimension helpers

  defp broadcast_dimensions(left, right) do
    {min, max} = if left <= right, do: {left, right}, else: {right, left}
    min_size = tuple_size(min)
    max_size = tuple_size(max)
    # To reproduce Nx broadcast, we simply match the lower dimensions to the highest ones.
    List.to_tuple(count_down(min_size, max_size - min_size))
  end

  defp count_down(0, _n), do: []
  defp count_down(i, n), do: [n | count_down(i - 1, n + 1)]

  ## Type helpers

  defp binary_op_type(builder, left_op, right_op, fun) do
    left_type = constant_or_type(left_op)
    right_type = constant_or_type(right_op)
    output_type = binary_op_type(left_type, right_type) |> fun.()

    {to_typed_operator(builder, left_op, left_type, output_type),
     to_typed_operator(builder, right_op, right_type, output_type)}
  end

  defp binary_op_type(left, right) when is_number(left) and is_number(right),
    do: Exla.Type.infer(left + right)

  defp binary_op_type(scalar, type) when is_number(scalar),
    do: Exla.Type.merge_scalar(type, scalar)

  defp binary_op_type(type, scalar) when is_number(scalar),
    do: Exla.Type.merge_scalar(type, scalar)

  defp binary_op_type(left, right),
    do: Exla.Type.merge(left, right)

  defp constant_or_type(number) when is_number(number), do: number
  defp constant_or_type(op), do: op_type(op)

  defp assert_integer_type!({:s, _} = type, _op), do: type
  defp assert_integer_type!({:u, _} = type, _op), do: type

  defp assert_integer_type!(type, op) do
    raise ArgumentError,
          "#{op} expects integer tensors as inputs and outputs an integer tensor, " <>
            "got: #{inspect(type)}"
  end

  defp op_type(op), do: Exla.Op.get_shape(op).dtype
end

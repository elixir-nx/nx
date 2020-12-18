defmodule Exla.Aot.Defn do
  @moduledoc false

  def sf_computation(module, name, arity, vars, options, fun) do
    buffers = for var <- vars, do: nx_to_buffer(var)

    shapes = Enum.map(buffers, & &1.shape)
    builder = Exla.Builder.new("#{name}/#{arity}")
    result = fun.(builder, shapes)
    ast = Exla.Op.tuple(builder, [to_block_result(builder, result)])

    {Exla.Builder.build(ast), name, shapes}
  end

  def sf_builder(name) do
    Exla.Builder.new(name)
  end

  def sf_parameter(builder, pos, shape, name) do
    Exla.Op.parameter(builder, pos, shape, name)
  end

  defp nx_to_buffer(%Nx.Tensor{data: {device, data}, type: type, shape: shape}) do
    case device do
      Nx.BitStringDevice when is_bitstring(data) ->
        Exla.Buffer.buffer(data, Exla.Shape.make_shape(type, shape))

      Exla.NxDevice when is_tuple(data) ->
        Exla.Buffer.buffer(data, Exla.Shape.make_shape(type, shape))

      true ->
        raise ArgumentError, "unknown device #{inspect(device)} given to defn compiled with Exla"
    end
  end

  defp nx_to_buffer(number) when is_integer(number) do
    Exla.Buffer.buffer(<<number::64-native>>, Exla.Shape.make_shape({:s, 64}, {}))
  end

  defp nx_to_buffer(number) when is_float(number) do
    Exla.Buffer.buffer(<<number::float-64-native>>, Exla.Shape.make_shape({:f, 64}, {}))
  end

  ## Special forms

  def sf_nx_tensor(builder, %Nx.Tensor{data: {Nx.BitStringDevice, data}, type: type, shape: shape}) do
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.constant_from_binary(builder, data, shape)
  end

  ## Aggregators

  @reduction_ops [:sum, :mean, :argmax, :argmin]

  ## Conversion functions

  defp to_block_result(builder, tuple) when is_tuple(tuple) do
    elements =
      tuple
      |> Tuple.to_list()
      |> Enum.map(&to_block_result(builder, &1))

    Exla.Op.tuple(builder, elements)
  end

  defp to_block_result(builder, operator), do: to_operator(builder, operator)

  defp to_operator(_builder, %Exla.Op{} = op), do: op
  defp to_operator(builder, constant) when is_number(constant), do: to_constant(builder, constant)

  defp to_operator(_builder, other) do
    raise ArgumentError, "expected a tensor, got: #{inspect(other)}"
  end

  defp to_constant(builder, int) when is_integer(int),
    do: Exla.Op.constant_r0(builder, int, {:s, 64})

  defp to_constant(builder, float) when is_float(float),
    do: Exla.Op.constant_r0(builder, float, {:f, 64})

  ## Callback

  def __compile__(env, _kind, _meta, vars, ast, options) do
    {name, arity} = env.function
    {ast, _state} = traverse(ast, %{})
    shapes = Macro.generate_arguments(length(vars), __MODULE__)

    quote do
      Exla.Aot.Defn.sf_computation(
        __MODULE__,
        unquote(name),
        unquote(arity),
        unquote(vars),
        unquote(options),
        fn builder, unquote(shapes) ->
          unquote_splicing(vars_to_parameters(vars, shapes))
          unquote(ast)
        end
      )
    end
  end

  @bin_float_arith_op [:divide, :arctan2]
  @bin_arith_op [:add, :subtract, :multiply, :divide, :min, :max, :remainder, :power]
  @bin_comparison_op [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]
  @bin_bitwise_op [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift]
  @unary_float_op [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt]
  @unary_bitwise_op [:bitwise_not, :count_leading_zeros, :population_count]
  @unary_noop_integer_op [:floor, :ceil, :round]

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @bin_float_arith_op do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_bin_float_arith_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @bin_arith_op do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_bin_arith_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @bin_bitwise_op do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_bin_bitwise_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @bin_comparison_op do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_bin_comparison_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @unary_float_op do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_unary_float_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @unary_bitwise_op do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_unary_bitwise_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @unary_noop_integer_op do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_unary_noop_integer_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state)
       when name in @reduction_ops do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :nx_reduction_op, [name | args]), state}
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state) do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :"nx_#{name}", args), state}
  end

  defp traverse({:%{}, meta, [__struct__: Nx.Tensor] ++ _} = struct, state) do
    {to_builder_call(meta, :sf_nx_tensor, [struct]), state}
  end

  # TODO: We need to implement pattern matching on tuple once we add conditionals
  defp traverse({name, meta, args}, state) do
    {args, state} = traverse(args, state)
    {{name, meta, args}, state}
  end

  defp traverse({left, right}, state) do
    {left, state} = traverse(left, state)
    {right, state} = traverse(right, state)
    {{left, right}, state}
  end

  defp traverse(list, state) when is_list(list) do
    Enum.map_reduce(list, state, &traverse/2)
  end

  defp traverse(other, state) do
    {other, state}
  end

  defp vars_to_parameters(args, shapes) do
    for {{var, shape}, index} <- Enum.with_index(Enum.zip(args, shapes)) do
      name = var_to_parameter_name(var)

      quote do
        unquote(var) =
          Exla.Defn.sf_parameter(builder, unquote(index), unquote(shape), unquote(name))
      end
    end
  end

  defp var_to_parameter_name({var, meta, ctx}) when is_atom(var) and is_atom(ctx) do
    "#{var}@#{Keyword.fetch!(meta, :counter)}"
  end

  defp to_builder_call(meta, fun, args), do: to_builder_call(meta, meta, fun, args)

  defp to_builder_call(dot_meta, meta, fun, args) do
    {{:., dot_meta, [Exla.Defn, fun]}, meta, [quote(do: builder) | args]}
  end
end

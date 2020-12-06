defmodule Exla.Defn do
  @moduledoc false
  # The Exla compiler for defn mixes runtime with compile-time
  # execution. The goal of the compiler pass is to convert all
  # special forms and Nx calls to calls to this module. All
  # calls are prefixed with either `sf_` or `nx_`.
  #
  # This module then builds the Exla AST as part of its code
  # evaluation process by invoking said functions.

  ## Builder and computations

  def sf_cached_def(module, name, arity, vars, options, fun) do
    cache_args = for var <- vars, do: nx_to_cache_key!(var)
    buffers = for var <- vars, do: nx_to_buffer(var)
    {client_name, options} = Keyword.pop(options, :client, :default)
    cache_key = {module, name, arity, cache_args, client_name}

    executable =
      Exla.LockedCache.run(cache_key, fn ->
        shapes = Enum.map(buffers, & &1.shape)
        builder = Exla.Builder.new("#{name}/#{arity}")
        result = fun.(builder, shapes)
        computation = Exla.Builder.build(to_block_result(builder, result))
        client = Exla.Client.fetch!(client_name)
        executable = Exla.Client.compile(client, computation, shapes)
        :persistent_term.put(cache_key, executable)
        executable
      end)

    executable
    |> Exla.Executable.run(buffers, options)
    |> buffer_to_nx()
  end

  def sf_builder(name) do
    Exla.Builder.new(name)
  end

  def sf_parameter(builder, pos, shape, name) do
    Exla.Op.parameter(builder, pos, shape, name)
  end

  ## Nx <-> Exla.Buffer

  defp buffer_to_nx(%Exla.Buffer{ref: nil, data: data, shape: shape}) do
    %Nx.Tensor{data: {Nx.BitStringDevice, data}, type: shape.dtype, shape: shape.dims}
  end

  defp buffer_to_nx(%Exla.Buffer{ref: ref, data: nil, shape: shape}) do
    %Nx.Tensor{data: {Exla.NxDevice, ref}, type: shape.dtype, shape: shape.dims}
  end

  defp buffer_to_nx({:tuple, buffers}) do
    List.to_tuple(Enum.map(buffers, &buffer_to_nx/1))
  end

  defp buffer_to_nx(other) do
    raise "invalid defn return type, make sure defn returns a tuple or a tensor, " <>
            "got: #{inspect(other)}"
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

  defp nx_to_cache_key!(number) when is_integer(number), do: {{:s, 64}, {}}
  defp nx_to_cache_key!(number) when is_float(number), do: {{:f, 64}, {}}
  defp nx_to_cache_key!(%Nx.Tensor{} = t), do: {t.type, t.shape}

  defp nx_to_cache_key!(other) do
    raise ArgumentError,
          "defn functions expects either numbers or Nx.Tensor's as arguments, " <>
            "got: #{inspect(other)}"
  end

  ## Special forms

  def sf_nx_tensor(builder, %Nx.Tensor{data: {Nx.BitStringDevice, data}, type: type, shape: shape}) do
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.constant_from_binary(builder, data, shape)
  end

  ## Operators

  def nx_bin_float_arith_op(builder, op, left, right) do
    type = binary_op_type(left, right) |> Nx.Type.to_floating()
    {left, left_dims} = to_typed_operator(builder, left, type)
    {right, right_dims} = to_typed_operator(builder, right, type)
    dims = broadcast_dimensions(left_dims, right_dims)
    apply(Exla.Op, op, [left, right, dims])
  end

  def nx_bin_arith_op(builder, op, left, right) do
    type = binary_op_type(left, right)
    {left, left_dims} = to_typed_operator(builder, left, type)
    {right, right_dims} = to_typed_operator(builder, right, type)
    dims = broadcast_dimensions(left_dims, right_dims)
    apply(Exla.Op, op, [left, right, dims])
  end

  def nx_bin_bitwise_op(builder, :right_shift, left, right) do
    op =
      # Perform logical operation if the left side is unsigned.
      # It can only be unsigned if it is an operator (numbers are always floats or signed).
      if match?(%Exla.Op{}, left) and match?({:u, _}, Exla.Op.get_shape(left).dtype),
        do: :right_shift_logical,
        else: :right_shift_arithmetic

    nx_bin_bitwise_op(builder, op, left, right)
  end

  def nx_bin_bitwise_op(builder, op, left, right) do
    type = binary_op_type(left, right) |> assert_integer_type!(op)
    {left, left_dims} = to_typed_operator(builder, left, type)
    {right, right_dims} = to_typed_operator(builder, right, type)
    dims = broadcast_dimensions(left_dims, right_dims)
    apply(Exla.Op, op, [left, right, dims])
  end

  def nx_unary_float_op(builder, op, arg) do
    apply(Exla.Op, op, [to_float_operator(builder, arg)])
  end

  def nx_unary_bitwise_op(builder, op, arg) do
    arg = to_operator(builder, arg)
    assert_integer_type!(Exla.Op.get_shape(arg).dtype, op)
    apply(Exla.Op, op, [arg])
  end

  def nx_unary_noop_integer_op(builder, op, arg) do
    arg = to_operator(builder, arg)

    case Exla.Op.get_shape(arg).dtype do
      {:s, _} -> arg
      {:u, _} -> arg
      _ -> apply(Exla.Op, op, [arg])
    end
  end

  def nx_abs(builder, arg) do
    arg = to_operator(builder, arg)

    case Exla.Op.get_shape(arg).dtype do
      {:u, _} -> arg
      _ -> Exla.Op.abs(arg)
    end
  end

  def nx_sign(builder, arg) do
    arg = to_operator(builder, arg)

    case Exla.Op.get_shape(arg).dtype do
      {:u, _} = type -> Exla.Op.min(arg, Exla.Op.constant_r0(builder, 1, type))
      _ -> Exla.Op.sign(arg)
    end
  end

  def nx_negate(_builder, %Exla.Op{} = op), do: Exla.Op.negate(op)
  def nx_negate(_builder, number) when is_number(number), do: -number

  ## Random

  def nx_random_uniform(builder, shape, min, max, opts)
      when is_number(min) and is_number(max) and is_tuple(shape) do
    type = opts[:type] || Nx.Type.infer(max - min)

    if match?({int, size} when int in [:s, :u] and size < 32, type) do
      raise ArgumentError,
            "Nx.random_uniform/4 for Exla requires signed and unsigned tensors to be " <>
              "at least of size 32, got: #{elem(type, 1)}"
    end

    {min, _} = to_typed_operator(builder, min, type)
    {max, _} = to_typed_operator(builder, max, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_uniform(min, max, shape)
  end

  def nx_random_uniform(builder, tensor, min, max, opts) do
    nx_random_uniform(builder, nx_shape(builder, tensor), min, max, opts)
  end

  def nx_random_normal(builder, shape, mu, sigma, opts)
      when is_float(mu) and is_float(sigma) and is_tuple(shape) do
    type = opts[:type] || {:f, 64}
    {mu, _} = to_typed_operator(builder, mu, type)
    {sigma, _} = to_typed_operator(builder, sigma, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_normal(mu, sigma, shape)
  end

  def nx_random_normal(builder, tensor, min, max, opts) do
    nx_random_normal(builder, nx_shape(builder, tensor), min, max, opts)
  end

  ## Reflection

  def nx_type(builder, op) do
    Exla.Op.get_shape(to_operator(builder, op)).dtype
  end

  def nx_shape(builder, op) do
    Exla.Op.get_shape(to_operator(builder, op)).dims
  end

  def nx_rank(builder, op) do
    tuple_size(nx_shape(builder, op))
  end

  def nx_size(builder, op) do
    tuple_product(nx_shape(builder, op))
  end

  ## Aggregators

  def nx_sum(builder, op, opts) do
    Exla.Lib.sum(builder, to_operator(builder, op), opts)
  end

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

  defp to_float_operator(_builder, %Exla.Op{} = op) do
    shape = Exla.Op.get_shape(op)
    type = Nx.Type.to_floating(shape.dtype)
    if shape.dtype != type, do: Exla.Op.convert_element_type(op, type), else: op
  end

  defp to_float_operator(builder, constant) when is_number(constant) do
    Exla.Op.constant_r0(builder, constant, {:f, 64})
  end

  defp to_float_operator(_builder, other) do
    raise ArgumentError, "expected a tensor, got: #{other}"
  end

  defp to_typed_operator(_builder, %Exla.Op{} = op, type) do
    shape = Exla.Op.get_shape(op)

    if shape.dtype != type do
      {Exla.Op.convert_element_type(op, type), shape.dims}
    else
      {op, shape.dims}
    end
  end

  defp to_typed_operator(builder, constant, type) when is_number(constant) do
    {Exla.Op.constant_r0(builder, constant, type), {}}
  end

  defp to_typed_operator(_builder, other, _type) do
    raise ArgumentError, "expected a tensor, got: #{other}"
  end

  ## Types

  defp binary_op_type(left, right) when is_number(left) and is_number(right),
    do: Nx.Type.merge(Nx.Type.infer(left), Nx.Type.infer(right))

  defp binary_op_type(scalar, op) when is_number(scalar),
    do: Nx.Type.merge_scalar(Exla.Op.get_shape(op).dtype, scalar)

  defp binary_op_type(op, scalar) when is_number(scalar),
    do: Nx.Type.merge_scalar(Exla.Op.get_shape(op).dtype, scalar)

  defp binary_op_type(left, right),
    do: Nx.Type.merge(Exla.Op.get_shape(left).dtype, Exla.Op.get_shape(right).dtype)

  defp assert_integer_type!({:s, _} = type, _op), do: type
  defp assert_integer_type!({:u, _} = type, _op), do: type

  defp assert_integer_type!(type, op) do
    raise ArgumentError,
          "#{op} expects integer tensors as inputs and outputs an integer tensor, " <>
            "got: #{inspect(type)}"
  end

  ## Constants

  defp to_constant(builder, int) when is_integer(int),
    do: Exla.Op.constant_r0(builder, int, {:s, 64})

  defp to_constant(builder, float) when is_float(float),
    do: Exla.Op.constant_r0(builder, float, {:f, 64})

  ## Dimensions

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  defp broadcast_dimensions(left, right) do
    {min, max} = if left <= right, do: {left, right}, else: {right, left}
    min_size = tuple_size(min)
    max_size = tuple_size(max)
    # To reproduce Nx broadcast, we simply match the lower dimensions to the highest ones.
    List.to_tuple(Enum.reverse(count_down(min_size, max_size - min_size, [])))
  end

  defp count_down(0, _n, acc), do: acc
  defp count_down(i, n, acc), do: count_down(i - 1, n + 1, [n | acc])

  ## Callback

  def __compile__(_kind, _meta, name, arity, vars, ast, options) do
    {ast, _state} = traverse(ast, %{})
    shapes = Macro.generate_arguments(length(vars), __MODULE__)

    quote do
      Exla.Defn.sf_cached_def(
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
    {{:., dot_meta, [__MODULE__, fun]}, meta, [quote(do: builder) | args]}
  end
end

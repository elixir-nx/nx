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

  def sf_cached_def(module, name, arity, args, options, fun) do
    cache_args = for arg <- args, do: nx_to_cache_key!(arg)
    buffers = for arg <- args, do: nx_to_buffer(arg)
    {client_name, options} = Keyword.pop(options, :client, :default)
    cache_key = {module, name, arity, cache_args, client_name}

    executable =
      Exla.LockedCache.run(cache_key, fn ->
        shapes = Enum.map(buffers, & &1.shape)
        builder = Exla.Builder.new("#{name}/#{arity}")
        result = fun.(builder, shapes)
        computation = Exla.Builder.build(to_operator(builder, result))
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
  # TODO: What to do when the tensor data is not a binary?

  defp buffer_to_nx(%Exla.Buffer{ref: nil, data: data, shape: shape}) do
    %Nx.Tensor{data: {Nx.BitStringDevice, data}, type: shape.dtype, shape: shape.dims}
  end

  defp buffer_to_nx(%Exla.Buffer{ref: ref, data: nil, shape: shape}) do
    %Nx.Tensor{data: {Exla.NxDevice, ref}, type: shape.dtype, shape: shape.dims}
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
          "defn functions expects either numbers or %Nx.Tensor{} as arguments, " <>
            "got: #{inspect(other)}"
  end

  ## Special forms

  def sf_nx_tensor(builder, %Nx.Tensor{data: {Nx.BitStringDevice, data}, type: type, shape: shape}) do
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.constant_from_binary(builder, data, shape)
  end

  ## Operators

  def nx_add(builder, left, right) do
    left = to_operator(builder, left)
    right = to_operator(builder, right)
    left_shape = Exla.Op.get_shape(left)
    right_shape = Exla.Op.get_shape(right)
    dims = broadcast_dimensions(left_shape.dims, right_shape.dims)
    Exla.Op.add(left, right, dims)
  end

  def nx_divide(builder, left, right) do
    left = to_operator(builder, left)
    right = to_operator(builder, right)
    Exla.Op.div(left, right)
  end

  def nx_exp(builder, op) do
    Exla.Op.exp(to_operator(builder, op))
  end

  def nx_sum(builder, op) do
    op = to_operator(builder, op)
    op_shape = Exla.Op.get_shape(op)
    reduction_shape = Exla.Shape.make_shape(op_shape.dtype, {})

    # Build the anonymous function
    unique = System.unique_integer([:positive])
    sub_builder = Exla.Builder.new(builder, builder.name <> "-sum-" <> Integer.to_string(unique))
    a = Exla.Op.parameter(sub_builder, 0, reduction_shape, "a")
    b = Exla.Op.parameter(sub_builder, 1, reduction_shape, "b")
    add = Exla.Op.add(a, b)
    reduction = Exla.Builder.build(add)

    init_value = to_typed_constant(builder, 0, reduction_shape.dtype)
    Exla.Op.reduce(op, init_value, reduction, all_dimensions(op_shape.dims))
  end

  # TODO: to_operator should actually call to_typed_constant
  # Implement this properly once we use convert_element_type.
  defp to_operator(_builder, %Exla.Op{} = op), do: op
  defp to_operator(builder, constant), do: to_constant(builder, constant)

  ## Constants

  defp to_typed_constant(builder, constant, type) when is_number(constant) do
    constant = Nx.Type.cast_scalar!(constant, type)
    Exla.Op.constant_r0(builder, constant, type)
  end

  defp to_typed_constant(_builder, other, _type) do
    raise(ArgumentError, "cannot compile constant #{inspect(other)}")
  end

  defp to_constant(builder, int) when is_integer(int),
    do: Exla.Op.constant_r0(builder, int, {:s, 64})

  defp to_constant(builder, float) when is_float(float),
    do: Exla.Op.constant_r0(builder, float, {:f, 64})

  defp to_constant(_builder, other),
    do: raise(ArgumentError, "cannot compile constant #{inspect(other)}")

  ## Dimensions

  # Converts {3, 255, 255} into {0, 1, 2}
  defp all_dimensions(dims), do: List.to_tuple(all_dimensions(0, tuple_size(dims)))
  defp all_dimensions(i, n) when i < n, do: [i | all_dimensions(i + 1, n)]
  defp all_dimensions(_, _), do: []

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

  def __compile__(_kind, _meta, name, args, ast, options) do
    state = %{}

    {ast, _state} = traverse(ast, state)
    arity = length(args)
    shapes = Macro.generate_arguments(arity, __MODULE__)

    quote do
      Exla.Defn.sf_cached_def(
        __MODULE__,
        unquote(name),
        unquote(arity),
        unquote(args),
        unquote(options),
        fn builder, unquote(shapes) ->
          unquote_splicing(args_to_parameters(args, shapes))
          unquote(ast)
        end
      )
    end
  end

  defp traverse({{:., dot_meta, [Nx, name]}, meta, args}, state) do
    {args, state} = traverse(args, state)
    {to_builder_call(dot_meta, meta, :"nx_#{name}", args), state}
  end

  defp traverse({:%{}, meta, [__struct__: Nx.Tensor] ++ _} = struct, state) do
    {to_builder_call(meta, :sf_nx_tensor, [struct]), state}
  end

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

  defp args_to_parameters(args, shapes) do
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

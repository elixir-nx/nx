defmodule Exla.Defn do
  @moduledoc false

  def __compile__(env, _kind, vars, fun, options) do
    %{module: module, function: {name, arity}} = env
    cache_args = for var <- vars, do: nx_to_cache_key!(var)
    buffers = for var <- vars, do: nx_to_buffer(var)

    # TODO: We should extract the client and device ordinal from buffers first
    # TODO: Rename :client to :default_client
    # TODO: Client_name plus device_ordinal must be part of the cache key
    {client_name, options} = Keyword.pop(options, :client, :default)
    cache_key = {module, name, arity, cache_args, client_name}

    executable =
      Exla.LockedCache.run(cache_key, fn ->
        builder = Exla.Builder.new("#{name}/#{arity}")

        params =
          for {buffer, i} <- Enum.with_index(buffers) do
            param = Exla.Op.parameter(builder, i, buffer.shape, "p#{i}")
            Nx.Defn.Expr.parameter(buffer.shape.dims, param)
          end

        computation =
          fun.(params)
          |> to_result(builder, %{})
          |> elem(0)
          |> Exla.Builder.build()

        client = Exla.Client.fetch!(client_name)
        executable = Exla.Client.compile(client, computation, Enum.map(buffers, & &1.shape))
        :persistent_term.put(cache_key, executable)
        executable
      end)

    executable
    |> Exla.Executable.run(buffers, options)
    |> buffer_to_nx()
  end

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

  ## to_operator

  defp to_operator(:tensor, [%Nx.Tensor{type: type, data: {_, data}}], shape, builder) do
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.constant_from_binary(builder, data, shape)
  end

  @bin_arith_op [:add, :subtract, :multiply, :divide, :min, :max, :remainder, :power]

  defp to_operator(op, [{left_expr, left}, {right_expr, right}], _shape, builder)
       when op in @bin_arith_op do
    {left, right} = binary_op_type(builder, left, right)
    dims = broadcast_dimensions(left_expr.shape, right_expr.shape)
    apply(Exla.Op, op, [left, right, dims])
  end

  ## constant/operator

  defp to_operator(_builder, %Exla.Op{} = op),
    do: op

  defp to_operator(builder, int) when is_integer(int),
    do: Exla.Op.constant_r0(builder, int, {:s, 64})

  defp to_operator(builder, float) when is_float(float),
    do: Exla.Op.constant_r0(builder, float, {:f, 64})

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

  defp binary_op_type(builder, left_op, right_op) do
    left_type = constant_or_type(left_op)
    right_type = constant_or_type(right_op)
    output_type = binary_op_type(left_type, right_type)

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
  defp constant_or_type(op), do: Exla.Op.get_shape(op).dtype

  ## Nx <-> Exla.Buffer

  defp buffer_to_nx(%Exla.Buffer{ref: nil, data: data, shape: shape}) do
    %Nx.Tensor{
      data: {Nx.BitStringDevice, data},
      type: Exla.Type.to_nx(shape.dtype),
      shape: shape.dims
    }
  end

  defp buffer_to_nx(%Exla.Buffer{ref: ref, data: nil, shape: shape}) do
    %Nx.Tensor{data: {Exla.NxDevice, ref}, type: Exla.Type.to_nx(shape.dtype), shape: shape.dims}
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

  defp nx_to_cache_key!(arg) when is_tuple(arg) do
    raise ArgumentError,
          "defn functions expects either numbers or %Nx.Tensor{} as arguments. " <>
            "If you want to pass a tuple, you must explicitly pattern match on the tuple in the signature. " <>
            "Got: #{inspect(arg)}"
  end

  defp nx_to_cache_key!(arg) do
    raise ArgumentError,
          "defn functions expects either numbers or %Nx.Tensor{} as arguments. " <>
            "Got: #{inspect(arg)}"
  end
end

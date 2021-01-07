defmodule Exla.Defn do
  @moduledoc false

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

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
          for {%{shape: shape}, i} <- Enum.with_index(buffers) do
            param = Exla.Op.parameter(builder, i, shape, "p#{i}")
            Nx.Defn.Expr.parameter(shape.dims, shape.dtype, param)
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
    recur_operator(expr, builder, cache)
  end

  defp recur_operator(%T{data: %Expr{op: :parameter, args: [param]}}, _builder, cache) do
    {param, cache}
  end

  defp recur_operator(%T{data: %Expr{id: id, op: op, args: args}} = ans, builder, cache) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {ops, cache} =
          Enum.map_reduce(args, cache, fn
            %T{data: %Expr{}} = arg, cache -> recur_operator(arg, builder, cache)
            arg, cache -> {arg, cache}
          end)

        op = to_operator(op, ops, ans, builder)
        {op, Map.put(cache, id, op)}
    end
  end

  ## to_operator creation

  defp to_operator(:tensor, [tensor], _ans, builder) do
    case tensor.shape do
      {} ->
        to_constant(builder, Nx.Util.to_scalar(tensor), tensor.type)

      shape ->
        shape = Exla.Shape.make_shape(tensor.type, shape)
        Exla.Op.constant_from_binary(builder, Nx.to_binary(tensor), shape)
    end
  end

  defp to_operator(:random_uniform, [shape, min, max, _opts], ans, builder) do
    type = ans.type

    if match?({int, size} when int in [:s, :u] and size < 32, type) do
      raise ArgumentError,
            "Nx.random_uniform/4 for Exla requires signed and unsigned tensors to be " <>
              "at least of size 32, got: #{elem(type, 1)}"
    end

    min = to_constant(builder, min, type)
    max = to_constant(builder, max, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_uniform(min, max, shape)
  end

  defp to_operator(:random_normal, [shape, mu, sigma, _opts], ans, builder) do
    type = ans.type
    mu = to_constant(builder, mu, type)
    sigma = to_constant(builder, sigma, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_normal(mu, sigma, shape)
  end

  defp to_operator(:iota, [shape, opts], ans, builder) do
    shape = Exla.Shape.make_shape(ans.type, shape)
    Exla.Lib.iota(builder, shape, opts)
  end

  ## to_operator shape

  defp to_operator(:reshape, [op, shape], _ans, _builder) do
    Exla.Op.reshape(op, shape)
  end

  defp to_operator(:pad, [op, value, padding_config], _ans, _builder) do
    Exla.Op.pad(op, value, padding_config)
  end

  defp to_operator(:broadcast, [op, _shape, axes], ans, _builder) do
    Exla.Op.broadcast_in_dim(op, ans.shape, List.to_tuple(axes))
  end

  defp to_operator(:transpose, [op, dims], _ans, _builder) do
    Exla.Op.transpose(op, List.to_tuple(dims))
  end

  defp to_operator(:squeeze, [op, _axes], ans, _builder) do
    Exla.Op.reshape(op, ans.shape)
  end

  ## to_operator others

  defp to_operator(:dot, [left, axes1, right, axes2], %{type: type}, _builder) do
    Exla.Op.dot_general(to_type(left, type), to_type(right, type), {axes1, axes2})
  end

  defp to_operator(
         :conv,
         [operand, kernel, strides, padding],
         %{shape: shape},
         _builder
       ) do
    rank = tuple_size(shape)
    # Build general conv dims
    input_dims = List.to_tuple(for i <- 0..(rank - 1), do: i)
    [out_features, in_features | kernel_spatial] = for i <- 0..(rank - 1), do: i
    kernel_dims = List.to_tuple([in_features, out_features | kernel_spatial])
    output_dims = input_dims

    conv_dim_nos = {input_dims, kernel_dims, output_dims}

    # No dilation for now
    lhs_dilation = List.to_tuple(for _ <- 0..(rank - 3), do: 1)
    rhs_dilation = lhs_dilation

    padding_config =
      case padding do
        :valid ->
          for _ <- 0..(rank - 3), do: {0, 0}

        :same ->
          :ok

        padding when is_list(padding) ->
          padding
      end

    Exla.Op.conv_general_dilated(
      operand,
      kernel,
      strides,
      padding_config,
      lhs_dilation,
      rhs_dilation,
      conv_dim_nos
    )
  end

  defp to_operator(:outer, [left, right], %{type: type, shape: shape}, _builder) do
    left =
      left
      |> to_type(type)
      |> Exla.Op.reshape({Nx.size(op_shape(left))})
      |> Exla.Op.broadcast_in_dim(shape, {0})

    right =
      right
      |> to_type(type)
      |> Exla.Op.reshape({Nx.size(op_shape(right))})
      |> Exla.Op.broadcast_in_dim(shape, {1})

    Exla.Op.multiply(left, right)
  end

  defp to_operator(:select, [pred, on_true, on_false], %{type: type, shape: shape}, _builder) do
    pred = to_type(pred, {:pred, 8})

    on_true =
      on_true
      |> to_type(type)
      |> Exla.Op.broadcast_in_dim(shape, broadcast_axes(op_shape(on_true), shape))

    on_false =
      on_false
      |> to_type(type)
      |> Exla.Op.broadcast_in_dim(shape, broadcast_axes(op_shape(on_false), shape))

    Exla.Op.select(pred, on_true, on_false)
  end

  ## to_operator element-wise

  defp to_operator(:negate, [op], _ans, _builder), do: Exla.Op.negate(op)

  defp to_operator(:abs, [op], _ans, _builder), do: Exla.Op.abs(op)

  defp to_operator(:sign, [op], %{type: type}, builder) do
    case type do
      {:u, _} -> Exla.Op.min(op, Exla.Op.constant_r0(builder, 1, type))
      _ -> Exla.Op.sign(op)
    end
  end

  defp to_operator(:right_shift, [left, right], %{type: type}, _builder) do
    dims = broadcast_axes(op_shape(left), op_shape(right))

    op =
      if match?({:u, _}, type),
        do: :right_shift_logical,
        else: :right_shift_arithmetic

    apply(Exla.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @bin_op [:add, :subtract, :multiply, :min, :max, :remainder, :power, :divide, :arctan2] ++
            [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift]

  defp to_operator(op, [left, right], %{type: type}, _builder) when op in @bin_op do
    dims = broadcast_axes(op_shape(left), op_shape(right))
    apply(Exla.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @bin_comp_op [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

  defp to_operator(op, [left, right], _ans, _builder) when op in @bin_comp_op do
    left_shape = Exla.Op.get_shape(left)
    right_shape = Exla.Op.get_shape(right)
    type = Nx.Type.merge(left_shape.dtype, right_shape.dtype)
    dims = broadcast_axes(left_shape.dims, right_shape.dims)
    apply(Exla.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @unary_op [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
              [:bitwise_not, :count_leading_zeros, :population_count] ++
              [:floor, :ceil, :round]

  defp to_operator(op, [arg], %{type: type}, _builder) when op in @unary_op do
    apply(Exla.Op, op, [to_type(arg, type)])
  end

  ## to_operator reduction

  @reduction_op [:sum, :argmax, :argmin]

  defp to_operator(op, [arg, opts], ans, builder)
       when op in @reduction_op do
    apply(Exla.Lib, op, [builder, arg, [type: ans.type] ++ opts])
  end

  ## Axes helpers

  defp broadcast_axes(left, right) do
    {min, max} = if left <= right, do: {left, right}, else: {right, left}
    min_size = tuple_size(min)
    max_size = tuple_size(max)

    # To reproduce Nx broadcast, we simply match the lower dimensions to the highest ones.
    List.to_tuple(count_up(min_size, max_size - min_size))
  end

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  ## Op Helpers

  defp op_type(op), do: Exla.Op.get_shape(op).dtype
  defp op_shape(op), do: Exla.Op.get_shape(op).dims

  defp to_type(op, type) do
    if op_type(op) == type, do: op, else: Exla.Op.convert_element_type(op, type)
  end

  defp to_constant(builder, constant, type) do
    Exla.Op.constant_r0(builder, constant, type)
  end

  ## Nx <-> Exla.Buffer

  defp buffer_to_nx(%Exla.Buffer{ref: nil, data: data, shape: shape}) do
    data
    |> Nx.from_binary(to_nx_type(shape.dtype))
    |> Map.replace!(:shape, shape.dims)
  end

  defp buffer_to_nx(%Exla.Buffer{ref: ref, data: nil, shape: shape}) do
    %Nx.Tensor{
      data: %Nx.BinaryTensor{device: Exla.NxDevice, state: ref},
      type: to_nx_type(shape.dtype),
      shape: shape.dims
    }
  end

  defp buffer_to_nx({:tuple, buffers}) do
    List.to_tuple(Enum.map(buffers, &buffer_to_nx/1))
  end

  defp buffer_to_nx(other) do
    raise "invalid defn return type, make sure defn returns a tuple or a tensor, " <>
            "got: #{inspect(other)}"
  end

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  defp nx_to_buffer(%Nx.Tensor{data: data, type: type, shape: shape} = tensor) do
    case data do
      %Nx.BinaryTensor{device: Exla.NxDevice, state: state} ->
        Exla.Buffer.buffer(state, Exla.Shape.make_shape(type, shape))

      _ ->
        Exla.Buffer.buffer(Nx.to_binary(tensor), Exla.Shape.make_shape(type, shape))
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

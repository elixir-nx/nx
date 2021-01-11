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

    {executable, holes} =
      Exla.LockedCache.run(cache_key, fn ->
        builder = Exla.Builder.new("#{name}/#{arity}")

        params =
          for {%{shape: shape}, i} <- Enum.with_index(buffers) do
            Exla.Op.parameter(builder, i, shape, "p#{i}")
          end

        state = %{
          precision: Keyword.get(options, :precision, :default),
          builder: builder,
          params: params
        }

        expr = apply(fun, vars)

        computation =
          expr
          |> to_result(state, %{})
          |> elem(0)
          |> Exla.Builder.build()

        client = Exla.Client.fetch!(client_name)
        executable = Exla.Client.compile(client, computation, Enum.map(buffers, & &1.shape))
        :persistent_term.put(cache_key, executable)
        {executable, holes(expr)}
      end)

    executable
    |> Exla.Executable.run(buffers, options)
    |> buffer_to_nx(holes)
  end

  defp holes(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&holes/1)

  defp holes(%T{} = t),
    do: %{t | data: nil}

  defp to_result(tuple, state, cache) when is_tuple(tuple) do
    {elements, cache} =
      tuple
      |> Tuple.to_list()
      |> Enum.map_reduce(cache, &to_result(&1, state, &2))

    {Exla.Op.tuple(state.builder, elements), cache}
  end

  defp to_result(expr, state, cache) do
    recur_operator(expr, state, cache)
  end

  defp recur_operator(%T{data: %Expr{op: :parameter, args: [i]}}, state, cache) do
    {Enum.fetch!(state.params, i), cache}
  end

  defp recur_operator(%T{data: %Expr{op: :fun}} = t, _state, cache) do
    {t, cache}
  end

  defp recur_operator(%T{data: %Expr{id: id, op: op, args: args}} = ans, state, cache) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        {ops, cache} =
          Enum.map_reduce(args, cache, fn
            %T{data: %Expr{}} = arg, cache -> recur_operator(arg, state, cache)
            arg, cache -> {arg, cache}
          end)

        op = to_operator(op, ops, ans, state)
        {op, Map.put(cache, id, op)}
    end
  end

  ## to_operator creation

  defp to_operator(:tensor, [tensor], _ans, state) do
    case tensor.shape do
      {} ->
        to_constant(state.builder, Nx.Util.to_scalar(tensor), tensor.type)

      shape ->
        shape = Exla.Shape.make_shape(tensor.type, shape)
        Exla.Op.constant_from_binary(state.builder, Nx.to_binary(tensor), shape)
    end
  end

  defp to_operator(:random_uniform, [min, max], %{type: type, shape: shape}, state) do
    if match?({int, size} when int in [:s, :u] and size < 32, type) do
      raise ArgumentError,
            "Nx.random_uniform/4 for Exla requires signed and unsigned tensors to be " <>
              "at least of size 32, got: #{elem(type, 1)}"
    end

    min = to_constant(state.builder, min, type)
    max = to_constant(state.builder, max, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_uniform(min, max, shape)
  end

  defp to_operator(:random_normal, [mu, sigma], %{type: type, shape: shape}, state) do
    mu = to_constant(state.builder, mu, type)
    sigma = to_constant(state.builder, sigma, type)
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Op.rng_normal(mu, sigma, shape)
  end

  defp to_operator(:iota, [axis], %{type: type, shape: shape}, state) do
    shape = Exla.Shape.make_shape(type, shape)
    Exla.Lib.iota(state.builder, shape, axis)
  end

  ## to_operator shape

  defp to_operator(:reshape, [op, shape], _ans, _state) do
    Exla.Op.reshape(op, shape)
  end

  defp to_operator(:pad, [op, value, padding_config], _ans, _state) do
    Exla.Op.pad(op, value, padding_config)
  end

  defp to_operator(:broadcast, [op, _shape, axes], ans, _state) do
    Exla.Op.broadcast_in_dim(op, ans.shape, List.to_tuple(axes))
  end

  defp to_operator(:transpose, [op, dims], _ans, _state) do
    Exla.Op.transpose(op, List.to_tuple(dims))
  end

  defp to_operator(:squeeze, [op, _axes], ans, _state) do
    Exla.Op.reshape(op, ans.shape)
  end

  ## to_operator others

  defp to_operator(:dot, [left, axes1, right, axes2], %{type: type}, state) do
    precision = state.precision
    Exla.Op.dot_general(to_type(left, type), to_type(right, type), {axes1, axes2}, precision)
  end

  defp to_operator(
         :conv,
         [operand, kernel, strides, padding, input_dilation, kernel_dilation],
         ans,
         state
       ) do
    %{type: output_type, shape: shape} = ans
    rank = tuple_size(shape)

    # Build general conv dims
    input_dims = List.to_tuple(for i <- 0..(rank - 1), do: i)
    [out_features, in_features | kernel_spatial] = for i <- 0..(rank - 1), do: i
    kernel_dims = List.to_tuple([in_features, out_features | kernel_spatial])
    output_dims = input_dims

    conv_dim_nos = {input_dims, kernel_dims, output_dims}

    # Ensure both types are floating
    operand = to_type(operand, output_type)
    kernel = to_type(kernel, output_type)

    Exla.Op.conv_general_dilated(
      operand,
      kernel,
      strides,
      padding,
      input_dilation,
      kernel_dilation,
      conv_dim_nos,
      state.precision
    )
  end

  defp to_operator(:outer, [left, right], %{type: type, shape: shape}, _state) do
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

  defp to_operator(:select, [pred, on_true, on_false], %{type: type, shape: shape}, _state) do
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

  defp to_operator(:negate, [op], _ans, _state), do: Exla.Op.negate(op)

  defp to_operator(:abs, [op], _ans, _state), do: Exla.Op.abs(op)

  defp to_operator(:sign, [op], %{type: type}, state) do
    case type do
      {:u, _} -> Exla.Op.min(op, Exla.Op.constant_r0(state.builder, 1, type))
      _ -> Exla.Op.sign(op)
    end
  end

  defp to_operator(:right_shift, [left, right], %{type: type}, _state) do
    dims = broadcast_axes(op_shape(left), op_shape(right))

    op =
      if match?({:u, _}, type),
        do: :right_shift_logical,
        else: :right_shift_arithmetic

    apply(Exla.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @bin_op [:add, :subtract, :multiply, :min, :max, :remainder, :power, :divide, :arctan2] ++
            [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift]

  defp to_operator(op, [left, right], %{type: type}, _state) when op in @bin_op do
    dims = broadcast_axes(op_shape(left), op_shape(right))
    apply(Exla.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @bin_comp_op [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

  defp to_operator(op, [left, right], _ans, _state) when op in @bin_comp_op do
    left_shape = Exla.Op.get_shape(left)
    right_shape = Exla.Op.get_shape(right)
    type = Nx.Type.merge(left_shape.dtype, right_shape.dtype)
    dims = broadcast_axes(left_shape.dims, right_shape.dims)
    apply(Exla.Op, op, [to_type(left, type), to_type(right, type), dims])
  end

  @unary_op [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
              [:bitwise_not, :count_leading_zeros, :population_count] ++
              [:floor, :ceil, :round]

  defp to_operator(op, [arg], %{type: type}, _state) when op in @unary_op do
    apply(Exla.Op, op, [to_type(arg, type)])
  end

  ## to_operator reduction

  defp to_operator(:sum, [arg, opts], %{type: type} = ans, state) do
    acc = Exla.Op.constant_r0(state.builder, 0, type)
    args = [Expr.parameter(type, {}, 0), Expr.parameter(type, {}, 1)]
    to_operator(:reduce, [arg, acc, opts, Expr.fun(:sum, args, &Nx.add/2)], ans, state)
  end

  defp to_operator(:reduce, [arg, acc, opts, fun], %{type: type}, state) do
    arg = to_type(arg, type)
    comp = to_computation(fun, state)
    Exla.Op.reduce(arg, to_type(acc, type), comp, reduce_axes(arg, opts[:axes]))
  end

  @reduction_op [:argmax, :argmin]

  defp to_operator(op, [arg, opts], ans, state)
       when op in @reduction_op do
    apply(Exla.Lib, op, [state.builder, arg, [type: ans.type] ++ opts])
  end

  defp to_operator(:clip, [operand, min, max], ans, _state) do
    min = to_type(min, ans.type)
    max = to_type(max, ans.type)
    operand = to_type(operand, ans.type)

    Exla.Op.clamp(operand, min, max)
  end

  defp to_operator(:slice, [tensor, start_indices, limit_indices, strides], _ans, _state) do
    Exla.Op.slice(tensor, start_indices, limit_indices, strides)
  end

  ## Computation helpers

  defp to_computation(%T{data: %Expr{op: :fun, args: [name, args, expr, _]}} = ans, state) do
    subbuilder = subbuilder(state.builder, "#{name}")

    params =
      for {%{type: type, shape: shape}, i} <- Enum.with_index(args) do
        fun_shape = Exla.Shape.make_shape(type, shape)
        Exla.Op.parameter(subbuilder, i, fun_shape, "p#{i}")
      end

    expr
    |> to_result(%{state | builder: subbuilder, params: params}, %{})
    |> elem(0)
    |> to_type(ans.type)
    |> Exla.Builder.build()
  end

  defp subbuilder(%Exla.Builder{name: name} = builder, desc) do
    suffix = System.unique_integer([:positive])
    Exla.Builder.new(builder, name <> "-" <> desc <> "-" <> Integer.to_string(suffix))
  end

  ## Axes helpers

  defp broadcast_axes(left, right) do
    {min, max} = if left <= right, do: {left, right}, else: {right, left}
    min_size = tuple_size(min)
    max_size = tuple_size(max)

    # To reproduce Nx broadcast, we simply match the lower dimensions to the highest ones.
    List.to_tuple(count_up(min_size, max_size - min_size))
  end

  defp reduce_axes(op, axes) do
    if axes do
      axes
      |> Enum.sort()
      |> List.to_tuple()
    else
      List.to_tuple(Nx.axes(op_shape(op)))
    end
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

  defp buffer_to_nx(%Exla.Buffer{shape: shape} = buffer, hole) do
    nx_type = to_nx_type(shape.dtype)
    nx_shape = shape.dims

    if hole.type != nx_type do
      raise "internal bug! Nx.Defn expected a tensor with type #{hole.type} " <>
              "but got #{inspect(nx_type)}"
    end

    if hole.shape != nx_shape do
      raise "internal bug! Nx.Defn expected a tensor with shape #{hole.shape} " <>
              "but got #{inspect(nx_shape)}"
    end

    %{hole | data: buffer_to_data(buffer)}
  end

  defp buffer_to_nx({:tuple, buffers}, holes) do
    # TODO: Use Enum.zip_with on Elixir v1.12
    buffers
    |> Enum.zip(holes)
    |> Enum.map(fn {buffer, hole} -> buffer_to_nx(buffer, hole) end)
    |> List.to_tuple()
  end

  defp buffer_to_data(%Exla.Buffer{ref: ref, data: nil}),
    do: %Nx.BinaryTensor{device: Exla.NxDevice, state: ref}

  defp buffer_to_data(%Exla.Buffer{ref: nil, data: data}),
    do: %Nx.BinaryTensor{device: Nx.BinaryDevice, state: data}

  defp to_nx_type({:pred, 8}), do: {:u, 8}
  defp to_nx_type(type), do: type

  defp nx_to_buffer(%T{data: data, type: type, shape: shape} = tensor) do
    case data do
      %Nx.BinaryTensor{device: Exla.NxDevice, state: state} ->
        Exla.Buffer.buffer(state, Exla.Shape.make_shape(type, shape))

      _ ->
        Exla.Buffer.buffer(Nx.to_binary(tensor), Exla.Shape.make_shape(type, shape))
    end
  end

  defp nx_to_cache_key!(%T{type: type, shape: shape, names: names}), do: {type, shape, names}
end

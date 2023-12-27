defmodule EXLA.MLIR.Value do
  @moduledoc """
  Representation of an MLIR Value.

  MLIR Values are SSA and generally are either operations or
  block arguments. This module is used to construct most of the
  MLIR operations.
  """
  defstruct [:ref, :function]

  alias __MODULE__, as: Value
  alias EXLA.MLIR.Function

  @bin_ops [:add, :subtract, :multiply, :divide, :pow, :min] ++
             [:max, :remainder, :atan2, :equal, :less, :less_equal] ++
             [:greater, :greater_equal, :not_equal, :bitwise_and] ++
             [:bitwise_or, :bitwise_xor] ++
             [:left_shift, :right_shift_arithmetic, :right_shift_logical]

  for op <- @bin_ops do
    mlir_op = :"mlir_#{op}"

    def unquote(op)(
          %Value{ref: lhs, function: %Function{} = func},
          %Value{ref: rhs, function: %Function{} = func}
        ) do
      ref = EXLA.NIF.unquote(mlir_op)(func.ref, lhs, rhs) |> unwrap!()
      %Value{ref: ref, function: func}
    end
  end

  @unary_ops [:abs, :exp, :expm1, :floor, :ceil, :round] ++
               [:log, :log1p, :sigmoid, :sign, :cos] ++
               [:sin, :tan, :acos, :asin, :atan, :cosh, :sinh] ++
               [:tanh, :acosh, :asinh, :atanh, :sqrt, :cbrt] ++
               [:bitwise_not, :erf, :erfc, :erf_inv] ++
               [:is_infinity, :is_nan, :rsqrt, :negate, :count_leading_zeros] ++
               [:population_count, :real, :imag, :conjugate]

  for op <- @unary_ops do
    mlir_op = :"mlir_#{op}"

    def unquote(op)(%Value{ref: operand, function: %Function{} = func}) do
      ref = EXLA.NIF.unquote(mlir_op)(func.ref, operand) |> unwrap!()
      %Value{ref: ref, function: func}
    end
  end

  def reshape(%Value{function: %Function{} = func} = op, shape_tuple) do
    ref = EXLA.NIF.mlir_reshape(func.ref, op.ref, shape_tuple) |> unwrap!()
    %Value{op | ref: ref}
  end

  def reverse(%Value{function: %Function{} = func} = op, dims) do
    ref = EXLA.NIF.mlir_reverse(func.ref, op.ref, dims) |> unwrap!()
    %Value{op | ref: ref}
  end

  def transpose(%Value{} = op, axes) when is_tuple(axes) do
    transpose(op, Tuple.to_list(axes))
  end

  def transpose(%Value{function: %Function{} = func} = op, axes) do
    ref = EXLA.NIF.mlir_transpose(func.ref, op.ref, axes) |> unwrap!()
    %Value{op | ref: ref}
  end

  def slice(%Value{function: %Function{} = func} = op, starts, limits, strides) do
    ref = EXLA.NIF.mlir_slice(func.ref, op.ref, starts, limits, strides) |> unwrap!()
    %Value{op | ref: ref}
  end

  def dynamic_slice(%Value{function: %Function{} = func} = op, starts, lengths) do
    starts = Enum.map(starts, fn %Value{ref: ref} -> ref end)
    ref = EXLA.NIF.mlir_dynamic_slice(func.ref, op.ref, starts, lengths) |> unwrap!()
    %Value{op | ref: ref}
  end

  def tuple(%Function{} = func, vals) when is_list(vals) do
    refs = Enum.map(vals, fn %Value{ref: ref} -> ref end)
    ref = EXLA.NIF.mlir_tuple(func.ref, refs) |> unwrap!()
    %Value{ref: ref, function: func}
  end

  def get_tuple_element(%Value{function: %Function{} = func, ref: ref}, index)
      when is_integer(index) do
    ref = EXLA.NIF.mlir_get_tuple_element(func.ref, ref, index) |> unwrap!()
    %Value{ref: ref, function: func}
  end

  def get_shape(%Value{ref: ref}) do
    shape_ref = EXLA.NIF.mlir_get_shape(ref) |> unwrap!()
    EXLA.Shape.get_shape_info(shape_ref)
  end

  def convert(%Value{ref: in_ref, function: %Function{} = func} = value, dtype) do
    out_ref =
      EXLA.NIF.mlir_convert(func.ref, in_ref, EXLA.Shape.dtype_to_charlist(dtype)) |> unwrap!()

    %Value{value | ref: out_ref}
  end

  def bitcast_convert(%Value{ref: in_ref, function: %Function{} = func} = value, dtype) do
    shape = get_shape(value)

    out_ref =
      EXLA.NIF.mlir_bitcast_convert(
        func.ref,
        in_ref,
        EXLA.Shape.dtype_to_charlist(dtype),
        shape.dims
      )
      |> unwrap!()

    %Value{value | ref: out_ref}
  end

  def top_k(%Value{function: %Function{ref: func_ref}, ref: op_ref} = val, k) do
    [val_ref, idx_ref] = EXLA.NIF.mlir_top_k(func_ref, op_ref, k) |> unwrap!()
    [%Value{val | ref: val_ref}, %Value{val | ref: idx_ref}]
  end

  def sort(%Value{} = value, comparator_fun, axis, stable) do
    [result] = sort([value], comparator_fun, axis, stable)
    result
  end

  def sort(
        [%Value{function: %Function{ref: func_ref}} | _] = values,
        %Function{ref: comparator_fun},
        axis,
        stable
      )
      when is_integer(axis) and is_boolean(stable) do
    stable = if stable, do: 1, else: 0

    in_refs =
      Enum.map(values, fn %Value{ref: ref, function: %Function{ref: ^func_ref}} -> ref end)

    out_refs =
      EXLA.NIF.mlir_sort(func_ref, in_refs, axis, comparator_fun, stable) |> unwrap!()

    Enum.zip_with(values, out_refs, fn value, out_ref -> %Value{value | ref: out_ref} end)
  end

  def iota(%Function{} = func, shape, dim) do
    ref = EXLA.NIF.mlir_iota(func.ref, shape.ref, dim) |> unwrap!()
    %Value{ref: ref, function: func}
  end

  def constant_r0(%Function{} = func, value, {:c, width} = type)
      when type in [{:c, 64}, {:c, 128}] do
    {re, im} =
      case value do
        %Complex{re: re, im: im} -> {re, im}
        n when is_float(n) -> {n, 0.0}
        n when is_integer(n) -> {n * 1.0, 0.0}
        true -> {1.0, 0.0}
        false -> {0.0, 0.0}
      end

    width = div(width, 2)

    data = <<re::float-native-size(width), im::float-native-size(width)>>

    ref =
      EXLA.NIF.mlir_constant_from_binary(
        func.ref,
        data,
        EXLA.Shape.dtype_to_charlist(type),
        {1}
      )
      |> unwrap!()

    reshape(%Value{ref: ref, function: func}, {})
  end

  def constant_r0(%Function{} = func, value, type)
      when value in [:infinity, :nan, :neg_infinity] do
    data =
      value
      |> Nx.tensor(backend: Nx.BinaryBackend, type: type)
      |> Nx.to_binary()

    ref =
      EXLA.NIF.mlir_constant_from_binary(
        func.ref,
        data,
        EXLA.Shape.dtype_to_charlist(type),
        {}
      )
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def constant_r0(%Function{} = func, value, type) do
    value =
      if Nx.Type.float?(type) and not is_float(value) do
        value * 1.0
      else
        value
      end

    ref =
      EXLA.NIF.mlir_constant_r0(func.ref, value, EXLA.Shape.dtype_to_charlist(type)) |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def constant_from_binary(%Function{} = func, data, shape) do
    ref =
      EXLA.NIF.mlir_constant_from_binary(
        func.ref,
        data,
        EXLA.Shape.dtype_to_charlist(shape.dtype),
        shape.dims
      )
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def dot_general(
        output_shape,
        %Value{function: func} = lhs,
        %Value{function: func} = rhs,
        dnums,
        precision_config
      ) do
    config = get_precision_config_int(precision_config)

    ref =
      EXLA.NIF.mlir_dot_general(func.ref, output_shape.ref, lhs.ref, rhs.ref, dnums, config)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def broadcast_in_dim(%Value{function: func} = operand, output_shape, axes) do
    ref =
      EXLA.NIF.mlir_broadcast_in_dim(func.ref, output_shape.ref, operand.ref, axes)
      |> unwrap!()

    %Value{function: func, ref: ref}
  end

  def concatenate([%Value{function: func} | _rest] = operands, dimension) do
    refs = Enum.map(operands, & &1.ref)

    ref =
      EXLA.NIF.mlir_concatenate(func.ref, refs, dimension)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def optimization_barrier(%Value{function: func} = operand) do
    ref =
      EXLA.NIF.mlir_optimization_barrier(func.ref, operand.ref)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def clamp(
        %Value{function: func} = operand,
        %Value{function: func} = min,
        %Value{function: func} = max
      ) do
    ref =
      EXLA.NIF.mlir_clamp(func.ref, operand.ref, min.ref, max.ref)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def select(
        %Value{function: func} = pred,
        %Value{function: func} = on_true,
        %Value{function: func} = on_false
      ) do
    ref =
      EXLA.NIF.mlir_select(func.ref, pred.ref, on_true.ref, on_false.ref)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def pad(%Value{function: func} = operand, %Value{function: func} = pad, padding_config) do
    {padding_low, padding_high, padding_mid} =
      Enum.reduce(padding_config, {[], [], []}, fn {low, high, mid},
                                                   {low_acc, high_acc, mid_acc} ->
        {[low | low_acc], [high | high_acc], [mid | mid_acc]}
      end)

    ref =
      EXLA.NIF.mlir_pad(
        func.ref,
        operand.ref,
        pad.ref,
        Enum.reverse(padding_low),
        Enum.reverse(padding_high),
        Enum.reverse(padding_mid)
      )
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def fft(%Value{function: func} = value, fft_kind, fft_length)
      when fft_kind in [:fft, :ifft]
      when is_list(fft_length) or is_integer(fft_length) do
    ref =
      EXLA.NIF.mlir_fft(
        func.ref,
        value.ref,
        if(fft_kind == :fft, do: 1, else: 0),
        List.wrap(fft_length)
      )
      |> unwrap!()

    %Value{value | ref: ref}
  end

  def scatter(
        %Value{function: func} = target,
        %Value{function: func} = indices,
        %Value{function: func} = updates,
        kind,
        indices_rank,
        update_window_dims,
        inserted_window_dims,
        index_dims_to_window_dims
      )
      when kind in [:add, :put] and is_integer(indices_rank) and is_list(update_window_dims) and
             is_list(inserted_window_dims) and is_list(index_dims_to_window_dims) do
    add_or_put = if(kind == :add, do: 1, else: 0)

    ref =
      EXLA.NIF.mlir_scatter(
        func.ref,
        target.ref,
        indices.ref,
        updates.ref,
        add_or_put,
        indices_rank,
        update_window_dims,
        inserted_window_dims,
        index_dims_to_window_dims
      )
      |> unwrap!()

    %Value{target | ref: ref}
  end

  def select_and_scatter(
        %Value{function: func} = target,
        %Value{function: func} = source,
        %Value{function: func} = init_value,
        comparison,
        window_dimensions,
        window_strides,
        padding
      )
      when comparison in [:gt, :lt] do
    gt_or_lt = if(comparison == :gt, do: 1, else: 0)

    ref =
      EXLA.NIF.mlir_select_and_scatter(
        func.ref,
        target.ref,
        source.ref,
        init_value.ref,
        gt_or_lt,
        window_dimensions,
        window_strides,
        padding
      )
      |> unwrap!()

    %Value{target | ref: ref}
  end

  def gather(
        %Value{function: func} = source,
        %Value{function: func} = indices,
        index_vector_dim,
        slice_sizes,
        offset_dims,
        collapsed_slice_dims,
        start_index_map
      ) do
    ref =
      EXLA.NIF.mlir_gather(
        func.ref,
        source.ref,
        indices.ref,
        slice_sizes,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        index_vector_dim
      )
      |> unwrap!()

    %Value{source | ref: ref}
  end

  defp get_precision_config_int(precision_config) do
    case precision_config do
      :default ->
        0

      :high ->
        1

      :highest ->
        2

      :packed_nibble ->
        3

      _ ->
        raise ArgumentError,
              "expected precision configuration to be one of" <>
                " :default, :high, :highest, or :packed_nibble," <>
                " got: #{inspect(precision_config)}"
    end
  end

  def convolution(
        %Value{function: func} = tensor,
        %Value{function: func} = kernel,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision_config,
        output_shape
      ) do
    precision_config = get_precision_config_int(precision_config)

    ref =
      EXLA.NIF.mlir_convolution(
        func.ref,
        tensor.ref,
        kernel.ref,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision_config,
        Tuple.to_list(output_shape)
      )
      |> unwrap!()

    %{tensor | ref: ref}
  end

  def triangular_solve(a, b, left_side, lower, transform) do
    ref =
      EXLA.NIF.mlir_triangular_solve(
        a.function.ref,
        a.ref,
        b.ref,
        if(left_side, do: 1, else: 0),
        if(lower, do: 1, else: 0),
        if(transform == :transpose, do: 1, else: 0)
      )
      |> unwrap!()

    %{a | ref: ref}
  end

  def dynamic_update_slice(operand, updates, starts) do
    ref =
      EXLA.NIF.mlir_dynamic_update_slice(
        operand.function.ref,
        operand.ref,
        updates.ref,
        Enum.map(starts, & &1.ref)
      )
      |> unwrap!()

    %{operand | ref: ref}
  end

  def reduce(
        %Function{ref: reducer},
        [%Value{function: func} | _] = init_values,
        [%Value{function: func} | _] = inputs,
        dimensions
      ) do
    init_value_refs = Enum.map(init_values, & &1.ref)
    input_refs = Enum.map(inputs, & &1.ref)

    refs =
      EXLA.NIF.mlir_reduce(func.ref, reducer, init_value_refs, input_refs, dimensions)
      |> unwrap!()

    Enum.map(refs, &%Value{ref: &1, function: func})
  end

  def window_reduce(
        %Function{ref: reducer},
        [%Value{function: func} | _] = init_values,
        [%Value{function: func} | _] = inputs,
        window_dimensions,
        window_strides,
        input_dilations,
        window_dilations,
        padding
      ) do
    init_value_refs = Enum.map(init_values, & &1.ref)
    input_refs = Enum.map(inputs, & &1.ref)

    refs =
      EXLA.NIF.mlir_window_reduce(
        func.ref,
        reducer,
        init_value_refs,
        input_refs,
        window_dimensions,
        window_strides,
        input_dilations,
        window_dilations,
        padding
      )
      |> unwrap!()

    Enum.map(refs, &%Value{ref: &1, function: func})
  end

  def map(
        %Function{ref: mapper},
        [%Value{function: func} | _] = inputs,
        dimensions
      ) do
    input_refs = Enum.map(inputs, & &1.ref)

    ref =
      EXLA.NIF.mlir_map(func.ref, mapper, input_refs, dimensions)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def if(
        %Value{} = pred,
        [%EXLA.Shape{} | _] = output_shapes,
        true_args,
        %Function{} = on_true,
        false_args,
        %Function{} = on_false
      ) do
    implicit_args_refs = Enum.map(true_args ++ false_args, & &1.ref)

    refs =
      EXLA.NIF.mlir_if(
        pred.function.ref,
        pred.ref,
        flatten_shapes(output_shapes),
        implicit_args_refs,
        on_true.ref,
        on_false.ref
      )
      |> unwrap!()

    Enum.map(refs, &%Value{ref: &1, function: pred.function})
  end

  def infeed(%Value{function: function} = token, %EXLA.Shape{} = shape) do
    ref = EXLA.NIF.mlir_infeed(function.ref, token.ref, shape.ref) |> unwrap!()

    %Value{token | ref: ref}
  end

  def outfeed(%Value{function: function} = token, inputs) do
    input_refs = Enum.map(inputs, & &1.ref)
    ref = EXLA.NIF.mlir_outfeed(function.ref, token.ref, input_refs) |> unwrap!()
    %{token | ref: ref}
  end

  def create_token(%Function{ref: ref} = function) do
    ref = EXLA.NIF.mlir_create_token(ref) |> unwrap!()
    %Value{ref: ref, function: function}
  end

  def call(%Function{ref: fun_ref} = function, args, %Function{ref: computation_ref}) do
    arg_refs = Enum.map(args, & &1.ref)
    ref = EXLA.NIF.mlir_call(fun_ref, arg_refs, computation_ref) |> unwrap!()
    %Value{ref: ref, function: function}
  end

  def while(
        %Function{ref: pred_ref},
        %Function{ref: body_ref},
        %Value{function: function} = initial
      ) do
    refs =
      EXLA.NIF.mlir_while(function.ref, pred_ref, body_ref, flatten_tuples(initial)) |> unwrap!()

    Enum.map(refs, &%Value{function: function, ref: &1})
  end

  def variadic_return([%Value{function: function} | _] = values, flatten_tuples? \\ false) do
    refs =
      if flatten_tuples? do
        flatten_tuples(values)
      else
        Enum.map(values, & &1.ref)
      end

    refs = EXLA.NIF.mlir_return(function.ref, refs) |> unwrap!()

    Enum.map(refs, fn ref -> %Value{function: function, ref: ref} end)
  end

  defp flatten_shapes(val) when is_list(val) do
    Enum.flat_map(val, &flatten_shapes/1)
  end

  defp flatten_shapes(val) do
    case val do
      %EXLA.Shape{dtype: {:tuple, element_shapes}} ->
        Enum.flat_map(element_shapes, fn shape -> flatten_shapes(shape) end)

      _ ->
        [val.ref]
    end
  end

  defp flatten_tuples(val) when is_list(val) do
    Enum.flat_map(val, &flatten_tuples/1)
  end

  defp flatten_tuples(val) do
    case get_shape(val) do
      %{dtype: {:tuple, _}, dims: {n}} ->
        Enum.flat_map(0..(n - 1), fn i -> val |> get_tuple_element(i) |> flatten_tuples() end)

      _ ->
        [val.ref]
    end
  end

  defp unwrap!({:ok, value}), do: value
  defp unwrap!(other), do: raise("#{inspect(other)}")
end

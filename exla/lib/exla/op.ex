defmodule EXLA.Op do
  @moduledoc """
  Wrapper around XLA's ops.
  """

  alias __MODULE__
  alias EXLA.{Builder, Computation, Shape}

  @enforce_keys [:builder, :ref]
  defstruct [:builder, :ref]

  ## Constructors

  @doc """
  Creates a scalar constant.
  """
  def constant_r0(%Builder{ref: builder}, value, dtype = {_, _}) when is_number(value) do
    value = cast_scalar!(dtype, value)
    ref = EXLA.NIF.constant_r0(builder, value, Shape.dtype_to_charlist(dtype)) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  defp cast_scalar!({:pred, 8}, 0), do: 0
  defp cast_scalar!({:pred, 8}, 1), do: 1
  defp cast_scalar!({:pred, 8}, n), do: raise("cannot cast #{inspect(n)} to {:pred, 8}")
  defp cast_scalar!(type, scalar), do: Nx.Type.cast_scalar!(type, scalar)

  @doc """
  Creates a n-dimensional constant from binary `data` with `shape`.
  """
  def constant_from_binary(%Builder{ref: builder}, data, %Shape{} = shape)
      when is_binary(data) do
    %{dims: dims, dtype: {_, size}, ref: shape_ref} = shape

    if bit_size(data) != size * tuple_product(dims) do
      raise ArgumentError, "binary does not match the given type and dimensions"
    end

    ref = EXLA.NIF.constant_from_binary(builder, data, shape_ref) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Specifies a parameter at position `i` with `shape` and `name`.
  """
  def parameter(%Builder{ref: builder}, i, %Shape{ref: shape}, name)
      when is_integer(i) and i >= 0 and is_binary(name) do
    ref = EXLA.NIF.parameter(builder, i, shape, name) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Builds a tuple with the given elements.
  """
  def tuple(%Builder{ref: builder}, elements) when is_list(elements) do
    element_refs = Enum.map(elements, & &1.ref)
    ref = EXLA.NIF.tuple(builder, element_refs) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Creates tensor with normal distribution.
  """
  def rng_normal(%Op{builder: builder, ref: mu}, %Op{builder: builder, ref: sigma}, %Shape{
        ref: shape
      }) do
    ref = EXLA.NIF.rng_normal(mu, sigma, shape) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Creates tensor with uniform distribution.
  """
  def rng_uniform(%Op{builder: builder, ref: a}, %Op{builder: builder, ref: b}, %Shape{ref: shape}) do
    ref = EXLA.NIF.rng_uniform(a, b, shape) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Creates iota tensor.
  """
  def iota(%Builder{ref: builder}, %Shape{ref: shape}, dim) do
    ref = EXLA.NIF.iota(builder, shape, dim) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  ## Shape

  @doc """
  Gets the shape of an operator.
  """
  def get_shape(%Op{builder: builder, ref: operand}) do
    ref = EXLA.NIF.get_shape(builder, operand) |> unwrap!()
    Shape.get_shape_info(ref)
  end

  @doc """
  Reshapes the tensor to `shape`.
  """
  def reshape(%Op{ref: ref} = op, shape) when is_tuple(shape) do
    ref = EXLA.NIF.reshape(ref, shape) |> unwrap!()
    %{op | ref: ref}
  end

  @doc """
  Pads the tensor with value and padding config.
  """
  def pad(%Op{ref: op, builder: builder}, %Op{ref: value, builder: builder}, padding_config) do
    ref = EXLA.NIF.pad(op, value, padding_config) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Broadcasts the tensor to `shape`.
  """
  def broadcast_in_dim(%Op{ref: ref} = op, shape, broadcast_dims)
      when is_tuple(shape) and is_tuple(broadcast_dims) do
    ref = EXLA.NIF.broadcast_in_dim(ref, shape, broadcast_dims) |> unwrap!()
    %{op | ref: ref}
  end

  ## Element-wise binary ops

  arith = [:add, :subtract, :multiply, :divide, :max, :min, :remainder, :atan2, :power]
  bitwise = [:bitwise_and, :bitwise_or, :bitwise_xor]
  shift = [:left_shift, :right_shift_arithmetic, :right_shift_logical]
  comparison = [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

  for fun <- arith ++ bitwise ++ shift ++ comparison do
    @doc """
    Element-wise #{fun} with broadcasting.
    """
    def unquote(fun)(
          %Op{builder: builder, ref: left},
          %Op{builder: builder, ref: right},
          broadcast_dims \\ {}
        )
        when is_tuple(broadcast_dims) do
      ref = EXLA.NIF.unquote(fun)(left, right, broadcast_dims) |> unwrap!()
      %Op{builder: builder, ref: ref}
    end
  end

  ## Element-wise unary ops

  returns_float =
    [:exp, :expm1, :log, :log1p, :logistic, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
      [:acosh, :asinh, :atanh, :acos, :asin, :atan, :cosh, :sinh] ++
      [:erf, :erfc, :erf_inv]

  returns_any = [:negate]
  requires_int = [:count_leading_zeros, :population_count, :bitwise_not]
  requires_signed = [:abs, :sign]
  requires_float = [:floor, :ceil, :round]

  for fun <- returns_float ++ returns_any ++ requires_int ++ requires_signed ++ requires_float do
    @doc """
    Unary #{fun}.
    """
    def unquote(fun)(%Op{ref: ref} = op) do
      ref = EXLA.NIF.unquote(fun)(ref) |> unwrap!()
      %{op | ref: ref}
    end
  end

  ## Ops

  def get_tuple_element(%Op{ref: operand} = op, index) when is_integer(index) do
    ref = EXLA.NIF.get_tuple_element(operand, index) |> unwrap!()
    %{op | ref: ref}
  end

  def conditional(
        %Op{builder: builder, ref: pred},
        %Op{builder: builder, ref: true_op},
        %Computation{ref: true_comp},
        %Op{builder: builder, ref: false_op},
        %Computation{ref: false_comp}
      ) do
    ref = EXLA.NIF.conditional(pred, true_op, true_comp, false_op, false_comp) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def conditional(%Op{builder: builder, ref: index}, branches, operands) do
    branches_refs =
      branches
      |> Enum.map(& &1.ref)

    operands_refs =
      operands
      |> Enum.map(& &1.ref)

    ref = EXLA.NIF.conditional(index, branches_refs, operands_refs) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def select(
        %Op{builder: builder, ref: pred},
        %Op{builder: builder, ref: on_true},
        %Op{builder: builder, ref: on_false}
      ) do
    ref = EXLA.NIF.select(pred, on_true, on_false) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def slice(
        %Op{builder: builder, ref: op},
        start_indices,
        limit_indices,
        strides
      )
      when is_list(start_indices) and is_list(limit_indices) and is_list(strides) do
    ref = EXLA.NIF.slice(op, start_indices, limit_indices, strides) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def dynamic_slice(
        %Op{builder: builder, ref: op},
        indices,
        slice_sizes
      )
      when is_list(indices) and is_list(slice_sizes) do
    indices_refs = Enum.map(indices, & &1.ref)
    ref = EXLA.NIF.dynamic_slice(op, indices_refs, slice_sizes) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def dynamic_update_slice(
        %Op{builder: builder, ref: op},
        %Op{builder: builder, ref: update},
        indices
      )
      when is_list(indices) do
    indices_refs = Enum.map(indices, & &1.ref)
    ref = EXLA.NIF.dynamic_update_slice(op, update, indices_refs) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def dot(
        %Op{builder: builder, ref: left},
        %Op{builder: builder, ref: right},
        precision_config
      ) do
    config = get_precision_config_int(precision_config)
    ref = EXLA.NIF.dot(left, right, config) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def dot_general(
        %Op{builder: builder, ref: left},
        %Op{builder: builder, ref: right},
        dimnos,
        precision_config
      ) do
    config = get_precision_config_int(precision_config)
    ref = EXLA.NIF.dot_general(left, right, dimnos, config) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def conv_general_dilated(
        %Op{builder: builder, ref: operand},
        %Op{builder: builder, ref: kernel},
        strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dim_nums,
        feature_group_count,
        batch_group_count,
        precision_config
      )
      when is_list(strides) and is_list(lhs_dilation) and is_list(rhs_dilation) do
    config = get_precision_config_int(precision_config)

    ref =
      EXLA.NIF.conv_general_dilated(
        operand,
        kernel,
        strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dim_nums,
        feature_group_count,
        batch_group_count,
        config
      )
      |> unwrap!()

    %Op{builder: builder, ref: ref}
  end

  def transpose(%Op{builder: builder, ref: operand}, permutation) when is_tuple(permutation) do
    ref = EXLA.NIF.transpose(operand, permutation) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def reduce(
        %Op{builder: builder, ref: operand},
        %Op{builder: builder, ref: init_value},
        %Computation{ref: reduction},
        reduction_dimensions
      ) do
    ref = EXLA.NIF.reduce(operand, init_value, reduction, reduction_dimensions) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def variadic_reduce(
        %Builder{ref: builder},
        operands,
        init_values,
        %Computation{ref: reduction},
        reduction_dimensions
      ) do
    operand_refs = Enum.map(operands, & &1.ref)
    init_value_refs = Enum.map(init_values, & &1.ref)

    ref =
      EXLA.NIF.variadic_reduce(
        builder,
        operand_refs,
        init_value_refs,
        reduction,
        reduction_dimensions
      )
      |> unwrap!()

    %Op{builder: builder, ref: ref}
  end

  def reduce_window(
        %Op{builder: builder, ref: operand},
        %Op{builder: builder, ref: init_value},
        %Computation{ref: reduction},
        window_dimensions,
        window_strides,
        window_dilations,
        padding_config
      )
      when is_tuple(window_dimensions) and is_list(window_strides) and is_list(window_dilations) do
    ref =
      EXLA.NIF.reduce_window(
        operand,
        init_value,
        reduction,
        window_dimensions,
        window_strides,
        window_dilations,
        padding_config
      )
      |> unwrap!()

    %Op{builder: builder, ref: ref}
  end

  def select_and_scatter(
    %Op{builder: builder, ref: operand},
    %Computation{ref: select_fn},
    window_dimensions,
    window_strides,
    padding_config,
    %Op{builder: builder, ref: source},
    %Op{builder: builder, ref: init_value},
    %Computation{ref: scatter_fn}) when is_tuple(window_dimensions) and is_list(window_strides) and is_list(padding_config) do
    ref =
      EXLA.NIF.select_and_scatter(
        operand,
        select_fn,
        window_dimensions,
        window_strides,
        padding_config,
        source,
        init_value,
        scatter_fn
      )
      |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def map(%Op{builder: builder, ref: operand}, %Computation{ref: function}, dimensions) do
    ref = EXLA.NIF.map(builder, operand, function, dimensions) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def convert_element_type(%Op{builder: builder, ref: operand}, dtype) do
    ref = EXLA.NIF.convert_element_type(operand, Shape.dtype_to_charlist(dtype)) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def bitcast_convert_type(%Op{builder: builder, ref: operand}, dtype) do
    ref = EXLA.NIF.bitcast_convert_type(operand, Shape.dtype_to_charlist(dtype)) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def clamp(%Op{builder: builder, ref: operand}, %Op{builder: builder, ref: min}, %Op{
        builder: builder,
        ref: max
      }) do
    ref = EXLA.NIF.clamp(operand, min, max) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def reverse(%Op{builder: builder, ref: operand}, dimensions) do
    ref = EXLA.NIF.reverse(operand, dimensions) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def concatenate([o1 | _] = operands, dimension) do
    %Op{builder: builder} = o1

    operand_refs =
      operands
      |> Enum.map(& &1.ref)

    ref = EXLA.NIF.concatenate(builder, operand_refs, dimension) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def cholesky(%Op{builder: builder, ref: operand}) do
    ref = EXLA.NIF.cholesky(operand) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def eigh(%Op{builder: builder, ref: operand}, lower) do
    {v_ref, w_ref} = EXLA.NIF.eigh(operand, lower) |> unwrap!()

    {
      %Op{builder: builder, ref: v_ref},
      %Op{builder: builder, ref: w_ref}
    }
  end

  def lu(%Op{builder: builder, ref: operand}) do
    {lu_ref, pivot_ref, permutation_ref} = EXLA.NIF.lu(operand) |> unwrap!()

    {
      %Op{builder: builder, ref: lu_ref},
      %Op{builder: builder, ref: pivot_ref},
      %Op{builder: builder, ref: permutation_ref}
    }
  end

  def qr(%Op{builder: builder, ref: operand}, full_matrices, precision)
      when is_boolean(full_matrices) do
    full_matrices = if full_matrices, do: 1, else: 0
    precision_config = get_precision_config_int(precision)
    {q_ref, r_ref} = EXLA.NIF.qr(operand, full_matrices, precision_config) |> unwrap!()

    {
      %Op{builder: builder, ref: q_ref},
      %Op{builder: builder, ref: r_ref}
    }
  end

  def svd(%Op{builder: builder, ref: operand}, precision) do
    precision_config = get_precision_config_int(precision)
    {u_ref, d_ref, v_ref} = EXLA.NIF.svd(operand, precision_config) |> unwrap!()

    {
      %Op{builder: builder, ref: u_ref},
      %Op{builder: builder, ref: d_ref},
      %Op{builder: builder, ref: v_ref}
    }
  end

  def triangular_solve(
        %Op{builder: builder, ref: a},
        %Op{builder: builder, ref: b},
        left_side,
        lower,
        unit_diagonal,
        transpose_a
      ) do
    transpose_a_int =
      case transpose_a do
        :none ->
          0

        :transpose ->
          1

        :conjugate ->
          2
      end

    ref =
      EXLA.NIF.triangular_solve(a, b, left_side, lower, unit_diagonal, transpose_a_int)
      |> unwrap!()

    %Op{builder: builder, ref: ref}
  end

  def sort(%Op{builder: builder, ref: operand}, %Computation{ref: comparator}, dimension) do
    ref = EXLA.NIF.sort(operand, comparator, dimension) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  ## Helpers

  defp get_precision_config_int(precision_config) do
    case precision_config do
      :default ->
        0

      :high ->
        1

      :highest ->
        2

      _ ->
        raise ArgumentError,
              "expected precision configuration to be one of" <>
                " :default, :high, or :highest, got: #{inspect(precision_config)}"
    end
  end

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end

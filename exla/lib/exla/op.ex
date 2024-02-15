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
  Creates a numeric constant.
  """
  def constant_r0(%Builder{} = builder, non_finite, dtype) when is_atom(non_finite) do
    binary = apply(Nx.Type, :"#{non_finite}_binary", [dtype])
    shape = EXLA.Shape.make_shape(dtype, {})
    constant_from_binary(builder, binary, shape)
  end

  def constant_r0(%Builder{} = builder, %Complex{re: r, im: i}, dtype = {:c, size}) do
    data =
      case size do
        64 -> <<r::32-float-native, i::32-float-native>>
        128 -> <<r::64-float-native, i::64-float-native>>
      end

    constant_from_binary(builder, data, Shape.make_shape(dtype, {}))
  end

  def constant_r0(%Builder{ref: builder}, value, dtype = {_, _}) when is_number(value) do
    value = cast_number!(dtype, value)
    ref = EXLA.NIF.constant_r0(builder, value, Shape.dtype_to_charlist(dtype)) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  defp cast_number!({:pred, 8}, 0), do: 0
  defp cast_number!({:pred, 8}, 1), do: 1
  defp cast_number!({:pred, 8}, n), do: raise("cannot cast #{inspect(n)} to {:pred, 8}")
  defp cast_number!(type, number), do: Nx.Type.cast_number!(type, number)

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

  def parameter(%EXLA.MLIR.Function{} = function, i, _shape, _name) do
    function
    |> EXLA.MLIR.Function.get_arguments()
    |> Enum.fetch!(i)
  end

  @doc """
  Builds a tuple with the given elements.
  """
  def tuple(%Builder{ref: builder}, elements) when is_list(elements) do
    element_refs = Enum.map(elements, & &1.ref)
    ref = EXLA.NIF.tuple(builder, element_refs) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def tuple(%EXLA.MLIR.Function{} = function, elements) when is_list(elements) do
    EXLA.MLIR.Value.tuple(function, List.flatten(elements))
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

  arith = [:add, :subtract, :multiply, :divide, :max, :min, :remainder, :atan2, :pow]
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
    [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tanh, :sqrt, :rsqrt, :cbrt] ++
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

  def fft(%Op{ref: ref} = op, fft_sizes) when is_list(fft_sizes) do
    ref = EXLA.NIF.fft(ref, fft_sizes) |> unwrap!()
    %{op | ref: ref}
  end

  def ifft(%Op{ref: ref} = op, fft_sizes) when is_list(fft_sizes) do
    ref = EXLA.NIF.ifft(ref, fft_sizes) |> unwrap!()
    %{op | ref: ref}
  end

  def is_nan(op, type, shape, builder),
    do: is_non_finite(&EXLA.NIF.is_nan/1, op, type, shape, builder)

  def is_infinity(op, type, shape, builder),
    do: is_non_finite(&EXLA.NIF.is_infinity/1, op, type, shape, builder)

  defp is_non_finite(nif_function, %{ref: ref} = op, {:c, _}, _shape, _builder) do
    re_part = ref |> EXLA.NIF.real() |> unwrap!() |> nif_function.() |> unwrap!()
    im_part = ref |> EXLA.NIF.imag() |> unwrap!() |> nif_function.() |> unwrap!()
    result_ref = EXLA.NIF.bitwise_or(re_part, im_part, {}) |> unwrap!()
    %{op | ref: result_ref}
  end

  defp is_non_finite(nif_function, op, {t, _}, _shape, _builder) when t in [:f, :bf] do
    %{ref: ref} = op
    result_ref = nif_function.(ref) |> unwrap!()
    %{op | ref: result_ref}
  end

  defp is_non_finite(_nif_function, _op, type, shape, builder) do
    # For non-floating types, we can just return
    # a boolean 0 tensor in the output shape

    out_type =
      case type do
        {:pred, 8} -> {:pred, 8}
        _ -> {:u, 8}
      end

    builder
    |> constant_r0(0, out_type)
    |> reshape(Tuple.duplicate(1, tuple_size(shape)))
    |> broadcast_in_dim(shape, List.to_tuple(Nx.axes(shape)))
  end

  ## Ops

  def get_tuple_element(%Op{ref: operand} = op, index) when is_integer(index) do
    ref = EXLA.NIF.get_tuple_element(operand, index) |> unwrap!()
    %{op | ref: ref}
  end

  def get_tuple_element(%EXLA.MLIR.Value{} = operand, index) when is_integer(index) do
    EXLA.MLIR.Value.get_tuple_element(operand, index)
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

  @doc """
  The XLA gather operation stitches together several slices
  of an input array.

  Note that this operation is extremely generic and far from
  intuitive for regular usage. However, it can be used to implement
  many specific operations that have to do with combining multiple
  tensor slices.

  ## Parameteres

  The XLA docs are rather cryptic unless already understood,
  so here's an attempt of a more intuitive description.

  ### `index_vector_dim`

  Determines which dimension contains index vectors. In most cases
  we want to set this to the last dimension.

      given
        start_indices = [[0, 1], [1, 1]]
      and given
        index_vector_dim = 1
      then
        index vectors are [0, 1] and [1, 1]

  Note that we can set this to `last_dimension + 1`, in which case
  `start_indices` are implicitly reshaped to have a trailing dimension
  of 1.

      given
        start_indices = [[0, 1], [1, 1]]
      and given
        index_vector_dim = 2
      then
        start_indices <- [[[0], [1]], [[1], [1]]]
        index vectors are [0], [1], [1], [1]

  ### `start_index_map`

  Note: though given as a list, it can be treated as a map of `list_idx -> value`.

  An index vector may have less elements than the operand tensor shape.
  For example:

      given
        operand = [[1, 2], [3, 4]]
        start_indices = [[1], [0]]
        index_vector_dim = 1

  As described above, in this case index vectors are `[1]`, `[0]` and they have
  length 1. However, the operand has rank 2, so we need vectors of the form `[_, _]`
  to point to a specific element in the operand. The `start_index_map` determines
  where indices go into this template:

      and given
        start_index_map = [0] # effectively %{0 => 0}
      then
        actual index vectors are [1, _] and [0, _]

      and given
        start_index_map = [1] # effectively %{0 => 1}
      then
        actual index vectors are [_, 1] and [_, 0]

  Finally, the missing elements (`_`) are assumed to be 0.

  Complete examples:

      given
        operand = [[1, 2], [3, 4]]
        start_indices = [[0], [1]]
        index_vector_dim = 1
      and given
        start_index_map = [1] # effectively %{0 => 1}
      then
        actual index vectors are [0, 0], [0, 1] (leading 0 is inserted)

      given
        operand = [[1, 2], [3, 4]]
        start_indices = [[0, 1], [1, 1]]
        index_vector_dim = 1
      and given
        start_index_map = [0, 1] # effectively %{0 => 0, 1 => 1}
      then
        actual index vectors are [0, 1], [1, 1] (as expected)

      given
        operand = [[1, 2], [3, 4]]
        start_indices = [[0, 1], [1, 1]]
        index_vector_dim = 1
      and given
        start_index_map = [1, 0] # effectively %{0 => 1, 1 => 0}
      then
        actual index vectors are [1, 0], [1, 1] (see how the first vector is reversed)

  ### `slice_sizes`

  For every starting point (as described above) we take a slice given
  by `slice_sizes`. Naturally, `slice_sizes` must have the same length
  as operand rank, so that we have one size per dimension.

      given
        operand = [[1, 2], [3, 4]]
        actual index vector [1, 0]
      and given
        slice_sizes = [1, 2]
      then
        slice for actual index vector is [[3, 4]]

  ### `collapsed_slice_dims`

  A list of dimensions that are collapsed (effectively removed) in
  the slice shape. Only dimensions of size 1 can be collapsed.

      given
        slice is [[3, 4]] # shape: [1][2]
      and given
        collapsed_slice_dims = [0]
      then
        actual slice is [3, 4] # shape [2]

  ### `offset_dims`

  A list of dimensions in the output tensor corresponding to the
  non-collapsed dimensions in slice tensors. In other words, these
  dimensions are used for indexing elements of the slice tensors.

      given
        operand = [[1, 2], [3, 4]]
        start_indices = [[1, 0], [0, 0], [1, 0]]
        index_vector_dim = 1
        start_index_map = [1, 2] # effectively %{0 => 0, 1 => 1}
        collapsed_slice_dims = [0]
      and given
        offset_dims = [1]
      then
        result is [[3, 4], [1, 2], [3, 4]]

  In the above example the collapsed slices are `[3, 4]`, `[1, 2]`, `[3, 4]`
  and have rank 1. Using `offset_dims` we specify that the first
  dimension in each slice corresponds to the second dimension in
  the output tensor.

  If we use the first output dimension instead, we get:

      and given
        offset_dims = [0]
      then
        result is [[3, 1, 3], [4, 2, 4]]

  ## Docs

  More formal specification can be found in [the XLA Gather docs](https://www.tensorflow.org/xla/operation_semantics#gather).
  """
  def gather(
        %Op{builder: builder, ref: op},
        %Op{builder: builder, ref: start_indices},
        index_vector_dim,
        slice_sizes,
        offset_dims,
        collapsed_slice_dims,
        start_index_map
      )
      when is_integer(index_vector_dim) and is_list(slice_sizes) and is_list(offset_dims) and
             is_list(collapsed_slice_dims) and is_list(start_index_map) do
    ref =
      EXLA.NIF.gather(
        op,
        start_indices,
        index_vector_dim,
        slice_sizes,
        offset_dims,
        collapsed_slice_dims,
        start_index_map
      )
      |> unwrap!()

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

  def window_reduce(
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
      EXLA.NIF.window_reduce(
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
        %Computation{ref: scatter_fn}
      )
      when is_tuple(window_dimensions) and is_list(window_strides) and is_list(padding_config) do
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

  def scatter(
        %Op{builder: builder, ref: target},
        %Op{ref: indices},
        %Op{ref: updates},
        %Computation{ref: scatter_fn},
        indices_rank,
        update_window_dims,
        inserted_window_dims,
        index_dims_to_window_dims
      )
      when is_integer(indices_rank) and is_list(update_window_dims) and
             is_list(inserted_window_dims) and is_list(index_dims_to_window_dims) do
    ref =
      EXLA.NIF.scatter(
        target,
        indices,
        updates,
        scatter_fn,
        indices_rank,
        update_window_dims,
        inserted_window_dims,
        index_dims_to_window_dims
      )
      |> unwrap!()

    %Op{builder: builder, ref: ref}
  end

  def map(%Op{builder: builder, ref: operand}, %Computation{ref: function}, dimensions) do
    ref = EXLA.NIF.map(builder, operand, function, dimensions) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def while(
        %Computation{ref: cond_fn},
        %Computation{ref: body_fn},
        %Op{builder: builder, ref: init_value}
      ) do
    ref = EXLA.NIF.while(cond_fn, body_fn, init_value) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def call(
        %Builder{ref: builder},
        args,
        %Computation{ref: body_fn}
      ) do
    args_ref = Enum.map(args, & &1.ref)

    # wrap args in an n-tuple to avoid nif variadic limitations
    ref = EXLA.NIF.call(builder, args_ref, body_fn) |> unwrap!()
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

  def conjugate(%Op{builder: builder, ref: operand}) do
    ref = EXLA.NIF.conj(operand) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def real(%Op{builder: builder, ref: operand}) do
    ref = EXLA.NIF.real(operand) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def imag(%Op{builder: builder, ref: operand}) do
    ref = EXLA.NIF.imag(operand) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def lu(%Op{builder: builder, ref: operand}) do
    {lu_ref, pivot_ref, permutation_ref} = EXLA.NIF.lu(operand) |> unwrap!()

    {
      %Op{builder: builder, ref: lu_ref},
      %Op{builder: builder, ref: pivot_ref},
      %Op{builder: builder, ref: permutation_ref}
    }
  end

  def triangular_solve(
        %Op{builder: builder, ref: a},
        %Op{builder: builder, ref: b},
        left_side,
        lower,
        unit_diagonal,
        transpose_a
      )
      when is_boolean(left_side) and is_boolean(lower) and is_boolean(unit_diagonal) do
    left_side = boolean_to_int(left_side)
    lower = boolean_to_int(lower)
    unit_diagonal = boolean_to_int(unit_diagonal)

    transpose_a_int =
      case transpose_a do
        :none -> 0
        :transpose -> 1
        :conjugate -> 2
      end

    ref =
      EXLA.NIF.triangular_solve(a, b, left_side, lower, unit_diagonal, transpose_a_int)
      |> unwrap!()

    %Op{builder: builder, ref: ref}
  end

  def sort(%Op{builder: builder, ref: operand}, %Computation{ref: comparator}, dimension, stable)
      when is_integer(dimension) and is_boolean(stable) do
    stable = if stable, do: 1, else: 0
    ref = EXLA.NIF.sort(operand, comparator, dimension, stable) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def top_k(%Op{builder: builder, ref: operand}, k) do
    ref = EXLA.NIF.top_k(operand, k) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def variadic_sort(
        %Builder{ref: builder},
        operands,
        %Computation{ref: comparator},
        dimension,
        stable
      )
      when is_integer(dimension) and is_boolean(stable) do
    stable = if stable, do: 1, else: 0
    operand_refs = Enum.map(operands, & &1.ref)
    ref = EXLA.NIF.variadic_sort(operand_refs, comparator, dimension, stable) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def create_token(%EXLA.MLIR.Function{} = function) do
    # TO-DO (mlir): actually do something here
    function
  end

  def create_token(%Builder{ref: builder}) do
    ref = EXLA.NIF.create_token(builder) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def infeed(%Op{builder: builder, ref: token}, %Shape{ref: shape}) do
    ref = EXLA.NIF.infeed(token, shape) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def outfeed(%Op{builder: builder, ref: operand}, %Op{builder: builder, ref: token}) do
    shape_ref = EXLA.NIF.get_shape(builder, operand) |> unwrap!()
    ref = EXLA.NIF.outfeed(operand, token, shape_ref) |> unwrap!()
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

  defp boolean_to_int(true), do: 1
  defp boolean_to_int(false), do: 0

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end

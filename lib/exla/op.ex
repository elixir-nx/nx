defmodule Exla.Op do
  alias __MODULE__
  alias Exla.{Builder, Computation, Shape}

  @enforce_keys [:builder, :ref]
  defstruct [:builder, :ref]

  ## Constructors

  @doc """
  Creates a scalar constant.
  """
  def constant_r0(%Builder{ref: builder}, value, dtype = {_, _}) when is_number(value) do
    ref = Exla.NIF.constant_r0(builder, value, Shape.dtype_to_charlist(dtype)) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Creates a n-dimensional constant from binary `data` with `shape`.
  """
  def constant_from_binary(%Builder{ref: builder}, data, %Shape{} = shape)
      when is_binary(data) do
    %{dims: dims, dtype: {_, size}, ref: shape_ref} = shape

    if bit_size(data) != size * tuple_product(dims) do
      raise ArgumentError, "binary does not match the given type and dimensions"
    end

    ref = Exla.NIF.constant_from_binary(builder, data, shape_ref) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Specifies a parameter at position `i` with `shape` and `name`.
  """
  def parameter(%Builder{ref: builder}, i, %Shape{ref: shape}, name)
      when is_integer(i) and i >= 0 and is_binary(name) do
    ref = Exla.NIF.parameter(builder, i, shape, name) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  @doc """
  Builds a tuple with the given elements.
  """
  def tuple(%Builder{ref: builder}, elements) when is_tuple(elements) do
    element_refs =
      elements
      |> Tuple.to_list()
      |> Enum.map(& &1.ref)

    ref = Exla.NIF.tuple(builder, element_refs) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  ## Reflection

  @doc """
  Gets the shape of an operator.
  """
  def get_shape(%Op{builder: builder, ref: operand}) do
    ref = Exla.NIF.get_shape(builder, operand) |> unwrap!()
    Shape.get_shape_info(ref)
  end

  ## Element-wise ops

  arith = [:add, :subtract, :multiply, :divide, :max, :min, :remainder, :arctan2, :power]
  bitwise = [:bitwise_and, :bitwise_or, :bitwise_xor]
  shift = [:left_shift, :right_shift_arithmetic, :right_shift_logical]

  for fun <- arith ++ bitwise ++ shift do
    @doc """
    Element-wise #{fun} with broadcasting.
    """
    def unquote(fun)(
          %Op{builder: builder, ref: left},
          %Op{builder: builder, ref: right},
          broadcast_dims \\ {}
        ) do
      ref = Exla.NIF.unquote(fun)(left, right, broadcast_dims) |> unwrap!()
      %Op{builder: builder, ref: ref}
    end
  end

  ## Ops

  def get_tuple_element(%Op{builder: builder, ref: operand}, index) when is_integer(index) do
    ref = Exla.NIF.get_tuple_element(operand, index) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def conditional(
        %Op{builder: builder, ref: pred},
        %Op{builder: builder, ref: true_op},
        %Computation{ref: true_comp},
        %Op{builder: builder, ref: false_op},
        %Computation{ref: false_comp}
      ) do
    ref = Exla.NIF.conditional(pred, true_op, true_comp, false_op, false_comp) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def ne(
        %Op{builder: builder, ref: left},
        %Op{builder: builder, ref: right},
        broadcast_dims \\ {}
      ) do
    ref = Exla.NIF.ne(left, right, broadcast_dims) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def conditional(%Op{builder: builder, ref: index}, branches, operands) do
    branches_refs =
      branches
      |> Enum.map(& &1.ref)

    operands_refs =
      operands
      |> Enum.map(& &1.ref)

    ref = Exla.NIF.conditional(index, branches_refs, operands_refs) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  # TODO: Bounds checks!
  def slice(
        %Op{builder: builder, ref: op},
        start_indices,
        limit_indices,
        strides \\ {}
      ) do
    ref = Exla.NIF.slice(op, start_indices, limit_indices, strides) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  # TODO: Needs dim, index checks, will SegFault without error messages on bad dims/index!
  def slice_in_dim(
        %Op{builder: builder, ref: op},
        start_index,
        end_index,
        stride,
        dimno
      ) do
    ref = Exla.NIF.slice_in_dim(op, start_index, end_index, stride, dimno) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  # TODO: Indices as tuple.
  def dynamic_slice(
        %Op{builder: builder, ref: op},
        indices,
        slice_sizes
      ) do
    indices_refs =
      indices
      |> Enum.map(& &1.ref)

    ref = Exla.NIF.dynamic_slice(op, indices_refs, slice_sizes) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def dynamic_update_slice(
        %Op{builder: builder, ref: op},
        %Op{builder: builder, ref: update},
        indices
      ) do
    indices_refs =
      indices
      |> Enum.map(& &1.ref)

    ref = Exla.NIF.dynamic_update_slice(op, update, indices_refs) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def rng_normal(%Op{builder: builder, ref: mu}, %Op{builder: builder, ref: sigma}, %Shape{ref: shape}) do
    ref = Exla.NIF.rng_normal(mu, sigma, shape) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def rng_uniform(%Op{builder: builder, ref: a}, %Op{builder: builder, ref: b}, %Shape{ref: shape}) do
    ref = Exla.NIF.rng_uniform(a, b, shape) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def dot(%Op{builder: builder, ref: left}, %Op{builder: builder, ref: right}) do
    ref = Exla.NIF.dot(left, right) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def exp(%Op{builder: builder, ref: op}) do
    ref = Exla.NIF.exp(op) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def reduce(
        %Op{builder: builder, ref: operand},
        %Op{builder: builder, ref: init_value},
        %Computation{ref: reduction},
        reduction_dimensions
      ) do
    ref = Exla.NIF.reduce(operand, init_value, reduction, reduction_dimensions) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def reduce_all(
        %Op{builder: builder, ref: operand},
        %Op{builder: builder, ref: init_value},
        %Computation{ref: reduction}
      ) do
    ref = Exla.NIF.reduce_all(operand, init_value, reduction) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  def convert_element_type(%Op{builder: builder, ref: operand}, dtype) do
    ref = Exla.NIF.convert_element_type(operand, Shape.dtype_to_charlist(dtype)) |> unwrap!()
    %Op{builder: builder, ref: ref}
  end

  ## Helpers

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end

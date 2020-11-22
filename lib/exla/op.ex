defmodule Exla.Op do
  alias __MODULE__, as: Op
  alias Exla.Shape
  alias Exla.Builder
  alias Exla.Computation

  @enforce_keys [:builder, :ref]
  defstruct [:builder, :ref]

  # The XLA API is explicit about the rank of the constant being created e.g. ConstantR0, ConstantR1
  # We can be just as explicit, or we can use pattern matching on the inputs, I lean pattern matching
  # as I think it makes the API feel more flexible
  def constant(%Builder{ref: builder}, value) when is_number(value) do
    {:ok, ref} = Exla.NIF.constant_r0(builder, value)
    %Op{builder: builder, ref: ref}
  end

  def constant_r1(%Builder{ref: builder}, length, value) when is_number(length) and is_number(value) do
    {:ok, ref} = Exla.NIF.constant_r1(builder, length, value)
    %Op{builder: builder, ref: ref}
  end

  def zero(%Builder{ref: builder}, dtype) when is_atom(dtype) do
    {:ok, ref} = Exla.NIF.zero(builder, dtype)
    %Op{builder: builder, ref: ref}
  end

  def parameter(%Builder{ref: builder}, i, %Shape{ref: shape}, name)
      when is_integer(i) and i >= 0 and is_binary(name) do
    {:ok, ref} = Exla.NIF.parameter(builder, i, shape, name)
    %Op{builder: builder, ref: ref}
  end

      def conditional(%Op{builder: builder, ref: pred}, %Op{builder: builder, ref: true_op}, %Computation{ref: true_comp}, %Op{builder: builder, ref: false_op}, %Computation{ref: false_comp}) do
        {:ok, ref} = Exla.NIF.conditional(pred, true_op, true_comp, false_op, false_comp)
        %Op{builder: builder, ref: ref}
      end
  def ne(%Op{builder: builder, ref: left}, %Op{builder: builder, ref: right}, broadcast_dims \\ {}) do
        {:ok, ref} = Exla.NIF.ne(left, right, broadcast_dims)
        %Op{builder: builder, ref: ref}
      end

  def conditional(%Op{builder: builder, ref: index}, branches, operands) do
    # TODO: Both branches and operands need to be lists!
    branches_refs =
      branches
      |> Tuple.to_list()
      |> Enum.map(&(&1.ref))
      |> List.to_tuple()

    operands_refs =
      operands
      |> Tuple.to_list()
      |> Enum.map(&(&1.ref))
      |> List.to_tuple()

    {:ok, ref} = Exla.NIF.conditional(index, branches_refs, operands_refs)
    %Op{builder: builder, ref: ref}
  end

  def add(
        %Op{builder: builder, ref: left},
        %Op{builder: builder, ref: right},
        broadcast_dims \\ {}
      ) do
    {:ok, ref} = Exla.NIF.add(left, right, broadcast_dims)
    %Op{builder: builder, ref: ref}
  end

  # TODO: Bounds checks!
  def slice(
        %Op{builder: builder, ref: op},
        start_indices,
        limit_indices,
        strides \\ {}
      ) do
    {:ok, ref} = Exla.NIF.slice(op, start_indices, limit_indices, strides)
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
    {:ok, ref} = Exla.NIF.slice_in_dim(op, start_index, end_index, stride, dimno)
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
      |> Tuple.to_list()
      |> Enum.map(&(&1.ref))
      |> List.to_tuple()
    {:ok, ref} = Exla.NIF.dynamic_slice(op, indices_refs, slice_sizes)
    %Op{builder: builder, ref: ref}
  end

  def dynamic_update_slice(%Op{builder: builder, ref: op}, %Op{builder: builder, ref: update}, indices) do
    indices_refs =
      indices
      |> Tuple.to_list()
      |> Enum.map(&(&1.ref))
      |> List.to_tuple()
    {:ok, ref} = Exla.NIF.dynamic_update_slice(op, update, indices_refs)
    %Op{builder: builder, ref: ref}
  end

  def div(%Op{builder: builder, ref: left}, %Op{builder: builder, ref: right}, broadcast_dims \\ {}) do
    {:ok, ref} = Exla.NIF.div(left, right, broadcast_dims)
    %Op{builder: builder, ref: ref}
  end

  def dot(%Op{builder: builder, ref: left}, %Op{builder: builder, ref: right}) do
    {:ok, ref} = Exla.NIF.dot(left, right)
    %Op{builder: builder, ref: ref}
  end

  def exp(%Op{builder: builder, ref: op}) do
    {:ok, ref} = Exla.NIF.exp(op)
    %Op{builder: builder, ref: ref}
  end

  def reduce(%Op{builder: builder, ref: operand}, %Op{builder: builder, ref: init_value}, %Computation{ref: reduction}, reduction_dimensions) do
    {:ok, ref} = Exla.NIF.reduce(operand, init_value, reduction, reduction_dimensions)
    %Op{builder: builder, ref: ref}
  end

  def reduce_all(%Op{builder: builder, ref: operand}, %Op{builder: builder, ref: init_value}, %Computation{ref: reduction}) do
    {:ok, ref} = Exla.NIF.reduce_all(operand, init_value, reduction)
    %Op{builder: builder, ref: ref}
  end
end

defmodule Nx.Util do
  @moduledoc """
  A collection of helper functions for working with tensors.

  Generally speaking, the usage of these functions is discouraged,
  as they don't work inside `defn` and the `Nx` module often has
  higher-level features around them.
  """

  import Nx.Shared

  ## Conversions

  @doc """
  Returns the underlying tensor as a flat list.

  The list is returned as is (which is row-major).

    ## Examples

      iex> Nx.Util.to_flat_list(1)
      [1]

      iex> Nx.Util.to_flat_list(Nx.tensor([1.0, 2.0, 3.0]))
      [1.0, 2.0, 3.0]

  """
  def to_flat_list(tensor) do
    tensor = Nx.tensor(tensor)

    match_types [tensor.type] do
      for <<match!(var, 0) <- Nx.to_binary(tensor)>>, do: read!(var, 0)
    end
  end

  @doc """
  Returns the underlying tensor as a scalar.

  If the tensor has a dimension, it raises.

    ## Examples

      iex> Nx.Util.to_scalar(1)
      1

      iex> Nx.Util.to_scalar(Nx.tensor([1.0, 2.0, 3.0]))
      ** (ArgumentError) cannot convert tensor of shape {3} to scalar

  """
  def to_scalar(tensor) do
    tensor = Nx.tensor(tensor)

    if tensor.shape != {} do
      raise ArgumentError, "cannot convert tensor of shape #{inspect(tensor.shape)} to scalar"
    end

    match_types [tensor.type] do
      <<match!(x, 0)>> = Nx.to_binary(tensor)
      read!(x, 0)
    end
  end

  # TODO: Move it to Nx.reduce_window

  #  @doc """
  #  Reduces elements in a window.

  #  The rank of the input tensor, window dimensions, and window
  #  strides must match.

  #  ## Examples

  #      iex> Nx.Util.reduce_window(Nx.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [11, 12, 13, 14]]),
  #      ...> :first,
  #      ...> fn x, acc -> if acc == :first, do: x, else: max(x, acc) end,
  #      ...> {2, 2}, {1, 1}
  #      ...> )
  #      #Nx.Tensor<
  #        s64[3][3]
  #        [
  #          [5, 6, 7],
  #          [8, 9, 10],
  #          [12, 13, 14]
  #        ]
  #      >

  #      iex> Nx.Util.reduce_window(Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
  #      ...> :first,
  #      ...> fn x, acc -> if acc == :first, do: x, else: max(x, acc) end,
  #      ...> {2, 2}, {1, 1}, :same
  #      ...> )
  #      #Nx.Tensor<
  #        s64[3][3]
  #        [
  #          [5, 6, 6],
  #          [8, 9, 9],
  #          [8, 9, 9]
  #        ]
  #      >

  #      iex> Nx.Util.reduce_window(Nx.tensor([[1, 2, 3], [4, 5, 6]]),
  #      ...> 0,
  #      ...> fn x, acc -> x + acc end,
  #      ...> {1, 2}, {1, 1}, :same
  #      ...> )
  #      #Nx.Tensor<
  #        s64[2][3]
  #        [
  #          [3, 5, 3],
  #          [9, 11, 6]
  #        ]
  #      >
  #  """
  #  def reduce_window(tensor, acc, fun, window_dimensions, window_strides, padding \\ :valid) do
  #    %T{type: {_, size} = type, shape: shape} = t = Nx.tensor(tensor)

  #    %T{shape: padded_shape} =
  #      t =
  #      case padding do
  #        :valid ->
  #          t

  #        :same ->
  #          padding_values = Nx.Shape.calculate_padding(shape, window_dimensions, window_strides)
  #          Nx.pad(t, 0, padding_values)
  #      end

  #    output_shape = Nx.Shape.window(padded_shape, window_dimensions, window_strides)

  #    data = Nx.to_binary(t)

  #    weighted_shape = weighted_shape(padded_shape, size, window_dimensions)
  #    anchors = Enum.sort(make_anchors(padded_shape, window_strides, window_dimensions, []))

  #    data =
  #      for anchor <- anchors, into: <<>> do
  #        offset = weighted_offset(weighted_shape, anchor)

  #        window = IO.iodata_to_binary(weighted_traverse(weighted_shape, data, size, offset))

  #        match_types [type] do
  #          window_val =
  #            for <<match!(x, 0) <- window>>,
  #              reduce: acc,
  #              do: (acc -> fun.(read!(x, 0), acc))

  #          <<write!(window_val, 0)>>
  #        end
  #      end

  #    %{t | data: {Nx.BinaryDevice, data}, shape: output_shape}
  #  end
end

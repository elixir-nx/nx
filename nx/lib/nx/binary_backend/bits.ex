defmodule Nx.BinaryBackend.Bits do
  import Nx.Shared

  alias Nx.BinaryBackend.Index

  @compile {:inline, from_number: 2, to_number: 2, number_at: 3}

  @doc """
  Encodes a number into a bitstring according to the type.

  ## Examples

      iex> Bits.from_number(0.0, {:f, 32})
      <<0, 0, 0, 0>>

      iex> Bits.from_number(255, {:s, 64})
      <<255::64-native-signed>>

      iex> Bits.from_number(1.0e-3, {:f, 64})
      <<252, 169, 241, 210, 77, 98, 80, 63>>
  """
  def from_number(number, type) do
    match_types([type], do: <<write!(number, 0)>>)
  end

  @doc """
  Decodes a bitstring into a number according to the type.

  ## Examples

      iex> Bits.to_number(<<0, 0, 0, 0>>, {:f, 32})
      0.0

      iex> Bits.to_number(<<255::64-native-unsigned>>, {:u, 64})
      255

      iex> Bits.to_number(<<252, 169, 241, 210, 77, 98, 80, 63>>, {:f, 64})
      1.0e-3
  """
  def to_number(bin, type) do
    match_types [type] do
      <<match!(value, 0)>> = bin
      read!(value, 0)
    end
  end

  @doc """
  Decodes the number at the given index of the bitstring
  according to the type.

  ## Examples

      iex> Bits.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {:f, 32}, 0)
      1.0

      iex> Bits.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {:f, 32}, 1)
      2.0

      iex> Bits.number_at(<<127, 128>>, {:u, 8}, 1)
      128
  """
  def number_at(bin, {_, sizeof} = type, i) when is_integer(i) do
    bits_offset = i * sizeof
    <<_::size(bits_offset), numbits::size(sizeof)-bitstring, _::bitstring>> = bin
    to_number(numbits, type)
  end

  @doc """
  Returns the number from the binary at the given coords according
  to the shape and type.

  ## Examples
      iex> Bits.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {2}, {:f, 32}, {0})
      1.0

      iex> Bits.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {2}, {:f, 32}, {1})
      2.0

      iex> Bits.number_at(<<0, 1, 2, 3, 4, 5>>, {3, 2}, {:u, 8}, {2, 1})
      5
  """
  def number_at(bin, shape, type, coords) when tuple_size(shape) == tuple_size(coords) do
    number_at(bin, type, Index.coords_to_i(shape, coords))
  end

  @doc """
  Slices a bitstring from the start to the start + len according to the type.
  """
  def slice(bin, {_, sizeof}, start, len) when is_integer(len) and len > 0 do
    bits_offset = start * sizeof
    sizeof_slice = len * sizeof
    <<_::size(bits_offset), sliced::size(sizeof_slice)-bitstring, _::bitstring>> = bin
    sliced
  end

  @doc """
  Zips and reduces the data of the two tensors
  """
  def zip_reduce(bits_acc, type_out, _shape1, type1, data1, [], _shape2, type2, data2, [], acc, fun) do
    match_types [type1, type2] do
      for <<match!(bits_n1, 0) <- data1>>, <<match!(bits_n2, 1) <- data2>>, into: bits_acc do
        result = fun.(read!(bits_n1, 0), read!(bits_n2, 1), acc)
        from_number(result, type_out)
      end
    end
  end

  def zip_reduce(bits_acc, type_out, shape1, type1, data1, [_ | _] = axes1, shape2, type2, data2, [_ | _] = axes2, init_acc, fun) do
    weights1 = Nx.Shape.weights(shape1)
    weights2 = Nx.Shape.weights(shape2)

    tagged_dims1 = tagged_dims(shape1, axes1)
    tagged_dims2 = tagged_dims(shape2, axes2)

    indices1 = tagged_dims_to_indices(tagged_dims1, weights1, 0)
    indices2 = tagged_dims_to_indices(tagged_dims2, weights2, 0)

    traverse_reduce(bits_acc, type_out,  type1, data1, indices1, type2, data2, indices2, init_acc, fun)
  end

  defp tagged_dims(shape, axes) do
    0..(tuple_size(shape) - 1)
    |> Enum.map(fn axis ->
      dim = Nx.Shape.dimension(shape, axis)
      if axis in axes do
        {:contract, axis, dim}
      else
        {:normal, axis, dim}
      end
    end)
    |> Enum.sort(&(&1 >= &2))
  end


  defp tagged_dims_to_indices([], _weights, _i) do
    []
  end

  defp tagged_dims_to_indices([{tag, axis, dim} | tagged_dims], weights, offset) do
    w = weight_of_axis(weights, axis)

    idxs = 
      for i <- 0..(dim - 1) do
        i = offset + i * w
        [i | tagged_dims_to_indices(tagged_dims, weights, i)]
      end

    aggregate(tag, idxs)
  end

  def aggregate(:contract, [_ | idxs]) do
    List.flatten(idxs)
  end

  def aggregate(:normal, idxs) do
    idxs
  end
  
  defp traverse_reduce(bits_acc, type_out, type1, data1, [i1 | _] = indices1, type2, data2, indices2, acc, fun) when is_list(i1) do
    for idx1 <- indices1, reduce: bits_acc do
      bits_acc2 ->
        traverse_reduce(bits_acc2, type_out, type1, data1, idx1, type2, data2, indices2, acc, fun)
    end
  end

  defp traverse_reduce(bits_acc, type_out, type1, data1, indices1, type2, data2, [i2 | _ ] = indices2, acc, fun) when is_list(i2) do
    for idx2 <- indices2, reduce: bits_acc do
      bits_acc2 ->
        traverse_reduce(bits_acc2, type_out, type1, data1, indices1, type2, data2, idx2, acc, fun)
    end
  end

  defp traverse_reduce(bits_acc, type_out, type1, data1, [i1 | rest1], type2, data2, [i2 | rest2], acc, fun) when is_integer(i1) and is_integer(i2) do
    {_, sizeof1} = type1
    {_, sizeof2} = type2
    acc2 = 
      match_types [type1, type2] do
        offset1 = i1 * sizeof1
        offset2 = i2 * sizeof2
        <<_::size(offset1), match!(bits_n1, 0), _::bitstring>> = data1
        <<_::size(offset2), match!(bits_n2, 1), _::bitstring>> = data2
        n1 = read!(bits_n1, 0)
        n2 = read!(bits_n2, 1)
        fun.(n1, n2, acc)
      end
    traverse_reduce(bits_acc, type_out, type1, data1, rest1, type2, data2, rest2, acc2, fun)
  end

  defp traverse_reduce(bits_acc, type_out, _type1, _data1, [], _type2, _data2, [], acc, _fun) do
    bits_acc <> from_number(acc, type_out)
  end

  def weight_of_axis({_size, weights}, axis) do
    elem(weights, axis)
  end

  def map_i_to_axis({size, ws}, axis, i) do
    rem(i * elem(ws, axis), size)
  end
end
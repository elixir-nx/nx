defmodule Nx.BinaryBackend.Binary do
  import Nx.Shared

  @compile {:inline, from_number: 2, to_number: 2, number_at: 3}

  @doc """
  Encodes a number into a bitstring according to the type.

  ## Examples

      iex> Binary.from_number(0.0, {:f, 32})
      <<0, 0, 0, 0>>

      iex> Binary.from_number(255, {:s, 64})
      <<255::64-native-signed>>

      iex> Binary.from_number(1.0e-3, {:f, 64})
      <<252, 169, 241, 210, 77, 98, 80, 63>>
  """
  def from_number(number, type) do
    match_types([type], do: <<write!(number, 0)>>)
  end

  @doc """
  Decodes a bitstring into a number according to the type.

  ## Examples

      iex> Binary.to_number(<<0, 0, 0, 0>>, {:f, 32})
      0.0

      iex> Binary.to_number(<<255::64-native-unsigned>>, {:u, 64})
      255

      iex> Binary.to_number(<<252, 169, 241, 210, 77, 98, 80, 63>>, {:f, 64})
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

      iex> Binary.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {:f, 32}, 0)
      1.0

      iex> Binary.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {:f, 32}, 1)
      2.0

      iex> Binary.number_at(<<127, 128>>, {:u, 8}, 1)
      128
  """
  def number_at(bin, {_, sizeof} = type, i) when is_integer(i) do
    bits_offset = i * sizeof
    <<_::size(bits_offset), numbits::size(sizeof)-bitstring, _::bitstring>> = bin
    to_number(numbits, type)
  end

  @doc """
  Returns a number from a binary at the given coords according
  to the shape and type.

  ## Examples
      iex> Binary.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {2}, {:f, 32}, {0})
      1.0

      iex> Binary.number_at(<<0, 0, 128, 63, 0, 0, 0, 64>>, {2}, {:f, 32}, {1})
      2.0

      iex> Binary.number_at(<<0, 1, 2, 3, 4, 5>>, {3, 2}, {:u, 8}, {2, 1})
      5
  """
  def number_at(bin, shape, type, coords) when tuple_size(shape) == tuple_size(coords) do
    number_at(bin, type, Nx.Shape.coords_to_i(shape, coords))
  end
end
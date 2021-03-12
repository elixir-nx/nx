defmodule Nx.BinaryBackend.Bits do
  @moduledoc """
  For manipulating bitstrings in BinaryBackend.
  """
  import Nx.Shared

  alias Nx.BinaryBackend
  alias Nx.Tensor, as: T

  @compile {:inline, from_number: 2, to_number: 2, number_at: 3, slice: 4, from_scalar: 2}

  @doc """
  Encodes a scalar number or tensor into a bitstring according to the type.

  ## Examples

      iex> Bits.from_scalar(1, {:u, 8})
      <<1>>

      iex> Bits.from_scalar(Nx.tensor(10, type: {:u, 8}), {:u, 8})
      <<10>>

  ### Errors

      iex> Bits.from_scalar(<<0>>, {:u, 8})
      ** (ArgumentError) expected a number or a scalar tensor of type {:u, 8}, got: <<0>>
  """
  def from_scalar(value, type) when is_number(value) do
    from_number(value, type)
  end

  def from_scalar(%T{shape: {}, type: type} = t, type) do
    BinaryBackend.to_binary(t)
  end

  def from_scalar(t, type) do
    raise ArgumentError,
          "expected a number or a scalar tensor of type #{inspect(type)}, got: #{inspect(t)}"
  end

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
  def number_at(bin, type, i) when is_integer(i) do
    to_number(slice(bin, type, i, 1), type)
  end

  @doc """
  Slices a bitstring from the start to the start + len
  according to the type.

  ## Examples

      iex> Bits.slice(<<1, 2, 3, 4>>, {:u, 8}, 1, 2)
      <<2, 3>>

  """
  def slice(bin, {_, sizeof}, start, len) when is_integer(len) and len > 0 do
    bits_offset = start * sizeof
    sizeof_slice = len * sizeof
    <<_::size(bits_offset), sliced::size(sizeof_slice)-bitstring, _::bitstring>> = bin
    sliced
  end
end

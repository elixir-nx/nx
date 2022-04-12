defmodule Nx.Type do
  @moduledoc """
  Conveniences for working with types.

  A type is a two-element tuple with the name and the size.
  The respective sizes for the types are the following:

      * `:s` - signed integer (8, 16, 32, 64)
      * `:u` - unsigned integer (8, 16, 32, 64)
      * `:f` - float (16, 32, 64)
      * `:bf` - a brain floating point (16)
      * `:c` - a complex number, represented as a pair of floats (64, 128)

  Note: there is a special type used by the `defn` compiler
  which is `{:tuple, size}`, that represents a tuple. Said types
  do not appear on user code, only on compiler implementations,
  and therefore are not handled by the functions in this module.
  """

  @type t ::
          {:s, 8}
          | {:s, 16}
          | {:s, 32}
          | {:s, 64}
          | {:u, 8}
          | {:u, 16}
          | {:u, 32}
          | {:u, 64}
          | {:f, 16}
          | {:f, 32}
          | {:f, 64}
          | {:bf, 16}
          | {:c, 64}
          | {:c, 128}
          | {:tuple, non_neg_integer}

  @doc """
  Returns the minimum possible value for the given type.
  """
  def min_finite_binary(type)

  def min_finite_binary({:s, 8}), do: <<-128::8-signed-native>>
  def min_finite_binary({:s, 16}), do: <<-32678::16-signed-native>>
  def min_finite_binary({:s, 32}), do: <<-2_147_483_648::32-signed-native>>
  def min_finite_binary({:s, 64}), do: <<-9_223_372_036_854_775_808::64-signed-native>>
  def min_finite_binary({:u, size}), do: <<0::size(size)-native>>
  def min_finite_binary({:bf, 16}), do: <<0xFF80::16-native>>
  def min_finite_binary({:f, 16}), do: <<0xFBFF::16-native>>
  def min_finite_binary({:f, 32}), do: <<0xFF7FFFFF::32-native>>
  def min_finite_binary({:f, 64}), do: <<0xFFEFFFFFFFFFFFFF::64-native>>

  @doc """
  Returns the maximum possible value for the given type.
  """
  def max_finite_binary(type)

  def max_finite_binary({:s, 8}), do: <<127::8-signed-native>>
  def max_finite_binary({:s, 16}), do: <<32677::16-signed-native>>
  def max_finite_binary({:s, 32}), do: <<2_147_483_647::32-signed-native>>
  def max_finite_binary({:s, 64}), do: <<9_223_372_036_854_775_807::64-signed-native>>
  def max_finite_binary({:u, 8}), do: <<255::8-native>>
  def max_finite_binary({:u, 16}), do: <<65535::16-native>>
  def max_finite_binary({:u, 32}), do: <<4_294_967_295::32-native>>
  def max_finite_binary({:u, 64}), do: <<18_446_744_073_709_551_615::64-native>>
  def max_finite_binary({:bf, 16}), do: <<0x7F80::16-native>>
  def max_finite_binary({:f, 16}), do: <<0x7BFF::16-native>>
  def max_finite_binary({:f, 32}), do: <<0x7F7FFFFF::32-native>>
  def max_finite_binary({:f, 64}), do: <<0x7FEFFFFFFFFFFFFF::64-native>>

  @doc """
  Returns infinity as a binary for the given type.
  """
  def nan_binary(type)
  def nan_binary({:bf, 16}), do: <<0x7F81::16-native>>
  def nan_binary({:f, 16}), do: <<0x7C01::16-native>>
  def nan_binary({:f, 32}), do: <<0x7F800001::32-native>>
  def nan_binary({:f, 64}), do: <<0x7FF0000000000001::64-native>>

  @doc """
  Returns infinity as a binary for the given type.
  """
  def infinity_binary(type)
  def infinity_binary({:bf, 16}), do: <<0x7F80::16-native>>
  def infinity_binary({:f, 16}), do: <<0x7C00::16-native>>
  def infinity_binary({:f, 32}), do: <<0x7F800000::32-native>>
  def infinity_binary({:f, 64}), do: <<0x7FF0000000000000::64-native>>

  @doc """
  Returns negative infinity as a binary for the given type.
  """
  def neg_infinity_binary(type)
  def neg_infinity_binary({:bf, 16}), do: <<0xFF80::16-native>>
  def neg_infinity_binary({:f, 16}), do: <<0xFC00::16-native>>
  def neg_infinity_binary({:f, 32}), do: <<0xFF800000::32-native>>
  def neg_infinity_binary({:f, 64}), do: <<0xFFF0000000000000::64-native>>

  @doc """
  Infers the type of the given number.

  ## Examples

      iex> Nx.Type.infer(1)
      {:s, 64}
      iex> Nx.Type.infer(1.0)
      {:f, 32}
      iex> Nx.Type.infer(Complex.new(1))
      {:c, 64}

  """
  def infer(value) when is_integer(value), do: {:s, 64}
  def infer(value) when is_float(value), do: {:f, 32}
  def infer(%Complex{}), do: {:c, 64}

  @doc """
  Validates the given type tuple.

  It returns the type itself or raises.

  ## Examples

      iex> Nx.Type.normalize!({:u, 8})
      {:u, 8}

      iex> Nx.Type.normalize!({:u, 0})
      ** (ArgumentError) invalid numerical type: {:u, 0} (see Nx.Type docs for all supported types)

      iex> Nx.Type.normalize!({:k, 8})
      ** (ArgumentError) invalid numerical type: {:k, 8} (see Nx.Type docs for all supported types)

  """
  def normalize!(type) do
    case validate(type) do
      :error ->
        raise ArgumentError,
              "invalid numerical type: #{inspect(type)} (see Nx.Type docs for all supported types)"

      type ->
        type
    end
  end

  defp validate({:s, size} = type) when size in [8, 16, 32, 64], do: type
  defp validate({:u, size} = type) when size in [8, 16, 32, 64], do: type
  defp validate({:f, size} = type) when size in [16, 32, 64], do: type
  defp validate({:bf, size} = type) when size in [16], do: type
  defp validate({:c, size} = type) when size in [64, 128], do: type
  defp validate(_type), do: :error

  @doc """
  Converts the given type to a floating point representation
  with the minimum size necessary.

  Note both float and complex are floating point representations.

  ## Examples

      iex> Nx.Type.to_floating({:s, 8})
      {:f, 32}
      iex> Nx.Type.to_floating({:s, 32})
      {:f, 32}
      iex> Nx.Type.to_floating({:bf, 16})
      {:bf, 16}
      iex> Nx.Type.to_floating({:f, 32})
      {:f, 32}
      iex> Nx.Type.to_floating({:c, 64})
      {:c, 64}

  """
  def to_floating({:bf, size}), do: {:bf, size}
  def to_floating({:f, size}), do: {:f, size}
  def to_floating({:c, size}), do: {:c, size}
  def to_floating(type), do: merge(type, {:f, 32})

  @doc """
  Converts the given type to a complex representation with
  the minimum size necessary.

  ## Examples

      iex> Nx.Type.to_complex({:s, 64})
      {:c, 64}
      iex> Nx.Type.to_complex({:bf, 16})
      {:c, 64}
      iex> Nx.Type.to_complex({:f, 32})
      {:c, 64}
      iex> Nx.Type.to_complex({:c, 64})
      {:c, 64}
      iex> Nx.Type.to_complex({:f, 64})
      {:c, 128}
      iex> Nx.Type.to_complex({:c, 128})
      {:c, 128}

  """
  def to_complex({:c, size}), do: {:c, size}
  def to_complex({:f, 64}), do: {:c, 128}
  def to_complex(_type), do: {:c, 64}

  @doc """
  Converts the given type to a real number representation
  with the minimum size necessary.

  ## Examples

      iex> Nx.Type.to_real({:s, 8})
      {:f, 32}
      iex> Nx.Type.to_real({:s, 64})
      {:f, 32}
      iex> Nx.Type.to_real({:bf, 16})
      {:bf, 16}
      iex> Nx.Type.to_real({:c, 64})
      {:f, 32}
      iex> Nx.Type.to_real({:c, 128})
      {:f, 64}
      iex> Nx.Type.to_real({:f, 32})
      {:f, 32}
      iex> Nx.Type.to_real({:f, 64})
      {:f, 64}
  """
  def to_real({:f, size}), do: {:f, size}
  def to_real({:c, s}), do: {:f, div(s, 2)}
  def to_real({:bf, size}), do: {:bf, size}
  def to_real(_type), do: {:f, 32}

  @doc """
  Converts the given type to an aggregation precision.

  ## Examples

      iex> Nx.Type.to_aggregate({:s, 8})
      {:s, 64}
      iex> Nx.Type.to_aggregate({:u, 32})
      {:u, 64}
      iex> Nx.Type.to_aggregate({:bf, 16})
      {:bf, 16}
      iex> Nx.Type.to_aggregate({:f, 32})
      {:f, 32}
      iex> Nx.Type.to_aggregate({:c, 64})
      {:c, 64}

  """
  def to_aggregate({:u, _size}), do: {:u, 64}
  def to_aggregate({:s, _size}), do: {:s, 64}
  def to_aggregate(type), do: type

  @doc """
  Casts the given number to type.

  It does not handle overflow/underflow,
  returning the number as is, but cast.

  ## Examples

      iex> Nx.Type.cast_number!({:u, 8}, 10)
      10
      iex> Nx.Type.cast_number!({:s, 8}, 10)
      10
      iex> Nx.Type.cast_number!({:s, 8}, -10)
      -10
      iex> Nx.Type.cast_number!({:f, 32}, 10)
      10.0
      iex> Nx.Type.cast_number!({:bf, 16}, -10)
      -10.0

      iex> Nx.Type.cast_number!({:f, 32}, 10.0)
      10.0
      iex> Nx.Type.cast_number!({:bf, 16}, -10.0)
      -10.0

      iex> Nx.Type.cast_number!({:c, 64}, 10)
      %Complex{im: 0.0, re: 10.0}

      iex> Nx.Type.cast_number!({:u, 8}, -10)
      ** (ArgumentError) cannot cast number -10 to {:u, 8}

      iex> Nx.Type.cast_number!({:s, 8}, 10.0)
      ** (ArgumentError) cannot cast number 10.0 to {:s, 8}
  """
  def cast_number!({type, _}, int) when type in [:u] and is_integer(int) and int >= 0, do: int
  def cast_number!({type, _}, int) when type in [:s] and is_integer(int), do: int
  def cast_number!({type, _}, int) when type in [:f, :bf] and is_integer(int), do: int * 1.0
  def cast_number!({type, _}, float) when type in [:f, :bf] and is_float(float), do: float
  def cast_number!({:c, _}, number), do: Complex.new(number)

  def cast_number!(type, other) do
    raise ArgumentError, "cannot cast number #{inspect(other)} to #{inspect(type)}"
  end

  @doc """
  Merges the given types finding a suitable representation for both.

  Types have the following precedence:

      c > f > bf > s > u

  If the types are the same, they are merged to the highest size.
  If they are different, the one with the highest precedence wins,
  as long as the size of the `max(big, small * 2))` fits under 64
  bits. Otherwise it casts to f64.

  In the case of complex numbers, the maximum bit size is 128 bits
  because they are composed of two floats.

  ## Examples

      iex> Nx.Type.merge({:s, 8}, {:s, 8})
      {:s, 8}
      iex> Nx.Type.merge({:s, 8}, {:s, 64})
      {:s, 64}

      iex> Nx.Type.merge({:s, 8}, {:u, 8})
      {:s, 16}
      iex> Nx.Type.merge({:s, 16}, {:u, 8})
      {:s, 16}
      iex> Nx.Type.merge({:s, 8}, {:u, 16})
      {:s, 32}
      iex> Nx.Type.merge({:s, 32}, {:u, 8})
      {:s, 32}
      iex> Nx.Type.merge({:s, 8}, {:u, 32})
      {:s, 64}
      iex> Nx.Type.merge({:s, 64}, {:u, 8})
      {:s, 64}
      iex> Nx.Type.merge({:s, 8}, {:u, 64})
      {:s, 64}

      iex> Nx.Type.merge({:u, 8}, {:f, 32})
      {:f, 32}
      iex> Nx.Type.merge({:u, 64}, {:f, 32})
      {:f, 32}
      iex> Nx.Type.merge({:s, 8}, {:f, 32})
      {:f, 32}
      iex> Nx.Type.merge({:s, 64}, {:f, 32})
      {:f, 32}

      iex> Nx.Type.merge({:u, 8}, {:f, 64})
      {:f, 64}
      iex> Nx.Type.merge({:u, 64}, {:f, 64})
      {:f, 64}
      iex> Nx.Type.merge({:s, 8}, {:f, 64})
      {:f, 64}
      iex> Nx.Type.merge({:s, 64}, {:f, 64})
      {:f, 64}

      iex> Nx.Type.merge({:u, 8}, {:bf, 16})
      {:bf, 16}
      iex> Nx.Type.merge({:u, 64}, {:bf, 16})
      {:bf, 16}
      iex> Nx.Type.merge({:s, 8}, {:bf, 16})
      {:bf, 16}
      iex> Nx.Type.merge({:s, 64}, {:bf, 16})
      {:bf, 16}

      iex> Nx.Type.merge({:f, 32}, {:bf, 16})
      {:f, 32}
      iex> Nx.Type.merge({:f, 64}, {:bf, 16})
      {:f, 64}

      iex> Nx.Type.merge({:c, 64}, {:f, 32})
      {:c, 64}
      iex> Nx.Type.merge({:c, 64}, {:c, 64})
      {:c, 64}
      iex> Nx.Type.merge({:c, 128}, {:c, 64})
      {:c, 128}
  """
  def merge({type, left_size}, {type, right_size}) do
    {type, max(left_size, right_size)}
  end

  def merge(left, right) do
    case sort(left, right) do
      {{:u, size1}, {:s, size2}} -> {:s, max(min(size1 * 2, 64), size2)}
      {_, type2} -> type2
    end
  end

  defp type_to_int(:c), do: 4
  defp type_to_int(:f), do: 3
  defp type_to_int(:bf), do: 2
  defp type_to_int(:s), do: 1
  defp type_to_int(:u), do: 0

  defp sort({left_type, _} = left, {right_type, _} = right) do
    if type_to_int(left_type) < type_to_int(right_type) do
      {left, right}
    else
      {right, left}
    end
  end

  @doc """
  Merges the given types with the type of a number.

  We attempt to keep the original type and its size as best
  as possible.

  ## Examples

      iex> Nx.Type.merge_number({:u, 8}, 0)
      {:u, 8}
      iex> Nx.Type.merge_number({:u, 8}, 255)
      {:u, 8}
      iex> Nx.Type.merge_number({:u, 8}, 256)
      {:u, 16}
      iex> Nx.Type.merge_number({:u, 8}, -1)
      {:s, 16}
      iex> Nx.Type.merge_number({:u, 8}, -32767)
      {:s, 16}
      iex> Nx.Type.merge_number({:u, 8}, -32768)
      {:s, 16}
      iex> Nx.Type.merge_number({:u, 8}, -32769)
      {:s, 32}

      iex> Nx.Type.merge_number({:s, 8}, 0)
      {:s, 8}
      iex> Nx.Type.merge_number({:s, 8}, 127)
      {:s, 8}
      iex> Nx.Type.merge_number({:s, 8}, -128)
      {:s, 8}
      iex> Nx.Type.merge_number({:s, 8}, 128)
      {:s, 16}
      iex> Nx.Type.merge_number({:s, 8}, -129)
      {:s, 16}
      iex> Nx.Type.merge_number({:s, 8}, 1.0)
      {:f, 32}
      iex> Nx.Type.merge_number({:u, 64}, -1337)
      {:s, 64}

      iex> Nx.Type.merge_number({:f, 32}, 1)
      {:f, 32}
      iex> Nx.Type.merge_number({:f, 32}, 1.0)
      {:f, 32}
      iex> Nx.Type.merge_number({:f, 64}, 1.0)
      {:f, 64}

  """
  def merge_number({:u, size}, integer) when is_integer(integer) and integer >= 0 do
    {:u, max(unsigned_size(integer), size)}
  end

  def merge_number({:u, size}, integer) when is_integer(integer) do
    merge_number({:s, min(size * 2, 64)}, integer)
  end

  def merge_number({:s, size}, integer) when is_integer(integer) do
    {:s, max(signed_size(integer), size)}
  end

  def merge_number({:bf, size}, number) when is_number(number) do
    {:bf, size}
  end

  def merge_number({:f, size}, number) when is_number(number) do
    {:f, size}
  end

  def merge_number({:c, size}, _number), do: {:c, size}

  def merge_number(_, number) when is_number(number) do
    {:f, 32}
  end

  @doc """
  Returns true if the type is an integer in Elixir.

  ## Examples

      iex> Nx.Type.integer?({:s, 8})
      true
      iex> Nx.Type.integer?({:u, 64})
      true
      iex> Nx.Type.integer?({:f, 64})
      false
  """
  def integer?({:u, _}), do: true
  def integer?({:s, _}), do: true
  def integer?({_, _}), do: false

  @doc """
  Returns true if the type is a float in Elixir.

  ## Examples

      iex> Nx.Type.float?({:f, 32})
      true
      iex> Nx.Type.float?({:bf, 16})
      true
      iex> Nx.Type.float?({:u, 64})
      false
  """
  def float?({:f, _}), do: true
  def float?({:bf, _}), do: true
  def float?({:c, _}), do: true
  def float?({_, _}), do: false

  @doc """
  Returns a string representation of the given type.

  ## Examples

      iex> Nx.Type.to_string({:s, 8})
      "s8"
      iex> Nx.Type.to_string({:s, 16})
      "s16"
      iex> Nx.Type.to_string({:s, 32})
      "s32"
      iex> Nx.Type.to_string({:s, 64})
      "s64"
      iex> Nx.Type.to_string({:u, 8})
      "u8"
      iex> Nx.Type.to_string({:u, 16})
      "u16"
      iex> Nx.Type.to_string({:u, 32})
      "u32"
      iex> Nx.Type.to_string({:u, 64})
      "u64"
      iex> Nx.Type.to_string({:f, 16})
      "f16"
      iex> Nx.Type.to_string({:bf, 16})
      "bf16"
      iex> Nx.Type.to_string({:f, 32})
      "f32"
      iex> Nx.Type.to_string({:f, 64})
      "f64"
  """
  def to_string({type, size}), do: Atom.to_string(type) <> Integer.to_string(size)

  defp unsigned_size(x) when x <= 1, do: 1
  defp unsigned_size(x) when x <= 255, do: 8
  defp unsigned_size(x) when x <= 65535, do: 16
  defp unsigned_size(x) when x <= 4_294_967_295, do: 32
  defp unsigned_size(_), do: 64

  defp signed_size(x) when x < 0, do: signed_size(-x - 1)
  defp signed_size(x) when x <= 1, do: 1
  defp signed_size(x) when x <= 127, do: 8
  defp signed_size(x) when x <= 32767, do: 16
  defp signed_size(x) when x <= 2_147_483_647, do: 32
  defp signed_size(_), do: 64
end

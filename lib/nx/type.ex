defmodule Nx.Type do
  @moduledoc """
  Conveniences for working with types.

  A type is a two-element tuple. The first element is one of:

      * `:s` - signed integer
      * `:u` - unsigned integer
      * `:f` - float

  The second element is the size. Integers can have any size
  between 1 and 64. Floats may be 32 and 64 bits.
  """

  @doc """
  Infers the type of the given value.

  The value may be a number, boolean, or an arbitrary list with
  any of the above. Integers are by default signed and of size 64.
  Floats have size of 64. Booleans are unsigned integers of size 1
  (also known as predicates).

  In case mixed types are given, the one with highest space
  requirements is used (i.e. float > integer > boolean).

  ## Examples

      iex> Nx.Type.infer(true)
      {:u, 1}
      iex> Nx.Type.infer(false)
      {:u, 1}

      iex> Nx.Type.infer([1, 2, 3])
      {:s, 64}
      iex> Nx.Type.infer([[1, 2], [3, 4]])
      {:s, 64}

      iex> Nx.Type.infer([1.0, 2.0, 3.0])
      {:f, 64}

      iex> Nx.Type.infer([1, true, 2])
      {:s, 64}
      iex> Nx.Type.infer([1, true, 2.0])
      {:f, 64}

      iex> Nx.Type.infer([])
      {:f, 64}

      iex> Nx.Type.infer("string")
      ** (ArgumentError) cannot infer the numerical type of "string"

  """
  def infer(value) do
    case infer(value, -1) do
      -1 -> {:f, 64}
      0 -> {:u, 1}
      1 -> {:s, 64}
      2 -> {:f, 64}
    end
  end

  defp infer(arg, inferred) when is_list(arg), do: Enum.reduce(arg, inferred, &infer/2)
  defp infer(arg, inferred) when is_boolean(arg), do: max(inferred, 0)
  defp infer(arg, inferred) when is_integer(arg), do: max(inferred, 1)
  defp infer(arg, inferred) when is_float(arg), do: max(inferred, 2)

  defp infer(other, _inferred),
    do: raise(ArgumentError, "cannot infer the numerical type of #{inspect(other)}")

  @doc """
  Validates the given type tuple.

  It returns the type itself or raises.

  ## Examples

      iex> Nx.Type.validate!({:s, 1})
      {:s, 1}

      iex> Nx.Type.validate!({:s, 0})
      ** (ArgumentError) invalid numerical type: {:s, 0} (see Nx.Type docs for all supported types)

      iex> Nx.Type.validate!({:k, 1})
      ** (ArgumentError) invalid numerical type: {:k, 1} (see Nx.Type docs for all supported types)

  """
  def validate!(type)

  def validate!({:s, size} = type) when size in 1..64, do: type
  def validate!({:u, size} = type) when size in 1..64, do: type
  def validate!({:f, size} = type) when size in [32, 64], do: type

  def validate!(type) do
    raise ArgumentError,
          "invalid numerical type: #{inspect(type)} (see Nx.Type docs for all supported types)"
  end

  @doc """
  Merges the given types finding a suitable representation for both.

  Types have the following precedence:

      u < s < f

  If the types are the same, they are merged to the highest size.
  If they are different, the one with the highest precedence wins,
  as long as the size of the `max(big, small * 2))` fits under 64
  bits. Otherwise it casts to f64.

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
      {:f, 64}

      iex> Nx.Type.merge({:u, 8}, {:f, 32})
      {:f, 32}
      iex> Nx.Type.merge({:u, 16}, {:f, 32})
      {:f, 32}
      iex> Nx.Type.merge({:u, 32}, {:f, 32})
      {:f, 64}
      iex> Nx.Type.merge({:s, 8}, {:f, 32})
      {:f, 32}
      iex> Nx.Type.merge({:s, 16}, {:f, 32})
      {:f, 32}
      iex> Nx.Type.merge({:s, 32}, {:f, 32})
      {:f, 64}

  """
  def merge(left, right) do
    # Sorting right now is straight-forward because
    # the type ordering is also the lexical ordering.
    merge_sorted(min(left, right), max(left, right))
  end

  defp merge_sorted({type, size1}, {type, size2}) do
    {type, max(size1, size2)}
  end

  defp merge_sorted({type1, size1}, {_type2, size2}) do
    max = max(size1, size2 * 2)
    if max > 64, do: {:f, 64}, else: {type1, max}
  end

  @doc """
  Merges the given types with the type of a scalar.

  We attempt to keep the original type and its size as best
  as possible.

  ## Examples

      iex> Nx.Type.merge_scalar({:u, 8}, 0)
      {:u, 8}
      iex> Nx.Type.merge_scalar({:u, 8}, 255)
      {:u, 8}
      iex> Nx.Type.merge_scalar({:u, 8}, 256)
      {:u, 16}
      iex> Nx.Type.merge_scalar({:u, 8}, -1)
      {:s, 16}
      iex> Nx.Type.merge_scalar({:u, 8}, -32767)
      {:s, 16}
      iex> Nx.Type.merge_scalar({:u, 8}, -32768)
      {:s, 16}
      iex> Nx.Type.merge_scalar({:u, 8}, -32769)
      {:s, 32}

      iex> Nx.Type.merge_scalar({:s, 8}, 0)
      {:s, 8}
      iex> Nx.Type.merge_scalar({:s, 8}, 127)
      {:s, 8}
      iex> Nx.Type.merge_scalar({:s, 8}, -128)
      {:s, 8}
      iex> Nx.Type.merge_scalar({:s, 8}, 128)
      {:s, 16}
      iex> Nx.Type.merge_scalar({:s, 8}, -129)
      {:s, 16}
      iex> Nx.Type.merge_scalar({:s, 8}, 1.0)
      {:f, 64}

      iex> Nx.Type.merge_scalar({:f, 32}, 1)
      {:f, 32}
      iex> Nx.Type.merge_scalar({:f, 32}, 1.0)
      {:f, 32}
      iex> Nx.Type.merge_scalar({:f, 64}, 1.0)
      {:f, 64}

  """
  def merge_scalar({:u, size}, integer) when is_integer(integer) and integer >= 0 do
    {:u, max(unsigned_size(integer), size)}
  end

  def merge_scalar({:u, size}, integer) when is_integer(integer) do
    merge_scalar({:s, size * 2}, integer)
  end

  def merge_scalar({:s, size}, integer) when is_integer(integer) do
    {:s, max(signed_size(integer), size)}
  end

  def merge_scalar({:f, size}, number) when is_number(number) do
    {:f, size}
  end

  def merge_scalar(_, number) when is_number(number) do
    {:f, 64}
  end

  defp unsigned_size(x) when x <= 1, do: 1
  defp unsigned_size(x) when x <= 255, do: 8
  defp unsigned_size(x) when x <= 65535, do: 16
  defp unsigned_size(x) when x <= 4294967295, do: 32
  defp unsigned_size(_), do: 64

  defp signed_size(x) when x < 0, do: signed_size(-x - 1)
  defp signed_size(x) when x <= 1, do: 1
  defp signed_size(x) when x <= 127, do: 8
  defp signed_size(x) when x <= 32767, do: 16
  defp signed_size(x) when x <= 2147483647, do: 32
  defp signed_size(_), do: 64
end

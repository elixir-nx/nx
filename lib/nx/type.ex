defmodule Nx.Type do
  @moduledoc """
  Conveniences for working with types.

  A type is a two-element tuple with the name and the size.
  The first element must be one of followed by the respective
  sizes:

      * `:s` - signed integer (8, 16, 32, 64)
      * `:u` - unsigned integer (1, 8, 16, 32, 64)
      * `:f` - float (32, 64)
      * `:bf` - a brain floating point (16)

  """

  @doc """
  Infers the type of the given value.

  The value may be a number, boolean, or an arbitrary list with
  any of the above. Integers are by default signed and of size 64.
  Floats have size of 64. Booleans are unsigned integers of size 1
  (also known as predicates).

  In case mixed types are given, the one with highest space
  requirements is used (i.e. float > brain floating > integer > boolean).

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

      iex> Nx.Type.validate!({:u, 1})
      {:u, 1}

      iex> Nx.Type.validate!({:u, 0})
      ** (ArgumentError) invalid numerical type: {:u, 0} (see Nx.Type docs for all supported types)

      iex> Nx.Type.validate!({:k, 1})
      ** (ArgumentError) invalid numerical type: {:k, 1} (see Nx.Type docs for all supported types)

  """
  def validate!(type) do
    case validate(type) do
      :error ->
        raise ArgumentError,
              "invalid numerical type: #{inspect(type)} (see Nx.Type docs for all supported types)"

      type ->
        type
    end
  end

  def validate({:s, size} = type) when size in [8, 16, 32, 64], do: type
  def validate({:u, size} = type) when size in [1, 8, 16, 32, 64], do: type
  def validate({:f, size} = type) when size in [32, 64], do: type
  def validate({:bf, size} = type) when size in [16], do: type
  def validate(_type), do: :error

  @doc """
  Converts the given type to a float with the minimum size necessary.

  ## Examples

      iex> Nx.Type.to_float({:s, 8})
      {:f, 32}
      iex> Nx.Type.to_float({:s, 32})
      {:f, 64}
      iex> Nx.Type.to_float({:bf, 16})
      {:bf, 16}
      iex> Nx.Type.to_float({:f, 32})
      {:f, 32}

  """
  def to_float({:bf, size}), do: {:bf, size}
  def to_float({:f, size}), do: {:f, size}
  def to_float(type), do: merge(type, {:f, 32})

  @doc """
  Casts scalar the given scalar to type.

  It does not handle overflows/underfows,
  returning the scalar as is, but cast.

  ## Examples

      iex> Nx.Type.cast_scalar!(10, {:u, 8})
      10
      iex> Nx.Type.cast_scalar!(10, {:s, 8})
      10
      iex> Nx.Type.cast_scalar!(-10, {:s, 8})
      -10
      iex> Nx.Type.cast_scalar!(10, {:f, 32})
      10.0
      iex> Nx.Type.cast_scalar!(-10, {:bf, 16})
      -10.0

      iex> Nx.Type.cast_scalar!(10.0, {:f, 32})
      10.0
      iex> Nx.Type.cast_scalar!(-10.0, {:bf, 16})
      -10.0

      iex> Nx.Type.cast_scalar!(-10, {:u, 8})
      ** (ArgumentError) cannot cast scalar -10 to {:u, 8}

      iex> Nx.Type.cast_scalar!(10.0, {:s, 8})
      ** (ArgumentError) cannot cast scalar 10.0 to {:s, 8}
  """
  def cast_scalar!(int, {type, _}) when type in [:u] and is_integer(int) and int >= 0, do: int
  def cast_scalar!(int, {type, _}) when type in [:s] and is_integer(int), do: int
  def cast_scalar!(int, {type, _}) when type in [:f, :bf] and is_integer(int), do: int * 1.0
  def cast_scalar!(float, {type, _}) when type in [:f, :bf] and is_float(float), do: float

  def cast_scalar!(other, type) do
    raise ArgumentError, "cannot cast scalar #{inspect(other)} to #{inspect(type)}"
  end

  @doc """
  Merges the given types finding a suitable representation for both.

  Types have the following precedence:

      f > bf > s > u

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

      iex> Nx.Type.merge({:s, 8}, {:bf, 16})
      {:bf, 16}
      iex> Nx.Type.merge({:s, 16}, {:bf, 16})
      {:f, 32}
      iex> Nx.Type.merge({:s, 32}, {:bf, 16})
      {:f, 64}

      iex> Nx.Type.merge({:f, 32}, {:bf, 16})
      {:f, 32}
      iex> Nx.Type.merge({:f, 64}, {:bf, 16})
      {:f, 64}

  """
  def merge({type, left_size}, {type, right_size}) do
    {type, max(left_size, right_size)}
  end

  def merge(left, right) do
    # Sorting right now is straight-forward because
    # the type ordering is also the lexical ordering.
    {{_type1, size1}, {type2, size2}} = sort(left, right)
    candidate = {type2, max(size1 * 2, size2)}

    case validate(candidate) do
      :error ->
        case candidate do
          {:bf, 32} -> {:f, 32}
          _ -> {:f, 64}
        end

      type ->
        type
    end
  end

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

  def merge_scalar({:bf, size}, number) when is_number(number) do
    {:bf, size}
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
  defp unsigned_size(x) when x <= 4_294_967_295, do: 32
  defp unsigned_size(_), do: 64

  defp signed_size(x) when x < 0, do: signed_size(-x - 1)
  defp signed_size(x) when x <= 1, do: 1
  defp signed_size(x) when x <= 127, do: 8
  defp signed_size(x) when x <= 32767, do: 16
  defp signed_size(x) when x <= 2_147_483_647, do: 32
  defp signed_size(_), do: 64
end

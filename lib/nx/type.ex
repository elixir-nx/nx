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
  Floats have size of 64. Booleans are integers of size 1 (also known
  as predicates).

  In case mixed types are given, the one with highest space
  requirements are taken (i.e. float > integer > boolean).
  """
  def infer(value) do
    case infer(value, -1) do
      -1 -> {:f, 64}
      0 -> {:s, 1}
      1 -> {:s, 64}
      2 -> {:f, 64}
    end
  end

  defp infer(arg, inferred) when is_list(arg), do: Enum.reduce(arg, inferred, &infer/2)
  defp infer(arg, inferred) when is_boolean(arg), do: max(inferred, 0)
  defp infer(arg, inferred) when is_integer(arg), do: max(inferred, 1)
  defp infer(arg, inferred) when is_float(arg), do: max(inferred, 2)

  defp infer(other, _inferred),
    do: raise(ArgumentError, "cannot infer the type for #{inspect(other)}")

  @doc """
  Validates the given type tuple.
  """
  def validate!(type)

  def validate!({:s, size}) when size in 1..64, do: :ok
  def validate!({:u, size}) when size in 1..64, do: :ok
  def validate!({:f, size}) when size in [32, 64], do: :ok

  @doc """
  Merges the given types finding a suitable representation for both.

  Types have the following precedence:

      u < s < f

  If the types are the same, they are merged to the highest size.
  If they are different, the one with the highest precedence wins,
  as long as the size of the `max(big, small * 2))` fits under 64
  bits. Otherwise it casts to f64.
  """
  def merge(left, right) do
    # Sorting right now is straight-forward because
    # the type ordering is also the lexical ordering.
    merge_sorted(min(left, right), max(left, right))
  end

  defp merge_sorted({type, size1}, {type, size2}) do
    {type, max(size1, size2)}
  end

  defp merge_sorted({type1, size1}, {type2, size2}) do
    max = max(size1, size2 * 2)
    if max > 64, do: {:f, 64}, else: {type1, max}
  end
end

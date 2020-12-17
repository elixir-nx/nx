<<<<<<< HEAD
defmodule EXLA.Computation do
  @moduledoc """
  Wrapper around XLA's computation.
  """
=======
defmodule Exla.Computation do
  alias __MODULE__
>>>>>>> Initial example of AOT Compilation

  @enforce_keys [:ref, :output_shape]
  defstruct [:ref, :output_shape]

  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end

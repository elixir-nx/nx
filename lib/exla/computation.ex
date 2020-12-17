defmodule EXLA.Computation do
  @moduledoc """
  Wrapper around XLA's computation.
  """
  @enforce_keys [:ref, :output_shape]
  defstruct [:ref, :output_shape]
end

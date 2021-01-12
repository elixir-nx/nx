defmodule EXLA.Computation do
  @enforce_keys [:ref, :output_shape]
  defstruct [:ref, :output_shape]
end

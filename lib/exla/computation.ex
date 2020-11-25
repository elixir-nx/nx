defmodule Exla.Computation do
  alias __MODULE__, as: Computation
  alias Exla.Shape

  @enforce_keys [:ref, :output_shape]
  defstruct [:ref, :output_shape]
end

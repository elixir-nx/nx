defmodule Exla.Builder do
  alias Exla.Op
  alias Exla.Computation
  @enforce_keys [:ref]
  defstruct [:ref]

  # TODO: When we get rid of the global instance of `xla::XlaBuilder` in `exla.cc`, we
  # need to add a creation method for this module. It's actually okay to have multiple
  # builders for different computations. One use case I can see is if there are multiple
  # numerical functions in a module, we can spawn a builder for each function. There's a
  # stipulation though that operations are owned by whatever builder created them. If I
  # defined a constant/parameter on one builder, every operation that uses those constants/parameters
  # is owned by the builder that created them.

  # TODO: Add builder attribute once we get rid of global builder
  def build(root = %Op{}) do
    case Exla.NIF.build(root.ref) do
      {:ok, ref} -> {:ok, %Computation{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end
end

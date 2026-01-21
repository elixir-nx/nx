defmodule Nx.Defn.Mesh do
  @moduledoc """
  A mesh is a named collection of devices arranged in a logical shape.

  `name` is a string identifier for the mesh in the lowered program so that
  sharding annotations can refer to a specific device topology without
  embedding concrete device handles directly in the intermediate
  representation.

  `shape` is a tuple describing the logical layout of devices, where each
  element is the size of a mesh dimension. For instance, a shape like
  `{2, 4}` represents a 2x4 logical grid of devices.
  """
  defstruct [:name, :shape]
  @type t :: %__MODULE__{name: String.t(), shape: tuple()}
end

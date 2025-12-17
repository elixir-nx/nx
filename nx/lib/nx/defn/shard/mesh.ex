defmodule Nx.Defn.Shard.Mesh do
  defstruct [:name, :shape]
  @type t :: %__MODULE__{name: String.t(), shape: tuple()}
end

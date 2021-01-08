defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.
  """

  @type data :: {module, term}
  @type type :: Nx.Type.t()
  @type shape :: tuple()

  @enforce_keys [:type, :shape]
  defstruct [:data, :type, :shape]

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(tensor, opts) do
      inner = tensor.data.__struct__.inspect(tensor, opts)

      color("#Nx.Tensor<", :map, opts)
      |> concat(nest(concat(line(), inner), 2))
      |> concat(color("\n>", :map, opts))
    end
  end
end

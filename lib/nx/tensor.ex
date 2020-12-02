defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.

  All of its fields are private. You can access tensor
  metadata via the functions in the Nx module.
  """
  # TODO: Implementation is inefficient as is

  @type data :: {module, term}
  @type type :: Nx.Type.t
  @type shape :: tuple()

  @enforce_keys [:data, :type, :shape]
  defstruct [:data, :type, :shape]

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(tensor, _opts) do

      case tensor.data do
        {Nx.BitStringDevice, bin} ->
          data =
            tensor.shape
            |> Tuple.to_list()
            |> Enum.reverse()
            |> Enum.reduce(bin_to_list(bin, tensor.type),
                fn x, acc ->
                  Enum.chunk_every(acc, x)
                end
              )
            |> hd()

          concat([
            "#Nx.Tensor<\n",
            Nx.Type.to_string(tensor.type),
            shape_to_string(tensor.shape),
            "\n",
            Inspect.Algebra.nest(Inspect.Algebra.to_doc(data, opts), 2),
            "\n>"
          ])
        {device, _} ->
          concat([
            "Nx.Tensor\n",
            Nx.Type.to_string(tensor.type),
            shape_to_string(tensor.shape),
            "\n",
            inspect(device),
            "\n>"
          ])
      end
    end

    defp bin_to_list(bin, type) do
      case type do
        {:s, size} -> for <<x::size(size)-native <- bin>>, do: x
        {:u, size} -> for <<x::size(size)-unsigned-native <- bin>>, do: x
        {:f, size} -> for <<x::size(size)-float-native <- bin>>, do: x
        {_, size} -> for <<x::size(size)-float-native <- bin>>, do: x
      end
    end

    defp shape_to_string(shape) do
      shape
      |> Tuple.to_list()
      |> Enum.map(&"[#{&1}]")
      |> IO.iodata_to_binary()
    end
  end
end

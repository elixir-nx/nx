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

    def inspect(tensor, opts) do

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
            Inspect.Algebra.nest(format_list_or_numbers(data, opts), 2),
            "\n>"
          ])
        {device, _} ->
          concat([
            "#Nx.Tensor<\n",
            Nx.Type.to_string(tensor.type),
            shape_to_string(tensor.shape),
            "\n",
            Inspect.Algebra.nest(Inspect.Algebra.to_doc(device, opts), 2),
            "\n>"
          ])
      end
    end

    @compile {:inline, read_bf16: 1}
    if System.endianness() == :little do
      defp read_bf16(bf16) do
        <<x::float-little-32>> = <<0::16, bf16::binary>>
        Float.to_string(x)
      end
    else
      defp read_bf16(bf16) do
        <<x::float-big-32>> = <<bf16::binary, 0::16>>
        Float.to_string(x)
      end
    end

    defp read_float(data, size) do
      case data do
        <<0xFF800000::size(size)-float-native>> -> "-Infinity"
        <<0x7F800000::size(size)-float-native>> -> "Infinity"
        <<0x7FC00000::size(size)-float-native>> -> "NaN"
        <<x::size(size)-float-native>> -> Float.to_string(x)
      end
    end

    defp bin_to_list(bin, type) do
      case type do
        {:s, size} -> for <<x::size(size)-native <- bin>>, do: Integer.to_string(x)
        {:u, size} -> for <<x::size(size)-unsigned-native <- bin>>, do: Integer.to_string(x)
        {:f, size} -> for <<x::size(size)-bitstring <- bin>>, do: read_float(x, size)
        {:bf, 16} -> for <<x::16-bitstring <- bin>>, do: read_bf16(x)
      end
    end

    defp format_list_or_numbers(x, opts) when is_list(x) do
      open = Inspect.Algebra.color("[", :list, opts)
      sep = Inspect.Algebra.color(",", :list, opts)
      close = Inspect.Algebra.color("]", :list, opts)
      Inspect.Algebra.container_doc(open, x, close, opts, &format_list_or_numbers/2, separator: sep)
    end
    defp format_list_or_numbers(x, _opts), do: Inspect.Algebra.string(x)

    defp shape_to_string(shape) do
      shape
      |> Tuple.to_list()
      |> Enum.map(&"[#{&1}]")
      |> IO.iodata_to_binary()
    end
  end
end

defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.

  All of its fields are private. You can access tensor
  metadata via the functions in the Nx module.
  """

  @type data :: {module, term}
  @type type :: Nx.Type.t
  @type shape :: tuple()

  @enforce_keys [:data, :type, :shape]
  defstruct [:data, :type, :shape]

  defimpl Inspect do
    import Inspect.Algebra

    # TODO: Implementation is inefficient as is, we should limit
    # the amount of data alongside each dimension
    # TODO: Support inspect data on the device
    def inspect(tensor, opts) do
      dims_list = Tuple.to_list(tensor.shape)

      open = color("#Nx.Tensor<", :map, opts)
      close = color("\n>", :map, opts)
      type = color(Nx.Type.to_string(tensor.type), :atom, opts)
      shape = shape_to_algebra(dims_list, opts)

      data =
        case tensor.data do
          {Nx.BitStringDevice, bin} ->
            {_, size} = tensor.type
            total_size = Enum.reduce(dims_list, size, &*/2)

            dims_list
            |> chunk(bin, total_size, tensor.type)
            |> format_list_or_numbers(opts)

          {device, _} ->
            to_doc(device, opts)
        end

      inner = concat([line(), type, shape, line(), data])
      concat([open, nest(inner, 2), close])
    end

    defp shape_to_algebra(dims_list, opts) do
      open = color("[", :list, opts)
      close = color("]", :list, opts)

      dims_list
      |> Enum.map(fn number -> concat([open, Integer.to_string(number), close]) end)
      |> concat()
    end

    defp chunk([], data, size, type) do
      # Pretend it has a dimension
      hd(chunk([size], data, size, type))
    end

    defp chunk([_], data, _size, {type, size}) do
      case type do
        :s -> for <<x::size(size)-native <- data>>, do: Integer.to_string(x)
        :u -> for <<x::size(size)-unsigned-native <- data>>, do: Integer.to_string(x)
        :f -> for <<x::size(size)-bitstring <- data>>, do: read_float(x, size)
        :bf -> for <<x::16-bitstring <- data>>, do: read_bf16(x)
      end
    end

    defp chunk([size | rest], data, total_size, type) do
      chunk_size = div(total_size, size)

      for <<chunk::size(chunk_size)-bitstring <- data>> do
        chunk(rest, chunk, chunk_size, type)
      end
    end

    defp read_bf16(<<0xFF80::16-native>>), do: "-Inf"
    defp read_bf16(<<0x7F80::16-native>>), do: "Inf"
    defp read_bf16(<<0xFFC1::16-native>>), do: "NaN"
    defp read_bf16(<<0xFF81::16-native>>), do: "NaN"

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

    defp read_float(data, 32) do
      case data do
        <<0xFF800000::32-native>> -> "-Inf"
        <<0x7F800000::32-native>> -> "Inf"
        <<0xFF800001::32-native>> -> "NaN"
        <<0xFFC00001::32-native>> -> "NaN"
        <<x::float-32-native>> -> Float.to_string(x)
      end
    end

    defp read_float(data, 64) do
      case data do
        <<0xFFF0000000000000::64-native>> -> "-Inf"
        <<0x7FF0000000000000::64-native>> -> "Inf"
        <<0x7FF0000000000001::64-native>> -> "NaN"
        <<0x7FF8000000000001::64-native>> -> "NaN"
        <<x::float-64-native>> -> Float.to_string(x)
      end
    end

    defp format_list_or_numbers(x, opts) when is_list(x) do
      open = color("[", :list, opts)
      sep = color(",", :list, opts)
      close = color("]", :list, opts)
      container_doc(open, x, close, opts, &format_list_or_numbers/2, separator: sep)
    end

    defp format_list_or_numbers(x, _opts), do: Inspect.Algebra.string(x)
  end
end

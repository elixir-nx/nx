defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.

  All of its fields are private. You can access tensor
  metadata via the functions in the Nx module.
  """
  # TODO: implement the inspect protocol.

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
            case tensor.type do
              {:s, size} -> decode(bin, div(size, 8), 0, byte_size(bin), &decode_signed/1)
              {:u, size} -> decode(bin, div(size, 8), 0, byte_size(bin), &decode_unsigned/1)
              {:f, size} -> decode(bin, div(size, 8), 0, byte_size(bin), &decode_float/1)
              {_, size} -> decode(bin, div(size, 8), 0, byte_size(bin), &decode_float/1)
            end

          concat([
            "#Nx.Tensor<",
            break("\n\t"),
            string(List.to_string(Nx.Type.dtype_to_charlist(tensor.type))),
            inspect(Tuple.to_list(tensor.shape)),
            break("\n\t"),
            inspect(data),
            break("\n"),
            ">"
          ])

        {device, _} ->
          concat([
            "Nx.Tensor<",
            break("\n\t"),
            string(List.to_string(Nx.Type.dtype_to_charlist(tensor.type))),
            inspect(Tuple.to_list(tensor.shape)),
            break("\n\t"),
            inspect(device),
            break("\n"),
            ">"
          ])
      end
    end

    defp decode(binary, bytes, pos, bytes, decode_fn),
      do: [decode_segment(binary, bytes, pos, decode_fn)]

    defp decode(binary, bytes, pos, left, decode_fn) do
      [decode_segment(binary, bytes, pos, decode_fn) | decode(binary, bytes, pos+bytes, left-bytes, decode_fn)]
    end

    defp decode_segment(binary, bytes, pos, decode_fn) do
      binary
      |> :binary.part(pos, bytes)
      |> decode_fn.()
    end

    defp decode_signed(data) do
      case data do
        <<x::8-native>> -> x
        <<x::16-native>> -> x
        <<x::32-native>> -> x
        <<x::64-native>> -> x
      end
    end

    defp decode_unsigned(data) do
      case data do
        <<x::8-unsigned-native>> -> x
        <<x::16-unsigned-native>> -> x
        <<x::32-unsigned-native>> -> x
        <<x::64-unsigned-native>> -> x
      end
    end

    defp decode_float(data) do
      case data do
        <<x::float-32-native>> -> x
        <<x::float-64-native>> -> x
      end
    end
  end
end

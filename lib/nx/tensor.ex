defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.

  All of its fields are private. You can access tensor
  metadata via the functions in the Nx module.
  """

  @type data :: {module, term}
  @type type :: Nx.Type.t()
  @type shape :: tuple()

  @enforce_keys [:type, :shape]
  defstruct [:data, :type, :shape]

  defimpl Inspect do
    import Inspect.Algebra

    # TODO: To print data on device, we can support reading a slice
    # from the device which we will compute with:
    #
    #     min(opts.limit, Nx.Shape.size(shape)) * size
    #
    def inspect(tensor, opts) do
      dims_list = Tuple.to_list(tensor.shape)

      open = color("[", :list, opts)
      sep = color(",", :list, opts)
      close = color("]", :list, opts)
      type = color(Nx.Type.to_string(tensor.type), :atom, opts)
      shape = shape_to_algebra(dims_list, open, close)

      {data, _limit} =
        case tensor.data do
          {Nx.BitStringDevice, bin} ->
            {_, size} = tensor.type
            total_size = Enum.reduce(dims_list, size, &*/2)
            chunk(dims_list, bin, opts.limit, total_size, tensor.type, {open, sep, close})

          {device, _} ->
            {to_doc(device, opts), opts.limit}
        end

      inner = concat([line(), type, shape, line(), data])

      color("#Nx.Tensor<", :map, opts)
      |> concat(nest(inner, 2))
      |> concat(color("\n>", :map, opts))
    end

    defp shape_to_algebra(dims_list, open, close) do
      dims_list
      |> Enum.map(fn number -> concat([open, Integer.to_string(number), close]) end)
      |> concat()
    end

    defp chunk([], data, limit, size, {type, size}, _docs) do
      doc =
        case type do
          :s ->
            <<x::size(size)-signed-native>> = data
            Integer.to_string(x)

          :u ->
            <<x::size(size)-unsigned-native>> = data
            Integer.to_string(x)

          :pred ->
            <<x::size(size)-unsigned-native>> = data
            Integer.to_string(x)

          :f ->
            <<x::size(size)-bitstring>> = data
            read_float(x, size)

          :bf ->
            <<x::16-bitstring>> = data
            read_bf16(x)
        end

      if limit == :infinity, do: {doc, limit}, else: {doc, limit - 1}
    end

    defp chunk([dim | dims], data, limit, total_size, type, {open, sep, close} = docs) do
      chunk_size = div(total_size, dim)

      {acc, limit} =
        chunk_each(dim, data, [], limit, chunk_size, fn chunk, limit ->
          chunk(dims, chunk, limit, chunk_size, type, docs)
        end)

      {open, sep, close, nest} =
        if dims == [] do
          {open, concat(sep, " "), close, 0}
        else
          {concat(open, line()), concat(sep, line()), concat(line(), close), 2}
        end

      doc =
        open
        |> concat(concat(Enum.intersperse(acc, sep)))
        |> nest(nest)
        |> concat(close)

      {doc, limit}
    end

    defp chunk_each(0, "", acc, limit, _size, _fun) do
      {Enum.reverse(acc), limit}
    end

    defp chunk_each(_dim, _data, acc, 0, _size, _fun) do
      {Enum.reverse(["..." | acc]), 0}
    end

    defp chunk_each(dim, data, acc, limit, size, fun) when dim > 0 do
      <<chunk::size(size)-bitstring, rest::bitstring>> = data
      {doc, limit} = fun.(chunk, limit)
      chunk_each(dim - 1, rest, [doc | acc], limit, size, fun)
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
  end
end

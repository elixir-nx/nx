defmodule Nx.BinaryBackend.Orig do
  ## Aggregation helpers

  def aggregate_axes(binary, axes, shape, size) do
    {chunk_size, read_size, path} = aggregate_axes(axes, shape, size)

    view =
      for <<chunk::size(chunk_size)-bitstring <- binary>> do
        weighted_traverse(path, chunk, read_size)
      end

    List.flatten(view)
  end

  def aggregate_axes([_ | _] = axes, shape, size) do
    axes = Enum.sort(axes)
    min = hd(axes)
    weighted_shape = weighted_shape(shape, size)
    [{axis_count, axis_weight} | _] = weighted_shape = Enum.drop(weighted_shape, min)
    chunk_size = axis_count * axis_weight

    # The goal of aggregate path is to split the paths
    # we are reducing from the ones we are keeping as is.
    {reverse_pre, reverse_pos} = aggregate_path(weighted_shape, axes, min, [], [])

    # Now if we are reducing on the last dimensions, we
    # can increase the read size.
    {reverse_pos, read_size} =
      aggregate_read(reverse_pos, tuple_size(shape) - 1, Enum.reverse(axes), size)

    path = Enum.reverse(reverse_pre, [(&IO.iodata_to_binary/1) | Enum.reverse(reverse_pos)])
    {chunk_size, read_size, path}
  end

  def aggregate_axes(axes, _shape, _size) do
    raise ArgumentError, ":axes must be a non empty list, got: #{inspect(axes)}"
  end

  defp aggregate_path([pair | shape], [i | axes], i, pre, pos),
    do: aggregate_path(shape, axes, i + 1, pre, [pair | pos])

  defp aggregate_path([pair | shape], axes, i, pre, pos),
    do: aggregate_path(shape, axes, i + 1, [pair | pre], pos)

  defp aggregate_path([], [], _i, pre, pos), do: {pre, pos}

  defp aggregate_read([{axis, weight} | shape], i, [i | axis], _size),
    do: aggregate_read(shape, i - 1, axis, axis * weight)

  defp aggregate_read(shape, _i, _axis, size),
    do: {shape, size}

  ## Weighted shapes

  # Converts the shape to a weight shape list.
  #
  # A weighted shape is a list of tuples where the first
  # element is the number of elements in the dimension
  # and the second element is the size to be traversed in
  # the binary to fetch the next element.
  #
  # This is often given to `weighted_traverse/4` as a general
  # mechanism to traverse binaries.
  defp weighted_shape(shape, size, limits \\ :none, dilations \\ 1) do
    rank = tuple_size(shape)

    dilations =
      if is_list(dilations),
        do: Enum.reverse(dilations),
        else: List.duplicate(dilations, rank)

    weighted_shape(shape, rank, size, limits, dilations, [])
  end

  defp weighted_shape(_shape, 0, _weight, _limits, [], acc), do: acc

  defp weighted_shape(shape, pos, weight, limits, [dilation | dilations], acc) do
    shape_elem = :erlang.element(pos, shape)

    element =
      if limits == :none, do: shape_elem, else: min(:erlang.element(pos, limits), shape_elem)

    dilation_factor =
      if element == 1,
        do: 1,
        else: dilation

    acc = [{element, dilation_factor * weight} | acc]
    weighted_shape(shape, pos - 1, weight * shape_elem, limits, dilations, acc)
  end

  # Reads the chunk size from a weighted list at the given position.
  defp weighted_chunk(list, at, size) do
    {element, size} = Enum.at(list, at, {1, size})
    element * size
  end

  # Traverses a binary using the elements and shape given by `weighted_shape`.
  #
  # When all dimensions are traversed, we read `read_size`.
  #
  # The `weighted_shape` can also contain functions, which are applied to the
  # result of the remaining of the weighted shape.
  defp weighted_traverse(weighted_shape, binary, read_size, offset \\ 0)

  defp weighted_traverse([], data, read_size, offset) do
    <<_::size(offset)-bitstring, chunk::size(read_size)-bitstring, _::bitstring>> = data
    chunk
  end

  defp weighted_traverse([{dim, size} | dims], data, read_size, offset) do
    weighted_traverse(dim, size, dims, data, read_size, offset)
  end

  defp weighted_traverse([fun | dims], data, read_size, offset) do
    fun.(weighted_traverse(dims, data, read_size, offset))
  end

  defp weighted_traverse(dim, dim_size, dims, data, read_size, offset) do
    head = weighted_traverse(dims, data, read_size, offset)

    case dim do
      1 ->
        [head]

      _ ->
        <<_::size(dim_size)-bitstring, data::bitstring>> = data
        [head | weighted_traverse(dim - 1, dim_size, dims, data, read_size, offset)]
    end
  end
end
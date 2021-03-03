
# defmodule Nx.BinaryBackend.Agg do
#   defmodule Traversal do
#     defstruct data: ""
#               shape: nil,

#               dilations: nil,
#               offset: 0,
#               orientation: {:forward, :normal}

#     def new(shape, offset \\ 0, dilations \\ 1) do
#       %View{
#         shape: shape,
#         dilations: resolve_dilations(shape, dilations),
#       }
#     end

#     defp calc_weights(shape) do
#       shape
#     end

#     defp resolve_dilations(shape, dilations) when length(dilations) == tuple_size(shape) do
#       dilations
#       |> Enum.with_index()
#       |> Enum.map(fn {dilation, i} ->
#         if Nx.Shape.dimension(shape, i) == 1, do: 1, else: dilation
#       end)
#       |> List.to_tuple()
#     end

#     defp resolve_dilations(shape, dilation) when is_integer(dilation) do
#       rank = Nx.Shape.rank(shape)
#       dilations = List.duplicate(dilation, rank)
#       resolve_dilation_factors(shape, dilation)
#     end
#   end

#   def axes(binary, axes, shape) do
#     {chunk_size, read_size, path} = axes(axes, shape)

#     # view =
#     #   for <<chunk::size(chunk_size)-bitstring <- binary>> do
#     #     weighted_traverse(path, chunk, read_size)
#     #   end

#     # List.flatten(view)
#   end

#   def axes([_ | _] = axes, shape) do
#     size = 8
#     axes = Enum.sort(axes)
#     min = hd(axes)
#     weighted_shape = weighted_shape(shape)
#     [{axis_count, axis_weight} | _] = weighted_shape = Enum.drop(weighted_shape, min)
#     chunk_size = axis_count * axis_weight

#     # The goal of aggregate path is to split the paths
#     # we are reducing from the ones we are keeping as is.
#     {reverse_pre, reverse_pos} = aggregate_path(weighted_shape, axes, min, [], [])

#     # Now if we are reducing on the last dimensions, we
#     # can increase the read size.
#     {reverse_pos, read_size} =
#       aggregate_read(reverse_pos, tuple_size(shape) - 1, Enum.reverse(axes), size)

#     path = Enum.reverse(reverse_pre, [(&IO.iodata_to_binary/1) | Enum.reverse(reverse_pos)])
#     {chunk_size, read_size, path}
#   end

#   def axes(axes, _shape) do
#     raise ArgumentError, ":axes must be a non empty list, got: #{inspect(axes)}"
#   end

#   defp aggregate_path([pair | shape], [i | axes], i, pre, pos),
#     do: aggregate_path(shape, axes, i + 1, pre, [pair | pos])

#   defp aggregate_path([pair | shape], axes, i, pre, pos),
#     do: aggregate_path(shape, axes, i + 1, [pair | pre], pos)

#   defp aggregate_path([], [], _i, pre, pos), do: {pre, pos}

#   defp aggregate_read([{axis, weight} | shape], i, [i | axis], _size),
#     do: aggregate_read(shape, i - 1, axis, axis * weight)

#   defp aggregate_read(shape, _i, _axis, size),
#     do: {shape, size}

#   ## Weighted shapes

#   # Converts the shape to a weight shape list.
#   #
#   # A weighted shape is a list of tuples where the first
#   # element is the number of elements in the dimension
#   # and the second element is the size to be traversed in
#   # the binary to fetch the next element.
#   #
#   # This is often given to `weighted_traverse/4` as a general
#   # mechanism to traverse binaries.
#   def weighted_shape(shape, limits \\ :none, dilations \\ 1) do
#     rank = tuple_size(shape)

#     dilations =
#       if is_list(dilations),
#         do: Enum.reverse(dilations),
#         else: List.duplicate(dilations, rank)

#     weighted_shape(shape, rank, 1, limits, dilations, [])
#   end

#   defp weighted_shape(_shape, 0, _weight, _limits, [], acc), do: acc

#   defp weighted_shape(shape, pos, weight, limits, [dilation | dilations], acc) do
#     shape_elem = :erlang.element(pos, shape)

#     element =
#       if limits == :none, do: shape_elem, else: min(:erlang.element(pos, limits), shape_elem)

#     dilation_factor =
#       if element == 1,
#         do: 1,
#         else: dilation

#     acc = [{element, weight, dilation_factor} | acc]
#     weighted_shape(shape, pos - 1, weight * shape_elem, limits, dilations, acc)
#   end



#   # Reads the chunk size from a weighted list at the given position.
#   defp weighted_chunk(list, at) do
#     Enum.at(list, at, 1)
#   end

#   # Traverses a binary using the elements and shape given by `weighted_shape`.
#   #
#   # When all dimensions are traversed, we read `read_size`.
#   #
#   # The `weighted_shape` can also contain functions, which are applied to the
#   # result of the remaining of the weighted shape.
#   defp weighted_traverse(weighted_shape)

#   defp weighted_traverse([], data, read_size, offset) do
#     <<_::size(offset)-bitstring, chunk::size(read_size)-bitstring, _::bitstring>> = data
#     chunk
#   end

#   defp weighted_traverse([{dim, size} | dims], data, read_size, offset) do
#     weighted_traverse(dim, size, dims, data, read_size, offset)
#   end

#   defp weighted_traverse([fun | dims], data, read_size, offset) do
#     fun.(weighted_traverse(dims, data, read_size, offset))
#   end

#   defp weighted_traverse(dim, dim_size, dims, data, read_size, offset) do
#     head = weighted_traverse(dims, data, read_size, offset)

#     case dim do
#       1 ->
#         [head]

#       _ ->
#         <<_::size(dim_size)-bitstring, data::bitstring>> = data
#         [head | weighted_traverse(dim - 1, dim_size, dims, data, read_size, offset)]
#     end
#   end

#   # # Makes anchors for traversing a binary with a window.
#   # defp make_anchors(shape, strides, window, dilations \\ 1)

#   # defp make_anchors(shape, strides, window, dilations)
#   #      when is_tuple(shape) and is_tuple(window) and is_list(strides) do
#   #   dilations =
#   #     if is_integer(dilations),
#   #       do: List.duplicate(dilations, tuple_size(shape)),
#   #       else: dilations

#   #   make_anchors(Tuple.to_list(shape), strides, Tuple.to_list(window), dilations, :init)
#   # end

#   # defp make_anchors([], [], _window, _dilations, anchors), do: anchors

#   # defp make_anchors([dim | shape], [s | strides], [w | window], [dil | dilation], :init) do
#   #   dims = for i <- 0..(dim - 1), rem(i, s) == 0 and i + (w - 1) * dil < dim, do: [i]
#   #   make_anchors(shape, strides, window, dilation, dims)
#   # end

#   # defp make_anchors([dim | shape], [s | strides], [w | window], [dil | dilation], anchors) do
#   #   anchors =
#   #     for i <- 0..(dim - 1),
#   #         rem(i, s) == 0 and i + (w - 1) * dil < dim,
#   #         anchor <- anchors,
#   #         do: anchor ++ [i]

#   #   make_anchors(shape, strides, window, dilation, anchors)
#   # end

#   # Calculates the offset needed to reach a specified position
#   # in the binary from a weighted shape list.
#   defp weighted_offset(weighted_shape, pos, dilations \\ 1)

#   defp weighted_offset(weighted_shape, pos, dilations) when is_list(pos) do
#     dilations =
#       if is_list(dilations),
#         do: dilations,
#         else: List.duplicate(dilations, length(weighted_shape))

#     sum_weighted_offset(weighted_shape, pos, dilations)
#   end

#   defp sum_weighted_offset([], [], []), do: 0

#   defp sum_weighted_offset([{s, size} | dims], [x | pos], [d | dilation]) do
#     # Account for the dilation
#     dilation_factor =
#       if s == 1,
#         do: 1,
#         else: d

#     div(size, dilation_factor) * x + weighted_offset(dims, pos, dilation)
#   end
# end
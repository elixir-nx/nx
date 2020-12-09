defmodule Nx.Util do
  @moduledoc """
  A collection of helper functions for working with tensors.

  Generally speaking, the usage of these functions is discouraged,
  as they don't work inside `defn` and the `Nx` module often has
  higher-level features around them.
  """

  alias Nx.Tensor, as: T
  import Nx.Shared

  @doc """
  Returns the underlying tensor as a bitstring.

  The bitstring is returned as is (which is row-major).
  """
  def to_bitstring(%T{data: {Nx.BitStringDevice, data}}), do: data

  def to_bitstring(%T{data: {device, _data}}) do
    raise ArgumentError,
          "cannot read Nx.Tensor data because the data is allocated on device #{inspect(device)}. " <>
            "Please use Nx.device_transfer/1 to transfer data back to Elixir"
  end

  @doc """
  Creates a one-dimensional tensor from a `bitstring` with the given `type`.

  If the bitstring size does not match its type, an error is raised.

  ## Examples

      iex> Nx.Util.from_bitstring(<<1, 2, 3, 4>>, {:s, 8})
      #Nx.Tensor<
        s8[4]
        [1, 2, 3, 4]
      >

      iex> Nx.Util.from_bitstring(<<12.3::float-64-native>>, {:f, 64})
      #Nx.Tensor<
        f64[1]
        [12.3]
      >

      iex> Nx.Util.from_bitstring(<<1, 2, 3, 4>>, {:f, 64})
      ** (ArgumentError) bitstring does not match the given size

  """
  def from_bitstring(bitstring, type) when is_bitstring(bitstring) do
    {_, size} = Nx.Type.validate!(type)
    dim = div(bit_size(bitstring), size)

    if bitstring == "" do
      raise ArgumentError, "cannot build an empty tensor"
    end

    if rem(bit_size(bitstring), size) != 0 do
      raise ArgumentError, "bitstring does not match the given size"
    end

    %T{data: {Nx.BitStringDevice, bitstring}, type: type, shape: {dim}}
  end

  @doc """
  Reduces over a tensor with the given accumulator.

  The tensor may be reduced in parallel. The evaluation
  order of the reduction function is arbitrary and may
  be non-deterministic. Therefore, the reduction function
  should not be overly sensitive to reassociation.

  If the `:axis` option is given, it aggregates over
  that dimension, effectively removing it. `axis: 0`
  implies aggregating over the highest order dimension
  and so forth. If the axis is negative, then counts
  the axis from the back. For example, `axis: -1` will
  always aggregate all rows.

  ## Examples

      iex> Nx.Util.reduce(Nx.tensor(42), 0, &+/2)
      #Nx.Tensor<
        s64
        42
      >

      iex> Nx.Util.reduce(Nx.tensor([1, 2, 3]), 0, &+/2)
      #Nx.Tensor<
        s64
        6
      >

      iex> Nx.Util.reduce(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, &+/2)
      #Nx.Tensor<
        f64
        10.0
      >

  ### Aggregating over an axis

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.Util.reduce(t, 0, [axis: 0], &+/2)
      #Nx.Tensor<
        s64
        6
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> Nx.Util.reduce(t, 0, [axis: 0], &+/2)
      #Nx.Tensor<
        s64[2][3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> Nx.Util.reduce(t, 0, [axis: 1], &+/2)
      #Nx.Tensor<
        s64[2][3]
        [
          [5, 7, 9],
          [17, 19, 21]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> Nx.Util.reduce(t, 0, [axis: 2], &+/2)
      #Nx.Tensor<
        s64[2][2]
        [
          [6, 15],
          [24, 33]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> Nx.Util.reduce(t, 0, [axis: -1], &+/2)
      #Nx.Tensor<
        s64[2][2]
        [
          [6, 15],
          [24, 33]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> Nx.Util.reduce(t, 0, [axis: -3], &+/2)
      #Nx.Tensor<
        s64[2][3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

  ### Errors

      iex> Nx.Util.reduce(Nx.tensor([1, 2, 3]), 0, [axis: 1], &+/2)
      ** (ArgumentError) unknown axis 1 for shape {3} (axis is zero-indexed)
  """
  def reduce(tensor, acc, opts \\ [], fun)

  def reduce(number, acc, opts, fun) when is_number(number) do
    reduce(Nx.tensor(number), acc, opts, fun)
  end

  def reduce(%T{type: {_, size} = type, shape: shape} = t, acc, opts, fun)
      when is_list(opts) and is_function(fun, 2) do
    data = to_bitstring(t)

    {data, shape} =
      if axis = opts[:axis] do
        {gap_count, chunk_count, chunk_size, new_shape} = aggregate_axis(shape, axis, size)

        new_data =
          aggregate_gaps(chunk_count, chunk_size, fn pre, pos ->
            <<_::size(pre)-bitstring, chunk::size(chunk_size)-bitstring, _::size(pos)-bitstring>> =
              data

            aggregate_gaps(gap_count, size, fn pre, pos ->
              match_types [type] do
                value =
                  for <<_::size(pre)-bitstring, match!(var, 0), _::size(pos)-bitstring <- chunk>>,
                    reduce: acc,
                    do: (acc -> fun.(read!(var, 0), acc))

                <<write!(value, 0)>>
              end
            end)
          end)

        {new_data, new_shape}
      else
        new_data =
          match_types [type] do
            value =
              for <<match!(var, 0) <- data>>,
                reduce: acc,
                do: (acc -> fun.(read!(var, 0), acc))

            <<write!(value, 0)>>
          end

        {new_data, {}}
      end

    %{t | data: {Nx.BitStringDevice, IO.iodata_to_binary(data)}, shape: shape}
  end

  @doc """
  Performs a reduction of multiple tensors, producing a single
  tensor and the final accumulator.

  If the `:axis` option is given, it aggregates over
  that dimension, effectively removing it. `axis: 0`
  implies aggregating over the highest order dimension
  and so forth. If the axis is negative, then counts
  the axis from the back. For example, `axis: -1` will
  always aggregate all rows.

  ## Examples

    iex> t1 = Nx.tensor([[1, 2, 3], [4, 5, 6]])
    iex> {new_tensor, accs} = Nx.Util.zip_map_reduce([{t1, 0}], {0, :first, -1},
    ...> fn {x}, {i, cur_max_i, cur_max} ->
    ...>  if x > cur_max or cur_max == :first, do: {i, {i+1, i, x}}, else: {cur_max_i, {i + 1, cur_max_i, cur_max}}
    ...> end
    ...> )
    iex> new_tensor
    #Nx.Tensor<
      s64[3]
      [1, 1, 1]
    >
    iex> accs
    [{2, 1, 4}, {2, 1, 5}, {2, 1, 6}]
  """
  def zip_map_reduce(tensors_and_axes, acc, fun)

  def zip_map_reduce([{folding_tensor, _} | rest] = tensors_and_axes, acc, fun)
      when is_list(tensors_and_axes) and is_function(fun, 2) do

    output_type = Enum.reduce(rest, folding_tensor.type, fn {t, _}, acc -> Nx.Type.merge(t.type, acc) end)

    {zipped_axes, new_shape} = bin_zip(tensors_and_axes)

    data_and_acc =
      for zipped_axis <- zipped_axes do
        {tensor_data, acc} = bin_reduce_many(zipped_axis, output_type, acc, fun)
        {scalar_to_bin(tensor_data, output_type), acc}
      end

    {final_data, final_acc} = Enum.unzip(data_and_acc)

    {%T{
       data: {Nx.BitStringDevice, IO.iodata_to_binary(final_data)},
       shape: new_shape,
       type: output_type
     }, final_acc}
  end

  ## Binary Helpers

  # Helper for zipping tensors along given axis/axes.
  # Given we always reduce on the first tensor provided,
  # the "new_shape" returned is always the "new_shape" of
  # the first tensor reduced along it's provided axis.
  #
  # Validates that subsequent shapes are compatible with the
  # "folding" shape by determining if the dimension of the given
  # axis equals the "folding" dimension of the "folding" shape.
  #
  # If the shapes do not match, but they are aligned correctly,
  # "broadcasts" them to match by repeating a view the necessary
  # number of times.
  defp bin_zip([{folding_tensor, folding_axis} | []]) do
    {_, folding_size} = folding_tensor.type
    {folding_view, new_shape} = bin_view_axis(to_bitstring(folding_tensor), folding_axis, folding_tensor.shape, folding_size)
    {Enum.zip([folding_view]), new_shape}
  end

  defp bin_zip([{folding_tensor, folding_axis} | rest]) do
    {_, folding_size} = folding_tensor.type
    folding_dim = if folding_axis, do: elem(folding_tensor.shape, folding_axis), else: tuple_product(folding_tensor.shape)
    {folding_view, folding_shape} = bin_view_axis(to_bitstring(folding_tensor), folding_axis, folding_tensor.shape, folding_size)

    data =
      for {tensor, axis} <- rest do
        dim = if axis, do: elem(tensor.shape, axis), else: tuple_product(tensor.shape)

        unless dim == folding_dim,
          do: raise ArgumentError, "expected dimensions to match"

        {_, size} = tensor.type
        {view, new_shape} = bin_view_axis(to_bitstring(tensor), axis, tensor.shape, size)

        views =
          for f_axis <- folding_view,
              axis <- view, do: {axis, f_axis}

        {views, Tuple.to_list(new_shape)}
      end

    {zipped_views, shapes} = Enum.unzip(data)

    new_shape = combine_shapes([Tuple.to_list(folding_shape) | shapes], {})

    {List.flatten(zipped_views), new_shape}
  end

  # TODO: This will become unnecessary when we restrict zip_map_reduce to 2 tensors
  defp combine_shapes([shape | []], acc) do
    shape
    |> Enum.reduce(acc, & Tuple.append(&2, &1))
  end
  defp combine_shapes([shape | rest], acc) do
    acc =
      shape
      |> Enum.reduce(acc, & Tuple.append(&2, &1))
    combine_shapes(rest, acc)
  end

  # Helper for "viewing" a tensor along a given axis.
  # Returns the view and the expected new shape when
  # reducing down the axis.
  #
  # If the axis isn't provided, the "view" is just the
  # entire binary as it is layed out in memory and we
  # expect the entire tensor to be reduced down to a scalar.
  defp bin_view_axis(binary, axis, shape, size) do
    if axis do
      {gap_count, chunk_count, chunk_size, new_shape} = aggregate_axis(shape, axis, size)
      view =
        aggregate_gaps(chunk_count, chunk_size, fn pre, _pos ->
          <<_::size(pre)-bitstring, chunk::size(chunk_size)-bitstring, _::bitstring>> = binary

          aggregate_gaps(gap_count, size, fn pre, pos ->
            for <<_::size(pre)-bitstring, var::size(size)-bitstring,
                  _::size(pos)-bitstring <- chunk>>,
                into: <<>>,
                do: var
          end)
        end)
      {List.flatten(view), new_shape}
    else
      {[binary], {}}
    end
  end

  # Helper for reducing multiple binaries at once.
  # Expects `fun` to return an accumulator and data
  # for building a new tensor.
  defp bin_reduce_many(binaries, type, acc, fun) do
    {heads, tails} =
      binaries
      |> Tuple.to_list()
      |> Enum.map(fn data ->
        match_types [type] do
          <<match!(x, 0), rest::bitstring>> = data
          {read!(x, 0), rest}
        end
      end)
      |> Enum.unzip()

    if empty?(tails) do
      fun.(List.to_tuple(heads), acc)
    else
      {_cur_tensor_data, acc} = fun.(List.to_tuple(heads), acc)
      bin_reduce_many(List.to_tuple(tails), type, acc, fun)
    end
  end

  defp empty?([<<>> | _]), do: true
  defp empty?([_ | _]), do: false

  ## Dimension helpers

  # The goal of aggregate axis is to find a chunk_count and gap_count
  # that allows us to traverse the tensor. Consider this input tensor:
  #
  #     #Nx.Tensor<
  #       s64[2][2][3]
  #       [
  #         [
  #           [1, 2, 3],
  #           [4, 5, 6]
  #         ],
  #         [
  #           [7, 8, 9],
  #           [10, 11, 12]
  #         ]
  #       ]
  #     >
  #
  # When computing the sum on axis 0, we have:
  #
  #     #Nx.Tensor<
  #       s64[2][3]
  #       [
  #         [8, 10, 12],
  #         [14, 16, 18]
  #       ]
  #     >
  #
  # The first time element is 1 + 7, which means the gap between
  # them is 6. Given we have to traverse the whole tensor, the
  # chunk_count is 1.
  #
  # For axis 1, we have:
  #
  #     #Nx.Tensor<
  #       s64[2][3]
  #       [
  #         [5, 7, 9],
  #         [17, 19, 21]
  #       ]
  #     >
  #
  # The first element is 1 + 4 within the "first quadrant". 17 is
  # is 10 + 17 in the second quadrant. Therefore the gap is 3 and
  # the chunk_count is 2.
  #
  # Finally, for axis 2, we have:
  #
  #     #Nx.Tensor<
  #       s64[2][2]
  #       [
  #         [6, 15],
  #         [24, 33]
  #       ]
  #     >
  #
  # 6 is the sum of the first row, 15 the sum of the second row, etc.
  # Therefore the gap is 1 and the chunk count is 4.
  #
  # Computing the aggregate is a matter of mapping the binary over
  # each chunk and then mapping gap times, moving the computation root
  # by size over each gap.
  defp aggregate_axis(shape, axis, size) do
    total = tuple_product(shape)
    {gap_count, chunk_count, new_shape} = total_aggregate_axis(shape, axis, total)
    {gap_count, chunk_count, div(size * total, chunk_count), new_shape}
  end

  defp total_aggregate_axis(shape, axis, total) when axis >= 0 and axis < tuple_size(shape) do
    aggregate_axis(Tuple.to_list(shape), 0, axis, 1, total, [])
  end

  defp total_aggregate_axis(shape, axis, total) when axis < 0 and axis >= -tuple_size(shape) do
    aggregate_axis(Tuple.to_list(shape), 0, tuple_size(shape) + axis, 1, total, [])
  end

  defp total_aggregate_axis(shape, axis, _total) do
    raise ArgumentError, "unknown axis #{axis} for shape #{inspect(shape)} (axis is zero-indexed)"
  end

  defp aggregate_axis([dim | dims], axis, chosen, chunk, gap, acc) do
    gap = div(gap, dim)

    if axis == chosen do
      {gap, chunk, List.to_tuple(Enum.reverse(acc, dims))}
    else
      aggregate_axis(dims, axis + 1, chosen, chunk * dim, gap, [dim | acc])
    end
  end

  defp aggregate_gaps(gap_count, size, fun),
    do: aggregate_gaps(0, gap_count * size, size, fun)

  defp aggregate_gaps(_pre, 0, _size, _fun),
    do: []

  defp aggregate_gaps(pre, pos, size, fun),
    do: [fun.(pre, pos - size) | aggregate_gaps(pre + size, pos - size, size, fun)]
end

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

  If multiple tensors are given, performs the reduction
  side-by-side and returns a tuple of size `N` which matches
  the number of tensors given. Reduction function must
  be an arity-2 function which accepts equal-sized tuples.
  Number of initial values must match the number of tensors
  passed.

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

  ### Aggregating multiple tensors

      iex> t1 = Nx.tensor([1, 2, 3])
      iex> t2 = Nx.tensor([4, 5, 6])
      iex> {t1, t2} = Nx.Util.reduce([t1, t2], [0, 0], fn {x, y}, {acc1, acc2} -> {x + acc1, y + acc2} end)
      iex> t1
      #Nx.Tensor<
        s64
        6
      >
      iex> t2
      #Nx.Tensor<
        s64
        15
      >

      iex> t1 = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> t2 = Nx.tensor([[7, 8, 9], [10, 11, 12]])
      iex> {t1, t2} = Nx.Util.reduce([t1, t2], [0, 0], [axis: 1], fn {x, y}, {acc1, acc2} -> {x*y + acc1, y - x + acc2} end)
      iex> t1
      #Nx.Tensor<
        s64[2]
        [50, 167]
      >
      iex> t2
      #Nx.Tensor<
        s64[2]
        [18, 18]
      >

  ### Errors

      iex> Nx.Util.reduce(Nx.tensor([1, 2, 3]), 0, [axis: 1], &+/2)
      ** (ArgumentError) unknown axis 1 for shape {3} (axis is zero-indexed)

      iex> Nx.Util.reduce([Nx.tensor([1, 2, 3]), Nx.tensor([1, 2])], [0, 0], [axis: 0], fn {x, y}, {_, _} -> {x, y} end)
      ** (ArgumentError) attempt to pass mixed shapes to reduce/4, all tensor shapes must match

      iex> Nx.Util.reduce([Nx.tensor([1, 2, 3])], [0, 0], [axis: 0], fn {x}, {_, _} -> {x, x+x} end)
      ** (ArgumentError) number of tensors must match number of initial values passed to reduce/4

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

  def reduce(tensors, accs, opts, fun)
      when is_list(tensors) and is_list(accs) and is_list(opts) and is_function(fun, 2) do
    unless length(tensors) == length(accs),
      do:
        raise(
          ArgumentError,
          "number of tensors must match number of initial values passed to reduce/4"
        )

    {type, shape} =
      tensors
      |> Enum.reduce(
        {{:u, 8}, :empty},
        fn t, {type, shape} ->
          unless shape == :empty or t.shape == shape,
            do:
              raise(
                ArgumentError,
                "attempt to pass mixed shapes to reduce/4, all tensor shapes must match"
              )

          {Nx.Type.merge(t.type, type), t.shape}
        end
      )

    # TODO: Merge all to highest type when we have a `cast!` that works
    # over an entire tensor
    data =
      tensors
      |> Enum.map(&to_bitstring/1)

    {zipped_binaries, new_shape} = bin_zip(data, type, opts[:axis], shape)

    zipped_data =
      for bins <- zipped_binaries do
        res = bin_reduce_side_by_side(bins, type, List.to_tuple(accs), fun)

        res
        |> Tuple.to_list()
        |> Enum.map(fn val ->
          match_types [type] do
            <<write!(val, 0)>>
          end
        end)
        |> List.to_tuple()
      end

    zipped_data
    |> Enum.unzip()
    |> Tuple.to_list()
    |> Enum.map(
      &%T{data: {Nx.BitStringDevice, IO.iodata_to_binary(&1)}, type: type, shape: new_shape}
    )
    |> List.to_tuple()
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
    iex> t2 = Nx.iota(t1, axis: 0)
    iex> {new_tensor, accs} = Nx.Util.zip_map_reduce([t1, t2], {:first, -1}, [axis: 0],
    ...> fn {x, y}, {cur_max_i, cur_max} ->
    ...>  if x > cur_max or cur_max == :first, do: {y, {y, x}}, else: {cur_max_i, {cur_max_i, cur_max}}
    ...> end
    ...> )
    iex> new_tensor
    #Nx.Tensor<
      s64[3]
      [1, 1, 1]
    >
    iex> accs
    [{1, 4}, {1, 5}, {1, 6}]
  """
  def zip_map_reduce(tensors, acc, opts \\ [], fun)

  def zip_map_reduce([head | tail] = tensors, acc, opts, fun)
      when is_list(tensors) and is_list(opts) and is_function(fun, 2) do
    type = Enum.reduce(tail, head.type, &Nx.Type.merge(&1.type, &2))
    shape = head.shape

    if Enum.any?(tail, & &1.shape != shape) do
      raise ArgumentError,
              "attempt to pass mixed shapes to zip_map_reduce/4, all tensor shapes must match"
    end

    # TODO: Merge all to highest type when we have a `cast!` that works
    # over an entire tensor
    data =
      tensors
      |> Enum.map(&to_bitstring/1)

    {zipped_axes, new_shape} = bin_zip(data, type, opts[:axis], shape)

    data_and_acc =
      for zipped_axis <- zipped_axes do
        {tensor_data, acc} = bin_reduce(zipped_axis, type, acc, fun)

        tensor_bin =
          match_types [type] do
            <<write!(tensor_data, 0)>>
          end

        {tensor_bin, acc}
      end

    {final_data, final_acc} = Enum.unzip(data_and_acc)

    {%T{
       data: {Nx.BitStringDevice, IO.iodata_to_binary(final_data)},
       shape: new_shape,
       type: type
     }, final_acc}
  end

  ## Binary Helpers

  defp bin_zip(binaries, type, axis, shape) do
    {_, size} = type

    {new_data, new_shape} =
      if axis do
        {gap_count, chunk_count, chunk_size, new_shape} = aggregate_axis(shape, axis, size)

        data =
          aggregate_gaps(chunk_count, chunk_size, fn pre, pos ->
            axis =
              for binary <- binaries do
                <<_::size(pre)-bitstring, chunk::size(chunk_size)-bitstring,
                  _::size(pos)-bitstring>> = binary

                aggregate_gaps(gap_count, size, fn pre, pos ->
                  for <<_::size(pre)-bitstring, var::size(size)-bitstring,
                        _::size(pos)-bitstring <- chunk>>,
                      into: <<>>,
                      do: var
                end)
              end

            Enum.zip(axis)
          end)

        {List.flatten(data), new_shape}
      else
        {[List.to_tuple(binaries)], {}}
      end

    {new_data, new_shape}
  end

  defp bin_reduce_side_by_side(binaries, type, accs, fun) do
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

    if empty?(tails),
      do: fun.(List.to_tuple(heads), accs),
      else:
        bin_reduce_side_by_side(List.to_tuple(tails), type, fun.(List.to_tuple(heads), accs), fun)
  end

  defp bin_reduce(binaries, type, acc, fun) do
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
      bin_reduce(List.to_tuple(tails), type, acc, fun)
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

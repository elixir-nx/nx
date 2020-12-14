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
  Returns the underlying tensor as a float list.

  The list is returned as is (which is row-major).

    ## Examples

      iex> Nx.Util.to_flat_list(1)
      [1]

      iex> Nx.Util.to_flat_list(Nx.tensor([1.0, 2.0, 3.0]))
      [1.0, 2.0, 3.0]
  """
  def to_flat_list(tensor) do
    tensor = Nx.tensor(tensor)

    match_types [tensor.type] do
      for <<match!(var, 0) <- to_bitstring(tensor)>>, do: read!(var, 0)
    end
  end

  @doc """
  Returns the underlying tensor as a bitstring.

  The bitstring is returned as is (which is row-major).

  ## Examples

      iex> Nx.Util.to_bitstring(1)
      <<1::64-native>>

      iex> Nx.Util.to_bitstring(Nx.tensor([1.0, 2.0, 3.0]))
      <<1.0::float-native, 2.0::float-native, 3.0::float-native>>
  """
  def to_bitstring(%T{data: {Nx.BitStringDevice, data}}), do: data

  def to_bitstring(%T{data: {device, _data}}) do
    raise ArgumentError,
          "cannot read Nx.Tensor data because the data is allocated on device #{inspect(device)}. " <>
            "Please use Nx.device_transfer/1 to transfer data back to Elixir"
  end

  def to_bitstring(t), do: to_bitstring(Nx.tensor(t))

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
  Reduces over a tensor with the given accumulator, returning
  a new tensor and the final accumulators.

  Expects `fun` to return a tuple of tensor data and
  the accumulator.

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

      iex> {t, accs} = Nx.Util.reduce(Nx.tensor(42), 0, fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64
        42
      >
      iex> accs
      [42]

      iex> {t, accs} = Nx.Util.reduce(Nx.tensor([1, 2, 3]), 0, fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64
        6
      >
      iex> accs
      [6]

      iex> {t, accs} = Nx.Util.reduce(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        f64
        10.0
      >
      iex> accs
      [10.0]

  ### Aggregating over an axis

      iex> t = Nx.tensor([1, 2, 3])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axis: 0], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64
        6
      >
      iex> accs
      [6]

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axis: 0], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64[2][3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >
      iex> accs
      [8, 10, 12, 14, 16, 18]

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axis: 1], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64[2][3]
        [
          [5, 7, 9],
          [17, 19, 21]
        ]
      >
      iex> accs
      [5, 7, 9, 17, 19, 21]

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axis: 2], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64[2][2]
        [
          [6, 15],
          [24, 33]
        ]
      >
      iex> accs
      [6, 15, 24, 33]

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axis: -1], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64[2][2]
        [
          [6, 15],
          [24, 33]
        ]
      >
      iex> accs
      [6, 15, 24, 33]

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axis: -3], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64[2][3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >
      iex> accs
      [8, 10, 12, 14, 16, 18]

  ### Errors

      iex> Nx.Util.reduce(Nx.tensor([1, 2, 3]), 0, [axis: 1], fn x, acc -> {x+acc, x+acc} end)
      ** (ArgumentError) unknown axis 1 for shape {3} (axis is zero-indexed)
  """
  def reduce(tensor, acc, opts \\ [], fun)

  def reduce(number, acc, opts, fun) when is_number(number) do
    reduce(Nx.tensor(number), acc, opts, fun)
  end

  def reduce(%T{type: {_, size} = type, shape: shape} = t, acc, opts, fun)
      when is_list(opts) and is_function(fun, 2) do
    output_type = opts[:type] || t.type

    {view, new_shape} = bin_view_axis(to_bitstring(t), opts[:axis], shape, size)

    data_and_acc =
      for axis <- view do
        {bin, acc} =
          match_types [type] do
            for <<match!(var, 0) <- axis>>, reduce: {<<>>, acc} do
              {_, acc} -> fun.(read!(var, 0), acc)
            end
          end

        {scalar_to_bin(bin, output_type), acc}
      end

    {final_data, final_accs} = Enum.unzip(data_and_acc)

    {%T{
       data: {Nx.BitStringDevice, IO.iodata_to_binary(final_data)},
       shape: new_shape,
       type: output_type
     }, final_accs}
  end

  @doc """
  Zips two tensors along the specified axes, and then reduces
  them, returning a new tensor and the final accumulators.

  Expects `fun` to be a function with arity 2 that accepts a tuple
  and an accumulator. `fun` returns a tuple of tensor data
  used for building a new tensor, and an accumulator.

  The accumulator can be any generic term.

  The size of the dimensions of `t1` and `t2` must match along
  `axis1` and `axis2`. `axis1` and/or `axis2` may be `nil`, in
  which case, the entire tensor is reduced to a scalar.

  ## Examples

      iex> t1 = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> t2 = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      iex> {new_tensor, accs} = Nx.Util.zip_reduce(t1, 1, t2, 0, 0, fn {a, b}, acc -> {a*b+acc, a*b+acc} end)
      iex> new_tensor
      #Nx.Tensor<
        s64[2][2]
        [
          [22, 28],
          [49, 64]
        ]
      >
      iex> accs
      [22, 28, 49, 64]

  ### Aggregating over an entire tensor

      iex> t1 = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> t2 = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      iex> {new_tensor, accs} = Nx.Util.zip_reduce(t1, nil, t2, nil, 0, fn {a, b}, acc -> {a+b+acc, a+b+acc} end)
      iex> new_tensor
      #Nx.Tensor<
        s64
        42
      >
      iex> accs
      [42]

      iex> t1 = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> t2 = Nx.tensor([[1], [2], [3]])
      iex> {new_tensor, accs} = Nx.Util.zip_reduce(t1, 1, t2, nil, 0, fn {a, b}, acc -> {a*b+acc, a*b+acc} end)
      iex> new_tensor
      #Nx.Tensor<
        s64[2]
        [14, 32]
      >
      iex> accs
      [14, 32]

  ### Errors

      iex> t1 = Nx.tensor([1, 2, 3])
      iex> t2 = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.Util.zip_reduce(t1, 0, t2, 0, 0, fn {a, b}, acc -> {a+b+acc, a+b+acc} end)
      ** (ArgumentError) unable to zip tensors along the given axis or axes dimensions of zipped axes must match, got 3 and 2
  """
  def zip_reduce(t1, axis1, t2, axis2, acc, fun)

  def zip_reduce(t1, axis1, t2, axis2, acc, fun) when is_function(fun, 2) do
    output_type = Nx.Type.merge(t1.type, t2.type)
    {v1, v2, new_shape} = bin_zip_axis(t1, axis1, t2, axis2)

    data_and_acc =
      for b1 <- v1, b2 <- v2 do
        {bin, acc} = bin_zip_reduce_axis(b1, b2, t1.type, t2.type, <<>>, acc, fun)
        {scalar_to_bin(bin, output_type), acc}
      end

    {final_data, final_acc} = Enum.unzip(data_and_acc)

    {%T{
       data: {Nx.BitStringDevice, IO.iodata_to_binary(final_data)},
       shape: new_shape,
       type: output_type
     }, final_acc}
  end

  # Helper for zipping 2 tensors along given axis/axes.
  # Given we always reduce on the first tensor provided,
  # the "new_shape" returned is always the "new_shape" of
  # the first tensor reduced along it's provided axis.
  #
  # The "folded" tensor is "folded" into the "folding" tensor
  # along it's axis.
  #
  # Validates that subsequent shapes are compatible with the
  # "folding" shape by determining if the dimension of the given
  # axis equals the "folding" dimension of the "folding" shape.
  #
  # If the shapes do not match, but they are aligned correctly,
  # "broadcasts" them to match by repeating a view the necessary
  # number of times.
  defp bin_zip_axis(t1, axis1, t2, axis2) do
    {_, folding_size} = t1.type
    folding_dim = if axis1, do: elem(t1.shape, axis1), else: tuple_product(t1.shape)
    {folding_view, folding_shape} = bin_view_axis(to_bitstring(t1), axis1, t1.shape, folding_size)

    {_, folded_size} = t2.type
    folded_dim = if axis2, do: elem(t2.shape, axis2), else: tuple_product(t2.shape)
    {folded_view, folded_shape} = bin_view_axis(to_bitstring(t2), axis2, t2.shape, folded_size)

    unless folded_dim == folding_dim do
      raise ArgumentError,
            "unable to zip tensors along the given axis or axes" <>
              " dimensions of zipped axes must match, got #{folding_dim}" <>
              " and #{folded_dim}"
    end

    new_shape = Tuple.to_list(folding_shape) ++ Tuple.to_list(folded_shape)
    {folding_view, folded_view, List.to_tuple(new_shape)}
  end

  # Helper for reducing down a single axis over two tensors,
  # returning tensor data and a final accumulator.
  defp bin_zip_reduce_axis(<<>>, <<>>, _left_type, _right_type, bin, acc, _fun),
    do: {bin, acc}

  defp bin_zip_reduce_axis(b1, b2, left_type, right_type, _bin, acc, fun) do
    {head1, rest1, head2, rest2} =
      match_types [left_type, right_type] do
        <<match!(x, 0), rest1::bitstring>> = b1
        <<match!(y, 1), rest2::bitstring>> = b2
        {read!(x, 0), rest1, read!(y, 1), rest2}
      end

    {bin, acc} = fun.({head1, head2}, acc)
    bin_zip_reduce_axis(rest1, rest2, left_type, right_type, bin, acc, fun)
  end

  ## Axis helpers

  # Helper for "viewing" a tensor along a given axis.
  # Returns the view and the expected new shape when
  # reducing down the axis.
  #
  # If the axis isn't provided, the "view" is just the
  # entire binary as it is layed out in memory and we
  # expect the entire tensor to be reduced down to a scalar.
  defp bin_view_axis(binary, axis, shape, size) do
    if axis do
      {gap_count, chunk_size, new_shape} = aggregate_axis(shape, axis, size)

      view =
        for <<chunk::size(chunk_size)-bitstring <- binary>>,
            gap <-
              aggregate_gaps(gap_count, size, fn pre, pos ->
                for <<_::size(pre)-bitstring, var::size(size)-bitstring,
                      _::size(pos)-bitstring <- chunk>>,
                    into: <<>>,
                    do: var
              end),
            do: gap

      {view, new_shape}
    else
      {[binary], {}}
    end
  end

  # The goal of traverse axis is to find a chunk_count and gap_count
  # that allows us to traverse a given axis. Consider this input tensor:
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
    {gap_count, chunk_count, new_shape} = aggregate_axis(shape, axis)
    {gap_count, size * chunk_count, new_shape}
  end

  defp aggregate_axis(shape, axis) when axis >= 0 and axis < tuple_size(shape) do
    aggregate_axis(Tuple.to_list(shape), 0, axis, tuple_product(shape), [])
  end

  defp aggregate_axis(shape, axis) when axis < 0 and axis >= -tuple_size(shape) do
    aggregate_axis(Tuple.to_list(shape), 0, tuple_size(shape) + axis, tuple_product(shape), [])
  end

  defp aggregate_axis(shape, axis) do
    raise ArgumentError, "unknown axis #{axis} for shape #{inspect(shape)} (axis is zero-indexed)"
  end

  defp aggregate_axis([dim | dims], axis, chosen, prev_gap, acc) do
    next_gap = div(prev_gap, dim)

    if axis == chosen do
      {next_gap, prev_gap, List.to_tuple(Enum.reverse(acc, dims))}
    else
      aggregate_axis(dims, axis + 1, chosen, next_gap, [dim | acc])
    end
  end

  defp aggregate_gaps(gap_count, size, fun),
    do: aggregate_gaps(0, gap_count * size, size, fun)

  defp aggregate_gaps(_pre, 0, _size, _fun),
    do: []

  defp aggregate_gaps(pre, pos, size, fun),
    do: [fun.(pre, pos - size) | aggregate_gaps(pre + size, pos - size, size, fun)]
end

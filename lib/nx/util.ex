defmodule Nx.Util do
  @moduledoc """
  A collection of helper functions for working with tensors.

  Generally speaking, the usage of these functions is discouraged,
  as they don't work inside `defn` and the `Nx` module often has
  higher-level features around them.
  """

  alias Nx.Tensor, as: T
  import Nx.Shared

  ## Conversions

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

  ## High-level

  @doc """
  Computes the dot product of two tensors over the given axes.

  The axes must be non-empty lists and their dimensions must match.
  """
  def dot(%T{shape: s1} = t1, axes1, %T{shape: s2} = t2, axes2) do
    validate_dot_axes!(axes1, s1, axes2, s2)

    {tensor, _} =
      zip_reduce(t1, axes1, t2, axes2, 0, fn {lhs, rhs}, acc ->
        res = lhs * rhs + acc
        {res, res}
      end)

    tensor
  end

  defp validate_dot_axes!([a1 | axes1], s1, [a2 | axes2], s2) do
    d1 = elem(s1, a1)
    d2 = elem(s2, a2)

    if d1 == d2 do
      validate_dot_axes!(axes1, s1, axes2, s2)
    else
      raise ArgumentError,
            "dot product expects shapes to be compatible," <>
              " dimension #{a1} of left-side (#{d1}) does not equal" <>
              " dimension #{a2} of right-side (#{d2})"
    end
  end

  defp validate_dot_axes!([], _s1, [], _s2) do
    :ok
  end

  ## Reduces

  @doc """
  Reduces over a tensor with the given accumulator, returning
  a new tensor and the final accumulators.

  Expects `fun` to return a tuple of tensor data and
  the accumulator.

  The tensor may be reduced in parallel. The evaluation
  order of the reduction function is arbitrary and may
  be non-deterministic. Therefore, the reduction function
  should not be overly sensitive to reassociation.

  If the `:axes` option is given, it aggregates over
  multiple dimensions, effectively removing them. `axes: [0]`
  implies aggregating over the highest order dimension
  and so forth. If the axis is negative, then counts
  the axis from the back. For example, `axes: [-1]` will
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

  ### Aggregating over axes

      iex> t = Nx.tensor([1, 2, 3])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axes: [0]], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64
        6
      >
      iex> accs
      [6]

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axes: [0]], fn x, acc -> {x+acc, x+acc} end)
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
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axes: [1]], fn x, acc -> {x+acc, x+acc} end)
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
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axes: [0, 2]], fn x, acc -> {x+acc, x+acc} end)
      iex> t
      #Nx.Tensor<
        s64[2]
        [30, 48]
      >
      iex> accs
      [30, 48]

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axes: [-1]], fn x, acc -> {x+acc, x+acc} end)
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
      iex> {t, accs} = Nx.Util.reduce(t, 0, [axes: [-3]], fn x, acc -> {x+acc, x+acc} end)
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

      iex> Nx.Util.reduce(Nx.tensor([[1, 2, 3]]), 0, [axes: [2]], fn x, acc -> {x+acc, x+acc} end)
      ** (ArgumentError) axes [2] must be unique integers between 0 and 1
  """
  def reduce(tensor, acc, opts \\ [], fun)

  def reduce(number, acc, opts, fun) when is_number(number) do
    reduce(Nx.tensor(number), acc, opts, fun)
  end

  def reduce(%T{type: {_, size} = type, shape: shape} = t, acc, opts, fun)
      when is_list(opts) and is_function(fun, 2) do
    output_type = opts[:type] || t.type
    {view, new_shape} = bin_aggregate_axes(to_bitstring(t), opts[:axes], shape, size)

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
       shape: List.to_tuple(new_shape),
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
  `axes1` and `axes2`. `axes1` and/or `axes2` may be `nil`, in
  which case, the entire tensor is reduced to a scalar.

  ## Examples

  ### Aggregate over axes

      iex> t1 = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> t2 = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      iex> {new_tensor, accs} = Nx.Util.zip_reduce(t1, [1], t2, [0], 0, fn {a, b}, acc -> {a*b+acc, a*b+acc} end)
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
      iex> {new_tensor, accs} = Nx.Util.zip_reduce(t1, [1], t2, nil, 0, fn {a, b}, acc -> {a*b+acc, a*b+acc} end)
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
      iex> Nx.Util.zip_reduce(t1, [0], t2, [0], 0, fn {a, b}, acc -> {a+b+acc, a+b+acc} end)
      ** (ArgumentError) unable to zip tensors along the given axes. Dimensions of zipped axes must match, got 3 and 2
  """
  def zip_reduce(t1, axes1, t2, axes2, acc, fun)

  def zip_reduce(t1, axes1, t2, axes2, acc, fun) when is_function(fun, 2) do
    output_type = Nx.Type.merge(t1.type, t2.type)
    {v1, v2, new_shape} = bin_zip_axes(t1, axes1, t2, axes2)

    data_and_acc =
      for b1 <- v1, b2 <- v2 do
        {bin, acc} = bin_zip_reduce(b1, b2, t1.type, t2.type, <<>>, acc, fun)
        {scalar_to_bin(bin, output_type), acc}
      end

    {final_data, final_acc} = Enum.unzip(data_and_acc)

    {%T{
       data: {Nx.BitStringDevice, IO.iodata_to_binary(final_data)},
       shape: new_shape,
       type: output_type
     }, final_acc}
  end

  # Helper for zipping 2 tensors along given axes.
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
  defp bin_zip_axes(t1, axes1, t2, axes2) do
    {_, folding_size} = t1.type
    {_, folded_size} = t2.type

    {folding_view, folding_shape} =
      bin_aggregate_axes(to_bitstring(t1), axes1, t1.shape, folding_size)

    {folded_view, folded_shape} =
      bin_aggregate_axes(to_bitstring(t2), axes2, t2.shape, folded_size)

    folding_dim = zip_dims(t1, axes1)
    folded_dim = zip_dims(t2, axes2)

    unless folding_dim == folded_dim do
      raise ArgumentError,
            "unable to zip tensors along the given axes. " <>
              "Dimensions of zipped axes must match, got #{folding_dim} and #{folded_dim}"
    end

    {folding_view, folded_view, List.to_tuple(folding_shape ++ folded_shape)}
  end

  defp zip_dims(t, nil), do: tuple_product(t.shape)
  defp zip_dims(_, []), do: 1
  defp zip_dims(t, [axis | axes]), do: elem(t.shape, axis) * zip_dims(t, axes)

  # Helper for reducing down a single axis over two tensors,
  # returning tensor data and a final accumulator.
  defp bin_zip_reduce(<<>>, <<>>, _left_type, _right_type, bin, acc, _fun),
    do: {bin, acc}

  defp bin_zip_reduce(b1, b2, left_type, right_type, _bin, acc, fun) do
    {head1, rest1, head2, rest2} =
      match_types [left_type, right_type] do
        <<match!(x, 0), rest1::bitstring>> = b1
        <<match!(y, 1), rest2::bitstring>> = b2
        {read!(x, 0), rest1, read!(y, 1), rest2}
      end

    {bin, acc} = fun.({head1, head2}, acc)
    bin_zip_reduce(rest1, rest2, left_type, right_type, bin, acc, fun)
  end

  ## Axes helpers

  # Helper for "viewing" a tensor along the given axes.
  # Returns the view and the expected new shape when
  # reducing down the axes.
  #
  # If the axes isn't provided, the "view" is just the
  # entire binary as it is layed out in memory and we
  # expect the entire tensor to be reduced down to a scalar.
  defp bin_aggregate_axes(binary, axes, shape, size) do
    if axes do
      {chunk_size, read_size, path, shape} = aggregate_axes(axes, shape, size)

      view =
        for <<chunk::size(chunk_size)-bitstring <- binary>> do
          weighted_traverse(path, chunk, read_size)
        end

      {List.flatten(view), shape}
    else
      {[binary], []}
    end
  end

  defp aggregate_axes([_ | _] = given_axes, shape, size) do
    rank = tuple_size(shape)
    axes = Enum.sort(Enum.map(given_axes, &if(&1 >= 0, do: &1, else: rank + &1)))

    unless Enum.all?(axes, &(&1 >= 0 and &1 < rank)) and
             length(Enum.uniq(axes)) == length(given_axes) do
      raise ArgumentError,
            "axes #{inspect(axes)} must be unique integers between 0 and #{rank - 1}"
    end

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
    {chunk_size, read_size, path, aggregate_shape(shape, axes, rank)}
  end

  defp aggregate_axes(axes, _shape, _size) do
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

  defp aggregate_shape(shape, axes, rank), do: aggregate_shape(shape, axes, 0, rank)

  defp aggregate_shape(_shape, _axes, n, n), do: []

  defp aggregate_shape(shape, axes, i, n) do
    if i in axes do
      aggregate_shape(shape, axes, i + 1, n)
    else
      [elem(shape, i) | aggregate_shape(shape, axes, i + 1, n)]
    end
  end
end

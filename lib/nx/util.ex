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
  Returns the underlying tensor as a flat list.

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
  Returns the underlying tensor as a scalar.

  If the tensor has a dimension, it raises.

    ## Examples

      iex> Nx.Util.to_scalar(1)
      1

      iex> Nx.Util.to_scalar(Nx.tensor([1.0, 2.0, 3.0]))
      ** (ArgumentError) cannot convert tensor of shape {3} to scalar
  """
  def to_scalar(tensor) do
    tensor = Nx.tensor(tensor)

    if tensor.shape != {} do
      raise ArgumentError, "cannot convert tensor of shape #{inspect(tensor.shape)} to scalar"
    end

    data = to_bitstring(tensor)

    match_types [tensor.type] do
      <<match!(x, 0)>> = data
      read!(x, 0)
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
      ** (ArgumentError) given axis (2) invalid for shape with rank 2
  """
  def reduce(tensor, acc, opts \\ [], fun)
      when is_list(opts) and is_function(fun, 2) do
    %T{type: {_, size} = type, shape: shape} = t = Nx.tensor(tensor)
    output_type = opts[:type] || type

    {view, new_shape} =
      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes)
        view = bin_aggregate_axes(to_bitstring(t), axes, shape, size)
        {view, Nx.Shape.contract(shape, axes)}
      else
        {[to_bitstring(t)], {}}
      end

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

  The size of the dimensions of `t1` and `t2` must match
  along `axes1` and `axes2`.

  ## Examples

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

  """
  def zip_reduce(t1, [_ | _] = axes1, t2, [_ | _] = axes2, acc, fun)
      when is_function(fun, 2) do
    %T{type: left_type} = t1 = Nx.tensor(t1)
    %T{type: right_type} = t2 = Nx.tensor(t2)
    output_type = Nx.Type.merge(left_type, right_type)

    axes1 = Nx.Shape.normalize_axes(t1.shape, axes1)
    axes2 = Nx.Shape.normalize_axes(t2.shape, axes2)
    new_shape = Nx.Shape.zip_reduce(t1.shape, axes1, t2.shape, axes2)
    {v1, v2} = bin_zip_axes(t1, axes1, t2, axes2)

    data_and_acc =
      for b1 <- v1, b2 <- v2 do
        {bin, acc} = bin_zip_reduce(b1, b2, left_type, right_type, <<>>, acc, fun)
        {scalar_to_bin(bin, output_type), acc}
      end

    {final_data, final_acc} = Enum.unzip(data_and_acc)

    {%T{
       data: {Nx.BitStringDevice, IO.iodata_to_binary(final_data)},
       shape: new_shape,
       type: output_type
     }, final_acc}
  end

  @doc """
  Reduces elements in a window.

  The rank of the input tensor, window dimensions, and window
  strides must match.

  ## Examples

      iex> Nx.Util.reduce_window(Nx.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [11, 12, 13, 14]]),
      ...> :first,
      ...> fn x, acc -> if acc == :first, do: x, else: max(x, acc) end,
      ...> {2, 2}, {1, 1}
      ...> )
      #Nx.Tensor<
        s64[3][3]
        [
          [5, 6, 7],
          [8, 9, 10],
          [12, 13, 14]
        ]
      >

      iex> Nx.Util.reduce_window(Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
      ...> :first,
      ...> fn x, acc -> if acc == :first, do: x, else: max(x, acc) end,
      ...> {2, 2}, {1, 1}, :same
      ...> )
      #Nx.Tensor<
        s64[3][3]
        [
          [5, 6, 6],
          [8, 9, 9],
          [8, 9, 9]
        ]
      >

      iex> Nx.Util.reduce_window(Nx.tensor([[1, 2, 3], [4, 5, 6]]),
      ...> 0,
      ...> fn x, acc -> x + acc end,
      ...> {1, 2}, {1, 1}, :same
      ...> )
      #Nx.Tensor<
        s64[2][3]
        [
          [3, 5, 3],
          [9, 11, 6]
        ]
      >
  """
  def reduce_window(tensor, acc, fun, window_dimensions, window_strides, padding \\ :valid) do
    %T{type: {_, size} = type, shape: shape} = t = Nx.tensor(tensor)

    %T{shape: padded_shape} =
      t =
      case padding do
        :valid ->
          t

        :same ->
          padding_values = Nx.Shape.calculate_padding(shape, window_dimensions, window_strides)
          Nx.pad(t, 0, padding_values)
      end

    output_shape = Nx.Shape.window(padded_shape, window_dimensions, window_strides)

    data = Nx.Util.to_bitstring(t)

    weighted_shape = weighted_shape(padded_shape, size, window_dimensions)
    anchors = Enum.sort(make_anchors(padded_shape, window_strides, window_dimensions, []))

    data =
      for anchor <- anchors, into: <<>> do
        offset = weighted_offset(weighted_shape, anchor)

        window = IO.iodata_to_binary(weighted_traverse(weighted_shape, data, size, offset))

        match_types [type] do
          window_val =
            for <<match!(x, 0) <- window>>,
              reduce: acc,
              do: (acc -> fun.(read!(x, 0), acc))

          <<write!(window_val, 0)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}, shape: output_shape}
  end

  defp make_anchors(shape, strides, window, anchors)
       when is_tuple(shape) and is_tuple(strides) and is_tuple(window),
       do:
         make_anchors(
           Tuple.to_list(shape),
           Tuple.to_list(strides),
           Tuple.to_list(window),
           anchors
         )

  defp make_anchors([], [], _window, anchors), do: anchors

  defp make_anchors([dim | shape], [s | strides], [w | window], []) do
    dims = for i <- 0..(dim - 1), rem(i, s) == 0 and i + w - 1 < dim, do: {i}
    make_anchors(shape, strides, window, dims)
  end

  defp make_anchors([dim | shape], [s | strides], [w | window], anchors) do
    dims =
      for i <- 0..(dim - 1), rem(i, s) == 0 and i + w - 1 < dim do
        Enum.map(anchors, &Tuple.append(&1, i))
      end

    make_anchors(shape, strides, window, List.flatten(dims))
  end

  # Helper used in Nx.pad to add padding to the high and low
  # ends of the last dimension of a tensor
  @doc false
  def pad_last_dim(%T{shape: shape, type: {_, size} = type} = t, value, edge_low, edge_high) do
    data = Nx.Util.to_bitstring(t)

    view = bin_aggregate_axes(data, [tuple_size(shape) - 1], shape, size)

    new_shape = pad_in_dim(shape, tuple_size(shape) - 1, edge_low, edge_high)

    {edge_low_padding, edge_high_padding} =
      match_types [type] do
        edge_high_padding =
          if edge_high <= 0,
            do: <<>>,
            else: for(_ <- 1..edge_high, into: <<>>, do: <<write!(value, 0)>>)

        edge_low_padding =
          if edge_low <= 0,
            do: <<>>,
            else: for(_ <- 1..edge_low, into: <<>>, do: <<write!(value, 0)>>)

        {edge_low_padding, edge_high_padding}
      end

    data =
      for bin <- view, into: <<>> do
        cond do
          edge_low < 0 and edge_high < 0 ->
            low_byte = abs(edge_low) * size
            high_byte = abs(edge_high) * size
            new_bytes = (byte_size(bin) * div(size, 8)) - high_byte - low_byte
            <<_::size(low_byte)-bitstring, new_bin::size(new_bytes)-bitstring, _::bitstring>> = bin
            new_bin

          edge_low < 0 and edge_high >= 0 ->
            low_byte = abs(edge_low) * size
            <<_::size(low_byte)-bitstring, new_bin::bitstring>> = bin
            <<new_bin::bitstring, edge_high_padding::bitstring>>

          edge_low >= 0 and edge_high < 0 ->
            high_byte = abs(edge_high) * size
            new_bytes = (byte_size(bin) * div(size, 8)) - high_byte
            <<new_bin::size(new_bytes)-bitstring, _::size(high_byte)-bitstring>> = bin
            <<edge_low_padding::bitstring, new_bin::bitstring>>

          true ->
            <<edge_low_padding::bitstring, bin::bitstring, edge_high_padding::bitstring>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}, type: type, shape: new_shape}
  end

  defp pad_in_dim(shape, dim, edge_low, edge_high) do
    dim = Nx.Shape.normalize_axis(shape, dim)
    dim_size = elem(shape, dim)
    new_dim = dim_size + edge_high + edge_low
    if new_dim <= 0 do
      raise ArgumentError, "invalid padding widths, edge low and edge high padding" <>
                           " cannot cause zero or negative dimension size"
    else
      :erlang.setelement(dim + 1, shape, new_dim)
    end
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

    folding_view = bin_aggregate_axes(to_bitstring(t1), axes1, t1.shape, folding_size)
    folded_view = bin_aggregate_axes(to_bitstring(t2), axes2, t2.shape, folded_size)
    {folding_view, folded_view}
  end

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
    {chunk_size, read_size, path} = aggregate_axes(axes, shape, size)

    view =
      for <<chunk::size(chunk_size)-bitstring <- binary>> do
        weighted_traverse(path, chunk, read_size)
      end

    List.flatten(view)
  end

  defp aggregate_axes([_ | _] = axes, shape, size) do
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
end

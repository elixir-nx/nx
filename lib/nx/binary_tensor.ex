defmodule Nx.BinaryTensor do
  # TODO: Document me and define a behaviour for tensor backends
  @moduledoc false

  defstruct [:device, :state]

  alias Nx.Tensor, as: T
  alias Nx.BinaryTensor, as: BT

  import Nx.Shared
  import Bitwise, only: [>>>: 2, &&&: 2]

  ## Creation

  @doc false
  def tensor(arg, type, names) do
    {shape, data} = flatten(arg, type)

    names = Nx.Names.validate!(names, shape)

    if data == "" do
      raise "cannot build empty tensor"
    end

    from_binary(%T{shape: shape, type: type, names: names}, data)
  end

  defp flatten(list, type) when is_list(list) do
    {dimensions, acc} = flatten_list(list, type, [], [])

    {dimensions |> Enum.reverse() |> List.to_tuple(),
     acc |> Enum.reverse() |> :erlang.list_to_binary()}
  end

  defp flatten(other, type), do: {{}, number_to_binary(other, type)}

  defp flatten_list([], _type, dimensions, acc) do
    {[0 | dimensions], acc}
  end

  defp flatten_list([head | rest], type, parent_dimensions, acc) when is_list(head) do
    {child_dimensions, acc} = flatten_list(head, type, [], acc)

    {n, acc} =
      Enum.reduce(rest, {1, acc}, fn list, {count, acc} ->
        case flatten_list(list, type, [], acc) do
          {^child_dimensions, acc} ->
            {count + 1, acc}

          {other_dimensions, _acc} ->
            raise ArgumentError,
                  "cannot build tensor because lists have different shapes, got " <>
                    inspect(List.to_tuple(child_dimensions)) <>
                    " at position 0 and " <>
                    inspect(List.to_tuple(other_dimensions)) <> " at position #{count + 1}"
        end
      end)

    {child_dimensions ++ [n | parent_dimensions], acc}
  end

  defp flatten_list(list, type, dimensions, acc) do
    {[length(list) | dimensions], Enum.reduce(list, acc, &[number_to_binary(&1, type) | &2])}
  end

  @doc false
  def random_uniform(%{type: type, shape: shape} = out, min, max) do
    gen =
      case type do
        {:s, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {:u, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {_, _} -> fn -> (max - min) * :rand.uniform() + min end
      end

    data = for _ <- 1..Nx.size(shape), into: "", do: number_to_binary(gen.(), type)
    from_binary(out, data)
  end

  @doc false
  def random_normal(%{type: type, shape: shape} = out, mu, sigma) do
    data =
      for _ <- 1..Nx.size(shape),
          into: "",
          do: number_to_binary(:rand.normal(mu, sigma), type)

    from_binary(out, data)
  end

  @doc false
  def iota(%{shape: {n}, type: type} = out, 0) do
    data = for i <- 0..(n - 1), do: number_to_binary(i, type)
    from_binary(out, data)
  end

  def iota(%{shape: shape, type: type} = out, axis) do
    {dims_before, [dim | dims_after]} =
      shape
      |> Tuple.to_list()
      |> Enum.split(axis)

    # Number of repetitions of an index in memory
    repeat_blocks =
      dims_after
      |> Enum.reduce(1, &*/2)

    # Number of cycles of the counting pattern
    cycles =
      dims_before
      |> Enum.reduce(1, &*/2)

    data =
      for _ <- 1..cycles,
          i <- 0..(dim - 1),
          _ <- 1..repeat_blocks,
          into: "",
          do: number_to_binary(i, type)

    from_binary(out, data)
  end

  ## Device API

  @doc false
  def from_binary(t, binary) when is_binary(binary) do
    %{t | data: %BT{device: Nx.BinaryDevice, state: binary}}
  end

  def from_binary(t, other) do
    %{t | data: %BT{device: Nx.BinaryDevice, state: IO.iodata_to_binary(other)}}
  end

  @doc false
  def to_binary(%T{data: %{device: Nx.BinaryDevice, state: data}}), do: data

  def to_binary(%T{data: %{device: device}}) do
    raise ArgumentError,
          "cannot read Nx.Tensor data because the data is allocated on device #{inspect(device)}. " <>
            "Please use Nx.device_transfer/1 to transfer data back to Elixir"
  end

  @doc false
  def device_read(%T{data: %{device: device, state: state}} = tensor) do
    from_binary(tensor, device.read(state))
  end

  @doc false
  def device_deallocate(%T{data: %{device: device, state: state}}) do
    device.deallocate(state)
  end

  @doc false
  def device_transfer(%T{data: %{device: Nx.BinaryDevice}} = tensor, device, opts) do
    %{type: type, shape: shape} = tensor
    {device, state} = device.allocate(to_binary(tensor), type, shape, opts)
    %{tensor | data: %BT{device: device, state: state}}
  end

  def device_transfer(%T{} = tensor, Nx.BinaryDevice, _opts) do
    new = device_read(tensor)
    _ = device_deallocate(tensor)
    new
  end

  def device_transfer(%T{data: %{device: data_device}}, device, _opts) do
    raise ArgumentError, "cannot transfer from #{inspect(data_device)} to #{inspect(device)}"
  end

  ## Shape

  @doc false
  def reshape(out, tensor, _shape), do: from_binary(out, to_binary(tensor))

  @doc false
  def squeeze(out, tensor, _axes), do: from_binary(out, to_binary(tensor))

  ## Broadcast

  @doc false
  def broadcast(out, t, shape, axes) do
    from_binary(out, broadcast_data(t, shape, axes))
  end

  defp broadcast_data(%{shape: shape} = t, shape),
    do: to_binary(t)

  defp broadcast_data(t, shape),
    do: broadcast_data(t, shape, Nx.Shape.broadcast_axes(t.shape, shape))

  defp broadcast_data(%T{shape: {}} = t, shape, []) do
    t
    |> to_binary()
    |> :binary.copy(Nx.size(shape))
  end

  defp broadcast_data(%T{shape: old_shape, type: {_, size}} = t, new_shape, axes) do
    chunk_size = size * Nx.size(old_shape)

    new_shape
    |> Tuple.to_list()
    |> unary_broadcast(0, old_shape, 0, axes, to_binary(t), chunk_size)
    |> IO.iodata_to_binary()
  end

  # Old and new match
  defp unary_broadcast([dim | dims], axis, old_shape, old_pos, [axis | axes], data, chunk_size)
       when elem(old_shape, old_pos) == dim do
    chunk_size = div(chunk_size, dim)

    for <<chunk::size(chunk_size)-bitstring <- data>> do
      unary_broadcast(dims, axis + 1, old_shape, old_pos + 1, axes, chunk, chunk_size)
    end
  end

  # Implicit broadcasting
  defp unary_broadcast([dim | dims], axis, old_shape, old_pos, [axis | axes], data, chunk_size)
       when elem(old_shape, old_pos) == 1 do
    for _ <- 1..dim do
      unary_broadcast(dims, axis + 1, old_shape, old_pos + 1, axes, data, chunk_size)
    end
  end

  # Explicit broadcasting (unmapped axes)
  defp unary_broadcast([dim | dims], axis, old_shape, old_pos, axes, data, chunk_size) do
    for _ <- 1..dim do
      unary_broadcast(dims, axis + 1, old_shape, old_pos, axes, data, chunk_size)
    end
  end

  defp unary_broadcast([], _axis, _old_shape, _old_pos, [], data, _chunk_size) do
    data
  end

  ## Shape

  @doc false
  def transpose(out, %T{shape: shape, type: {_, size}} = t, axes) do
    data = to_binary(t)
    {list, min, max} = transpose_axes(shape, axes)
    weighted_shape = weighted_shape(shape, size)

    # The chunk size is computed based on all dimensions
    # before the minimum one being changed. For example,
    # for {0, 1, 2, 3} and the swap is between 1 and 2,
    # the chunk_size will be d1 * d2 * d3 * size.
    chunk_size = weighted_chunk(weighted_shape, min, size)

    # All of the major dimensions not being transposed can be
    # read at once. For example, for {0, 1, 2, 3} and the swap
    # is between 1 and 2, the read_size will be d3 * size.
    read_size = weighted_chunk(weighted_shape, max + 1, size)

    # And now how we will traverse
    traverse_list = Enum.map(list, &Enum.fetch!(weighted_shape, &1))

    data =
      for <<chunk::size(chunk_size)-bitstring <- data>> do
        weighted_traverse(traverse_list, chunk, read_size)
      end

    from_binary(out, data)
  end

  defp transpose_axes(shape, axes) do
    size = tuple_size(shape)
    {axes, min} = transpose_min(axes, 0)
    {axes, max} = transpose_max(Enum.reverse(axes), size - 1)
    {axes, min, max}
  end

  defp transpose_min([head | tail], head), do: transpose_min(tail, head + 1)
  defp transpose_min(tail, head), do: {tail, head}

  defp transpose_max([head | tail], head), do: transpose_max(tail, head - 1)
  defp transpose_max(tail, head), do: {Enum.reverse(tail), head}

  ## Pad

  # We ignore the out because we need to recur over the shape
  # as we transpose and build the rest.
  @doc false
  def pad(_out, t, pad_value, padding_config) do
    pad_value = Nx.Util.to_scalar(pad_value)

    case t.shape do
      {} ->
        t

      {_} ->
        [{edge_low, edge_high, interior}] = padding_config
        pad_last_dim(t, pad_value, edge_low, edge_high, interior)

      _ ->
        permutation = for i <- 0..(Nx.rank(t) - 2), do: i
        permutation = [Nx.rank(t) - 1 | permutation]

        for {edge_low, edge_high, interior} <- Enum.reverse(padding_config), reduce: t do
          acc ->
            Nx.transpose(pad_last_dim(acc, pad_value, edge_low, edge_high, interior), permutation)
        end
    end
  end

  # Add padding to the high and low ends of the last dimension of a tensor
  defp pad_last_dim(
         %T{shape: shape, type: {_, size} = type} = t,
         value,
         edge_low,
         edge_high,
         interior
       ) do
    view = aggregate_axes(to_binary(t), [tuple_size(shape) - 1], shape, size)
    new_shape = pad_in_dim(shape, tuple_size(shape) - 1, edge_low, edge_high, interior)

    {edge_low_padding, edge_high_padding, interior_padding} =
      match_types [type] do
        edge_high_padding =
          if edge_high <= 0,
            do: <<>>,
            else: for(_ <- 1..edge_high, into: <<>>, do: <<write!(value, 0)>>)

        edge_low_padding =
          if edge_low <= 0,
            do: <<>>,
            else: for(_ <- 1..edge_low, into: <<>>, do: <<write!(value, 0)>>)

        interior_padding =
          if interior == 0,
            do: <<>>,
            else: for(_ <- 1..interior, into: <<>>, do: <<write!(value, 0)>>)

        {edge_low_padding, edge_high_padding, interior_padding}
      end

    interior_padding_size = interior * size

    interior_padded =
      for bin <- view do
        match_types [type] do
          padded =
            for <<match!(dim, 0) <- bin>>, into: <<>> do
              <<write!(dim, 0), interior_padding::bitstring>>
            end

          new_bytes = byte_size(padded) * 8 - interior_padding_size
          <<new_bin::size(new_bytes)-bitstring, _::bitstring>> = padded
          new_bin
        end
      end

    data =
      for bin <- interior_padded, into: <<>> do
        cond do
          edge_low < 0 and edge_high < 0 ->
            low_byte = abs(edge_low) * size
            high_byte = abs(edge_high) * size
            new_bytes = byte_size(bin) * 8 - high_byte - low_byte

            <<_::size(low_byte)-bitstring, new_bin::size(new_bytes)-bitstring, _::bitstring>> =
              bin

            new_bin

          edge_low < 0 and edge_high >= 0 ->
            low_byte = abs(edge_low) * size
            <<_::size(low_byte)-bitstring, new_bin::bitstring>> = bin
            <<new_bin::bitstring, edge_high_padding::bitstring>>

          edge_low >= 0 and edge_high < 0 ->
            high_byte = abs(edge_high) * size
            new_bytes = byte_size(bin) * 8 - high_byte
            <<new_bin::size(new_bytes)-bitstring, _::bitstring>> = bin
            <<edge_low_padding::bitstring, new_bin::bitstring>>

          true ->
            <<edge_low_padding::bitstring, bin::bitstring, edge_high_padding::bitstring>>
        end
      end

    from_binary(%{t | type: type, shape: new_shape}, data)
  end

  defp pad_in_dim(shape, dim, edge_low, edge_high, interior) do
    dim_size = elem(shape, dim)
    interior_padding_factor = if interior == 0, do: 0, else: dim_size * interior - 1
    new_dim = dim_size + interior_padding_factor + edge_high + edge_low
    put_elem(shape, dim, new_dim)
  end

  ## Two-element

  @doc false
  def outer(out, %{type: left_type} = t1, %{type: right_type} = t2) do
    b1 = to_binary(t1)
    b2 = to_binary(t2)

    data =
      match_types [left_type, right_type] do
        for <<match!(left, 0) <- b1>>,
            <<match!(right, 1) <- b2>>,
            into: <<>>,
            do: number_to_binary(read!(left, 0) * read!(right, 1), out.type)
      end

    from_binary(out, data)
  end

  @doc false
  def dot(out, %{type: t1} = left, axes1, %{type: t2} = right, axes2) do
    bin_zip_reduce(out, left, axes1, right, axes2, 0, fn lhs, rhs, acc ->
      res = binary_to_number(lhs, t1) * binary_to_number(rhs, t2) + acc
      {res, res}
    end)
  end

  ## Element wise ternary ops

  @doc false
  def select(out, %{shape: {}} = pred, on_true, on_false) do
    if Nx.Util.to_scalar(pred) == 0,
      do: from_binary(out, broadcast_data(on_false, out.shape)),
      else: from_binary(out, broadcast_data(on_true, out.shape))
  end

  def select(%{shape: shape, type: type} = out, pred, on_true, on_false) do
    %T{type: {_, pred_size} = pred_type} = pred
    %T{type: {_, left_size} = left_type} = on_true
    %T{type: {_, right_size} = right_type} = on_false

    pred_data = to_binary(pred)
    on_true_data = broadcast_data(on_true, shape)
    on_false_data = broadcast_data(on_false, shape)

    data =
      for i <- 0..(Nx.size(shape) - 1), into: <<>> do
        pred =
          match_types [pred_type] do
            consumed = i * pred_size
            <<_::size(consumed)-bitstring, match!(pred, 0), _::bitstring>> = pred_data
            read!(pred, 0)
          end

        result =
          if pred == 0 do
            match_types [right_type] do
              consumed = i * right_size
              <<_::size(consumed)-bitstring, match!(x, 0), _::bitstring>> = on_false_data
              read!(x, 0)
            end
          else
            match_types [left_type] do
              consumed = i * left_size
              <<_::size(consumed)-bitstring, match!(x, 0), _::bitstring>> = on_true_data
              read!(x, 0)
            end
          end

        number_to_binary(result, type)
      end

    from_binary(out, data)
  end

  ## Element wise bin ops

  for fun <-
        [:add, :subtract, :multiply, :power, :remainder, :divide, :arctan2, :min, :max] ++
          [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
          [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] do
    capture = Macro.var(:"element_#{fun}", __MODULE__)

    @doc false
    def unquote(fun)(out, left, right) do
      element_wise_bin_op(out, left, right, &(unquote(capture) / 3))
    end
  end

  defp element_wise_bin_op(%{shape: shape, type: type} = out, left, right, fun) do
    %T{type: {_, left_size} = left_type} = left
    %T{type: {_, right_size} = right_type} = right

    count = Nx.size(shape)
    left_data = broadcast_data(left, shape)
    right_data = broadcast_data(right, shape)

    data =
      for i <- 0..(count - 1), into: <<>> do
        x =
          match_types [left_type] do
            left_consumed = i * left_size
            <<_::size(left_consumed)-bitstring, match!(x, 0), _::bitstring>> = left_data
            read!(x, 0)
          end

        y =
          match_types [right_type] do
            right_consumed = i * right_size
            <<_::size(right_consumed)-bitstring, match!(y, 0), _::bitstring>> = right_data
            read!(y, 0)
          end

        match_types [type] do
          <<write!(fun.(type, x, y), 0)>>
        end
      end

    from_binary(out, data)
  end

  defp element_add(_, a, b), do: a + b
  defp element_subtract(_, a, b), do: a - b
  defp element_multiply(_, a, b), do: a * b
  defp element_divide(_, a, b), do: a / b
  defp element_arctan2(_, a, b), do: :math.atan2(a, b)
  defp element_max(_, a, b), do: :erlang.max(a, b)
  defp element_min(_, a, b), do: :erlang.min(a, b)

  defp element_remainder(_, a, b) when is_integer(a) and is_integer(b), do: rem(a, b)
  defp element_remainder(_, a, b), do: :math.fmod(a, b)

  defp element_power({type, _}, a, b) when type in [:s, :u], do: integer_pow(a, b)
  defp element_power(_, a, b), do: :math.pow(a, b)

  # TODO: Use Integer.pow on Elixir v1.12
  defp integer_pow(base, exponent) when is_integer(base) and is_integer(exponent) do
    if exponent < 0, do: :erlang.error(:badarith, [base, exponent])
    guarded_pow(base, exponent)
  end

  defp guarded_pow(_, 0), do: 1
  defp guarded_pow(b, 1), do: b
  defp guarded_pow(b, e) when (e &&& 1) == 0, do: guarded_pow(b * b, e >>> 1)
  defp guarded_pow(b, e), do: b * guarded_pow(b * b, e >>> 1)

  defp element_bitwise_and(_, a, b), do: :erlang.band(a, b)
  defp element_bitwise_or(_, a, b), do: :erlang.bor(a, b)
  defp element_bitwise_xor(_, a, b), do: :erlang.bxor(a, b)

  defp element_left_shift(_, a, b) when b >= 0, do: :erlang.bsl(a, b)
  defp element_left_shift(_, _, b), do: raise(ArgumentError, "cannot left shift by #{b}")

  defp element_right_shift(_, a, b) when b >= 0, do: :erlang.bsr(a, b)
  defp element_right_shift(_, _, b), do: raise(ArgumentError, "cannot right shift by #{b}")

  defp element_equal(_, a, b), do: if(a == b, do: 1, else: 0)
  defp element_not_equal(_, a, b), do: if(a != b, do: 1, else: 0)
  defp element_greater(_, a, b), do: if(a > b, do: 1, else: 0)
  defp element_less(_, a, b), do: if(a < b, do: 1, else: 0)
  defp element_greater_equal(_, a, b), do: if(a >= b, do: 1, else: 0)
  defp element_less_equal(_, a, b), do: if(a <= b, do: 1, else: 0)

  ## Element wiwse unary ops

  for {name, {_desc, code}} <- Nx.Shared.unary_math_funs() do
    @doc false
    def unquote(name)(out, tensor) do
      element_wise_unary_op(out, tensor, fn x -> unquote(code) end)
    end
  end

  @doc false
  def count_leading_zeros(out, %{type: {_, size} = type} = tensor) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_clz(seg, size), 0)>>
        end
      end

    from_binary(out, data)
  end

  @doc false
  def population_count(out, %{type: {_, size} = type} = tensor) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_popcount(seg, 0), 0)>>
        end
      end

    from_binary(out, data)
  end

  @doc false
  def abs(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.abs/1)
  @doc false
  def bitwise_not(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.bnot/1)
  @doc false
  def ceil(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.ceil/1)
  @doc false
  def floor(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.floor/1)
  @doc false
  def negate(out, tensor), do: element_wise_unary_op(out, tensor, &-/1)
  @doc false
  def round(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.round/1)
  @doc false
  def sign(out, tensor), do: element_wise_unary_op(out, tensor, &element_sign/1)

  defp element_sign(n) when n < 0, do: -1
  defp element_sign(n) when n > 0, do: 1
  defp element_sign(n), do: n

  # https://en.wikipedia.org/wiki/Hamming_weight
  # There are algorithms with faster worst case but they are size specific.
  # The implementation below is also the most efficient for low counts. Given
  # our integers are always 64 bits internally, we will have a lot of zeros
  # internally, so this should be the fastest.
  defp element_popcount(0, count), do: count
  defp element_popcount(n, count), do: element_popcount(n &&& n - 1, count + 1)

  defp element_wise_unary_op(out, tensor, fun) do
    data =
      match_types [tensor.type, out.type] do
        for <<match!(seg, 0) <- to_binary(tensor)>>, into: <<>> do
          <<write!(fun.(read!(seg, 0)), 1)>>
        end
      end

    from_binary(out, data)
  end

  defp element_clz(0, size), do: size
  defp element_clz(n, 64), do: element_clz64(n)
  defp element_clz(n, 32), do: element_clz32(n)
  defp element_clz(n, 16), do: element_clz16(n)
  defp element_clz(n, 8), do: element_clz8(n)

  defp element_clz64(num) do
    case num &&& 0xFFFFFFFF00000000 do
      0 -> 32 + element_clz32(num)
      _ -> element_clz32(num >>> 32)
    end
  end

  defp element_clz32(num) do
    case num &&& 0xFFFF0000 do
      0 -> 16 + element_clz16(num)
      _ -> element_clz16(num >>> 16)
    end
  end

  defp element_clz16(num) do
    case num &&& 0xFF00 do
      0 -> 8 + element_clz8(num)
      _ -> element_clz8(num >>> 8)
    end
  end

  defp element_clz8(num) do
    case num &&& 0xF0 do
      0 -> 4 + element_clz4(num)
      _ -> element_clz4(num >>> 4)
    end
  end

  defp element_clz4(num) do
    case num &&& 0xC do
      0 -> 2 + element_clz2(num)
      _ -> element_clz2(num >>> 2)
    end
  end

  defp element_clz2(0), do: 2
  defp element_clz2(1), do: 1
  defp element_clz2(_), do: 0

  ## Inspect

  alias Inspect.Algebra, as: IA

  @doc false
  def inspect(tensor, opts) do
    dims = Tuple.to_list(tensor.shape)
    names = tensor.names

    open = IA.color("[", :list, opts)
    sep = IA.color(",", :list, opts)
    close = IA.color("]", :list, opts)
    type = IA.color(Nx.Type.to_string(tensor.type), :atom, opts)
    shape = shape_to_algebra(dims, names, open, close)

    {data, _limit} =
      case tensor.data do
        %Nx.BinaryTensor{device: Nx.BinaryDevice, state: bin} ->
          {_, size} = tensor.type
          total_size = Enum.reduce(dims, size, &*/2)
          chunk(dims, bin, opts.limit, total_size, tensor.type, {open, sep, close})

        # TODO: To print data on device, we can support reading a slice
        # from the device which we will compute with:
        #
        #     min(opts.limit, Nx.size(shape)) * size
        #
        %Nx.BinaryTensor{device: device} ->
          {IA.to_doc(device, opts), opts.limit}
      end

    IA.concat([type, shape, IA.line(), data])
  end

  defp shape_to_algebra(dims, names, open, close) do
    dims
    |> Enum.zip(names)
    |> Enum.map(fn
      {number, nil} ->
        IA.concat([open, Integer.to_string(number), close])

      {number, name} ->
        IA.concat([open, Atom.to_string(name), ": ", Integer.to_string(number), close])
    end)
    |> IA.concat()
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
        {open, IA.concat(sep, " "), close, 0}
      else
        {IA.concat(open, IA.line()), IA.concat(sep, IA.line()), IA.concat(IA.line(), close), 2}
      end

    doc =
      open
      |> IA.concat(IA.concat(Enum.intersperse(acc, sep)))
      |> IA.nest(nest)
      |> IA.concat(close)

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

  ## Conv

  def conv(out, t, k, strides, padding, input_dilation, kernel_dilation) do
    # Consider an image representation, the input shape should be:
    # {batch, channels, height, width}
    #
    # The kernel then has the following shape:
    # {num_filters, channels, filter_height, filter_width}
    #
    # The output type is merged between the input tensor
    # and the input kernel, both inputs must be floating types
    #
    # The output shape is a product of the batch size,
    # number of filters in the input kernel, and the transformation
    # on the spatial dimensions.
    #
    # The shape of each spatial dimension is calculated as:
    #
    # (spatial_dim - filter_size + 2 * padding_size) / stride + 1
    #
    # where spatial dim is the current spatial dimension size, filter size is
    # the size of the corresponding spatial dimension in the kernel,
    # padding size is the amount of padding applied to either side of the
    # spatial dimension, and stride is the input stride
    #
    # The final shape is then given as:
    # {batch, num_filters, spatial_dim0, spatial_dim1, ...}
    %T{type: {_, input_size} = input_type, shape: input_shape} = t
    %T{type: {_, kernel_size} = kernel_type} = k

    %{type: output_type} = out

    # We need to dilate the spatial dimensions of the kernel first...
    dilation_padding = [
      {0, 0, 0},
      {0, 0, 0} | Enum.map(Tuple.to_list(kernel_dilation), &{0, 0, &1 - 1})
    ]

    %T{shape: kernel_shape} = k = Nx.pad(k, 0, dilation_padding)

    # The size of the "window" or the size of each filter
    # removes the first two dimensions of the kernel
    #
    # This "window" is applied `num_filters` times across
    # the input tensors spatial dimensions
    filter_shape =
      kernel_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    num_input_channels = elem(input_shape, 1)

    filter_spatial_dims = Tuple.delete_at(kernel_shape, 0)

    filter_size =
      tuple_product(filter_spatial_dims, tuple_size(filter_spatial_dims)) * kernel_size

    # We first pad the input tensor with the following semantics
    #   :valid - no padding
    #   :same - pad with 0 such that the input's spatial dims
    #           remain unchanged WITHOUT considering strides
    #   general - pad with the given padding configuration
    #
    # Padding configuration is guaranteed to always be provided
    # as a list because it is handled in the Nx module before
    # lowering to the implementation; however, the padding
    # configuration only accounts for spatial dims
    spatial_padding_config =
      padding
      |> Enum.zip(Tuple.to_list(input_dilation))
      |> Enum.map(fn {{lo, hi}, dilation} -> {lo, hi, dilation - 1} end)

    padding_config = [
      {0, 0, 0},
      {0, 0, 0} | spatial_padding_config
    ]

    %T{shape: padded_shape} = padded_t = Nx.pad(t, 0, padding_config)

    single_data_dims = Tuple.delete_at(padded_shape, 0)
    batch_size = tuple_product(single_data_dims, tuple_size(single_data_dims)) * input_size

    # We will traverse the input tensor exactly the same as we traversed
    # the binary in reduce_window, but the window is equal to the filter
    # size of the kernel plus the channel size of the input tensor
    window_shape = Tuple.insert_at(filter_shape, 0, num_input_channels)

    input_data = to_binary(padded_t)
    kernel_data = to_binary(k)

    batch_weighted_shape =
      weighted_shape(Tuple.delete_at(padded_shape, 0), input_size, window_shape)

    # We calculate our "anchors" using just the spatial dimensions
    # but they also need to consider the depth or channels of the input
    # tensor, so we always anchor on the `0th` channel
    padded_spatial_dims =
      padded_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    anchors = Enum.sort(make_anchors(padded_spatial_dims, strides, filter_shape, []))
    anchors = Enum.map(anchors, &Tuple.insert_at(&1, 0, 0))

    # Traverse the batch dim first
    output_data =
      for <<batch::size(batch_size)-bitstring <- input_data>>,
          # Traverse the filters next, this allows us to rebuild
          # the resulting binary correctly
          <<filter::size(filter_size)-bitstring <- kernel_data>>,
          # Then we traverse the spatial dimension, applying
          # the filter at each step
          anchor <- anchors,
          into: <<>> do
        offset = weighted_offset(batch_weighted_shape, anchor)
        # The shape of the window is {channels} + filter_shape
        # The shape of the kernel is {num_filters, channels} + filter_shape
        window =
          IO.iodata_to_binary(weighted_traverse(batch_weighted_shape, batch, input_size, offset))

        # The receptive field size of each binary in bytes
        input_field_size = tuple_product(filter_shape, tuple_size(filter_shape)) * input_size
        filter_field_size = tuple_product(filter_shape, tuple_size(filter_shape)) * kernel_size
        # For each channel in both filter and input...
        # The output from a single filter being applied over a window
        # of the input tensor is the sum of the element-wise products
        values =
          for i <- 0..(num_input_channels - 1) do
            current_input_pos = i * input_field_size
            current_filter_pos = i * filter_field_size
            <<_::size(current_input_pos)-bitstring, input_receptive_field::bitstring>> = window
            <<_::size(current_filter_pos)-bitstring, filter_receptive_field::bitstring>> = filter

            for j <- 0..(tuple_product(filter_shape, tuple_size(filter_shape)) - 1) do
              x =
                match_types [input_type] do
                  left_consumed = j * input_size

                  <<_::size(left_consumed)-bitstring, match!(x, 0), _::bitstring>> =
                    input_receptive_field

                  read!(x, 0)
                end

              y =
                match_types [kernel_type] do
                  right_consumed = j * kernel_size

                  <<_::size(right_consumed)-bitstring, match!(y, 0), _::bitstring>> =
                    filter_receptive_field

                  read!(y, 0)
                end

              x * y
            end
          end

        match_types [output_type] do
          sum = Enum.sum(List.flatten(values))
          <<write!(sum, 0)>>
        end
      end

    from_binary(out, output_data)
  end

  ## Aggregation

  @doc false
  def sum(out, %{type: type} = tensor, opts) do
    bin_reduce(out, tensor, 0, opts, fn bin, acc ->
      res = binary_to_number(bin, type) + acc
      {res, res}
    end)
  end

  @doc false
  def argmin(out, tensor, opts) do
    comparator =
      case opts[:tie_break] do
        :high -> &<=/2
        :low -> &</2
      end

    argmin_or_max(out, tensor, comparator, opts[:axis])
  end

  @doc false
  def argmax(out, tensor, opts) do
    comparator =
      case opts[:tie_break] do
        :high -> &>=/2
        :low -> &>/2
      end

    argmin_or_max(out, tensor, comparator, opts[:axis])
  end

  defp argmin_or_max(out, %{type: type} = tensor, comparator, axis) do
    opts = if axis, do: [axes: [axis]], else: []

    bin_reduce(out, tensor, {0, :first, -1}, opts, fn bin, {i, cur_extreme_x, cur_extreme_i} ->
      x = binary_to_number(bin, type)

      if comparator.(x, cur_extreme_x) or cur_extreme_x == :first do
        {i, {i + 1, x, i}}
      else
        {cur_extreme_i, {i + 1, cur_extreme_x, cur_extreme_i}}
      end
    end)
  end

  @doc false
  def reduce(out, tensor, acc, opts, fun) do
    each = %{tensor | shape: {}}

    bin_reduce(out, tensor, acc, opts, fn bin, acc ->
      res = fun.(from_binary(each, bin), acc)
      {res, res}
    end)
  end

  ## Binary reducers

  defp bin_reduce(out, tensor, acc, opts, fun) do
    %T{type: {_, size}, shape: shape} = tensor

    view =
      if axes = opts[:axes] do
        aggregate_axes(to_binary(tensor), axes, shape, size)
      else
        [to_binary(tensor)]
      end

    data =
      for axis <- view do
        {result, _} =
          for <<bin::size(size)-bitstring <- axis>>, reduce: {<<>>, acc} do
            {_, acc} -> fun.(bin, acc)
          end

        scalar_to_binary(result, out.type)
      end

    from_binary(out, data)
  end

  defp bin_zip_reduce(%{type: type} = out, t1, [], t2, [], acc, fun) do
    %{type: {_, s1}} = t1
    %{type: {_, s2}} = t2
    b1 = to_binary(t1)
    b2 = to_binary(t2)

    data =
      match_types [t1.type, t2.type] do
        for <<d1::size(s1)-bitstring <- b1>>, <<d2::size(s2)-bitstring <- b2>>, into: <<>> do
          {result, _} = fun.(d1, d2, acc)
          scalar_to_binary(result, type)
        end
      end

    from_binary(out, data)
  end

  defp bin_zip_reduce(%{type: type} = out, t1, [_ | _] = axes1, t2, [_ | _] = axes2, acc, fun) do
    {_, s1} = t1.type
    {_, s2} = t2.type

    v1 = aggregate_axes(to_binary(t1), axes1, t1.shape, s1)
    v2 = aggregate_axes(to_binary(t2), axes2, t2.shape, s2)

    data =
      for b1 <- v1, b2 <- v2 do
        {bin, _acc} = bin_zip_reduce_axis(b1, b2, s1, s2, <<>>, acc, fun)
        scalar_to_binary(bin, type)
      end

    from_binary(out, data)
  end

  # Helper for reducing down a single axis over two tensors,
  # returning tensor data and a final accumulator.
  defp bin_zip_reduce_axis(<<>>, <<>>, _s1, _s2, bin, acc, _fun),
    do: {bin, acc}

  defp bin_zip_reduce_axis(b1, b2, s1, s2, _bin, acc, fun) do
    <<x::size(s1)-bitstring, rest1::bitstring>> = b1
    <<y::size(s2)-bitstring, rest2::bitstring>> = b2
    {bin, acc} = fun.(x, y, acc)
    bin_zip_reduce_axis(rest1, rest2, s1, s2, bin, acc, fun)
  end

  ## Scalar helpers

  @compile {:inline, number_to_binary: 2, binary_to_number: 2}

  defp scalar_to_binary(value, type) when is_number(value),
    do: number_to_binary(value, type)

  defp scalar_to_binary(%T{shape: {}, type: type} = t, type),
    do: to_binary(t)

  defp scalar_to_binary(t, type) do
    raise ArgumentError,
          "expected a number or a scalar tensor of type #{inspect(type)}, got: #{inspect(t)}"
  end

  defp number_to_binary(number, type),
    do: match_types([type], do: <<write!(number, 0)>>)

  defp binary_to_number(bin, type) do
    match_types [type] do
      <<match!(value, 0)>> = bin
      read!(value, 0)
    end
  end

  ## Aggregation helpers

  defp aggregate_axes(binary, axes, shape, size) do
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

  ## Weighted shapes

  # Converts the shape to a weight shape list.
  #
  # A weighted shape is a list of tuples where the first
  # element is the number of elements in the dimension
  # and the second element is the size to be traversed in
  # the binary to fetch the next element.
  #
  # This is often given to `weighted_traverse/3` as a general
  # mechanism to traverse binaries.
  def weighted_shape(shape, size, limits \\ :none) do
    Enum.reverse(weighted_shape(shape, tuple_size(shape), size, limits))
  end

  defp weighted_shape(_shape, 0, _weight, _limits) do
    []
  end

  defp weighted_shape(shape, pos, weight, limits) do
    shape_elem = :erlang.element(pos, shape)

    element =
      if limits == :none, do: shape_elem, else: min(:erlang.element(pos, limits), shape_elem)

    [{element, weight} | weighted_shape(shape, pos - 1, weight * shape_elem, limits)]
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

  # Makes anchors for traversing a binary with a window.
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

  # Calculates the offset needed to reach a specified position
  # in the binary from a weighted shape list.
  defp weighted_offset(weighted_shape, pos) when is_tuple(pos),
    do: weighted_offset(weighted_shape, Tuple.to_list(pos))

  defp weighted_offset([], []), do: 0

  defp weighted_offset([{_, size} | dims], [x | pos]),
    do: size * x + weighted_offset(dims, pos)

  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)
end

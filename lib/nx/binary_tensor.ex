defmodule Nx.BinaryTensor do
  # TODO: Document me and @doc false everything.
  @moduledoc false

  defstruct [:device, :state]

  alias Nx.Tensor, as: T
  alias Nx.BinaryTensor, as: BT

  import Nx.Shared
  import Bitwise, only: [>>>: 2, &&&: 2]

  ## Creation

  def tensor(arg, type) do
    {shape, data} = flatten(arg, type)

    if data == "" do
      raise "cannot build empty tensor"
    end

    from_binary(%T{shape: shape, type: type}, data)
  end

  defp flatten(list, type) when is_list(list) do
    {dimensions, acc} = flatten_list(list, type, [], [])

    {dimensions |> Enum.reverse() |> List.to_tuple(),
     acc |> Enum.reverse() |> :erlang.list_to_binary()}
  end

  defp flatten(other, type), do: {{}, scalar_to_binary(other, type)}

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
    {[length(list) | dimensions], Enum.reduce(list, acc, &[scalar_to_binary(&1, type) | &2])}
  end

  def random_uniform(%{type: type, shape: shape} = out, min, max) do
    gen =
      case type do
        {:s, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {:u, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {_, _} -> fn -> (max - min) * :rand.uniform() + min end
      end

    data = for _ <- 1..Nx.Shape.size(shape), into: "", do: scalar_to_binary(gen.(), type)
    from_binary(out, data)
  end

  def random_normal(%{type: type, shape: shape} = out, mu, sigma) do
    data =
      for _ <- 1..Nx.Shape.size(shape),
          into: "",
          do: scalar_to_binary(:rand.normal(mu, sigma), type)

    from_binary(out, data)
  end

  def iota(%{shape: {n}, type: type} = out, 0) do
    data = for i <- 0..(n - 1), do: scalar_to_binary(i, type)
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
          do: scalar_to_binary(i, type)

    from_binary(out, data)
  end

  defp scalar_to_binary(value, type) do
    match_types([type], do: <<write!(value, 0)>>)
  end

  ## Device API

  def from_binary(t, binary) when is_binary(binary) do
    %{t | data: %BT{device: Nx.BinaryDevice, state: binary}}
  end

  def from_binary(t, other) do
    %{t | data: %BT{device: Nx.BinaryDevice, state: IO.iodata_to_binary(other)}}
  end

  def to_binary(%T{data: %{device: Nx.BinaryDevice, state: data}}), do: data

  def to_binary(%T{data: %{device: device}}) do
    raise ArgumentError,
          "cannot read Nx.Tensor data because the data is allocated on device #{inspect(device)}. " <>
            "Please use Nx.device_transfer/1 to transfer data back to Elixir"
  end

  def device_read(%T{data: %{device: device, state: state}} = tensor) do
    from_binary(tensor, device.read(state))
  end

  def device_deallocate(%T{data: %{device: device, state: state}}) do
    device.deallocate(state)
  end

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

  def reshape(out, tensor, _shape), do: from_binary(out, to_binary(tensor)) 
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
    |> :binary.copy(Nx.Shape.size(shape))
  end

  defp broadcast_data(%T{shape: old_shape, type: {_, size}} = t, new_shape, axes) do
    chunk_size = size * Nx.Shape.size(old_shape)

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
  def pad(_out, t, pad_value, padding_config) do
    pad_value = Nx.Util.to_scalar(pad_value)

    case t.shape do
      {} ->
        t

      {_} ->
        [{edge_low, edge_high}] = padding_config
        pad_last_dim(t, pad_value, edge_low, edge_high)

      _ ->
        permutation = for i <- 0..(Nx.rank(t) - 2), do: i
        permutation = [Nx.rank(t) - 1 | permutation]

        for {edge_low, edge_high} <- Enum.reverse(padding_config), reduce: t do
          acc -> Nx.transpose(pad_last_dim(acc, pad_value, edge_low, edge_high), permutation)
        end
    end
  end

  # Add padding to the high and low ends of the last dimension of a tensor
  defp pad_last_dim(%T{shape: shape, type: {_, size} = type} = t, value, edge_low, edge_high) do
    view = aggregate_axes(to_binary(t), [tuple_size(shape) - 1], shape, size)
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
            new_bytes = byte_size(bin) * div(size, 8) - high_byte - low_byte

            <<_::size(low_byte)-bitstring, new_bin::size(new_bytes)-bitstring, _::bitstring>> =
              bin

            new_bin

          edge_low < 0 and edge_high >= 0 ->
            low_byte = abs(edge_low) * size
            <<_::size(low_byte)-bitstring, new_bin::bitstring>> = bin
            <<new_bin::bitstring, edge_high_padding::bitstring>>

          edge_low >= 0 and edge_high < 0 ->
            high_byte = abs(edge_high) * size
            new_bytes = byte_size(bin) * div(size, 8) - high_byte
            <<new_bin::size(new_bytes)-bitstring, _::size(high_byte)-bitstring>> = bin
            <<edge_low_padding::bitstring, new_bin::bitstring>>

          true ->
            <<edge_low_padding::bitstring, bin::bitstring, edge_high_padding::bitstring>>
        end
      end

    from_binary(%{t | type: type, shape: new_shape}, data)
  end

  defp pad_in_dim(shape, dim, edge_low, edge_high) do
    dim = Nx.Shape.normalize_axis(shape, dim)
    dim_size = elem(shape, dim)
    new_dim = dim_size + edge_high + edge_low
    put_elem(shape, dim, new_dim)
  end

  ## Two-element

  def outer(out, %{type: left_type} = t1, %{type: right_type} = t2) do
    b1 = to_binary(t1)
    b2 = to_binary(t2)

    data =
      match_types [left_type, right_type] do
        for <<match!(left, 0) <- b1>>,
            <<match!(right, 1) <- b2>>,
            into: <<>>,
            do: scalar_to_binary(read!(left, 0) * read!(right, 1), out.type)
      end

    from_binary(out, data)
  end

  def dot(out, t1, axes1, t2, axes2) do
    zip_reduce(out, t1, axes1, t2, axes2, 0, fn {lhs, rhs}, acc ->
      res = lhs * rhs + acc
      {res, res}
    end)
  end

  ## Element wise ternary ops

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
      for i <- 0..(Nx.Shape.size(shape) - 1), into: <<>> do
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

        scalar_to_binary(result, type)
      end

    from_binary(out, data)
  end

  ## Element wise bin ops

  for fun <-
        [:add, :subtract, :multiply, :power, :remainder, :divide, :arctan2, :min, :max] ++
          [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
          [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] do
    capture = Macro.var(:"element_#{fun}", __MODULE__)

    def unquote(fun)(out, left, right) do
      element_wise_bin_op(out, left, right, &(unquote(capture) / 3))
    end
  end

  defp element_wise_bin_op(%{shape: shape, type: type} = out, left, right, fun) do
    %T{type: {_, left_size} = left_type} = left
    %T{type: {_, right_size} = right_type} = right

    count = Nx.Shape.size(shape)
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
    def unquote(name)(out, tensor) do
      element_wise_unary_op(out, tensor, fn x -> unquote(code) end)
    end
  end

  def count_leading_zeros(out, %{type: {_, size} = type} = tensor) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_clz(seg, size), 0)>>
        end
      end

    from_binary(out, data)
  end

  def population_count(out, %{type: {_, size} = type} = tensor) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_popcount(seg, 0), 0)>>
        end
      end

    from_binary(out, data)
  end

  def abs(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.abs/1)
  def bitwise_not(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.bnot/1)
  def ceil(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.ceil/1)
  def floor(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.floor/1)
  def negate(out, tensor), do: element_wise_unary_op(out, tensor, &-/1)
  def round(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.round/1)
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

  def inspect(tensor, opts) do
    dims = Tuple.to_list(tensor.shape)

    open = IA.color("[", :list, opts)
    sep = IA.color(",", :list, opts)
    close = IA.color("]", :list, opts)
    type = IA.color(Nx.Type.to_string(tensor.type), :atom, opts)
    shape = shape_to_algebra(dims, open, close)

    {data, _limit} =
      case tensor.data do
        %Nx.BinaryTensor{device: Nx.BinaryDevice, state: bin} ->
          {_, size} = tensor.type
          total_size = Enum.reduce(dims, size, &*/2)
          chunk(dims, bin, opts.limit, total_size, tensor.type, {open, sep, close})

        # TODO: To print data on device, we can support reading a slice
        # from the device which we will compute with:
        #
        #     min(opts.limit, Nx.Shape.size(shape)) * size
        #
        %Nx.BinaryTensor{device: device} ->
          {IA.to_doc(device, opts), opts.limit}
      end

    IA.concat([type, shape, IA.line(), data])
  end

  defp shape_to_algebra(dims, open, close) do
    dims
    |> Enum.map(fn number -> IA.concat([open, Integer.to_string(number), close]) end)
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

  ## Aggregation

  def sum(out, tensor, opts) do
    reduce(out, tensor, 0, opts, fn x, acc -> {x + acc, x + acc} end)
  end

  def argmin(out, tensor, opts) do
    comparator =
      case opts[:tie_break] do
        :high -> &<=/2
        :low -> &</2
      end

    argmin_or_max(out, tensor, comparator, opts[:axis])
  end

  def argmax(out, tensor, opts) do
    comparator =
      case opts[:tie_break] do
        :high -> &>=/2
        :low -> &>/2
      end

    argmin_or_max(out, tensor, comparator, opts[:axis])
  end

  defp argmin_or_max(out, tensor, comparator, axis) do
    opts = if axis, do: [axes: [axis]], else: []

    reduce(out, tensor, {0, :first, -1}, opts, fn x, {i, cur_extreme_x, cur_extreme_i} ->
      if comparator.(x, cur_extreme_x) or cur_extreme_x == :first do
        {i, {i + 1, x, i}}
      else
        {cur_extreme_i, {i + 1, cur_extreme_x, cur_extreme_i}}
      end
    end)
  end

  ## Reduce

  def reduce(out, tensor, acc, opts, fun) when is_list(opts) and is_function(fun, 2) do
    %T{type: {_, size} = type, shape: shape} = tensor

    view =
      if axes = opts[:axes] do
        aggregate_axes(to_binary(tensor), axes, shape, size)
      else
        [to_binary(tensor)]
      end

    data =
      for axis <- view do
        {result, _} =
          match_types [type] do
            for <<match!(var, 0) <- axis>>, reduce: {<<>>, acc} do
              {_, acc} -> fun.(read!(var, 0), acc)
            end
          end

        scalar_to_binary(result, out.type)
      end

    from_binary(out, data)
  end

  ## Zip reduce

  def zip_reduce(%{type: type} = out, t1, [], t2, [], acc, fun) do
    b1 = to_binary(t1)
    b2 = to_binary(t2)

    data =
      match_types [t1.type, t2.type] do
        for <<match!(left, 0) <- b1>>, <<match!(right, 1) <- b2>>, into: <<>> do
          {result, _} = fun.({read!(left, 0), read!(right, 1)}, acc)
          scalar_to_binary(result, type)
        end
      end

    from_binary(out, data)
  end

  def zip_reduce(%{type: type} = out, t1, [_ | _] = axes1, t2, [_ | _] = axes2, acc, fun)
      when is_function(fun, 2) do
    {_, s1} = left_type = t1.type
    {_, s2} = right_type = t2.type

    v1 = aggregate_axes(to_binary(t1), axes1, t1.shape, s1)
    v2 = aggregate_axes(to_binary(t2), axes2, t2.shape, s2)

    data =
      for b1 <- v1, b2 <- v2, into: <<>> do
        {bin, _acc} = bin_zip_reduce(b1, b2, left_type, right_type, <<>>, acc, fun)
        scalar_to_binary(bin, type)
      end

    from_binary(out, data)
  end

  # Helper for reducing down a single axis over two tensors,
  # returning tensor data and a final accumulator.
  defp bin_zip_reduce(<<>>, <<>>, _left_type, _right_type, bin, acc, _fun),
    do: {bin, acc}

  defp bin_zip_reduce(b1, b2, left_type, right_type, _bin, acc, fun) do
    {head1, rest1} =
      match_types [left_type] do
        <<match!(x, 0), rest1::binary>> = b1
        {read!(x, 0), rest1}
      end

    {head2, rest2} =
      match_types [right_type] do
        <<match!(y, 0), rest2::binary>> = b2
        {read!(y, 0), rest2}
      end

    {bin, acc} = fun.({head1, head2}, acc)
    bin_zip_reduce(rest1, rest2, left_type, right_type, bin, acc, fun)
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
end

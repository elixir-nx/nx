defmodule Nx.BinaryBackend do
  @moduledoc """
  An opaque backend written in pure Elixir that stores
  the data in Elixir's binaries.

  This is the default backend used by the `Nx` module.
  The backend itself (and its data) is private and must
  not be accessed directly.
  """

  @behaviour Nx.Backend

  @doc false
  defstruct [:state]

  alias Nx.Tensor, as: T
  alias Nx.BinaryBackend, as: B

  import Nx.Shared
  import Bitwise, only: [>>>: 2, &&&: 2]

  @default_eps 1.0e-10

  ## Creation

  @impl true
  def scalar(%{type: type, shape: shape} = out, scalar, _backend_options) do
    data = :binary.copy(number_to_binary(scalar, type), Nx.size(shape))
    from_binary(out, data)
  end

  @impl true
  def random_uniform(%{type: type, shape: shape} = out, min, max, _backend_options) do
    gen =
      case type do
        {:s, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {:u, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {_, _} -> fn -> (max - min) * :rand.uniform() + min end
      end

    data = for _ <- 1..Nx.size(shape), into: "", do: number_to_binary(gen.(), type)
    from_binary(out, data)
  end

  @impl true
  def random_normal(%{type: type, shape: shape} = out, mu, sigma, _backend_options) do
    data =
      for _ <- 1..Nx.size(shape),
          into: "",
          do: number_to_binary(:rand.normal(mu, sigma), type)

    from_binary(out, data)
  end

  @impl true
  def iota(%{shape: {}, type: type} = out, nil, _backend_options) do
    from_binary(out, number_to_binary(0, type))
  end

  def iota(%{shape: shape, type: type} = out, nil, backend_options) do
    t = iota(%T{type: type, shape: {Nx.size(shape)}, names: [nil]}, 0, backend_options)
    %{out | data: t.data}
  end

  def iota(%{shape: {n}, type: type} = out, 0, _backend_options) do
    data = for i <- 0..(n - 1), do: number_to_binary(i, type)
    from_binary(out, data)
  end

  def iota(%{shape: shape, type: type} = out, axis, _backend_options) do
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

  @impl true
  def eye(%{shape: {n, n}, type: type} = out, _backend_options) do
    one = number_to_binary(1, type)
    zero = number_to_binary(0, type)

    data =
      for i <- 1..n, j <- 1..n, into: <<>> do
        if i == j, do: one, else: zero
      end

    from_binary(out, data)
  end

  ## Conversions

  @impl true
  def from_binary(t, binary, _backend_options), do: from_binary(t, binary)

  defp from_binary(t, binary) when is_binary(binary), do: %{t | data: %B{state: binary}}
  defp from_binary(t, other), do: %{t | data: %B{state: IO.iodata_to_binary(other)}}

  @impl true
  def to_binary(%{type: {_, size}} = t, limit) do
    limit = limit * div(size, 8)
    binary = to_binary(t)

    if byte_size(binary) == limit do
      binary
    else
      binary_part(binary, 0, limit)
    end
  end

  defp to_binary(%T{data: %{state: data}}), do: data

  @impl true
  def backend_copy(tensor, backend, opts) do
    backend_transfer(tensor, backend, opts)
  end

  @impl true
  def backend_transfer(tensor, Nx.Tensor, _opts) do
    tensor
  end

  def backend_transfer(tensor, Nx.BinaryBackend, _opts) do
    tensor
  end

  def backend_transfer(tensor, backend, opts) do
    backend.from_binary(tensor, to_binary(tensor), opts)
  end

  @impl true
  def backend_deallocate(_tensor) do
    :ok
  end

  @impl true
  def to_batched_list(out, %{type: {_, size}} = tensor) do
    bitsize = Nx.size(out) * size

    for <<data::bitstring-size(bitsize) <- to_binary(tensor)>> do
      from_binary(out, data)
    end
  end

  defp to_scalar(n) when is_number(n), do: n
  defp to_scalar(t), do: binary_to_number(to_binary(t), t.type)

  ## Shape

  @impl true
  def reshape(out, tensor, _shape), do: from_binary(out, to_binary(tensor))

  @impl true
  def squeeze(out, tensor, _axes), do: from_binary(out, to_binary(tensor))

  ## Broadcast

  @impl true
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

  @impl true
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
  @impl true
  def pad(_out, t, pad_value, padding_config) do
    pad_value = to_scalar(pad_value)

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
            Nx.transpose(pad_last_dim(acc, pad_value, edge_low, edge_high, interior),
              axes: permutation
            )
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
        padded =
          for <<dim::size(size)-bitstring <- bin>>, into: <<>> do
            <<dim::size(size)-bitstring, interior_padding::bitstring>>
          end

        new_bytes = byte_size(padded) * 8 - interior_padding_size
        <<new_bin::size(new_bytes)-bitstring, _::bitstring>> = padded
        new_bin
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
    interior_padding_factor = (dim_size - 1) * interior
    new_dim = dim_size + interior_padding_factor + edge_high + edge_low
    put_elem(shape, dim, new_dim)
  end

  @impl true
  def reverse(out, %{type: {_, size}, shape: shape} = t, axes) do
    data = to_binary(t)
    weighted_shape = weighted_shape(shape, size)

    # Nx guaranteex axes is sorted and non-empty.
    min = List.first(axes)
    max = List.last(axes) + 1

    # The chunk size is computed based on all dimensions
    # before the minimum one being changed. For example,
    # for {0, 1, 2, 3} and the reverse is between 1 and 2,
    # the chunk_size will be d1 * d2 * d3 * size.
    chunk_size = weighted_chunk(weighted_shape, min, size)

    # All of the major dimensions not being reverse can be
    # read at once. For example, for {0, 1, 2, 3} and the reverse
    # is between 1 and 2, the read_size will be d3 * size.
    read_size = weighted_chunk(weighted_shape, max, size)

    # And now how we will traverse
    traverse =
      weighted_shape
      |> Enum.take(max)
      |> Enum.drop(min)
      |> reverse_traverse(min, axes)

    data =
      for <<chunk::size(chunk_size)-bitstring <- data>> do
        weighted_traverse(traverse, chunk, read_size)
      end

    from_binary(out, data)
  end

  defp reverse_traverse([head | tail], axis, axes) do
    if axis in axes do
      [&Enum.reverse/1, head | reverse_traverse(tail, axis + 1, axes)]
    else
      [head | reverse_traverse(tail, axis + 1, axes)]
    end
  end

  defp reverse_traverse([], _axis, _axes), do: []

  ## Two-element

  @impl true
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

  @impl true
  def dot(out, %{type: t1} = left, axes1, %{type: t2} = right, axes2) do
    bin_zip_reduce(out, left, axes1, right, axes2, 0, fn lhs, rhs, acc ->
      res = binary_to_number(lhs, t1) * binary_to_number(rhs, t2) + acc
      {res, res}
    end)
  end

  ## Element wise ternary ops

  @impl true
  def select(out, %{shape: {}} = pred, on_true, on_false) do
    if to_scalar(pred) == 0,
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
        [:add, :subtract, :multiply, :power, :remainder, :divide, :atan2, :min, :max, :quotient] ++
          [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
          [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
          [:logical_and, :logical_or, :logical_xor] do
    capture = Macro.var(:"element_#{fun}", __MODULE__)

    @impl true
    def unquote(fun)(out, left, right) do
      element_wise_bin_op(out, left, right, &(unquote(capture) / 3))
    end
  end

  defp element_wise_bin_op(%{type: type} = out, %{shape: {}} = left, right, fun) do
    scalar = to_scalar(left)

    data =
      match_types [right.type, type] do
        for <<match!(x, 0) <- to_binary(right)>>, into: <<>> do
          <<write!(fun.(type, scalar, read!(x, 0)), 1)>>
        end
      end

    from_binary(out, data)
  end

  defp element_wise_bin_op(%{type: type} = out, left, %{shape: {}} = right, fun) do
    scalar = to_scalar(right)

    data =
      match_types [left.type, type] do
        for <<match!(x, 0) <- to_binary(left)>>, into: <<>> do
          <<write!(fun.(type, read!(x, 0), scalar), 1)>>
        end
      end

    from_binary(out, data)
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
  defp element_quotient(_, a, b), do: div(a, b)
  defp element_atan2(_, a, b), do: :math.atan2(a, b)
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

  defp element_logical_and(_, l, _) when l == 0, do: 0
  defp element_logical_and(_, _, r) when r == 0, do: 0
  defp element_logical_and(_, _, _), do: 1

  defp element_logical_or(_, l, r) when l == 0 and r == 0, do: 0
  defp element_logical_or(_, _, _), do: 1

  defp element_logical_xor(_, l, r) when l == 0 and r == 0, do: 0
  defp element_logical_xor(_, l, _) when l == 0, do: 1
  defp element_logical_xor(_, _, r) when r == 0, do: 1
  defp element_logical_xor(_, _, _), do: 0

  ## Element wise unary ops

  for {name, {_desc, code}} <- Nx.Shared.unary_math_funs() do
    @impl true
    def unquote(name)(out, tensor) do
      element_wise_unary_op(out, tensor, fn x -> unquote(code) end)
    end
  end

  @impl true
  def count_leading_zeros(out, %{type: {_, size} = type} = tensor) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_clz(seg, size), 0)>>
        end
      end

    from_binary(out, data)
  end

  @impl true
  def population_count(out, %{type: {_, size} = type} = tensor) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_popcount(seg, 0), 0)>>
        end
      end

    from_binary(out, data)
  end

  @impl true
  def abs(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.abs/1)

  @impl true
  def bitwise_not(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.bnot/1)

  @impl true
  def ceil(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.ceil/1)

  @impl true
  def floor(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.floor/1)

  @impl true
  def negate(out, tensor), do: element_wise_unary_op(out, tensor, &-/1)

  @impl true
  def round(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.round/1)

  @impl true
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

  @impl true
  def inspect(tensor, inspect_opts) do
    limit = inspect_opts.limit
    binary = Nx.to_binary(tensor, if(limit == :infinity, do: [], else: [limit: limit + 1]))
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  ## Conv

  @impl true
  def conv(out, t, k, opts) do
    padding = opts[:padding]
    strides = opts[:strides]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    feature_groups = opts[:feature_group_size]
    batch_groups = opts[:batch_group_size]
    input_permutation = opts[:input_permutation]
    kernel_permutation = opts[:kernel_permutation]
    output_permutation = opts[:output_permutation]

    output_permutation =
      output_permutation
      |> Enum.with_index()
      |> Enum.sort()
      |> Enum.map(&elem(&1, 1))

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
    %T{type: {_, input_size} = input_type} = t = Nx.transpose(t, axes: input_permutation)
    %T{type: {_, kernel_size} = kernel_type} = k = Nx.transpose(k, axes: kernel_permutation)

    %{type: output_type, shape: output_shape, names: output_names} = out

    inverse_output_permutation =
      output_permutation |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))

    {untransposed_shape, untransposed_names} =
      Nx.Shape.transpose(output_shape, inverse_output_permutation, output_names)

    untransposed_out = %{out | names: untransposed_names, shape: untransposed_shape}

    # We need to dilate the spatial dimensions of the kernel first...
    dilation_padding = [
      {0, 0, 0},
      {0, 0, 0} | Enum.map(kernel_dilation, &{0, 0, &1 - 1})
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

    num_filters = elem(kernel_shape, 0)
    num_input_channels = elem(kernel_shape, 1)

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
    # TODO: Use Enum.zip_with on Elixir v1.12
    spatial_padding_config =
      padding
      |> Enum.zip(input_dilation)
      |> Enum.map(fn {{lo, hi}, dilation} -> {lo, hi, dilation - 1} end)

    padding_config = [
      {0, 0, 0},
      {0, 0, 0} | spatial_padding_config
    ]

    %T{shape: padded_shape} = padded_t = Nx.pad(t, 0, padding_config)

    single_data_dims = Tuple.delete_at(padded_shape, 0)
    batch_size = Nx.size(single_data_dims) * input_size

    # We will traverse the input tensor exactly the same as we traversed
    # the binary in reduce_window, but the window is equal to the filter
    # size of the kernel plus the channel size of the input tensor
    window_shape = Tuple.insert_at(filter_shape, 0, num_input_channels)

    input_data = to_binary(padded_t)
    kernel_data = to_binary(k)

    filter_groups_with_index =
      split_into_feature_groups(kernel_data, kernel_shape, kernel_type, feature_groups)

    batch_groups_with_index =
      split_into_batch_groups(input_data, padded_shape, input_type, batch_groups)

    # If we're not splitting across input channels, we're splitting across input batches
    # So we split the filters into groups to apply to the corresponding batch
    filter_groups_with_index =
      if batch_groups > 1,
        do: Enum.chunk_every(filter_groups_with_index, div(num_filters, batch_groups)),
        else: filter_groups_with_index

    batch_weighted_shape =
      weighted_shape(Tuple.delete_at(padded_shape, 0), input_size, window_shape)

    # We calculate our "anchors" using just the spatial dimensions
    # but they also need to consider the depth or channels of the input
    # tensor, so we always anchor on the `0th` channel
    padded_spatial_dims =
      padded_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    anchors = Enum.sort(make_anchors(padded_spatial_dims, strides, filter_shape))

    output_data =
      for {batch_group, batch_group_index} <- batch_groups_with_index, into: <<>> do
        current_filter_group_with_index =
          if batch_groups > 1,
            do: Enum.at(filter_groups_with_index, batch_group_index),
            else: filter_groups_with_index

        # Traverse the batch dim first
        for <<batch::size(batch_size)-bitstring <- batch_group>>,
            # Traverse the filters next, this allows us to rebuild
            # the resulting binary correctly
            {filter, feature_group} <- current_filter_group_with_index,
            # Then we traverse the spatial dimension, applying
            # the filter at each step
            anchor <- anchors,
            into: <<>> do
          offset =
            weighted_offset(batch_weighted_shape, [feature_group * num_input_channels | anchor])

          # The shape of the window is {channels} + filter_shape
          # The shape of the kernel is {num_filters, channels} + filter_shape
          window =
            batch_weighted_shape
            |> weighted_traverse(batch, input_size, offset)
            |> IO.iodata_to_binary()

          # The receptive field size of each binary in bytes
          input_field_size = Nx.size(filter_shape) * input_size
          filter_field_size = Nx.size(filter_shape) * kernel_size

          # For each channel in both filter and input...
          # The output from a single filter being applied over a window
          # of the input tensor is the sum of the element-wise products
          values =
            for i <- 0..(num_input_channels - 1) do
              current_input_pos = i * input_field_size
              current_filter_pos = i * filter_field_size
              <<_::size(current_input_pos)-bitstring, input_receptive_field::bitstring>> = window

              <<_::size(current_filter_pos)-bitstring, filter_receptive_field::bitstring>> =
                filter

              for j <- 0..(Nx.size(filter_shape) - 1) do
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
      end

    Nx.transpose(from_binary(untransposed_out, output_data), axes: output_permutation)
  end

  defp split_into_feature_groups(data, shape, {_, size}, groups) do
    group_size = div(Nx.size(shape) * size, groups)
    data_size = div(Nx.size(shape) * size, elem(shape, 0))

    for i <- 0..(groups - 1),
        offset = group_size * i,
        <<_::size(offset)-bitstring, data_group::size(group_size)-bitstring, _::bitstring>> =
          data,
        <<out_data::size(data_size)-bitstring <- data_group>>,
        do: {out_data, i}
  end

  defp split_into_batch_groups(data, shape, {_, size}, groups) do
    item_size = div(Nx.size(shape) * size, elem(shape, 0))
    num_groups = div(elem(shape, 0), groups)

    output_batch_groups =
      for i <- 0..(num_groups - 1),
          j <- 0..(groups - 1) do
        offset = (i + j * num_groups) * item_size
        <<_::size(offset)-bitstring, item::size(item_size)-bitstring, _::bitstring>> = data
        item
      end

    output_batch_groups |> Enum.with_index() |> Enum.map(fn {x, i} -> {x, rem(i, groups)} end)
  end

  @impl true
  def cholesky(%{type: output_type, shape: {rows, cols}} = out, tensor) do
    %T{type: {_, size} = input_type} = tensor

    row_chunk_size = size * cols
    data = to_binary(tensor)

    output_data =
      for i <- 0..(rows - 1), reduce: <<>> do
        cur_binary ->
          lower_triangle_part =
            for j <- 1..(i + 1), reduce: cur_binary do
              acc ->
                current_element_offset = i * row_chunk_size + (j - 1) * size
                current_element_opp_offset = (j - 1) * row_chunk_size + i * size
                diagonal_element_offset = (j - 1) * row_chunk_size + (j - 1) * size

                lhs_dot_offset = i * row_chunk_size
                lhs_dot_size = (j - 1) * size

                <<_::size(lhs_dot_offset)-bitstring, lhs::size(lhs_dot_size)-bitstring,
                  _::bitstring>> = acc

                rhs_dot_offset = (j - 1) * row_chunk_size
                rhs_dot_size = (j - 1) * size

                <<_::size(rhs_dot_offset)-bitstring, rhs::size(rhs_dot_size)-bitstring,
                  _::bitstring>> = acc

                elem =
                  match_types [input_type] do
                    <<_::size(current_element_offset)-bitstring, match!(x, 0), _::bitstring>> =
                      data

                    <<_::size(current_element_opp_offset)-bitstring, match!(y, 0), _::bitstring>> =
                      data

                    if x != y do
                      raise ArgumentError,
                            "matrix must be symmetric, a matrix is symmetric iff X = X.T"
                    end

                    fun = fn <<match!(left, 0)>>, <<match!(right, 0)>>, acc ->
                      {<<>>, read!(left, 0) * read!(right, 0) + acc}
                    end

                    {_, tmp_sum} = bin_zip_reduce_axis(lhs, rhs, size, size, <<>>, 0, fun)

                    if i == j - 1 do
                      value = :math.sqrt(Kernel.max(read!(x, 0) - tmp_sum, 0))
                      scalar_to_binary(value, output_type)
                    else
                      <<_::size(diagonal_element_offset)-bitstring, match!(diag, 0),
                        _::bitstring>> = acc

                      value = 1.0 / read!(diag, 0) * (read!(x, 0) - tmp_sum)
                      scalar_to_binary(value, output_type)
                    end
                  end

                <<acc::bitstring, elem::bitstring>>
            end

          col_zeros = IO.iodata_to_binary(List.duplicate(<<0::size(size)>>, cols - i - 1))

          <<lower_triangle_part::bitstring, col_zeros::bitstring>>
      end

    from_binary(out, output_data)
  end

  @impl true
  def qr(
        {%{shape: {m, k}, type: output_type} = q_holder,
         %{shape: {k, n}, type: output_type} = r_holder},
        %{type: {_, input_num_bits} = input_type, shape: input_shape} = tensor,
        opts
      ) do
    mode = opts[:mode]
    eps = opts[:eps] || @default_eps

    r_matrix =
      if mode == :reduced do
        # Since we want the first k rows of r, we can
        # just slice the binary by taking the first
        # n * k * output_type_num_bits bits from the binary.
        # Trick for r = tensor[[0..(k - 1), 0..(n - 1)]]
        slice_size = n * k * input_num_bits
        <<r_bin::bitstring-size(slice_size), _::bitstring>> = to_binary(tensor)
        binary_to_matrix(r_bin, input_type, {k, n})
      else
        tensor |> to_binary() |> binary_to_matrix(input_type, input_shape)
      end

    {q_matrix, r_matrix} =
      for i <- 0..(n - 1), reduce: {nil, r_matrix} do
        {q, r} ->
          # a = r[[i..(k - 1), i]]
          a = r |> Enum.slice(i..(k - 1)) |> Enum.map(fn row -> Enum.at(row, i) end)

          h = householder_reflector(a, k, eps)

          # If we haven't allocated Q yet, let Q = H1
          q =
            if is_nil(q) do
              zero_padding = 0 |> List.duplicate(k) |> List.duplicate(m - k)
              h ++ zero_padding
            else
              dot_matrix(q, h)
            end

          r = dot_matrix(h, r)

          {q, r}
      end

    q_bin = matrix_to_binary(q_matrix, output_type)
    r_bin = matrix_to_binary(r_matrix, output_type)

    q = from_binary(q_holder, q_bin)
    r = from_binary(r_holder, r_bin)

    {q, r}
  end

  @impl true
  def svd(
        {u_holder, %{type: output_type} = s_holder, v_holder},
        %{type: input_type, shape: input_shape} = tensor,
        opts
      ) do
    # This implementation is a mixture of concepts described in [1] and the
    # algorithmic descriptions found in [2], [3] and [4]
    #
    # [1] - Parallel One-Sided Block Jacobi SVD Algorithm: I. Analysis and Design,
    #       by Gabriel Oksa and Marian Vajtersic
    #       Source: https://www.cosy.sbg.ac.at/research/tr/2007-02_Oksa_Vajtersic.pdf
    # [2] - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/client/lib/svd.cc#L784
    # [3] - https://github.com/tensorflow/tensorflow/blob/dcdc6b2f9015829cde2c02b111c04b2852687efc/tensorflow/compiler/xla/client/lib/svd.cc#L386
    # [4] - http://drsfenner.org/blog/2016/03/householder-bidiagonalization/
    # [5] - http://www.mymathlib.com/c_source/matrices/linearsystems/singular_value_decomposition.c

    a = tensor |> to_binary() |> binary_to_matrix(input_type, input_shape)

    eps = opts[:eps] || @default_eps
    max_iter = opts[:max_iter] || 1000
    {u, d, v} = householder_bidiagonalization(a, input_shape, eps)

    {fro_norm, off_diag_norm} = get_frobenius_norm(d)

    {u, s_matrix, v, _, _} =
      Enum.reduce_while(1..max_iter, {u, d, v, off_diag_norm, fro_norm}, fn
        _, {u, d, v, off_diag_norm, fro_norm} ->
          eps = 1.0e-9 * fro_norm

          if off_diag_norm > eps do
            # Execute a round of jacobi rotations on u, d and v
            {u, d, v} = svd_jacobi_rotation_round(u, d, v, input_shape, eps)

            # calculate a posteriori norms for d, so the next iteration of Enum.reduce_while can decide to halt
            {fro_norm, off_diag_norm} = get_frobenius_norm(d)

            {:cont, {u, d, v, off_diag_norm, fro_norm}}
          else
            {:halt, {u, d, v, nil, nil}}
          end
      end)

    # Make s a vector
    s =
      s_matrix
      |> Enum.with_index()
      |> Enum.map(fn {row, idx} -> Enum.at(row, idx) end)
      |> Enum.reject(&is_nil/1)

    {s, v} = apply_singular_value_corrections(s, v)

    s_bin = matrix_to_binary(s, output_type)
    s = from_binary(s_holder, s_bin)

    u_bin = matrix_to_binary(u, output_type)
    v_bin = matrix_to_binary(v, output_type)

    u = from_binary(u_holder, u_bin)
    v = from_binary(v_holder, v_bin)

    {u, s, v}
  end

  @impl true
  def lu(
        {%{type: p_type} = p_holder, %{type: l_type} = l_holder, %{type: u_type} = u_holder},
        %{type: input_type, shape: {n, n} = input_shape} = tensor,
        opts \\ []
      ) do
    a = tensor |> to_binary() |> binary_to_matrix(input_type, input_shape)

    eps = opts[:eps] || @default_eps

    {p, a_prime} = lu_validate_and_pivot(a, n)

    # We'll work with linear indices because of the way each matrix
    # needs to be updated/accessed
    zeros_matrix = List.duplicate(List.duplicate(0, n), n)

    {l, u} =
      for j <- 0..(n - 1), reduce: {zeros_matrix, zeros_matrix} do
        {l, u} ->
          i_range = if j == 0, do: [0], else: 0..j
          [l_row_j] = get_matrix_rows(l, [j])
          l = replace_rows(l, [j], [replace_vector_element(l_row_j, j, 1.0)])

          u_col_j =
            for i <- i_range,
                reduce: get_matrix_column(u, j) do
              u_col_j ->
                l_row_slice = slice_matrix(l, [i, 0], [1, i])

                u_col_slice = slice_vector(u_col_j, 0, i)

                s = dot_matrix(l_row_slice, u_col_slice)
                [a_elem] = get_matrix_elements(a_prime, [[i, j]])
                replace_vector_element(u_col_j, i, a_elem - s)
            end

          u = replace_cols(u, [j], transpose_matrix(u_col_j))

          [u_jj] = get_matrix_elements(u, [[j, j]])

          l =
            if u_jj < eps do
              l
            else
              i_range = if j == n - 1, do: [n - 1], else: j..(n - 1)

              for i <- i_range, reduce: l do
                l ->
                  u_col_slice = slice_matrix(u, [0, j], [j, 1])
                  [l_row_i] = get_matrix_rows(l, [i])
                  s = dot_matrix(u_col_slice, l_row_i)
                  [a_elem] = get_matrix_elements(a_prime, [[i, j]])

                  l_updated_row = replace_vector_element(l_row_i, j, (a_elem - s) / u_jj)

                  replace_rows(l, [i], [l_updated_row])
              end
            end

          {l, u}
      end

    # Transpose because since P is orthogonal, inv(P) = tranpose(P)
    # and we want to return P such that A = P.L.U
    p_bin = p |> transpose_matrix() |> matrix_to_binary(p_type)
    l_bin = matrix_to_binary(l, l_type)
    u_bin = matrix_to_binary(u, u_type)

    {from_binary(p_holder, p_bin), from_binary(l_holder, l_bin), from_binary(u_holder, u_bin)}
  end

  defp lu_validate_and_pivot(a, n) do
    # pivots a tensor so that the biggest elements of each column lie on the diagonal.
    # if any of the diagonal elements ends up being 0, raises an ArgumentError

    identity =
      Enum.map(0..(n - 1), fn i -> Enum.map(0..(n - 1), fn j -> if i == j, do: 1, else: 0 end) end)

    # For each row, find the max value by column.
    # If its index (max_idx) is not in the diagonal (i.e. j != max_idx)
    # we need to swap rows j and max_idx in both the permutation matrix
    # and in the a matrix.
    Enum.reduce(0..(n - 2), {identity, a}, fn j, {p, a} ->
      [max_idx | _] =
        Enum.sort_by(j..(n - 1), fn i -> a |> Enum.at(i) |> Enum.at(j) |> abs() end, &>=/2)

      if max_idx == j do
        {p, a}
      else
        p_row = Enum.at(p, max_idx)
        p_j = Enum.at(p, j)
        p = p |> List.replace_at(max_idx, p_j) |> List.replace_at(j, p_row)

        a_row = Enum.at(a, max_idx)
        a_j = Enum.at(a, j)
        a = a |> List.replace_at(max_idx, a_j) |> List.replace_at(j, a_row)
        {p, a}
      end
    end)
  end

  defp svd_jacobi_rotation_round(u, d, v, {_, n}, eps) do
    for p <- 0..(n - 2), q <- (p + 1)..(n - 1), reduce: {u, d, v} do
      {u, d, v} ->
        # We need the values for d_bin at indices [p,p], [p,q], [q,p], [q,q], [[p,q], 0..n-1] for this first iteration
        d_rows_pq = get_matrix_rows(d, [p, q])

        [d_pp, d_pq, d_qp, d_qq] = get_matrix_elements(d, [[p, p], [p, q], [q, p], [q, q]])

        {rot_l, rot_r} = jacobi_rotators(d_pp, d_pq, d_qp, d_qq, eps)

        updated_d_rows_pq = rot_l |> transpose_matrix() |> dot_matrix(d_rows_pq)

        d = replace_rows(d, [p, q], updated_d_rows_pq)
        d_cols = get_matrix_columns(d, [p, q])

        updated_d_cols = dot_matrix(d_cols, rot_r)
        d = replace_cols(d, [p, q], updated_d_cols)

        rotated_u_cols = u |> get_matrix_columns([p, q]) |> dot_matrix(rot_l)
        u = replace_cols(u, [p, q], rotated_u_cols)

        rotated_v_cols = v |> get_matrix_columns([p, q]) |> dot_matrix(rot_r)
        v = replace_cols(v, [p, q], rotated_v_cols)

        {u, d, v}
    end
  end

  defp apply_singular_value_corrections(s, v) do
    # Due to convention, the singular values must be positive.
    # This function fixes any negative singular values.
    # It's important to note that since s left-multiplies v_transposed in
    # the SVD result. Since it also represents a diagonal matrix,
    # changing a sign in s implies a sign change in the
    # corresponding row of v_transposed.

    # This function also sorts singular values from highest to lowest,
    # as this can be convenient.

    # TODO: Use Enum.zip_with on Elixir v1.12
    s
    |> Enum.zip(transpose_matrix(v))
    |> Enum.map(fn
      {singular_value, row} when singular_value < 0 ->
        {-singular_value, Enum.map(row, &(&1 * -1))}

      {singular_value, row} ->
        {singular_value, row}
    end)
    |> Enum.sort_by(fn {s, _} -> s end, &>=/2)
    |> Enum.unzip()
  end

  defp householder_reflector(a, target_k, eps)

  defp householder_reflector([], target_k, _eps) do
    flat_list =
      for col <- 0..(target_k - 1), row <- 0..(target_k - 1), into: [] do
        if col == row, do: 1, else: 0
      end

    Enum.chunk_every(flat_list, target_k)
  end

  defp householder_reflector([a_0 | tail] = a, target_k, eps) do
    # This is a trick so we can both calculate the norm of a_reverse and extract the
    # head a the same time we reverse the array
    # receives a_reverse as a list of numbers and returns the reflector as a
    # k x k matrix

    norm_a_squared = Enum.reduce(a, 0, fn x, acc -> x * x + acc end)
    norm_a_sq_1on = norm_a_squared - a_0 * a_0

    {v, scale} =
      if norm_a_sq_1on < eps do
        {[1 | tail], 0}
      else
        v_0 =
          if a_0 <= 0 do
            a_0 - :math.sqrt(norm_a_squared)
          else
            -norm_a_sq_1on / (a_0 + :math.sqrt(norm_a_squared))
          end

        v_0_sq = v_0 * v_0
        scale = 2 * v_0_sq / (norm_a_sq_1on + v_0_sq)
        v = [1 | Enum.map(tail, &(&1 / v_0))]
        {v, scale}
      end

    prefix_threshold = target_k - length(v)
    v = List.duplicate(0, prefix_threshold) ++ v

    # dot(v, v) = norm_v_squared, which can be calculated from norm_a as:
    # norm_v_squared = norm_a_squared - a_0^2 + v_0^2

    # execute I - 2 / norm_v_squared * outer(v, v)
    {_, _, reflector_reversed} =
      for col_factor <- v, row_factor <- v, reduce: {0, 0, []} do
        {row, col, acc} ->
          # The current element in outer(v, v) is given by col_factor * row_factor
          # and the current I element is 1 when row == col
          identity_element = if row == col, do: 1, else: 0

          result =
            if row >= prefix_threshold and col >= prefix_threshold do
              identity_element -
                scale * col_factor * row_factor
            else
              identity_element
            end

          acc = [result | acc]

          if col + 1 == target_k do
            {row + 1, 0, acc}
          else
            {row, col + 1, acc}
          end
      end

    # This is equivalent to reflector_reversed |> Enum.reverse() |> Enum.chunk_every(target_k)
    {reflector, _, _} =
      for x <- reflector_reversed, reduce: {[], [], 0} do
        {result_acc, row_acc, col} ->
          row_acc = [x | row_acc]

          if col + 1 == target_k do
            {[row_acc | result_acc], [], 0}
          else
            {result_acc, row_acc, col + 1}
          end
      end

    reflector
  end

  defp householder_bidiagonalization(tensor, {m, n}, eps) do
    # For each column in `tensor`, apply
    # the current column's householder reflector from the left to `tensor`.
    # if the current column is not the penultimate, also apply
    # the corresponding row's householder reflector from the right

    for col <- 0..(n - 1), reduce: {nil, tensor, nil} do
      {ll, a, rr} ->
        # a[[col..m-1, col]] -> take `m - col` rows from the `col`-th column
        row_length = if m < col, do: 0, else: m - col
        a_col = a |> slice_matrix([col, col], [row_length, 1])

        l = householder_reflector(a_col, m, eps)

        ll =
          if is_nil(ll) do
            l
          else
            dot_matrix(ll, l)
          end

        a = dot_matrix(l, a)

        {a, rr} =
          if col <= n - 2 do
            # r = householder_reflector(a[[col, col+1:]], n)

            r =
              a
              |> slice_matrix([col, col + 1], [1, n - col])
              |> householder_reflector(n, eps)

            rr =
              if is_nil(rr) do
                r
              else
                dot_matrix(r, rr)
              end

            a = dot_matrix(a, r)
            {a, rr}
          else
            {a, rr}
          end

        {ll, a, rr}
    end
  end

  defp get_frobenius_norm(tensor) do
    # returns a tuple with {frobenius_norm, off_diagonal_norm}
    {fro_norm_sq, off_diag_norm_sq, _row_idx} =
      Enum.reduce(tensor, {0, 0, 0}, fn row, {fro_norm_sq, off_diag_norm_sq, row_idx} ->
        {fro_norm_sq, off_diag_norm_sq, _} =
          Enum.reduce(row, {fro_norm_sq, off_diag_norm_sq, 0}, fn x,
                                                                  {fro_norm_sq, off_diag_norm_sq,
                                                                   col_idx} ->
            if col_idx == row_idx do
              {fro_norm_sq + x * x, off_diag_norm_sq, col_idx + 1}
            else
              {fro_norm_sq + x * x, off_diag_norm_sq + x * x, col_idx + 1}
            end
          end)

        {fro_norm_sq, off_diag_norm_sq, row_idx + 1}
      end)

    {:math.sqrt(fro_norm_sq), :math.sqrt(off_diag_norm_sq)}
  end

  defp jacobi_rotators(pp, pq, qp, qq, eps) do
    t = pp + qq
    d = qp - pq

    {s, c} =
      if abs(d) < eps do
        {0, 1}
      else
        u = t / d
        den = :math.sqrt(1 + u * u)
        {-1 / den, u / den}
      end

    [[m00, m01], [_, m11]] = dot_matrix([[c, s], [-s, c]], [[pp, pq], [qp, qq]])
    {c_r, s_r} = make_jacobi(m00, m11, m01, eps)
    c_l = c_r * c - s_r * s
    s_l = c_r * s + s_r * c

    rot_l = [[c_l, s_l], [-s_l, c_l]]
    rot_r = [[c_r, s_r], [-s_r, c_r]]

    {rot_l, rot_r}
  end

  defp make_jacobi(pp, qq, pq, eps) do
    if abs(pq) <= eps do
      {1, 0}
    else
      tau = (qq - pp) / (2 * pq)

      t =
        if tau >= 0 do
          1 / (tau + :math.sqrt(1 + tau * tau))
        else
          -1 / (-tau + :math.sqrt(1 + tau * tau))
        end

      c = 1 / :math.sqrt(1 + t * t)
      {c, t * c}
    end
  end

  ## Aggregation

  @impl true
  def all?(out, %{type: type} = tensor, opts) do
    bin_reduce(out, tensor, 1, opts, fn bin, acc ->
      res = if binary_to_number(bin, type) != 0, do: acc, else: 0
      {res, res}
    end)
  end

  @impl true
  def any?(out, %{type: type} = tensor, opts) do
    bin_reduce(out, tensor, 0, opts, fn bin, acc ->
      res = if binary_to_number(bin, type) != 0, do: 1, else: acc
      {res, res}
    end)
  end

  @impl true
  def sum(out, %{type: type} = tensor, opts) do
    bin_reduce(out, tensor, 0, opts, fn bin, acc ->
      res = binary_to_number(bin, type) + acc
      {res, res}
    end)
  end

  @impl true
  def product(out, %{type: type} = tensor, opts) do
    bin_reduce(out, tensor, 1, opts, fn bin, acc ->
      res = binary_to_number(bin, type) * acc
      {res, res}
    end)
  end

  @impl true
  def reduce_max(out, %{type: type} = tensor, opts) do
    bin_reduce(out, tensor, :first, opts, fn bin, acc ->
      val = binary_to_number(bin, type)
      res = if acc == :first, do: val, else: Kernel.max(acc, val)
      {res, res}
    end)
  end

  @impl true
  def reduce_min(out, %{type: type} = tensor, opts) do
    bin_reduce(out, tensor, :first, opts, fn bin, acc ->
      val = binary_to_number(bin, type)
      res = if acc == :first, do: val, else: Kernel.min(acc, val)
      {res, res}
    end)
  end

  @impl true
  def argmin(out, tensor, opts) do
    comparator =
      case opts[:tie_break] do
        :high -> &<=/2
        :low -> &</2
      end

    argmin_or_max(out, tensor, comparator, opts[:axis])
  end

  @impl true
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

  @impl true
  def reduce(out, tensor, acc, opts, fun) do
    each = %{tensor | shape: {}}

    bin_reduce(out, tensor, acc, opts, fn bin, acc ->
      res = fun.(from_binary(each, bin), acc)
      {res, res}
    end)
  end

  @impl true
  def reduce_window(out, tensor, acc, window_dimensions, opts, fun) do
    padding_config = opts[:padding]
    strides = opts[:strides]
    dilations = opts[:window_dilations]
    acc = to_scalar(acc)

    %T{shape: padded_shape, type: {_, size} = type} =
      tensor = Nx.pad(tensor, acc, Enum.map(padding_config, &Tuple.append(&1, 0)))

    data = to_binary(tensor)
    weighted_shape = weighted_shape(padded_shape, size, window_dimensions, dilations)
    anchors = Enum.sort(make_anchors(padded_shape, strides, window_dimensions, dilations))

    data =
      for anchor <- anchors, into: <<>> do
        offset = weighted_offset(weighted_shape, anchor, dilations)
        window = IO.iodata_to_binary(weighted_traverse(weighted_shape, data, size, offset))

        match_types [type] do
          window_val =
            for <<match!(x, 0) <- window>>,
              reduce: acc,
              do: (acc -> fun.(read!(x, 0), acc))

          <<write!(window_val, 0)>>
        end
      end

    from_binary(out, data)
  end

  @impl true
  def window_sum(out, tensor, window_dimensions, opts) do
    fun = fn a, b -> a + b end
    reduce_window(out, tensor, 0, window_dimensions, opts, fun)
  end

  @impl true
  def window_max(out, tensor, window_dimensions, opts) do
    %{type: type} = out

    init_value =
      match_types [type] do
        <<match!(x, 0)>> = Nx.Type.min_value_binary(type)
        read!(x, 0)
      end

    fun = fn a, b -> max(a, b) end
    reduce_window(out, tensor, init_value, window_dimensions, opts, fun)
  end

  @impl true
  def window_min(out, tensor, window_dimensions, opts) do
    %{type: type} = out

    init_value =
      match_types [type] do
        <<match!(x, 0)>> = Nx.Type.max_value_binary(type)
        read!(x, 0)
      end

    fun = fn a, b -> min(a, b) end
    reduce_window(out, tensor, init_value, window_dimensions, opts, fun)
  end

  @impl true
  def window_product(out, tensor, window_dimensions, opts) do
    fun = fn a, b -> a * b end
    reduce_window(out, tensor, 1, window_dimensions, opts, fun)
  end

  @impl true
  def map(%{type: output_type} = out, %{type: type} = tensor, fun) do
    data = to_binary(tensor)

    output_data =
      match_types [type, output_type] do
        for <<match!(x, 0) <- data>>, into: <<>> do
          <<write!(to_scalar(fun.(read!(x, 0))), 1)>>
        end
      end

    from_binary(out, output_data)
  end

  @impl true
  def scatter_window_max(out, tensor, source, window_dimensions, opts, init_value) do
    select_and_scatter(out, tensor, source, &Nx.greater_equal/2, window_dimensions, opts, init_value, &Nx.add/2)
  end

  @impl true
  def scatter_window_min(out, tensor, source, window_dimensions, opts, init_value) do
    select_and_scatter(out, tensor, source, &Nx.less_equal/2, window_dimensions, opts, init_value, &Nx.add/2)
  end

  defp select_and_scatter(
        %{type: {_, output_size} = output_type, shape: output_shape} = out,
        t,
        source,
        select_fn,
        window_dimensions,
        opts,
        init_value,
        scatter_fn
      ) do
    padding = opts[:padding]
    strides = opts[:strides]

    init_value = to_scalar(init_value)

    %T{shape: padded_shape, type: {_, size} = type} =
      tensor = Nx.pad(t, init_value, Enum.map(padding, &Tuple.append(&1, 0)))

    input_data = to_binary(tensor)
    input_weighted_shape = weighted_shape(padded_shape, size, window_dimensions)
    input_anchors = Enum.sort(make_anchors(padded_shape, strides, window_dimensions))

    %T{type: {_, source_size} = source_type} = source
    source_data = to_binary(source)

    output_windows =
      for {anchor, i} <- Enum.with_index(input_anchors) do
        offset = weighted_offset(input_weighted_shape, anchor)

        window =
          IO.iodata_to_binary(weighted_traverse(input_weighted_shape, input_data, size, offset))

        # Get the index where `select_fn` is true
        match_types [type] do
          <<match!(first_elem, 0), _::bitstring>> = window

          {_, index, _} =
            for <<match!(x, 0) <- window>>, reduce: {0, 0, read!(first_elem, 0)} do
              {cur_index, selected_index, acc} ->
                if select_fn.(read!(x, 0), acc) == Nx.tensor(1, type: {:u, 8}) do
                  {cur_index + 1, cur_index, read!(x, 0)}
                else
                  {cur_index + 1, selected_index, acc}
                end
            end

          offset_from_anchor =
            flattened_index_to_offset(index, Tuple.to_list(window_dimensions), 0, [])

          absolute_index =
            anchor
            |> Enum.zip(offset_from_anchor)
            |> Enum.map(fn {x, y} -> x + y end)

          source_consumed = i * source_size

          <<_::size(source_consumed)-bitstring, from_source::size(source_size)-bitstring,
            _::bitstring>> = source_data

          source_value = binary_to_number(from_source, source_type)

          {source_value, absolute_index}
        end
      end

    output_weighted_shape = weighted_shape(output_shape, output_size)

    # Fold duplicate indices into one another using `scatter_fn` and sort
    # by absolute offset
    values_with_indices =
      output_windows
      |> Enum.group_by(&elem(&1, 1), &elem(&1, 0))
      |> Enum.map(fn {index, value} ->
        offset = weighted_offset(output_weighted_shape, index)
        {offset, Enum.reduce(value, init_value, scatter_fn)}
      end)
      |> Enum.sort_by(&elem(&1, 0))

    output_data =
      match_types [output_type] do
        {final_offset, output_data} =
          for {offset, value} <- values_with_indices, reduce: {0, <<>>} do
            {acc_offset, acc_binary} ->
              num_vals_before = div(offset - acc_offset, output_size)
              vals_before = List.duplicate(<<write!(init_value, 0)>>, num_vals_before)
              source_val = to_binary(value)
              new_binary = IO.iodata_to_binary([vals_before, source_val])

              {offset + output_size,
               <<acc_binary::size(acc_offset)-bitstring, new_binary::bitstring>>}
          end

        num_vals_left = div(output_size * Nx.size(output_shape) - final_offset, output_size)
        vals_left = IO.iodata_to_binary(List.duplicate(<<write!(init_value, 0)>>, num_vals_left))
        <<output_data::size(final_offset)-bitstring, vals_left::bitstring>>
      end

    from_binary(out, output_data)
  end

  defp flattened_index_to_offset(leftover, [dim | []], _, acc),
    do: Enum.reverse([rem(leftover, dim) | acc])

  defp flattened_index_to_offset(leftover, [dim | rest], n, acc) do
    if leftover < Nx.size(List.to_tuple(rest)) do
      flattened_index_to_offset(leftover, rest, 0, [n | acc])
    else
      flattened_index_to_offset(
        leftover - Nx.size(List.to_tuple(rest)),
        [dim | rest],
        n + 1,
        acc
      )
    end
  end

  @impl true
  def clip(out, tensor, min, max) do
    %{type: out_type} = out
    %T{type: in_type} = tensor
    %T{type: min_type} = min
    %T{type: max_type} = max

    data = to_binary(tensor)
    min = to_binary(min)
    max = to_binary(max)

    out_data =
      match_types [in_type, min_type, max_type, out_type] do
        for <<match!(x, 0) <- data>>, into: <<>> do
          <<match!(min_binary, 1)>> = min
          <<match!(max_binary, 2)>> = max
          value = min(max(read!(x, 0), read!(min_binary, 1)), read!(max_binary, 2))
          <<write!(value, 3)>>
        end
      end

    from_binary(out, out_data)
  end

  @impl true
  def slice(out, tensor, start_indices, _lengths, strides) do
    # If you think of a slice as drawing a bounding box in the dimensions
    # of the tensor, then it's clear we can simply use a weighted
    # traversal to construct the new tensor
    %T{type: {_, size}, shape: shape} = tensor
    %{shape: output_shape} = out

    if top_dimension_slice?(Nx.rank(shape), shape, output_shape) do
      length = Nx.size(output_shape) * div(size, 8)
      offset = div(length, elem(output_shape, 0)) * hd(start_indices)
      from_binary(out, binary_part(to_binary(tensor), offset, length))
    else
      # Anchored around the start indices
      weighted_shape = weighted_shape(shape, size, output_shape)
      offset = weighted_offset(weighted_shape, start_indices)

      # The weighted shape is altered such that we traverse
      # with respect to the stride for each dimension
      # TODO: Use Enum.zip_with on Elixir v1.12
      weighted_shape =
        weighted_shape
        |> Enum.zip(strides)
        |> Enum.map(fn {{d, dim_size}, s} -> {d, dim_size + (s - 1) * dim_size} end)

      input_data = to_binary(tensor)

      output_data =
        IO.iodata_to_binary(weighted_traverse(weighted_shape, input_data, size, offset))

      from_binary(out, output_data)
    end
  end

  defp top_dimension_slice?(1, _, _), do: true

  defp top_dimension_slice?(i, is, os)
       when :erlang.element(i, is) == :erlang.element(i, os),
       do: top_dimension_slice?(i - 1, is, os)

  defp top_dimension_slice?(_, _, _), do: false

  @impl true
  def concatenate(out, tensors, axis) do
    %{shape: output_shape, type: {_, size} = output_type} = out
    tensors = Enum.map(tensors, fn t -> as_type(%{t | type: output_type}, t) end)

    output_data =
      if axis == tuple_size(output_shape) - 1 do
        aggregate_axes =
          tensors
          |> Enum.map(fn %T{shape: shape} = t ->
            aggregate_axes(to_binary(t), [axis], shape, size)
          end)
          |> Enum.zip()

        for axis <- aggregate_axes, into: <<>> do
          IO.iodata_to_binary(Tuple.to_list(axis))
        end
      else
        input_data = Enum.map(tensors, &to_binary/1) |> IO.iodata_to_binary()
        output_weighted_shape = weighted_shape(output_shape, size)
        IO.iodata_to_binary(weighted_traverse(output_weighted_shape, input_data, size))
      end

    from_binary(out, output_data)
  end

  @impl true
  def as_type(out, tensor) do
    %{type: output_type} = out

    case tensor do
      %T{type: ^output_type} ->
        tensor

      %T{type: input_type} ->
        data = to_binary(tensor)

        output_data =
          match_types [input_type] do
            for <<match!(x, 0) <- data>>, into: <<>> do
              scalar_to_binary(read!(x, 0), output_type)
            end
          end

        from_binary(out, output_data)
    end
  end

  @impl true
  def bitcast(out, tensor), do: from_binary(out, to_binary(tensor))

  @impl true
  def sort(_out, t, opts) do
    %T{shape: shape, type: type} = t
    last_axis = Nx.rank(t) - 1

    comparator =
      case opts[:comparator] do
        :desc ->
          &</2

        :asc ->
          &>/2

        fun ->
          fn a, b ->
            a = binary_to_number(a, type)
            b = binary_to_number(b, type)
            to_scalar(fun.(a, b)) != 0
          end
      end

    axis = opts[:axis]

    case shape do
      {} ->
        t

      _ when axis == last_axis ->
        sort_last_dim(t, comparator)

      _ ->
        permutation = Nx.axes(t)

        permutation =
          permutation
          |> List.delete(axis)
          |> List.insert_at(Nx.rank(t) - 1, axis)

        inverse_permutation =
          permutation
          |> Enum.with_index()
          |> Enum.sort_by(fn {x, _} -> x end)
          |> Enum.map(fn {_, i} -> i end)

        t
        |> Nx.transpose(axes: permutation)
        |> sort_last_dim(comparator)
        |> Nx.transpose(axes: inverse_permutation)
    end
  end

  defp sort_last_dim(%T{shape: shape, type: {_, size} = type} = t, comparator) do
    view = aggregate_axes(to_binary(t), [tuple_size(shape) - 1], shape, size)

    new_data =
      for bin <- view, into: <<>> do
        data =
          match_types [type] do
            for <<x::size(size)-bitstring <- bin>> do
              x
            end
          end

        IO.iodata_to_binary(Enum.sort(data, comparator))
      end

    from_binary(t, new_data)
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

  # Makes anchors for traversing a binary with a window.
  defp make_anchors(shape, strides, window, dilations \\ 1)

  defp make_anchors(shape, strides, window, dilations)
       when is_tuple(shape) and is_tuple(window) and is_list(strides) do
    dilations =
      if is_integer(dilations),
        do: List.duplicate(dilations, tuple_size(shape)),
        else: dilations

    make_anchors(Tuple.to_list(shape), strides, Tuple.to_list(window), dilations, :init)
  end

  defp make_anchors([], [], _window, _dilations, anchors), do: anchors

  defp make_anchors([dim | shape], [s | strides], [w | window], [dil | dilation], :init) do
    dims = for i <- 0..(dim - 1), rem(i, s) == 0 and i + (w - 1) * dil < dim, do: [i]
    make_anchors(shape, strides, window, dilation, dims)
  end

  defp make_anchors([dim | shape], [s | strides], [w | window], [dil | dilation], anchors) do
    anchors =
      for i <- 0..(dim - 1),
          rem(i, s) == 0 and i + (w - 1) * dil < dim,
          anchor <- anchors,
          do: anchor ++ [i]

    make_anchors(shape, strides, window, dilation, anchors)
  end

  # Calculates the offset needed to reach a specified position
  # in the binary from a weighted shape list.
  defp weighted_offset(weighted_shape, pos, dilations \\ 1)

  defp weighted_offset(weighted_shape, pos, dilations) when is_list(pos) do
    dilations =
      if is_list(dilations),
        do: dilations,
        else: List.duplicate(dilations, length(weighted_shape))

    sum_weighted_offset(weighted_shape, pos, dilations)
  end

  defp sum_weighted_offset([], [], []), do: 0

  defp sum_weighted_offset([{s, size} | dims], [x | pos], [d | dilation]) do
    # Account for the dilation
    dilation_factor =
      if s == 1,
        do: 1,
        else: d

    div(size, dilation_factor) * x + weighted_offset(dims, pos, dilation)
  end

  ## Matrix (2-D array) manipulation

  defp dot_matrix([], _), do: 0
  defp dot_matrix(_, []), do: 0

  defp dot_matrix([h1 | _] = v1, [h2 | _] = v2) when not is_list(h1) and not is_list(h2) do
    v1
    |> Enum.zip(v2)
    |> Enum.reduce(0, fn {x, y}, acc -> x * y + acc end)
  end

  defp dot_matrix(m1, m2) do
    # matrix multiplication which works on 2-D lists
    Enum.map(m1, fn row ->
      m2
      |> transpose_matrix()
      |> Enum.map(fn col ->
        row
        |> Enum.zip(col)
        |> Enum.reduce(0, fn {x, y}, acc -> acc + x * y end)
      end)
    end)
  end

  defp transpose_matrix([x | _] = m) when not is_list(x) do
    Enum.map(m, &[&1])
  end

  defp transpose_matrix(m) do
    m
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end

  defp matrix_to_binary(m, type) do
    m
    |> Enum.map(fn
      row when is_list(row) ->
        Enum.map(row, fn x ->
          scalar_to_binary(x, type)
        end)

      x ->
        scalar_to_binary(x, type)
    end)
    |> IO.iodata_to_binary()
  end

  defp binary_to_matrix(bin, type, {_, num_cols}) do
    match_types [type] do
      flat_list =
        for <<match!(x, 0) <- bin>>, into: [] do
          read!(x, 0)
        end

      Enum.chunk_every(flat_list, num_cols)
    end
  end

  defp slice_vector(a, start, length), do: Enum.slice(a, start, length)

  defp slice_matrix(a, [row_start, col_start], [row_length, col_length]) do
    a
    |> Enum.slice(row_start, row_length)
    |> Enum.flat_map(&Enum.slice(&1, col_start, col_length))
  end

  defp get_matrix_column(m, col) do
    Enum.map(m, fn row ->
      Enum.at(row, col)
    end)
  end

  defp get_matrix_columns(m, columns) do
    Enum.map(m, fn row ->
      for {item, i} <- Enum.with_index(row), i in columns, do: item
    end)
  end

  defp get_matrix_rows(m, rows) do
    for {row, i} <- Enum.with_index(m), i in rows, do: row
  end

  defp get_matrix_elements(m, row_col_pairs) do
    Enum.map(row_col_pairs, fn [row, col] ->
      m
      |> Enum.at(row, [])
      |> Enum.at(col)
      |> case do
        nil -> raise ArgumentError, "invalid index [#{row},#{col}] for matrix"
        item -> item
      end
    end)
  end

  defp replace_rows(m, rows, values) do
    rows
    |> Enum.zip(values)
    |> Enum.reduce(m, fn {idx, row}, m -> List.replace_at(m, idx, row) end)
  end

  defp replace_cols(m, [], _), do: m

  defp replace_cols(m, cols, values) do
    m
    |> transpose_matrix()
    |> replace_rows(cols, transpose_matrix(values))
    |> transpose_matrix()
  end

  defp replace_vector_element(m, row, value), do: List.replace_at(m, row, value)
end

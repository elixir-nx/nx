defmodule Nx.BinaryTensor do
  @moduledoc false

  # TODO: Remove me
  import Nx.Shared

  alias Nx.Tensor, as: T
  import Bitwise, only: [>>>: 2, &&&: 2]

  ## Reflection

  @unary_funs [
    exp: {"exponential", quote(do: :math.exp(var!(x)))},
    expm1: {"exponential minus one", quote(do: :math.exp(var!(x)) - 1)},
    log: {"natural log", quote(do: :math.log(var!(x)))},
    log1p: {"natural log plus one", quote(do: :math.log(var!(x) + 1))},
    logistic: {"standard logistic (a sigmoid)", quote(do: 1 / (1 + :math.exp(-var!(x))))},
    cos: {"cosine", quote(do: :math.cos(var!(x)))},
    sin: {"sine", quote(do: :math.sin(var!(x)))},
    tanh: {"hyperbolic tangent", quote(do: :math.tanh(var!(x)))},
    sqrt: {"square root", quote(do: :math.sqrt(var!(x)))},
    rsqrt: {"reverse square root", quote(do: 1 / :math.sqrt(var!(x)))},
    cbrt: {"cube root", quote(do: :math.pow(var!(x), 1 / 3))},
  ]

  @doc false
  def unary_funs, do: @unary_funs

  ## Broadcast

  @doc false
  def broadcast(t, %{shape: shape} = out, axes) do
    t
    |> broadcast_data(shape, axes)
    |> data(out)
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

  def transpose(%T{shape: shape, type: {_, size}} = t, out, axes) do
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

    data(data, out)
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

  # TODO: Use out
  def pad(t, _out, pad_value, padding_config) do
    pad_value = Nx.Util.to_scalar(pad_value)

    case t.shape do
      {} ->
        t

      {_} ->
        [{edge_low, edge_high}] = padding_config
        Nx.Util.pad_last_dim(t, pad_value, edge_low, edge_high)

      _ ->
        permutation = for i <- 0..(Nx.rank(t) - 2), do: i
        permutation = [Nx.rank(t) - 1 | permutation]

        for {edge_low, edge_high} <- Enum.reverse(padding_config), reduce: t do
          acc ->
            Nx.transpose(Nx.Util.pad_last_dim(acc, pad_value, edge_low, edge_high), permutation)
        end
    end
  end

  ## Two-element

  def outer(%{type: left_type} = t1, %{type: right_type} = t2, out) do
    b1 = to_binary(t1)
    b2 = to_binary(t2)

    data =
      match_types [left_type, right_type] do
        for <<match!(left, 0) <- b1>>,
            <<match!(right, 1) <- b2>>,
            into: <<>>,
            do: scalar_to_binary(read!(left, 0) * read!(right, 1), out.type)
      end

    data(data, out)
  end

  # TODO: Use out
  def dot(t1, axes1, t2, axes2, _out) do
    Nx.Util.zip_reduce(t1, axes1, t2, axes2, 0, fn {lhs, rhs}, acc ->
      res = lhs * rhs + acc
      {res, res}
    end)
  end

  ## Element wise ternary ops

  def select(%{shape: {}} = pred, on_true, on_false, out) do
    if Nx.Util.to_scalar(pred) == 0,
      do: broadcast_data(on_false, out.shape) |> data(out),
      else: broadcast_data(on_true, out.shape) |> data(out)
  end

  def select(pred, on_true, on_false, %{shape: shape, type: type} = out) do
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

    data(data, out)
  end

  ## Element wise bin ops

  for fun <-
        [:add, :subtract, :multiply, :power, :remainder, :divide, :arctan2, :min, :max] ++
          [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
          [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] do
    capture = Macro.var(:"element_#{fun}", __MODULE__)

    def unquote(fun)(left, right, out) do
      element_wise_bin_op(left, right, out, &(unquote(capture) / 3))
    end
  end

  defp element_wise_bin_op(left, right, %{shape: shape, type: type} = out, fun) do
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

    data(data, out)
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

  for {name, {_desc, code}} <- @unary_funs do
    def unquote(name)(tensor, out) do
      element_wise_unary_op(tensor, out, fn x -> unquote(code) end)
    end
  end

  def count_leading_zeros(%{type: {_, size} = type} = tensor, out) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_clz(seg, size), 0)>>
        end
      end

    data(data, out)
  end

  def population_count(%{type: {_, size} = type} = tensor, out) do
    data =
      for <<seg::unsigned-size(size)-native <- to_binary(tensor)>>, into: <<>> do
        match_types [type] do
          <<write!(element_popcount(seg, 0), 0)>>
        end
      end

    data(data, out)
  end

  def abs(tensor, out), do: element_wise_unary_op(tensor, out, &:erlang.abs/1)
  def bitwise_not(tensor, out), do: element_wise_unary_op(tensor, out, &:erlang.bnot/1)
  def ceil(tensor, out), do: element_wise_unary_op(tensor, out, &:erlang.ceil/1)
  def floor(tensor, out), do: element_wise_unary_op(tensor, out, &:erlang.floor/1)
  def negate(tensor, out), do: element_wise_unary_op(tensor, out, &-/1)
  def round(tensor, out), do: element_wise_unary_op(tensor, out, &:erlang.round/1)
  def sign(tensor, out), do: element_wise_unary_op(tensor, out, &element_sign/1)

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

  defp element_wise_unary_op(tensor, out, fun) do
    data =
      match_types [tensor.type, out.type] do
        for <<match!(seg, 0) <- to_binary(tensor)>>, into: <<>> do
          <<write!(fun.(read!(seg, 0)), 1)>>
        end
      end

    data(data, out)
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

  ## Aggregation

  # TODO: Use out
  def sum(tensor, out, opts) do
    opts = [type: out.type] ++ opts
    {tensor, _} = Nx.Util.reduce(tensor, 0, opts, fn x, acc -> {x + acc, x + acc} end)
    tensor
  end

  def argmin(tensor, out, opts) do
    comparator =
      case opts[:tie_break] do
        :high -> &<=/2
        :low -> &</2
      end

    argmin_or_max(tensor, out, comparator, opts[:axis])
  end

  def argmax(tensor, out, opts) do
    comparator =
      case opts[:tie_break] do
        :high -> &>=/2
        :low -> &>/2
      end

    argmin_or_max(tensor, out, comparator, opts[:axis])
  end

  # TODO: out
  defp argmin_or_max(tensor, out, comparator, axis) do
    axes = if axis, do: [axis], else: nil
    opts = [axes: axes, type: out.type]

    {tensor, _accs} =
      Nx.Util.reduce(tensor, {0, :first, -1}, opts, fn x, {i, cur_extreme_x, cur_extreme_i} ->
        if comparator.(x, cur_extreme_x) or cur_extreme_x == :first do
          {i, {i + 1, x, i}}
        else
          {cur_extreme_i, {i + 1, cur_extreme_x, cur_extreme_i}}
        end
      end)

    tensor
  end

  ## Conversions

  def to_binary(%T{data: {Nx.BitStringDevice, data}}), do: data

  def to_binary(%T{data: {device, _data}}) do
    raise ArgumentError,
          "cannot read Nx.Tensor data because the data is allocated on device #{inspect(device)}. " <>
            "Please use Nx.device_transfer/1 to transfer data back to Elixir"
  end

  ## Helpers

  defp data(binary, t) when is_binary(binary), do: %{t | data: {Nx.BitStringDevice, binary}}
  defp data(other, t), do: %{t | data: {Nx.BitStringDevice, IO.iodata_to_binary(other)}}
end

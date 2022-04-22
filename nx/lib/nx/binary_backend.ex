defmodule Nx.BinaryBackend do
  @moduledoc """
  An opaque backend written in pure Elixir that stores
  the data in Elixir's binaries.

  This is the default backend used by the `Nx` module.
  The backend itself (and its data) is private and must
  not be accessed directly.
  """
  use Complex.Kernel

  @behaviour Nx.Backend

  @doc false
  defstruct [:state]

  alias Nx.Tensor, as: T
  alias Nx.BinaryBackend, as: B

  import Nx.Shared
  import Bitwise, only: [>>>: 2, &&&: 2]

  ## Creation

  @impl true
  def constant(%{type: type, shape: shape} = out, constant, _backend_options) do
    data = :binary.copy(number_to_binary(constant, type), Nx.size(shape))
    from_binary(out, data)
  end

  @impl true
  def random_uniform(%{type: type, shape: shape} = out, min, max, _backend_options) do
    min = scalar_to_number(min)
    max = scalar_to_number(max)

    gen_float = fn -> (max - min) * :rand.uniform() + min end

    gen =
      case type do
        {:s, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {:u, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {:c, _} -> fn -> Complex.new(gen_float.(), gen_float.()) end
        {_, _} -> gen_float
      end

    data = for _ <- 1..Nx.size(shape), into: "", do: number_to_binary(gen.(), type)
    from_binary(out, data)
  end

  @impl true
  def random_normal(%{type: type, shape: shape} = out, mu, sigma, _backend_options) do
    mu = scalar_to_number(mu)
    sigma = scalar_to_number(sigma)

    gen =
      case type do
        {:c, _} ->
          fn ->
            Complex.new(:rand.normal(mu, sigma), :rand.normal(mu, sigma))
          end

        _ ->
          fn -> :rand.normal(mu, sigma) end
      end

    data =
      for _ <- 1..Nx.size(shape),
          into: "",
          do: number_to_binary(gen.(), type)

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
  def to_binary(%{type: {_backend_options, size}} = t, limit) do
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
  def to_batched_list(out, %{type: {_, size}} = tensor, opts) do
    leftover = opts[:leftover]

    input_size = Nx.size(tensor)

    batch_size = Nx.size(out)
    batch_bitsize = batch_size * size

    remainder = rem(input_size, batch_size)

    to_add = if remainder != 0, do: batch_size - remainder, else: 0

    tensor_bin =
      case to_binary(tensor) do
        bin when leftover == :repeat and to_add != 0 ->
          diff = to_add * size
          <<wrapped::size(diff)-bitstring, _::bitstring>> = bin
          bin <> wrapped

        bin ->
          bin
      end

    for <<batch::size(batch_bitsize)-bitstring <- tensor_bin>> do
      from_binary(out, batch)
    end
  end

  ## Shape

  @impl true
  def reshape(out, tensor), do: from_binary(out, to_binary(tensor))

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
  def pad(out, t, pad_value, padding_config) do
    pad_value = to_binary(as_type(out, pad_value))

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

    edge_high_padding =
      if edge_high <= 0,
        do: <<>>,
        else: for(_ <- 1..edge_high, into: <<>>, do: value)

    edge_low_padding =
      if edge_low <= 0,
        do: <<>>,
        else: for(_ <- 1..edge_low, into: <<>>, do: value)

    interior_padding =
      if interior == 0,
        do: <<>>,
        else: for(_ <- 1..interior, into: <<>>, do: value)

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
  def dot(out, left, contract_axes1, [], right, contract_axes2, []) do
    # dot/4 is directed to this specific clause so we can keep a more efficient implementation
    # for non-batched dot products. See the clause below for batched dot products
    data = bin_dot(left, contract_axes1, right, contract_axes2, out.type)
    from_binary(out, data)
  end

  def dot(
        out,
        %{shape: left_shape, type: {_, left_size}, names: left_names} = left,
        left_contract_axes,
        left_batch_axes,
        %{shape: right_shape, type: {_, right_size}, names: right_names} = right,
        right_contract_axes,
        right_batch_axes
      ) do
    left_binary = to_binary(left)
    right_binary = to_binary(right)

    left_batch_contract_axes =
      Enum.map(left_contract_axes, fn axis -> axis - length(left_batch_axes) end)

    right_batch_contract_axes =
      Enum.map(right_contract_axes, fn axis -> axis - length(right_batch_axes) end)

    {left_batch_shape, _left_batch_names} =
      Nx.Shape.contract(left_shape, left_batch_axes, left_names, false)

    {right_batch_shape, _right_batch_names} =
      Nx.Shape.contract(right_shape, right_batch_axes, right_names, false)

    left_batch_item_length = Nx.size(left_batch_shape)
    right_batch_item_length = Nx.size(right_batch_shape)

    batch_count = Enum.reduce(left_batch_axes, 1, fn x, acc -> elem(left_shape, x) * acc end)

    range = if batch_count == 0, do: [], else: 0..(batch_count - 1)

    left_batch_item_template = %{left | shape: left_batch_shape}
    right_batch_item_template = %{right | shape: right_batch_shape}

    bin_result =
      for index <- range do
        left_offset = index * left_batch_item_length
        right_offset = index * right_batch_item_length

        left_offset_bits = left_offset * left_size
        right_offset_bits = right_offset * right_size

        left_batch_item_bits = left_batch_item_length * left_size
        right_batch_item_bits = right_batch_item_length * right_size

        <<_::bitstring-size(left_offset_bits),
          left_batch_item_binary::bitstring-size(left_batch_item_bits),
          _::bitstring>> = left_binary

        <<_::bitstring-size(right_offset_bits),
          right_batch_item_binary::bitstring-size(right_batch_item_bits),
          _::bitstring>> = right_binary

        bin_dot(
          from_binary(left_batch_item_template, left_batch_item_binary),
          left_batch_contract_axes,
          from_binary(right_batch_item_template, right_batch_item_binary),
          right_batch_contract_axes,
          out.type
        )
      end

    from_binary(out, bin_result)
  end

  defp bin_dot(%{type: t1} = left, contract_axes1, %{type: t2} = right, contract_axes2, type) do
    {left, left_contract_axes} = bin_dot_transpose_contract_axes(left, contract_axes1)

    {right, right_contract_axes} = bin_dot_transpose_contract_axes(right, contract_axes2)

    bin_zip_reduce(left, left_contract_axes, right, right_contract_axes, type, 0, fn
      lhs, rhs, acc ->
        res = binary_to_number(lhs, t1) * binary_to_number(rhs, t2) + acc
        {res, res}
    end)
  end

  defp bin_dot_transpose_contract_axes(tensor, contract_axes) do
    # The intution here is that we can pre-condense the contracting axes into a
    # single dimension, which will then be contracted through bin_zip_reduce below.
    # This takes a shape {a, m, n, b} which contracts on m, n and turns it into
    # {a, b, m * n}, contracting on the last dimension. This is necessary because
    # bin_zip_reduce and aggregate_axes are order independent but dot depends
    # on the axes order.

    axes = Nx.axes(tensor)

    remaining_axes =
      contract_axes
      |> Enum.sort(:desc)
      |> Enum.reduce(axes, &List.delete_at(&2, &1))

    transpose_axes = remaining_axes ++ contract_axes

    transposed =
      if transpose_axes == axes do
        tensor
      else
        {shape, names} = Nx.Shape.transpose(tensor.shape, transpose_axes, tensor.names)
        transpose(%{tensor | shape: shape, names: names}, tensor, transpose_axes)
      end

    {kept, contracted} =
      transposed.shape
      |> Tuple.to_list()
      |> Enum.split(length(remaining_axes))

    kept_shape = List.to_tuple(kept)

    kept_size = tuple_size(kept_shape)

    reduced_shape = Tuple.insert_at(kept_shape, kept_size, Enum.product(contracted))

    {%{transposed | shape: reduced_shape, names: List.duplicate(nil, tuple_size(reduced_shape))},
     [kept_size]}
  end

  ## Element wise ternary ops

  @impl true
  def select(out, %{shape: {}} = pred, on_true, on_false) do
    if scalar_to_number(pred) == 0,
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
    number = scalar_to_number(left)

    data =
      match_types [right.type, type] do
        for <<match!(x, 0) <- to_binary(right)>>, into: <<>> do
          <<write!(fun.(type, number, read!(x, 0)), 1)>>
        end
      end

    from_binary(out, data)
  end

  defp element_wise_bin_op(%{type: type} = out, left, %{shape: {}} = right, fun) do
    number = scalar_to_number(right)

    data =
      match_types [left.type, type] do
        for <<match!(x, 0) <- to_binary(left)>>, into: <<>> do
          <<write!(fun.(type, read!(x, 0), number), 1)>>
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

  defp element_add(_, a, b), do: Complex.add(a, b)
  defp element_subtract(_, a, b), do: Complex.subtract(a, b)
  defp element_multiply(_, a, b), do: Complex.multiply(a, b)
  defp element_divide(_, a, b), do: Complex.divide(a, b)
  defp element_quotient(_, a, b), do: div(a, b)

  defp element_remainder(_, a, b) when is_integer(a) and is_integer(b), do: rem(a, b)
  defp element_remainder(_, a, b), do: :math.fmod(a, b)

  defp element_atan2(_, a, b), do: Complex.atan2(a, b)
  defp element_max(_, a, b), do: max(a, b)
  defp element_min(_, a, b), do: min(a, b)

  defp element_power({type, _}, a, b) when type in [:s, :u], do: Integer.pow(a, b)
  defp element_power(_, a, b), do: Complex.power(a, b)

  defp element_bitwise_and(_, a, b), do: :erlang.band(a, b)
  defp element_bitwise_or(_, a, b), do: :erlang.bor(a, b)
  defp element_bitwise_xor(_, a, b), do: :erlang.bxor(a, b)

  defp element_left_shift(_, a, b) when is_number(b) and b >= 0,
    do: :erlang.bsl(a, b)

  defp element_left_shift(_, _, b), do: raise(ArgumentError, "cannot left shift by #{b}")

  defp element_right_shift(_, a, b) when is_number(b) and b >= 0,
    do: :erlang.bsr(a, b)

  defp element_right_shift(_, _, b), do: raise(ArgumentError, "cannot right shift by #{b}")

  defp element_equal(_, a, b), do: boolean_as_number(a == b)
  defp element_not_equal(_, a, b), do: boolean_as_number(a != b)
  defp element_logical_and(_, a, b), do: boolean_as_number(as_boolean(a) and as_boolean(b))
  defp element_logical_or(_, a, b), do: boolean_as_number(as_boolean(a) or as_boolean(b))
  defp element_logical_xor(_, a, b), do: boolean_as_number(as_boolean(a) != as_boolean(b))

  defp element_greater(_, a, b) when is_number(a) and is_number(b), do: boolean_as_number(a > b)

  defp element_greater(_, a, b) do
    %Complex{re: a_re, im: a_im} = as_complex(a)
    %Complex{re: b_re, im: b_im} = as_complex(b)
    boolean_as_number(a_re > b_re and a_im > b_im)
  end

  defp element_less(_, a, b) when is_number(a) and is_number(b), do: boolean_as_number(a < b)

  defp element_less(_, a, b) do
    %Complex{re: a_re, im: a_im} = as_complex(a)
    %Complex{re: b_re, im: b_im} = as_complex(b)

    boolean_as_number(a_re < b_re and a_im < b_im)
  end

  defp element_greater_equal(_, a, b) when is_number(a) and is_number(b),
    do: boolean_as_number(a >= b)

  defp element_greater_equal(_, a, b) do
    %{re: a_re, im: a_im} = as_complex(a)
    %{re: b_re, im: b_im} = as_complex(b)

    boolean_as_number(a_re >= b_re and a_im >= b_im)
  end

  defp element_less_equal(_, a, b) when is_number(a) and is_number(b),
    do: boolean_as_number(a <= b)

  defp element_less_equal(_, a, b) do
    %{re: a_re, im: a_im} = as_complex(a)
    %{re: b_re, im: b_im} = as_complex(b)

    boolean_as_number(a_re <= b_re and a_im <= b_im)
  end

  defp as_boolean(n) when n == 0, do: false
  defp as_boolean(%Complex{re: re, im: im}) when re == 0 and im == 0, do: false
  defp as_boolean(_), do: true

  defp boolean_as_number(true), do: 1
  defp boolean_as_number(false), do: 0

  defp as_complex(%Complex{} = z), do: z
  defp as_complex(n), do: Complex.new(n)

  ## Element wise unary ops

  for {name, {_desc, code, _formula}} <- Nx.Shared.unary_math_funs() do
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
  def abs(out, tensor), do: element_wise_unary_op(out, tensor, &Complex.abs/1)

  @impl true
  def conjugate(out, tensor), do: element_wise_unary_op(out, tensor, &Complex.conjugate/1)

  @impl true
  def real(%{type: {_, component_size}} = out, %{type: {:c, _}} = tensor) do
    data = to_binary(tensor)

    result =
      for <<real::bitstring-size(component_size), _::bitstring-size(component_size) <- data>>,
        into: <<>>,
        do: real

    from_binary(out, result)
  end

  @impl true
  def imag(%{type: {_, component_size}} = out, %{type: {:c, _}} = tensor) do
    data = to_binary(tensor)

    result =
      for <<_::bitstring-size(component_size), imag::bitstring-size(component_size) <- data>>,
        into: <<>>,
        do: imag

    from_binary(out, result)
  end

  @impl true
  def bitwise_not(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.bnot/1)

  @impl true
  def ceil(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.ceil/1)

  @impl true
  def floor(out, tensor), do: element_wise_unary_op(out, tensor, &:erlang.floor/1)

  @impl true
  def negate(out, tensor), do: element_wise_unary_op(out, tensor, &Complex.negate/1)

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
    spatial_padding_config =
      Enum.zip_with(padding, input_dilation, fn {lo, hi}, dilation ->
        {lo, hi, dilation - 1}
      end)

    padding_config = [
      {0, 0, 0},
      {0, 0, 0} | spatial_padding_config
    ]

    %T{shape: padded_shape} = padded_t = Nx.pad(t, 0, padding_config)

    single_data_dims = Tuple.delete_at(padded_shape, 0)
    batch_size = Nx.size(single_data_dims) * input_size

    # We will traverse the input tensor exactly the same as we traversed
    # the binary in window_reduce, but the window is equal to the filter
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

                    if to_complex(read!(x, 0)) != Complex.conjugate(to_complex(read!(y, 0))) do
                      raise ArgumentError,
                            "matrix must be hermitian, a matrix is hermitian iff X = adjoint(X)"
                    end

                    fun = fn <<match!(left, 0)>>, <<match!(right, 0)>>, acc ->
                      {<<>>, read!(left, 0) * Complex.conjugate(read!(right, 0)) + acc}
                    end

                    {_, tmp_sum} = bin_zip_reduce_axis(lhs, rhs, size, size, <<>>, 0, fun)

                    if i == j - 1 do
                      value = Complex.sqrt(Kernel.max(read!(x, 0) - tmp_sum, 0))
                      number_to_binary(value, output_type)
                    else
                      <<_::size(diagonal_element_offset)-bitstring, match!(diag, 0),
                        _::bitstring>> = acc

                      value = 1.0 / read!(diag, 0) * (read!(x, 0) - tmp_sum)
                      number_to_binary(value, output_type)
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
        %{type: input_type, shape: input_shape} = tensor,
        opts
      ) do
    bin = to_binary(tensor)
    {q, r} = B.Matrix.qr(bin, input_type, input_shape, output_type, m, k, n, opts)
    {from_binary(q_holder, q), from_binary(r_holder, r)}
  end

  @impl true
  def eigh(
        {%{type: output_type} = eigenvals_holder, eigenvecs_holder},
        %{type: input_type, shape: input_shape} = tensor,
        opts
      ) do
    bin = to_binary(tensor)
    {eigenvals, eigenvecs} = B.Matrix.eigh(bin, input_type, input_shape, output_type, opts)

    {from_binary(eigenvals_holder, eigenvals), from_binary(eigenvecs_holder, eigenvecs)}
  end

  @impl true
  def svd(
        {%{shape: {m, _}} = u_holder, s_holder, %{shape: {_, n}} = v_holder} = outputs,
        %{type: input_type, shape: input_shape} = tensor,
        opts
      ) do
    if m < n do
      raise ArgumentError,
            "SVD not implemented for wide matrices (tensors with shape {m, n} where m < n)"
    end

    bin = to_binary(tensor)
    {u, s, v} = B.Matrix.svd(bin, input_type, input_shape, outputs, opts)
    {from_binary(u_holder, u), from_binary(s_holder, s), from_binary(v_holder, v)}
  end

  @impl true
  def lu(
        {%{type: p_type} = p_holder, %{type: l_type} = l_holder, %{type: u_type} = u_holder},
        %{type: input_type, shape: input_shape} = tensor,
        opts
      ) do
    bin = to_binary(tensor)
    {p, l, u} = B.Matrix.lu(bin, input_type, input_shape, p_type, l_type, u_type, opts)
    {from_binary(p_holder, p), from_binary(l_holder, l), from_binary(u_holder, u)}
  end

  @impl true
  def triangular_solve(
        %{type: output_type} = out,
        %{type: a_type, shape: {rows, rows} = a_shape} = a,
        %{type: b_type, shape: b_shape} = b,
        opts
      )
      when tuple_size(b_shape) == 2 or b_shape == {rows} do
    a_data = to_binary(a)
    b_data = to_binary(b)
    out_bin = B.Matrix.ts(a_data, a_type, a_shape, b_data, b_type, b_shape, output_type, opts)
    from_binary(out, out_bin)
  end

  ## Aggregation

  @impl true
  def all(out, %{type: type} = tensor, opts) do
    data =
      bin_reduce(tensor, out.type, 1, opts, fn bin, acc ->
        res = if binary_to_number(bin, type) != 0, do: acc, else: 0
        {res, res}
      end)

    from_binary(out, data)
  end

  @impl true
  def any(out, %{type: type} = tensor, opts) do
    data =
      bin_reduce(tensor, out.type, 0, opts, fn bin, acc ->
        res = if binary_to_number(bin, type) != 0, do: 1, else: acc
        {res, res}
      end)

    from_binary(out, data)
  end

  @impl true
  def sum(out, %{type: type} = tensor, opts) do
    data =
      bin_reduce(tensor, out.type, 0, opts, fn bin, acc ->
        res = binary_to_number(bin, type) + acc
        {res, res}
      end)

    from_binary(out, data)
  end

  @impl true
  def product(out, %{type: type} = tensor, opts) do
    data =
      bin_reduce(tensor, out.type, 1, opts, fn bin, acc ->
        res = binary_to_number(bin, type) * acc
        {res, res}
      end)

    from_binary(out, data)
  end

  @impl true
  def reduce_max(out, %{type: type} = tensor, opts) do
    data =
      bin_reduce(tensor, out.type, :first, opts, fn bin, acc ->
        val = binary_to_number(bin, type)
        res = if acc == :first, do: val, else: Kernel.max(acc, val)
        {res, res}
      end)

    from_binary(out, data)
  end

  @impl true
  def reduce_min(out, %{type: type} = tensor, opts) do
    data =
      bin_reduce(tensor, out.type, :first, opts, fn bin, acc ->
        val = binary_to_number(bin, type)
        res = if acc == :first, do: val, else: Kernel.min(acc, val)
        {res, res}
      end)

    from_binary(out, data)
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

    data =
      bin_reduce(tensor, out.type, {0, :first, -1}, opts, fn
        bin, {i, cur_extreme_x, cur_extreme_i} ->
          x = binary_to_number(bin, type)

          if comparator.(x, cur_extreme_x) or cur_extreme_x == :first do
            {i, {i + 1, x, i}}
          else
            {cur_extreme_i, {i + 1, cur_extreme_x, cur_extreme_i}}
          end
      end)

    from_binary(out, data)
  end

  @impl true
  def reduce(out, tensor, acc, opts, fun) do
    each = %{tensor | shape: {}}

    data =
      bin_reduce(tensor, out.type, acc, opts, fn bin, acc ->
        res = fun.(from_binary(each, bin), acc)
        {res, res}
      end)

    from_binary(out, data)
  end

  @impl true
  def window_reduce(out, tensor, acc, window_dimensions, opts, fun) do
    padding_config = opts[:padding]
    strides = opts[:strides]
    dilations = opts[:window_dilations]

    %T{shape: padded_shape, type: {_, size} = type} =
      tensor = Nx.pad(tensor, acc, Enum.map(padding_config, &Tuple.append(&1, 0)))

    acc = scalar_to_number(acc)

    data = to_binary(tensor)
    weighted_shape = weighted_shape(padded_shape, size, window_dimensions, dilations)
    anchors = Enum.sort(make_anchors(padded_shape, strides, window_dimensions, dilations))

    data =
      match_types [type] do
        for anchor <- anchors, into: <<>> do
          offset = weighted_offset(weighted_shape, anchor, dilations)
          window = IO.iodata_to_binary(weighted_traverse(weighted_shape, data, size, offset))

          window_val =
            for <<match!(x, 0) <- window>>,
              reduce: acc,
              do: (acc -> fun.(read!(x, 0), acc))

          <<write!(scalar_to_number(window_val), 0)>>
        end
      end

    from_binary(out, data)
  end

  @impl true
  def window_sum(out, tensor, window_dimensions, opts) do
    %{type: type} = out

    init_value = number_to_binary(0, type)
    init_value = from_binary(%{out | shape: {}, names: []}, init_value)

    fun = fn a, b -> element_add(type, a, b) end
    window_reduce(out, tensor, init_value, window_dimensions, opts, fun)
  end

  @impl true
  def window_max(out, tensor, window_dimensions, opts) do
    %{type: type} = out

    init_value = Nx.Type.min_finite_binary(type)
    init_value = from_binary(%{out | shape: {}, names: []}, init_value)

    fun = fn a, b -> element_max(type, a, b) end
    window_reduce(out, tensor, init_value, window_dimensions, opts, fun)
  end

  @impl true
  def window_min(out, tensor, window_dimensions, opts) do
    %{type: type} = out

    init_value = Nx.Type.max_finite_binary(type)
    init_value = from_binary(%{out | shape: {}, names: []}, init_value)

    fun = fn a, b -> element_min(type, a, b) end
    window_reduce(out, tensor, init_value, window_dimensions, opts, fun)
  end

  @impl true
  def window_product(out, tensor, window_dimensions, opts) do
    %{type: type} = out

    init_value = number_to_binary(1, type)
    init_value = from_binary(%{out | shape: {}, names: []}, init_value)

    fun = fn a, b -> element_multiply(type, a, b) end
    window_reduce(out, tensor, init_value, window_dimensions, opts, fun)
  end

  @impl true
  def map(%{type: output_type} = out, %{type: {_, size}} = tensor, _opts, fun) do
    data = to_binary(tensor)
    template = %{tensor | shape: {}}

    output_data =
      match_types [output_type] do
        for <<bin::size(size)-bitstring <- data>>, into: <<>> do
          tensor = put_in(template.data.state, bin)
          <<write!(scalar_to_number(fun.(tensor)), 0)>>
        end
      end

    from_binary(out, output_data)
  end

  @impl true
  def window_scatter_max(out, tensor, source, init_value, window_dimensions, opts) do
    select_and_scatter(
      out,
      tensor,
      source,
      &Nx.greater_equal/2,
      window_dimensions,
      opts,
      init_value,
      &Nx.add/2
    )
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, window_dimensions, opts) do
    select_and_scatter(
      out,
      tensor,
      source,
      &Nx.less_equal/2,
      window_dimensions,
      opts,
      init_value,
      &Nx.add/2
    )
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

    init_value = scalar_to_number(init_value)

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
  def indexed_add(
        %T{} = out,
        %T{shape: shape, type: {_, target_size}} = target,
        %T{shape: {indices_rows, _indices_cols} = indices_shape} = indices,
        %T{shape: {indices_rows}} = updates
      ) do
    indices_bin_list =
      indices |> to_binary() |> aggregate_axes([1], indices_shape, elem(indices.type, 1))

    offsets_list =
      match_types [indices.type] do
        for idx_bin <- indices_bin_list do
          idx = for <<match!(x, 0) <- idx_bin>>, do: read!(x, 0)
          offset = index_to_binary_offset(idx, shape)
          offset * target_size
        end
      end

    updates_list =
      match_types [updates.type] do
        for <<match!(x, 0) <- to_binary(updates)>>, do: read!(x, 0)
      end

    {offsets_with_updates, _last_offset} =
      offsets_list
      |> Enum.zip(updates_list)
      |> Enum.group_by(fn {off, _} -> off end, fn {_, upd} -> upd end)
      |> Enum.sort_by(fn {off, _} -> off end)
      |> Enum.map_reduce(0, fn {next_offset, upds}, previous_offset ->
        {{
           previous_offset + target_size,
           next_offset,
           Enum.sum(upds)
         }, next_offset}
      end)

    target_binary = to_binary(target)

    offsets_with_updates =
      List.update_at(offsets_with_updates, 0, fn {_prev, current, update} ->
        {0, current, update}
      end)

    {result, tail} =
      for {previous, current, update} <- offsets_with_updates, reduce: {<<>>, target_binary} do
        {traversed, to_traverse} ->
          before_slice_size = current - previous

          match_types [target.type, out.type] do
            <<before_offset::bitstring-size(before_slice_size), match!(element, 0),
              to_traverse::bitstring>> = to_traverse

            # this can be a list of binaries because we are accumulation an iodata list
            before_offset =
              if target.type == out.type do
                before_offset
              else
                for <<match!(x, 0) <- before_offset>>, do: number_to_binary(read!(x, 0), out.type)
              end

            updated_element = <<write!(read!(element, 0) + update, 1)>>

            {[traversed | [before_offset, updated_element]], to_traverse}
          end
      end

    # this can be a list of binaries because we are accumulation an iodata list
    tail =
      match_types [target.type] do
        if target.type == out.type do
          tail
        else
          for <<match!(x, 0) <- tail>>, do: number_to_binary(read!(x, 0), out.type)
        end
      end

    from_binary(out, IO.iodata_to_binary([result, tail]))
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
  def slice(out, tensor, start_indices, lengths, strides) do
    # If you think of a slice as drawing a bounding box in the dimensions
    # of the tensor, then it's clear we can simply use a weighted
    # traversal to construct the new tensor
    %T{type: {_, size}, shape: shape} = tensor
    %{shape: output_shape} = out

    tensor
    |> to_binary()
    |> bin_slice(shape, size, start_indices, lengths, strides, output_shape)
    |> then(&from_binary(out, &1))
  end

  defp bin_slice(data, shape, size, start_indices, lengths, strides, output_shape) do
    start_indices = clamp_indices(start_indices, shape, lengths)

    if hd(strides) == 1 and top_dimension_slice?(tuple_size(shape), shape, output_shape) do
      length = Nx.size(output_shape) * div(size, 8)
      offset = div(length, elem(output_shape, 0)) * hd(start_indices)
      binary_part(data, offset, length)
    else
      # Anchored around the start indices
      weighted_shape = weighted_shape(shape, size, output_shape)
      offset = weighted_offset(weighted_shape, start_indices)

      # The weighted shape is altered such that we traverse
      # with respect to the stride for each dimension
      weighted_shape =
        Enum.zip_with(weighted_shape, strides, fn {d, dim_size}, s ->
          {d, dim_size + (s - 1) * dim_size}
        end)

      IO.iodata_to_binary(weighted_traverse(weighted_shape, data, size, offset))
    end
  end

  defp clamp_indices(start_indices, shape, lengths) do
    Enum.zip_with([Tuple.to_list(shape), start_indices, lengths], fn [dim_size, idx, len] ->
      idx = scalar_to_number(idx)
      min(max(idx, 0), dim_size - len)
    end)
  end

  defp top_dimension_slice?(1, _, _), do: true

  defp top_dimension_slice?(i, is, os)
       when :erlang.element(i, is) == :erlang.element(i, os),
       do: top_dimension_slice?(i - 1, is, os)

  defp top_dimension_slice?(_, _, _), do: false

  @impl true
  def put_slice(out, tensor, start_indices, slice, combine_fn \\ fn _prev, new -> new end) do
    %T{type: {_, size}, shape: shape} = tensor = as_type(out, tensor)
    %T{shape: slice_shape} = slice = as_type(out, slice)

    start_indices = clamp_indices(start_indices, shape, Tuple.to_list(slice_shape))

    weighted_shape = weighted_shape(shape, size)

    rank = Nx.rank(shape)
    ones = List.duplicate(1, rank)

    offsets =
      slice_shape
      |> make_anchors(ones, List.to_tuple(ones))
      |> Enum.map(fn index ->
        pos =
          index
          |> Enum.zip(start_indices)
          |> Enum.map(fn {x, s} -> x + s end)

        weighted_offset(weighted_shape, pos)
      end)
      |> Enum.sort()

    {_, data} =
      for offset <- offsets, reduce: {to_binary(slice), to_binary(tensor)} do
        {<<cur_elem::size(size)-bitstring, rest_of_slice::bitstring>>, binary} ->
          <<before::size(offset)-bitstring, prev_elem::size(size)-bitstring,
            rest_of_tensor::bitstring>> = binary

          new_elem = combine_fn.(prev_elem, cur_elem)

          {rest_of_slice,
           <<before::size(offset)-bitstring, new_elem::size(size)-bitstring,
             rest_of_tensor::bitstring>>}
      end

    from_binary(out, data)
  end

  @impl true
  def take(out, tensor, indices, axis) do
    # We iterate over the indices in a flat manner,
    # and take a unit tensor slice along axis given
    # by each index. Then we concatenate the tensors
    # along the axis, which gives us the result with
    # index dimensions flattened and we just reshape.

    %T{type: {_, size}, shape: shape} = tensor
    %T{type: {_, idx_size}} = indices

    data = to_binary(tensor)
    tensor_rank = tuple_size(shape)
    slice_start = List.duplicate(0, tensor_rank)
    slice_lengths = shape |> Tuple.to_list() |> List.replace_at(axis, 1)
    slice_shape = List.to_tuple(slice_lengths)
    strides = List.duplicate(1, tensor_rank)

    slices =
      for <<bin::size(idx_size)-bitstring <- to_binary(indices)>> do
        idx = binary_to_number(bin, indices.type)

        if idx < 0 or idx >= elem(shape, axis) do
          raise ArgumentError,
                "index #{idx} is out of bounds for axis #{axis} in shape #{inspect(shape)}"
        end

        slice_start = List.replace_at(slice_start, axis, idx)

        slice_data =
          bin_slice(data, shape, size, slice_start, slice_lengths, strides, slice_shape)

        {slice_data, slice_shape}
      end

    concat_shape = put_elem(tensor.shape, axis, length(slices))
    result_data = bin_concatenate(slices, size, axis, concat_shape)

    from_binary(out, result_data)
  end

  @impl true
  def take_along_axis(
        %T{type: output_type} = output,
        %T{shape: t_shape, type: {_, t_size} = t_type} = tensor,
        %T{shape: idx_shape, type: {_, idx_size} = idx_type} = indices,
        axis
      ) do
    permutation =
      tensor
      |> Nx.axes()
      |> List.delete(axis)
      |> List.insert_at(Nx.rank(tensor) - 1, axis)

    inverse_permutation =
      permutation
      |> Enum.with_index()
      |> Enum.sort_by(fn {x, _} -> x end)
      |> Enum.map(fn {_, i} -> i end)

    shape_list = Tuple.to_list(output.shape)
    permuted_shape = permutation |> Enum.map(&Enum.at(shape_list, &1)) |> List.to_tuple()

    t_view = tensor |> to_binary() |> aggregate_axes([axis], t_shape, t_size)

    idx_view = indices |> to_binary() |> aggregate_axes([axis], idx_shape, idx_size)

    [t_view, idx_view]
    |> Enum.zip_with(fn [data_bin, idx_bin] ->
      match_types [t_type, idx_type, output_type] do
        data = for <<match!(x, 0) <- data_bin>>, do: read!(x, 0)

        for <<match!(x, 1) <- idx_bin>>, into: <<>> do
          idx = read!(x, 1)

          if idx < 0 or idx >= elem(tensor.shape, axis) do
            raise ArgumentError,
                  "index #{idx} is out of bounds for axis #{axis} in shape #{inspect(tensor.shape)}"
          end

          val = Enum.at(data, idx)
          <<write!(val, 2)>>
        end
      end
    end)
    |> then(&from_binary(%{output | shape: permuted_shape}, &1))
    |> then(&transpose(output, &1, inverse_permutation))
  end

  @impl true
  def gather(out, tensor, indices) do
    %T{type: {_, size}, shape: shape} = tensor
    %T{type: {_, idx_size}} = indices

    data = to_binary(tensor)
    rank = tuple_size(shape)
    byte_size = div(size, 8)

    idx_last_dim_bin_size = rank * idx_size

    new_data =
      for <<bin::size(idx_last_dim_bin_size)-bitstring <- to_binary(indices)>>, into: <<>> do
        slice_start =
          for <<bin::size(idx_size)-bitstring <- bin>>, do: binary_to_number(bin, indices.type)

        offset = index_to_binary_offset(slice_start, shape)
        binary_part(data, offset * byte_size, byte_size)
      end

    from_binary(out, new_data)
  end

  defp index_to_binary_offset(index, input_shape) when is_list(index) and is_tuple(input_shape) do
    {offset, []} =
      index
      |> Enum.with_index()
      |> Enum.reduce(
        {0, Tuple.to_list(input_shape)},
        fn {idx, axis}, {offset, [dim_size | shape]} ->
          if idx < 0 or idx >= dim_size do
            raise ArgumentError,
                  "index #{idx} is out of bounds for axis #{axis} in shape #{inspect(input_shape)}"
          end

          {offset + idx * Enum.product(shape), shape}
        end
      )

    offset
  end

  @impl true
  def concatenate(out, tensors, axis) do
    %{shape: output_shape, type: {_, size} = output_type} = out

    tensors
    |> Enum.map(fn %{shape: shape} = t ->
      t = as_type(%{t | type: output_type}, t)
      {to_binary(t), shape}
    end)
    |> bin_concatenate(size, axis, output_shape)
    |> then(&from_binary(out, &1))
  end

  defp bin_concatenate(binaries_with_shape, _size, 0, _output_shape) do
    binaries_with_shape |> Enum.map(&elem(&1, 0)) |> IO.iodata_to_binary()
  end

  defp bin_concatenate(binaries_with_shape, size, axis, output_shape) do
    rank = tuple_size(output_shape)
    steps = product_part(output_shape, 0, axis)

    data =
      for step <- 1..steps,
          {binary, shape} <- binaries_with_shape do
        product = product_part(shape, axis, rank) * size
        before = (step - 1) * product
        <<_::bitstring-size(before), part::bitstring-size(product), _::bitstring>> = binary
        part
      end

    IO.iodata_to_binary(data)
  end

  defp product_part(_tuple, n, n), do: 1
  defp product_part(tuple, n, limit), do: elem(tuple, n) * product_part(tuple, n + 1, limit)

  @impl true
  def as_type(out, tensor) do
    %{type: output_type} = out

    case tensor do
      %T{type: ^output_type} ->
        tensor

      %T{type: input_type} ->
        float_output? = Nx.Type.float?(output_type)
        data = to_binary(tensor)

        output_data =
          match_types [input_type] do
            for <<match!(x, 0) <- data>>, into: <<>> do
              x = read!(x, 0)

              case x do
                %Complex{re: re} when float_output? ->
                  number_to_binary(re, output_type)

                _ when float_output? ->
                  number_to_binary(x, output_type)

                %Complex{re: re} ->
                  number_to_binary(trunc(re), output_type)

                _ when is_number(x) ->
                  number_to_binary(trunc(x), output_type)

                :nan ->
                  number_to_binary(0, output_type)

                :infinity ->
                  Nx.Type.max_finite_binary(output_type)

                :neg_infinity ->
                  Nx.Type.min_finite_binary(output_type)
              end
            end
          end

        from_binary(out, output_data)
    end
  end

  @impl true
  def bitcast(out, tensor), do: from_binary(out, to_binary(tensor))

  @impl true
  def sort(output, t, opts), do: do_sort(output, t, opts, false)

  @impl true
  def argsort(output, t, opts), do: do_sort(output, t, opts, true)

  defp do_sort(output, t, opts, return_indices) do
    %T{shape: shape, type: type} = t
    last_axis = Nx.rank(t) - 1

    axis = opts[:axis]

    comparator =
      case opts[:direction] do
        :desc ->
          fn a, b ->
            a = binary_to_number(a, type)
            b = binary_to_number(b, type)
            a >= b
          end

        :asc ->
          fn a, b ->
            a = binary_to_number(a, type)
            b = binary_to_number(b, type)
            a <= b
          end
      end

    case shape do
      {} when return_indices ->
        sort_last_dim(t, comparator, output, return_indices)

      {} ->
        t

      _ when axis == last_axis ->
        sort_last_dim(t, comparator, output, return_indices)

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

        permuted_t = Nx.transpose(t, axes: permutation)

        permuted_t
        |> sort_last_dim(comparator, %{permuted_t | type: output.type}, return_indices)
        |> Nx.transpose(axes: inverse_permutation)
    end
  end

  defp sort_last_dim(
         %T{shape: shape, type: {_, size}} = t,
         comparator,
         output,
         return_indices
       ) do
    view = aggregate_axes(to_binary(t), [tuple_size(shape) - 1], shape, size)

    new_data =
      for bin <- view, into: <<>> do
        data = for <<x::size(size)-bitstring <- bin>>, do: x

        sorted =
          if return_indices do
            data
            |> Enum.with_index()
            |> Enum.sort_by(&elem(&1, 0), comparator)
            |> Enum.map(fn {_, index} -> number_to_binary(index, output.type) end)
          else
            Enum.sort(data, comparator)
          end

        IO.iodata_to_binary(sorted)
      end

    from_binary(output, new_data)
  end

  ## Binary reducers

  defp bin_reduce(tensor, type, acc, opts, fun) do
    %T{type: {_, size}, shape: shape} = tensor

    view =
      if axes = opts[:axes] do
        aggregate_axes(to_binary(tensor), axes, shape, size)
      else
        [to_binary(tensor)]
      end

    for axis <- view, into: <<>> do
      {result, _} =
        for <<bin::size(size)-bitstring <- axis>>, reduce: {<<>>, acc} do
          {_, acc} -> fun.(bin, acc)
        end

      scalar_to_binary(result, type)
    end
  end

  defp bin_zip_reduce(t1, [], t2, [], type, acc, fun) do
    %{type: {_, s1}} = t1
    %{type: {_, s2}} = t2
    b1 = to_binary(t1)
    b2 = to_binary(t2)

    match_types [t1.type, t2.type] do
      for <<d1::size(s1)-bitstring <- b1>>, <<d2::size(s2)-bitstring <- b2>>, into: <<>> do
        {result, _} = fun.(d1, d2, acc)
        scalar_to_binary(result, type)
      end
    end
  end

  defp bin_zip_reduce(t1, [_ | _] = axes1, t2, [_ | _] = axes2, type, acc, fun) do
    {_, s1} = t1.type
    {_, s2} = t2.type

    v1 = aggregate_axes(to_binary(t1), axes1, t1.shape, s1)
    v2 = aggregate_axes(to_binary(t2), axes2, t2.shape, s2)

    for b1 <- v1, b2 <- v2 do
      {bin, _acc} = bin_zip_reduce_axis(b1, b2, s1, s2, <<>>, acc, fun)
      scalar_to_binary(bin, type)
    end
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

  defp scalar_to_number(n) when is_number(n), do: n
  defp scalar_to_number(%Complex{} = n), do: n
  defp scalar_to_number(t), do: binary_to_number(to_binary(t), t.type)

  defp scalar_to_binary(%Complex{re: re, im: im}, type) do
    real_type = Nx.Type.to_real(type)
    number_to_binary(re, real_type) <> number_to_binary(im, real_type)
  end

  defp scalar_to_binary(value, type) when is_number(value),
    do: number_to_binary(value, type)

  defp scalar_to_binary(%T{shape: {}, type: type} = t, type),
    do: to_binary(t)

  defp scalar_to_binary(t, type) do
    raise ArgumentError,
          "expected a number or a scalar tensor of type #{inspect(type)}, got: #{inspect(t)}"
  end

  defp number_to_binary(number, type),
    do:
      match_types([type],
        do: <<write!(number, 0)>>
      )

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

  defp to_complex(%Complex{} = z), do: z
  defp to_complex(n), do: Complex.new(n)
end

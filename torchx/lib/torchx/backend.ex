defmodule Torchx.Backend do
  @moduledoc """
  An opaque backend Nx backend with bindings to libtorch/Pytorch.

  Torchx behaviour that is different from BinaryBackend:

    1. Torchx emulates the u16/u32/u64 unsigned integers using signed integers.
       In practice this means u64 actually overflows at u32. When accessing the
       underlying tensor, you will get signed integers back.

        iex> t = Nx.Constants.max_finite({:u, 32})
        #Nx.Tensor<
          u32
          4294967295
        >
        iex> t |> Torchx.from_nx() |> Torchx.to_nx()
        #Nx.Tensor<
          s64
          4294967295
        >

    2. Torchx rounds half-to-even, while Elixir rounds half-away-from-zero.
       So in Elixir `round(0.5) == 1.0` while in Torchx `round(0.5) == 0.0`.

    3. `Nx.as_type/2` converts non-finite values such as infinity becomes the
       maximum value for a type, negative infinity becomes the minimum value,
       and nan becomes zero. `Torchx` behaviour is type dependent with no clear
       rule across types.

  ## Options

    * `:device` - Defaults to `Torchx.default_device/0`. An atom representing the
      device for the allocation of a given tensor. Valid values can be seen at the
      main `Torchx` docs.
  """

  @behaviour Nx.Backend
  defstruct [:ref]

  require Application
  alias Nx.Tensor, as: T
  alias Torchx.Backend, as: TB

  @impl true
  def init(opts) do
    Keyword.validate!(opts, [:device])
  end

  ## Creation

  @impl true
  def constant(%T{shape: {}, type: type} = out, scalar, backend_options) do
    scalar
    |> constant_serialize_scalar()
    |> Torchx.scalar_tensor(to_torch_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  def constant(%T{shape: shape, type: type} = out, scalar, backend_options) do
    shape
    |> Torchx.full(
      constant_serialize_scalar(scalar),
      to_torch_type(type),
      device_option(backend_options)
    )
    |> to_nx(out)
  end

  defp constant_serialize_scalar(%Complex{re: real, im: imag}), do: {real, imag}
  defp constant_serialize_scalar(scalar), do: scalar

  @impl true
  def eye(%T{shape: shape, type: type} = out, backend_options) do
    rank = tuple_size(shape)
    m = elem(shape, rank - 2)
    n = elem(shape, rank - 1)

    m
    |> Torchx.eye(n, to_torch_type(type), device_option(backend_options))
    |> Torchx.broadcast_to(shape)
    |> to_nx(out)
  end

  @impl true
  def iota(%T{shape: {}, type: type} = out, nil, backend_options) do
    Torchx.scalar_tensor(0.0, to_torch_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  def iota(%T{shape: shape, type: type} = out, nil, backend_options) do
    Torchx.arange(
      0,
      Nx.size(shape),
      1,
      to_torch_type(type),
      device_option(backend_options),
      shape
    )
    |> to_nx(out)
  end

  @impl true
  def iota(%T{shape: {n}, type: type} = out, 0, backend_options) do
    Torchx.arange(0, n, 1, to_torch_type(type), device_option(backend_options)) |> to_nx(out)
  end

  def iota(%T{shape: shape, type: type} = out, axis, backend_options) do
    # gets the size of iota
    dim = elem(shape, axis)

    # build the iota in one dimension
    aten = Torchx.arange(0, dim, 1, to_torch_type(type), device_option(backend_options))

    # reshape the tensor above to be have shape where everything is 1, except for dim
    reshape = Tuple.duplicate(1, Nx.rank(shape)) |> put_elem(axis, dim)
    aten = Torchx.reshape(aten, reshape)

    # Now broadcast the tensor using the original shape
    Torchx.broadcast_to(aten, shape) |> to_nx(out)
  end

  @impl true
  def random_uniform(%T{type: {s, _} = type, shape: shape} = out, min, max, backend_options)
      when s in [:u, :s] do
    min = to_number(min)
    max = to_number(max)

    Torchx.randint(min, max, shape, to_torch_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  def random_uniform(%T{type: {:c, s}, shape: shape} = out, min, max, backend_options) do
    rand_type = {:f, div(s, 2)}

    real = random_uniform_float(min, max, shape, rand_type, backend_options)
    imag = random_uniform_float(min, max, shape, rand_type, backend_options)

    imag
    |> Torchx.multiply(
      Torchx.scalar_tensor(Complex.new(0, 1), :complex, device_option(backend_options))
    )
    |> Torchx.add(real)
    |> to_nx(out)
  end

  def random_uniform(%T{type: {f, _} = type, shape: shape} = out, min, max, backend_options)
      when f in [:f, :bf] do
    min
    |> random_uniform_float(max, shape, type, backend_options)
    |> to_nx(out)
  end

  defp random_uniform_float(min, max, shape, type, backend_options) do
    min = to_number(min)
    max = to_number(max)

    Torchx.rand(min, max, shape, to_torch_type(type), device_option(backend_options))
  end

  @impl true
  def random_normal(%T{type: type, shape: shape} = out, mu, sigma, backend_options) do
    mu = to_number(mu)
    sigma = to_number(sigma)

    Torchx.normal(mu, sigma, shape, to_torch_type(type), device_option(backend_options))
    |> to_nx(out)
  end

  ## Transfer

  @impl true
  def to_batched(%T{shape: shape} = out, %T{} = t, opts) do
    leftover = opts[:leftover]

    batch_size = elem(shape, 0)
    t_axis_0 = elem(t.shape, 0)

    remainder = rem(t_axis_0, batch_size)
    num_full_batches = div(t_axis_0, batch_size)

    num_batches =
      if leftover == :repeat and remainder != 0 do
        num_full_batches + 1
      else
        num_full_batches
      end

    to_batch =
      if remainder != 0 and leftover == :repeat do
        slice_shape = t.shape |> Tuple.delete_at(0) |> Tuple.insert_at(0, remainder)

        t_torchx = from_nx(t)

        slice = torchx_slice(t_torchx, t.shape, slice_shape, [0], [remainder], [1])

        Torchx.concatenate([t_torchx, slice], 0)
      else
        from_nx(t)
      end

    # torch::split returns a chunk with smaller size if the
    # tensor is not fully divisible by the batch_size.
    # We need to drop the last chunk in this case
    case Torchx.split(to_batch, batch_size) do
      batches when remainder != 0 ->
        batches |> Enum.take(num_batches) |> Stream.map(&to_nx(&1, out))

      batches ->
        Stream.map(batches, &to_nx(&1, out))
    end
  end

  @impl true
  def to_binary(tensor, limit) do
    blob = Torchx.to_blob(from_nx(tensor), limit)

    case tensor.type do
      {:u, 16} -> for <<x::32-native <- blob>>, do: <<x::16-native>>, into: <<>>
      {:u, 32} -> for <<x::64-native <- blob>>, do: <<x::32-native>>, into: <<>>
      _ -> blob
    end
  end

  @impl true
  def backend_deallocate(%T{} = t) do
    Torchx.delete_tensor(from_nx(t))
  rescue
    ArgumentError -> :already_deallocated
  end

  @impl true
  def backend_transfer(tensor, backend, opts) do
    backend_copy(tensor, backend, opts)
  after
    backend_deallocate(tensor)
  end

  @impl true
  def backend_copy(tensor, Torchx.Backend, opts) do
    Torchx.to_device(from_nx(tensor), device_option(opts)) |> to_nx(tensor)
  end

  def backend_copy(tensor, backend, opts) do
    backend.from_binary(tensor, Torchx.to_blob(from_nx(tensor)), opts)
  end

  @impl true
  def from_binary(%T{type: type, shape: shape} = out, binary, backend_options) do
    binary
    |> maybe_pad_binary(type)
    |> Torchx.from_blob(
      shape,
      to_torch_type(type),
      device_option(backend_options)
    )
    |> to_nx(out)
  end

  defp maybe_pad_binary(bin, {:u, size}) when size in [16, 32] do
    double_size = size * 2
    for <<x::native-size(size) <- bin>>, into: <<>>, do: <<x::native-size(double_size)>>
  end

  defp maybe_pad_binary(bin, _), do: bin

  ## Shape

  @impl true
  def reshape(%T{shape: shape} = out, %T{} = t),
    do: Torchx.reshape(from_nx(t), shape) |> to_nx(out)

  @impl true
  def as_type(%T{type: type} = out, %T{} = t),
    do: from_nx(t) |> Torchx.to_type(to_torch_type(type)) |> bitmask(type) |> to_nx(out)

  @impl true
  def squeeze(out, %T{} = t, axes) do
    # sort the axes desc so we don't have to decrease the axis numbers after each squeeze
    result =
      for axis <- Enum.sort(axes, :desc), reduce: from_nx(t) do
        t_tx ->
          Torchx.squeeze(t_tx, axis)
      end

    to_nx(result, out)
  end

  @impl true
  def broadcast(out, %T{} = t, shape, axes) do
    t
    |> maybe_reshape(shape, axes)
    |> from_nx()
    |> Torchx.broadcast_to(shape)
    |> to_nx(out)
  end

  defp maybe_reshape(%T{shape: {}} = t, target_shape, _axes) do
    shape = 1 |> List.duplicate(tuple_size(target_shape)) |> List.to_tuple()
    Nx.reshape(t, shape)
  end

  defp maybe_reshape(%T{shape: shape} = t, target_shape, axes) do
    base_broadcast_shape = 1 |> List.duplicate(tuple_size(target_shape)) |> List.to_tuple()

    new_shape =
      shape
      |> Tuple.to_list()
      |> Enum.zip(axes)
      |> Enum.reduce(base_broadcast_shape, fn {dim_size, target_axis}, shape_acc ->
        shape_acc
        |> Tuple.delete_at(target_axis)
        |> Tuple.insert_at(target_axis, dim_size)
      end)

    Nx.reshape(t, new_shape)
  end

  @impl true
  def transpose(out, %T{} = t, axes) do
    Torchx.permute(from_nx(t), axes) |> to_nx(out)
  end

  @impl true
  def slice(
        %T{shape: output_shape} = out,
        %T{shape: input_shape} = t,
        start_indices,
        lengths,
        strides
      ) do
    t
    |> from_nx()
    |> torchx_slice(input_shape, output_shape, start_indices, lengths, strides)
    |> to_nx(out)
  end

  defp torchx_slice(t, input_shape, output_shape, start_indices, lengths, strides) do
    t
    |> narrow(start_indices, lengths, 0, input_shape)
    |> stride(output_shape, lengths, strides)
  end

  defp narrow(ref, [start | starts], [length | lengths], axis, shape) do
    dim = elem(shape, axis)
    start = to_number(start)
    start = min(start, dim - length)

    # Nothing to narrow
    if start == 0 and length == dim do
      narrow(ref, starts, lengths, axis + 1, shape)
    else
      ref
      |> Torchx.narrow(axis, start, length)
      |> narrow(starts, lengths, axis + 1, shape)
    end
  end

  defp narrow(ref, [], [], _axis, _shape), do: ref

  defp stride(ref, shape, lengths, strides) do
    if Enum.all?(strides, &(&1 == 1)) do
      ref
    else
      ref
      |> Torchx.as_strided(shape, steps_to_strides(lengths, strides), 0)
    end
  end

  def steps_to_strides(shape, steps) do
    for {dim, step} <- Enum.zip(shape, steps) |> Enum.reverse(), reduce: {1, []} do
      {offset, strides} -> {offset * dim, [offset * step | strides]}
    end
    |> elem(1)
  end

  @impl true
  def put_slice(out, input, start_indices_unbounded, slice) do
    {device, _} = input_tx = from_nx(input)

    slice_shape_list = Tuple.to_list(slice.shape)

    zip_indices_input = [Tuple.to_list(input.shape), start_indices_unbounded, slice_shape_list]

    start_indices =
      Enum.zip_with(zip_indices_input, fn [dim_size, idx, len] ->
        idx = Nx.to_number(idx)
        min(max(idx, 0), dim_size - len)
      end)

    range_or_ranges =
      [start_indices, slice_shape_list]
      |> Enum.zip_with(fn [s, l] -> s..(s + l - 1)//1 end)
      |> Enum.reverse()
      |> Enum.reduce(fn range, acc -> for x <- range, y <- acc, do: List.flatten([x, y]) end)

    # if below is needed for when the reduce receives a single-element list
    linear_indices_tx =
      if is_list(range_or_ranges) do
        range_or_ranges
        |> Nx.tensor(backend: {__MODULE__, device: device})
        |> then(&as_torchx_linear_indices(input.shape, &1))
      else
        range_or_ranges
        |> Enum.to_list()
        |> Nx.tensor(backend: {__MODULE__, device: device})
        |> Torchx.from_nx()
      end

    slice_tx = slice |> from_nx() |> Torchx.to_type(to_torch_type(out.type))

    input_tx
    |> Torchx.to_type(to_torch_type(out.type))
    |> Torchx.put(linear_indices_tx, slice_tx)
    |> to_nx(out)
  end

  @impl true
  def concatenate(out, tensors, axis) do
    tensors
    |> Enum.map(&from_nx/1)
    |> Torchx.concatenate(axis)
    |> to_nx(out)
  end

  @impl true
  def take(out, t, i, axis) do
    axes_range = 0..(Nx.rank(t) - 1)//1

    indices_shape =
      axes_range
      |> Enum.flat_map(fn
        ^axis -> Tuple.to_list(i.shape)
        _ -> [1]
      end)
      |> List.to_tuple()

    idx_tiling =
      t.shape
      |> Tuple.to_list()
      |> Enum.with_index(fn
        _x, ^axis ->
          List.duplicate(1, Nx.rank(i))

        x, _ ->
          x
      end)
      |> List.flatten()

    indices_for_axis =
      i
      |> Nx.reshape(indices_shape)
      |> Nx.tile(idx_tiling)

    num_elements = Tuple.product(indices_for_axis.shape)

    axis_offset = Nx.rank(i) - 1

    indices =
      axes_range
      |> Enum.map(fn
        ^axis ->
          Nx.reshape(indices_for_axis, {num_elements, 1})

        current when current < axis ->
          indices_for_axis
          |> Nx.shape()
          |> Nx.iota(axis: current, backend: __MODULE__)
          |> Nx.reshape({num_elements, 1})

        current when current > axis ->
          indices_for_axis
          |> Nx.shape()
          |> Nx.iota(axis: current + axis_offset, backend: __MODULE__)
          |> Nx.reshape({num_elements, 1})
      end)
      |> Nx.concatenate(axis: 1)

    gather(out, t, indices)
  end

  @impl true
  def gather(out, tensor, idx) do
    linear_indices_tx = as_torchx_linear_indices(tensor.shape, idx)

    tensor
    |> from_nx()
    |> Torchx.reshape({Tuple.product(tensor.shape)})
    |> Torchx.gather(linear_indices_tx, 0)
    |> Torchx.reshape(out.shape)
    |> to_nx(out)
  end

  @impl true
  def indexed_add(out, tensor, indices, updates) do
    indexed(out, tensor, indices, updates, :indexed_add)
  end

  @impl true
  def indexed_put(out, tensor, indices, updates) do
    indexed(out, tensor, indices, updates, :indexed_put)
  end

  defp indexed(out, tensor, indices, updates, function) do
    linear_indices_tx = as_torchx_linear_indices(tensor.shape, indices)

    updates_tx =
      updates
      |> from_nx()
      |> Torchx.to_type(to_torch_type(out.type))

    tensor
    |> from_nx()
    |> Torchx.to_type(to_torch_type(out.type))
    |> Torchx.reshape({Tuple.product(tensor.shape)})
    |> then(&apply(Torchx, function, [&1, linear_indices_tx, updates_tx, 0]))
    |> Torchx.reshape(out.shape)
    |> to_nx(out)
  end

  defp as_torchx_linear_indices(shape, idx) do
    # Nx provides indices as a tensor of shape {*, input_dims}
    # However, torch expects indices to be a tensor of indices along a given axis.
    # As such, we need to convert the indices tensor to linear indices.
    # See the `linear_indices_offsets` function for an explanation on the offsets calculation.

    # Index limit validation

    ndims = tuple_size(shape)

    flattened_idx = Nx.reshape(idx, {div(Nx.size(idx), ndims), ndims})
    shape_tensor = shape |> Tuple.to_list() |> Nx.tensor()

    upper_clamped_idx =
      flattened_idx
      |> Nx.greater_equal(shape_tensor)
      |> Nx.select(Nx.subtract(shape_tensor, 1), flattened_idx)

    lower_clamp_selector = Nx.less(upper_clamped_idx, 0)

    fully_clamped_idx =
      lower_clamp_selector |> Nx.select(0, upper_clamped_idx) |> Nx.reshape(idx.shape)

    # Actual conversion algorithm

    linear_indices_offsets =
      shape
      |> linear_indices_offsets()
      |> from_nx()

    lin_idx_num_elements =
      idx.shape |> Tuple.delete_at(tuple_size(idx.shape) - 1) |> Tuple.product()

    fully_clamped_idx
    |> from_nx()
    |> Torchx.tensordot(linear_indices_offsets, [tuple_size(idx.shape) - 1], [0])
    |> Torchx.reshape({lin_idx_num_elements})
  end

  defp linear_indices_offsets(shape) do
    # The offsets tensor calculated below follows a formula in which we
    # multiply the index along each axis by the number of elements contained in all following axes
    # For example, for a {3, 5, 7, 2} tensor, the offsets tensor is [70, 14, 2, 1]

    # This offsets tensor is then applied to the indices tensor through matrix multiplication:
    # indices = [[0, 2, 1, 0], [0, 0, 0, 1], [1, 4, 3, 2]]
    # offsets = [70, 14, 2, 1]
    # linear_indices = [14 * 2 + 2 * 1, 1 * 1, 70 * 1 + 14 * 4 + 2 * 3 + 1 * 2] = [30, 1, 134]

    # By linear indices, we refer to the indices of a row-major representation of a tensor
    # it's easy to see the expected values using Nx.iota(tensor), which will output a tensor
    # which counts in exactly the same way, when provided no arguments. In effect, Nx.iota outputs
    # the corresponding linear indices for a given tensor shape.

    {offsets_list, _} =
      shape
      |> Tuple.to_list()
      |> Enum.reverse()
      |> Enum.reduce({[], 1}, fn x, {acc, multiplier} ->
        {[multiplier | acc], multiplier * x}
      end)

    Nx.tensor(offsets_list, backend: __MODULE__)
  end

  @impl true
  def take_along_axis(out, tensor, idx, axis) do
    idx_tx = idx |> from_nx() |> Torchx.to_type(:long)

    tensor
    |> from_nx()
    |> Torchx.gather(idx_tx, axis)
    |> to_nx(out)
  end

  @impl true
  def argsort(out, tensor, opts) do
    axis = opts[:axis]
    is_descending = opts[:direction] == :desc

    tensor
    |> from_nx()
    |> Torchx.argsort(axis, is_descending)
    |> to_nx(out)
  end

  @impl true
  def top_k({out_values, out_indices}, tensor, opts) do
    {values, indices} =
      tensor
      |> from_nx()
      |> Torchx.top_k(Keyword.fetch!(opts, :k))

    {to_nx(values, out_values), to_nx(indices, out_indices)}
  end

  @impl true
  def reverse(out, tensor, axes) do
    tensor
    |> from_nx()
    |> Torchx.flip(axes)
    |> to_nx(out)
  end

  ## Aggregators

  @impl true
  def sum(%T{} = out, %T{} = t, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    t
    |> from_nx()
    |> Torchx.sum(axes, keep_axes)
    |> to_nx(out)
  end

  @impl true
  def product(%T{} = out, %T{} = t, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    result =
      if axes == [] do
        aggregate_whole_tensor(t, keep_axes, &Torchx.product/1)
      else
        aggregate_over_axes(t, axes, keep_axes, &Torchx.product/3)
      end

    to_nx(result, out)
  end

  @impl true
  def any(%T{} = out, %T{} = t, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    result =
      if axes == [] do
        aggregate_whole_tensor(t, keep_axes, &Torchx.any/1)
      else
        aggregate_over_axes(t, axes, keep_axes, &Torchx.any/3)
      end

    to_nx(result, out)
  end

  @impl true
  def all(%T{} = out, %T{} = t, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    result =
      if axes == [] do
        aggregate_whole_tensor(t, keep_axes, &Torchx.all/1)
      else
        aggregate_over_axes(t, axes, keep_axes, &Torchx.all/3)
      end

    to_nx(result, out)
  end

  @impl true
  def all_close(%T{} = out, %T{} = a, %T{} = b, opts) do
    equal_nan = opts[:equal_nan]
    rtol = opts[:rtol]
    atol = opts[:atol]

    # Torch raises a cryptic error if the types are different,
    # so we need to upcast the tensors to the merged type
    type = a.type |> Nx.Type.merge(b.type) |> to_torch_type()
    a_tx = a |> from_nx() |> Torchx.to_type(type)
    b_tx = b |> from_nx() |> Torchx.to_type(type)

    a_tx
    |> Torchx.all_close(b_tx, rtol, atol, !!equal_nan)
    |> to_nx(out)
  end

  defp aggregate_whole_tensor(t, keep_axes, fun) when is_function(fun, 1) do
    result =
      t
      |> from_nx()
      |> then(fun)

    if keep_axes do
      shape = t.shape |> Tuple.delete_at(-1) |> Tuple.append(1)
      Torchx.reshape(result, shape)
    else
      result
    end
  end

  defp aggregate_over_axes(t, axes, keep_axes, fun) when is_function(fun, 3) do
    t_tx =
      case t do
        {_, _} -> t
        _ -> from_nx(t)
      end

    {_, result_tx} =
      for _ <- 1..length(axes), reduce: {axes, t_tx} do
        {[], t_tx} ->
          {[], t_tx}

        {[axis | axes], t_tx} ->
          # We need to offset all subsequent axes if keep_axes == false.
          # If keep_axes == true, we can use the same axis numbers as the
          # incoming tensor.
          axes =
            if keep_axes do
              axes
            else
              for x <- axes do
                if x > axis, do: x - 1, else: x
              end
            end

          {axes, fun.(t_tx, axis, keep_axes)}
      end

    result_tx
  end

  @impl true
  def determinant(out, tensor) do
    tensor
    |> from_nx()
    |> Torchx.to_type(to_torch_type(out.type))
    |> Torchx.determinant()
    |> to_nx(out)
  end

  @impl true
  def argmax(%T{} = out, %T{} = t, opts) do
    argminmax(:argmax, out, t, opts)
  end

  @impl true
  def argmin(out, t, opts) do
    argminmax(:argmin, out, t, opts)
  end

  defp argminmax(fun, %T{} = out, %T{} = t, opts) do
    tie_break = opts[:tie_break] || :low
    axis = opts[:axis] || -1
    keep_axis = opts[:keep_axis] || false

    t_tx = from_nx(t)

    if tie_break == :low do
      apply(Torchx, fun, [t_tx, axis, keep_axis])
      |> to_nx(out)
    else
      %{data: %{ref: {device, _}}, shape: shape} = t
      scalar = Torchx.scalar_tensor(elem(shape, axis) - 1, to_torch_type(out.type), device)

      flipped = Torchx.flip(t_tx, [axis])

      result = apply(Torchx, fun, [flipped, axis, keep_axis])

      scalar
      |> Torchx.subtract(result)
      |> to_nx(out)
    end
  end

  @impl true
  def cumulative_sum(%T{} = out, %T{} = t, opts) do
    cumulative_op(out, t, opts, &Torchx.cumulative_sum/2)
  end

  @impl true
  def cumulative_product(%T{} = out, %T{} = t, opts) do
    cumulative_op(out, t, opts, &Torchx.cumulative_product/2)
  end

  @impl true
  def cumulative_min(%T{} = out, %T{} = t, opts) do
    cumulative_op(out, t, opts, &Torchx.cumulative_min/2)
  end

  @impl true
  def cumulative_max(%T{} = out, %T{} = t, opts) do
    cumulative_op(out, t, opts, &Torchx.cumulative_max/2)
  end

  defp cumulative_op(out, t, opts, fun) when is_function(fun, 2) do
    axis = opts[:axis]
    reverse = opts[:reverse]

    t_tx =
      if reverse do
        t
        |> from_nx()
        |> Torchx.flip([axis])
      else
        from_nx(t)
      end

    result = apply(fun, [t_tx, axis])

    if reverse do
      result
      |> Torchx.flip([axis])
      |> to_nx(out)
    else
      to_nx(result, out)
    end
  end

  ## Ops

  @impl true
  def atan2(%{type: {:c, _}}, _l, _r) do
    raise ArithmeticError, "Torchx does not support complex values for atan2"
  end

  ops = [:add, :subtract, :multiply, :pow, :left_shift]

  for op <- ops do
    @impl true
    def unquote(op)(out, l, r) do
      {left, right} = maybe_upcast(l, r)
      {left_tx, right_tx} = maybe_broadcast_bin_args(out.shape, left, right)
      result = Torchx.unquote(op)(left_tx, right_tx)

      result
      |> bitmask(out.type)
      |> Torchx.to_type(to_torch_type(out.type))
      |> to_nx(out)
    end
  end

  defp bitmask({device, _} = tensor, {:u, 16}),
    do: Torchx.bitwise_and(tensor, Torchx.scalar_tensor(0xFFFF, :int, device))

  defp bitmask({device, _} = tensor, {:u, 32}),
    do: Torchx.bitwise_and(tensor, Torchx.scalar_tensor(0xFFFF_FFFF, :long, device))

  defp bitmask(tensor, {_, _}),
    do: tensor

  ops =
    [:min, :max, :divide, :quotient, :atan2] ++
      [:right_shift, :logical_and, :logical_or, :logical_xor] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal]

  for op <- ops do
    @impl true
    def unquote(op)(out, l, r) do
      {left, right} = maybe_upcast(l, r)
      {left_tx, right_tx} = maybe_broadcast_bin_args(out.shape, left, right)

      Torchx.unquote(op)(left_tx, right_tx)
      |> Torchx.to_type(to_torch_type(out.type))
      |> to_nx(out)
    end
  end

  @impl true
  def remainder(out, l, r) do
    {left, right} = maybe_upcast(l, r)
    {left_tx, right_tx} = maybe_broadcast_bin_args(out.shape, left, right)

    {device, _} = left_tx

    if l.type == {:u, 64} do
      # We emulate u64 numbers with s64.

      # Numbers smaller than max_s64 are kept as positive s64 and fmod
      # works fine for those.

      remainder_from_positive = Torchx.fmod(left_tx, right_tx)

      # Numbers bigger than max_s64 are kept as negative s64. Consider
      # such s64 number denoted as x. We can decompose x into two
      # positive s64 numbers:
      #
      #   x = max_s64 + rest
      #
      # We can obtain rest as follows:
      #
      #   rest = x - max_s64
      #
      # We also know that the following holds:
      #
      #   x mod y = ((max_s64 mod y) + (rest mod y)) mod y
      #
      # We can safely compute operations on the right hand side using
      # s64 numbers.

      max_s64_tx =
        Nx.Constants.max_finite(:s64, backend: {Nx.BinaryBackend, device: device}) |> from_nx()

      rest_tx = Torchx.subtract(left_tx, max_s64_tx)

      remainder_from_negative =
        Torchx.fmod(
          Torchx.add(
            Torchx.fmod(rest_tx, right_tx),
            Torchx.fmod(max_s64_tx, right_tx)
          ),
          right_tx
        )

      zero = Torchx.scalar_tensor(0, to_torch_type(l.type), device)

      left_tx
      |> Torchx.less(zero)
      |> Torchx.where(remainder_from_negative, remainder_from_positive)
      |> Torchx.to_type(to_torch_type(out.type))
      |> to_nx(out)
    else
      left_tx
      |> Torchx.fmod(right_tx)
      |> Torchx.to_type(to_torch_type(out.type))
      |> to_nx(out)
    end
  end

  defp maybe_upcast(%T{type: t} = left, %T{type: t} = right),
    do: {left, right}

  defp maybe_upcast(left, right) do
    type = Nx.Type.merge(left.type, right.type)
    {Nx.as_type(left, type), Nx.as_type(right, type)}
  end

  defp maybe_broadcast_bin_args(_out_shape, %{shape: {}} = l, r), do: {from_nx(l), from_nx(r)}
  defp maybe_broadcast_bin_args(_out_shape, l, %{shape: {}} = r), do: {from_nx(l), from_nx(r)}

  defp maybe_broadcast_bin_args(out_shape, l, r) do
    l_tx =
      case l.shape do
        ^out_shape ->
          from_nx(l)

        _ ->
          l |> from_nx() |> Torchx.broadcast_to(out_shape)
      end

    r_tx =
      case r.shape do
        ^out_shape -> from_nx(r)
        _ -> r |> from_nx() |> Torchx.broadcast_to(out_shape)
      end

    {l_tx, r_tx}
  end

  for op <- [:bitwise_and, :bitwise_or, :bitwise_xor] do
    @impl true
    def unquote(op)(out, l, r) do
      {left, right} = maybe_upcast(l, r)

      %T{type: {_, size_left}} = left
      %T{type: {_, size_right}} = right

      if size_left >= size_right do
        Torchx.unquote(op)(from_nx(left), from_nx(right))
      else
        Torchx.unquote(op)(from_nx(right), from_nx(left))
      end
      |> to_nx(out)
    end
  end

  @impl true
  def expm1(%{type: {:c, _}}, _t) do
    raise ArithmeticError, "Torchx does not support complex values for expm1"
  end

  @impl true
  def log1p(%{type: {:c, _}}, _t) do
    raise ArithmeticError, "Torchx does not support complex values for log1p"
  end

  @impl true
  def erf_inv(out, %{type: {:f, 16}} = tensor) do
    tensor
    |> from_nx()
    |> Torchx.to_type(:float)
    |> Torchx.erf_inv()
    |> Torchx.to_type(:half)
    |> to_nx(out)
  end

  unary_ops =
    [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh] ++
      [:tanh, :acos, :asin, :atan, :acosh, :asinh, :atanh, :sqrt, :rsqrt] ++
      [:erf, :erfc, :erf_inv, :abs, :bitwise_not, :ceil, :floor, :negate, :round, :sign] ++
      [:logical_not, :cbrt, :is_nan, :is_infinity]

  for op <- unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      Torchx.unquote(op)(from_nx(tensor)) |> to_nx(out)
    end
  end

  @impl true
  def conjugate(out, tensor) do
    tensor
    |> from_nx()
    |> Torchx.conjugate()
    |> Torchx.to_type(to_torch_type(out.type))
    |> to_nx(out)
  end

  @impl true
  def real(out, tensor) do
    get_complex_component(out, tensor, :real)
  end

  @impl true
  def imag(out, tensor) do
    get_complex_component(out, tensor, :imag)
  end

  defp get_complex_component(out, tensor, component) do
    tensor
    |> from_nx()
    |> get_complex_component_tx(tensor.shape, component)
    |> to_nx(out)
  end

  defp get_complex_component_tx(tensor_tx, shape, component) when component in [:real, :imag] do
    as_real = Torchx.view_as_real(tensor_tx)

    as_real_shape = Torchx.shape(as_real)

    starts =
      if component == :real do
        List.duplicate(0, tuple_size(as_real_shape))
      else
        0
        |> List.duplicate(tuple_size(as_real_shape))
        |> List.replace_at(-1, 1)
      end

    lengths = as_real_shape |> Tuple.to_list() |> List.replace_at(-1, 1)
    strides = List.duplicate(1, tuple_size(as_real_shape))

    as_real
    |> torchx_slice(as_real_shape, shape, starts, lengths, strides)
    |> Torchx.reshape(shape)
  end

  @impl true
  def fft(out, tensor, opts) do
    length = opts[:length]

    tensor
    |> from_nx()
    |> Torchx.fft(length)
    |> to_nx(out)
  end

  @impl true
  def ifft(out, tensor, opts) do
    length = opts[:length]

    tensor
    |> from_nx()
    |> Torchx.ifft(length)
    |> to_nx(out)
  end

  @impl true
  def dot(
        %T{type: out_type} = out,
        %T{type: left_type} = left,
        left_axes,
        left_batched_axes,
        %T{type: right_type} = right,
        right_axes,
        right_batched_axes
      ) do
    # since these lists aren't that big, we're probably fine
    # doing this but optimization is welcome
    left_axes = translate_to_inner_axes(left_axes, left_batched_axes)
    right_axes = translate_to_inner_axes(right_axes, right_batched_axes)

    left_tx = from_nx(left)
    right_tx = from_nx(right)

    Torchx.tensordot(
      to_typed_ref(left_tx, left_type, out_type),
      to_typed_ref(right_tx, right_type, out_type),
      left_axes,
      left_batched_axes,
      right_axes,
      right_batched_axes
    )
    |> to_nx(out)
  end

  defp translate_to_inner_axes(axes, []), do: axes
  # sort the batched_axes so we don't need to keep translating them as well
  defp translate_to_inner_axes(axes, batched_axes),
    do: do_translate_to_inner_axes(axes, Enum.sort(batched_axes, :desc))

  defp do_translate_to_inner_axes(axes, []), do: axes

  defp do_translate_to_inner_axes(axes, [batch_axis | batch_axes]) do
    axes
    |> Enum.map(fn axis -> if axis > batch_axis, do: axis - 1, else: axis end)
    |> translate_to_inner_axes(batch_axes)
  end

  @impl true
  def cholesky(%T{} = out, %T{} = t) do
    t
    |> from_nx()
    |> Torchx.cholesky()
    |> to_nx(out)
  end

  @impl true
  def eigh({eigenvals, eigenvecs}, tensor, _opts) do
    {q, r} =
      tensor
      |> from_nx()
      |> Torchx.to_type(to_torch_type(eigenvecs.type))
      |> Torchx.eigh()

    {to_nx(q, eigenvals), to_nx(r, eigenvecs)}
  end

  @impl true
  def qr({q_holder, r_holder}, tensor, opts) do
    {q, r} =
      tensor
      |> from_nx()
      |> Torchx.to_type(to_torch_type(q_holder.type))
      |> Torchx.qr(opts[:mode] == :reduced)

    {to_nx(q, q_holder), to_nx(r, r_holder)}
  end

  @impl true
  def svd({u_holder, s_holder, vt_holder}, tensor, opts) do
    {u, s, vt} =
      tensor
      |> from_nx()
      |> Torchx.to_type(to_torch_type(u_holder.type))
      |> Torchx.svd(opts[:full_matrices?] == true)

    {to_nx(u, u_holder), to_nx(s, s_holder), to_nx(vt, vt_holder)}
  end

  @impl true
  def lu(
        {p_holder, %{type: output_type} = l_holder, %{type: output_type} = u_holder},
        tensor,
        _opts
      ) do
    out_type = to_torch_type(output_type)

    {p_tx, l_tx, u_tx} =
      tensor
      |> from_nx()
      |> Torchx.to_type(out_type)
      |> Torchx.lu()

    p_type = to_torch_type(p_holder.type)

    # p_type can be an integer type, but we can
    # demote the floating-point torch tensor
    # without any loss because p_tx is a tensor
    # of zeros or ones only

    p =
      p_tx
      |> Torchx.to_type(p_type)
      |> to_nx(p_holder)

    l = to_nx(l_tx, l_holder)
    u = to_nx(u_tx, u_holder)

    {p, l, u}
  end

  @impl true
  def pad(out, tensor, constant, input_config) do
    config =
      input_config
      |> Enum.map(fn {a, b, c} ->
        if c < 0 do
          raise ArgumentError, "{#{a}, #{b}, #{c}} padding is not supported"
        end

        [max(a, 0), max(b, 0)]
      end)
      |> Enum.reverse()
      |> List.flatten()

    tensor
    |> from_nx()
    |> pad_internal(input_config)
    |> slice_negative_padding(input_config)
    |> Torchx.pad(from_nx(constant), config)
    |> to_nx(out)
  end

  defp pad_internal(t_tx, input_config) do
    pad_sizes = Enum.map(input_config, &elem(&1, 2))

    if Enum.all?(pad_sizes, &(&1 == 0)) do
      t_tx
    else
      pads = Enum.reduce(pad_sizes, [], fn size, acc -> [0, size, 0, 0 | acc] end)

      shape = Torchx.shape(t_tx)
      rank = tuple_size(shape)
      shape_list = Tuple.to_list(shape)
      expanded_shape = shape_list |> Enum.flat_map(&[&1, 1]) |> List.to_tuple()

      shape_after_pad =
        shape_list
        |> Enum.zip_with(pad_sizes, fn size, pad -> size + pad * size end)
        |> List.to_tuple()

      final_sizes =
        Enum.zip_with(shape_list, pad_sizes, fn size, pad -> size + pad * (size - 1) end)

      t_tx
      |> Torchx.reshape(expanded_shape)
      |> pad_zero(pads)
      |> Torchx.reshape(shape_after_pad)
      |> torchx_slice(
        shape_after_pad,
        List.to_tuple(final_sizes),
        List.duplicate(0, rank),
        final_sizes,
        List.duplicate(1, rank)
      )
    end
  end

  defp pad_zero({device, _} = t_tx, pads) do
    pad_value = Torchx.scalar_tensor(0, Torchx.scalar_type(t_tx), device)
    Torchx.pad(t_tx, pad_value, pads)
  end

  defp slice_negative_padding(t_tx, input_config) do
    if Enum.any?(input_config, fn {pre, post, _} -> pre < 0 or post < 0 end) do
      shape = Torchx.shape(t_tx)

      {starts, lengths} =
        input_config
        |> Enum.with_index(fn {pre, post, _inner}, axis ->
          start =
            if pre < 0 do
              -pre
            else
              0
            end

          axis_size = elem(shape, axis)

          len =
            if post < 0 do
              axis_size + post - start
            else
              axis_size - start
            end

          {start, len}
        end)
        |> Enum.unzip()

      strides = List.duplicate(1, tuple_size(shape))
      torchx_slice(t_tx, shape, List.to_tuple(lengths), starts, lengths, strides)
    else
      t_tx
    end
  end

  @impl true
  def triangular_solve(%T{} = out, %T{} = a, %T{} = b, opts) do
    transform = opts[:transform_a]
    upper = !opts[:lower]
    left_side = opts[:left_side]

    # We can support this eventually, but we'd need
    # to apply the same permutations BinaryBackend applies,
    # because this is not natively supported by libtorch
    unless left_side do
      raise ArgumentError, "left_side: false option not supported in Torchx"
    end

    batched_a_shape =
      case a.shape do
        {m, m} -> {1, m, m}
        shape -> shape
      end

    batched_b_shape =
      case b.shape do
        {n} -> {1, n, 1}
        {m, n} -> {1, m, n}
        shape -> shape
      end

    out_type = to_torch_type(out.type)

    a_tx =
      a
      |> from_nx()
      |> Torchx.reshape(batched_a_shape)
      |> Torchx.to_type(out_type)

    check_singular_matrix(a_tx)

    b_tx = b |> from_nx() |> Torchx.reshape(batched_b_shape) |> Torchx.to_type(out_type)

    a_tx
    |> Torchx.triangular_solve(b_tx, transform == :transpose, upper)
    |> Torchx.reshape(out.shape)
    |> Torchx.to_nx()
  end

  @impl true
  def solve(%T{type: type} = out, a, b) do
    a_tx = a |> from_nx |> Torchx.to_type(to_torch_type(type))
    b_tx = b |> from_nx |> Torchx.to_type(to_torch_type(type))

    check_singular_matrix(a_tx)

    a_tx
    |> Torchx.solve(b_tx)
    |> to_nx(out)
  end

  defp check_singular_matrix(tensor) do
    eps = 1.0e-10 |> Nx.tensor() |> Torchx.from_nx()

    # We need to manually validate if the A tensor is singular
    # (i.e. the tensor has its determinant equal to 0)
    # Otherwise, an exception will be thrown by libtorch.
    #
    # a non-zero eps value is chosen so we can account for possible rounding errors
    # in the determinant calculation
    is_singular =
      tensor
      |> Torchx.determinant()
      |> Torchx.abs()
      |> Torchx.less_equal(eps)
      |> Torchx.all()
      |> Torchx.to_nx()
      |> Nx.to_number()
      |> Kernel.==(1)

    if is_singular do
      raise ArgumentError, "can't solve for singular matrix"
    end
  end

  @impl true
  def sort(%T{} = out, %T{} = t, opts) do
    axis = opts[:axis]
    descending = opts[:direction] == :desc

    t
    |> from_nx()
    |> Torchx.sort(axis, descending)
    |> to_nx(out)
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    on_true = Nx.as_type(on_true, Nx.type(out))
    on_false = Nx.as_type(on_false, Nx.type(out))
    on_true_torch = from_nx(on_true)
    on_false_torch = from_nx(on_false)

    # Use logical_not to convert any tensor to a boolean tensor
    # because of that, we have to swap true/false tensor
    pred
    |> from_nx()
    |> Torchx.logical_not()
    |> Torchx.where(on_false_torch, on_true_torch)
    |> to_nx(out)
  end

  @impl true
  def clip(%T{} = out, %T{} = t, %T{} = min, %T{} = max) do
    t
    |> Nx.as_type(out.type)
    |> from_nx()
    |> Torchx.clip(from_nx(min), from_nx(max))
    |> to_nx(out)
  end

  @impl true
  def reduce_max(out, tensor, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    tensor
    |> from_nx()
    |> Torchx.amax(axes, keep_axes)
    |> to_nx(out)
  end

  @impl true
  def reduce_min(out, tensor, opts) do
    axes = opts[:axes] || []
    keep_axes = opts[:keep_axes] || false

    tensor
    |> from_nx()
    |> Torchx.amin(axes, keep_axes)
    |> to_nx(out)
  end

  @impl true
  def conv(%T{type: type} = out, t, k, opts) do
    unsupported_option!(opts, :batch_group_size, 1)

    input_dilation = opts[:input_dilation]

    input_inner_pads = [{0, 0, 0}, {0, 0, 0}] ++ Enum.map(input_dilation, &{0, 0, &1 - 1})

    padding = opts[:padding]
    strides = opts[:strides]
    kernel_dilation = opts[:kernel_dilation]
    feature_groups = opts[:feature_group_size]

    permute = fn tensor, permutation ->
      if permutation != nil do
        Torchx.permute(tensor, permutation)
      else
        tensor
      end
    end

    input_permutation = opts[:input_permutation]
    kernel_permutation = opts[:kernel_permutation]

    output_permutation =
      case opts[:output_permutation] do
        nil ->
          nil

        l ->
          # The permutation that Nx.Shape expects is actually the reverse permutation
          # for the given input
          l |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
      end

    pad_config = flatten_padding(padding)

    k_tx =
      k
      |> from_nx()
      |> Torchx.to_type(to_torch_type(type))
      |> permute.(kernel_permutation)

    t
    |> from_nx()
    |> permute.(input_permutation)
    |> pad_internal(input_inner_pads)
    |> pad_zero(pad_config)
    |> Torchx.to_type(to_torch_type(type))
    |> do_conv(k_tx, strides, kernel_dilation, feature_groups, type)
    |> permute.(output_permutation)
    |> to_nx(out)
  end

  defp do_conv(t_tx, k_tx, strides, kernel_dilation, feature_groups, {:c, _} = type) do
    # Torch doesn't support complex inputs,
    # so we rely on the fact that a convolution is basically
    # a sliding dot product. We can then decompose the dot product
    # of a complex-valued pair of tensors into dot products involving
    # their real and imaginary components.

    # For example, given a tensor v = [a1+b1i c1+d1i] and a
    # kernel k = [a2+b2i c2+d2i], we can expand the dot product into
    # (a1+b1i)(a2+b2i) + (c1+d1i)(c2+d2i)
    # = (a1a2 - b1b2) + (a1b2+a2b1)i + (c1c2 - d1d2) + (c1d2+c2d1)i
    # = (a1a2 + c1c2) - (b1b2 + d1d2) + i[(a1b2 + c1d2) + (a2b1  + c2d1)]
    # = ([a1 c1].[a2 c2] - [b1 d1].[b2 d2]) + i([a1 c1].[b2 d2] + [a2 c2].[b1 d1])
    # = (real(v).real(k) - imag(v).imag(k)) + i(real(v).imag(k) + imag(v).real(k))

    # With the result above, we can turn z = conv(t, k) where either t or k are complex
    # into:
    # real_part = conv(real(t), real(k)) - conv(imag(t), imag(k))
    # imag_part = conv(real(t), imag(k)) + conv(imag(t), real(k))
    # z = complex(real_part, imag_part)

    t_shape = Torchx.shape(t_tx)
    k_shape = Torchx.shape(k_tx)

    real_t = get_complex_component_tx(t_tx, t_shape, :real)
    imag_t = get_complex_component_tx(t_tx, t_shape, :imag)
    real_k = get_complex_component_tx(k_tx, k_shape, :real)
    imag_k = get_complex_component_tx(k_tx, k_shape, :imag)

    real_type = type |> Nx.Type.to_real() |> to_torch_type()

    real_part =
      Torchx.subtract(
        do_conv(real_t, real_k, strides, kernel_dilation, feature_groups, real_type),
        do_conv(imag_t, imag_k, strides, kernel_dilation, feature_groups, real_type)
      )

    {device, _} =
      imag_part =
      Torchx.add(
        do_conv(real_t, imag_k, strides, kernel_dilation, feature_groups, real_type),
        do_conv(imag_t, real_k, strides, kernel_dilation, feature_groups, real_type)
      )

    i = Torchx.scalar_tensor({0.0, 1.0}, :complex, {device, -1})

    imag_part
    |> Torchx.multiply(i)
    |> Torchx.add(real_part)
  end

  defp do_conv(t_tx, k_tx, strides, kernel_dilation, feature_groups, _type) do
    Torchx.conv(t_tx, k_tx, strides, [0], kernel_dilation, false, feature_groups)
  end

  @impl true
  def window_max(out, tensor, window_dims_tuple, opts) do
    window_op(
      out,
      tensor,
      window_dims_tuple,
      opts,
      Nx.Constants.min(tensor.type) |> from_nx(),
      &Torchx.amax(&1, &2, false)
    )
  end

  @impl true
  def window_min(out, tensor, window_dims_tuple, opts) do
    window_op(
      out,
      tensor,
      window_dims_tuple,
      opts,
      Nx.Constants.max(tensor.type) |> from_nx(),
      &Torchx.amin(&1, &2, false)
    )
  end

  @impl true
  def window_sum(out, tensor, window_dims_tuple, opts) do
    window_op(out, tensor, window_dims_tuple, opts, 0, &Torchx.sum(&1, &2, false))
  end

  @impl true
  def window_product(out, tensor, window_dims_tuple, opts) do
    window_op(out, tensor, window_dims_tuple, opts, 1, fn tensor, axes ->
      aggregate_over_axes(tensor, axes, false, &Torchx.product/3)
    end)
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, window_dims_tuple, opts) do
    window_scatter_function(
      &Nx.argmin(&1, axis: -1, tie_break: :high),
      out,
      tensor,
      source,
      init_value,
      window_dims_tuple,
      opts
    )
  end

  @impl true
  def window_scatter_max(out, tensor, source, init_value, window_dims_tuple, opts) do
    window_scatter_function(
      &Nx.argmax(&1, axis: -1),
      out,
      tensor,
      source,
      init_value,
      window_dims_tuple,
      opts
    )
  end

  defp window_scatter_function(function, out, tensor, source, init_value, window_dims_tuple, opts) do
    unfold_flat = fn tensor ->
      {device, _} = t_tx = from_nx(tensor)
      window_dilations = List.duplicate(1, tuple_size(window_dims_tuple))
      pad_constant = Torchx.scalar_tensor(0, Torchx.scalar_type(t_tx), device)

      unfolded =
        t_tx
        |> unfold_windows(
          opts[:padding],
          pad_constant,
          window_dims_tuple,
          opts[:strides],
          window_dilations,
          to_torch_type(out.type)
        )
        |> Torchx.to_nx()

      {to_keep, to_flatten} =
        unfolded
        |> Map.get(:shape)
        |> Tuple.to_list()
        |> Enum.split(-tuple_size(window_dims_tuple))

      flat_shape =
        to_keep
        |> List.to_tuple()
        |> then(&Tuple.insert_at(&1, tuple_size(&1), Enum.product(to_flatten)))

      Nx.reshape(unfolded, flat_shape)
    end

    arg_idx =
      tensor
      |> then(unfold_flat)
      |> then(function)

    indices_to_flatten =
      tensor
      |> Nx.axes()
      |> Enum.map(fn axis ->
        tensor
        |> Nx.shape()
        |> Nx.iota(axis: axis, backend: Torchx.Backend)
        |> then(unfold_flat)
        |> Nx.take_along_axis(Nx.new_axis(arg_idx, -1), axis: -1)
      end)
      |> Nx.concatenate(axis: -1)

    num_axes = tuple_size(out.shape)
    num_rows = div(Nx.size(indices_to_flatten), num_axes)
    indices = Nx.reshape(indices_to_flatten, {num_rows, num_axes})

    flat_source = Nx.flatten(source)

    init_value
    |> Nx.backend_transfer(Torchx.Backend)
    |> Nx.broadcast(out.shape)
    |> Nx.indexed_add(indices, flat_source)
    |> Nx.as_type(out.type)
  end

  defp window_op(out, tensor, window_dims_tuple, opts, pad_constant, reduce_fun)
       when is_function(reduce_fun, 2) do
    {device, _} = t_tx = from_nx(tensor)

    pad_constant =
      if is_number(pad_constant) do
        Torchx.scalar_tensor(pad_constant, Torchx.scalar_type(t_tx), device)
      else
        pad_constant
      end

    t_tx =
      unfold_windows(
        t_tx,
        opts[:padding],
        pad_constant,
        window_dims_tuple,
        opts[:strides],
        opts[:window_dilations],
        to_torch_type(out.type)
      )

    axes =
      Enum.map(tuple_size(window_dims_tuple)..1//-1, fn axis ->
        tuple_size(Torchx.shape(t_tx)) - axis
      end)

    reduce_fun
    |> apply([t_tx, axes])
    |> Torchx.reshape(out.shape)
    |> Torchx.to_type(to_torch_type(out.type))
    |> to_nx(out)
  end

  defp unfold_windows(
         {device, _} = tensor,
         padding,
         pad_constant,
         window_dims_tuple,
         strides,
         window_dilations,
         output_type
       ) do
    padding = flatten_padding(padding)
    padded = Torchx.pad(tensor, pad_constant, padding)
    window_pad_config = Enum.map(window_dilations, &{0, 0, &1 - 1})

    window =
      1
      |> Torchx.scalar_tensor(:bool, device)
      |> Torchx.broadcast_to(window_dims_tuple)
      |> pad_internal(window_pad_config)

    window_shape = Torchx.shape(window)

    {t_tx, _} =
      for {window_dim, stride} <- Enum.zip(Tuple.to_list(window_shape), strides),
          reduce: {padded, 0} do
        {t_tx, dim} ->
          {Torchx.unfold(t_tx, dim, window_dim, stride), dim + 1}
      end

    Torchx.where(
      window,
      Torchx.to_type(t_tx, output_type),
      pad_constant
    )
  end

  defp flatten_padding(padding) do
    Enum.reduce(padding, [], fn {a, b}, acc -> [a, b | acc] end)
  end

  @impl true
  # u64 is emulated with s64
  def bitcast(%{type: {:s, 64}} = out, %T{type: {:u, 64}, data: data}) do
    %{out | data: data}
  end

  def bitcast(%{type: {:u, 64}} = out, %T{type: {:s, 64}, data: data}) do
    %{out | data: data}
  end

  # u16/u32 are double the size of s16/u32
  def bitcast(%{type: {:u, bit}} = out, %T{type: {:s, bit}, data: %TB{ref: {device, _}}} = tensor)
      when bit in [16, 32] do
    output_size = 2 * bit

    blob =
      for <<x::size(bit)-signed-native <- to_binary(tensor, Nx.size(tensor))>>,
        into: <<>>,
        do: <<x::size(output_size)-signed-native>>

    blob
    |> Torchx.from_blob(out.shape, to_torch_type(out.type), device)
    |> to_nx(out)
  end

  # s16/s32 are half the size of s16/u32 but that's handled in to_binary
  def bitcast(out, %T{data: %TB{ref: {device, _}}} = tensor) do
    tensor
    |> to_binary(Nx.size(tensor))
    |> Torchx.from_blob(out.shape, to_torch_type(out.type), device)
    |> to_nx(out)
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  if Application.compile_env(:torchx, :add_backend_on_inspect, true) do
    defp maybe_add_signature(result, %T{data: %TB{ref: {device, _}}}) do
      Inspect.Algebra.concat([
        "Torchx.Backend(#{device})",
        Inspect.Algebra.line(),
        result
      ])
    end
  else
    defp maybe_add_signature(result, _tensor) do
      result
    end
  end

  ## Conversions

  @doc false
  def from_nx(%T{data: %TB{ref: device_ref}}), do: device_ref
  def from_nx(%T{} = tensor), do: Nx.backend_transfer(tensor, TB) |> from_nx()

  @doc false
  def to_nx({device, ref} = device_ref, %T{type: type, shape: shape} = t)
      when is_atom(device) and is_reference(ref) do
    cast_to_byte = type == {:u, 8} and Torchx.scalar_type(device_ref) == :bool

    # This cast is added because if the u8 type is bool, we get
    # boolean algebra instead of common arithmetic rules when
    # operating on a seemingly normal u8 tensor
    t_tx =
      if cast_to_byte do
        Torchx.to_type(device_ref, :byte)
      else
        device_ref
      end

    %{t | data: %__MODULE__{ref: check_shape_and_type!(t_tx, shape, type)}}
  end

  @doc false
  def from_torch_type(:char), do: {:s, 8}
  def from_torch_type(:byte), do: {:u, 8}
  def from_torch_type(:bool), do: {:u, 8}
  def from_torch_type(:short), do: {:s, 16}
  def from_torch_type(:int), do: {:s, 32}
  def from_torch_type(:long), do: {:s, 64}
  def from_torch_type(:brain), do: {:bf, 16}
  def from_torch_type(:half), do: {:f, 16}
  def from_torch_type(:float), do: {:f, 32}
  def from_torch_type(:double), do: {:f, 64}
  def from_torch_type(:complex), do: {:c, 64}
  def from_torch_type(:complex_double), do: {:c, 128}

  defp to_torch_type(nx_type, hint \\ "")
  defp to_torch_type({:u, 8}, _), do: :byte
  defp to_torch_type({:u, 16}, _), do: :int
  defp to_torch_type({:u, 32}, _), do: :long
  defp to_torch_type({:u, 64}, _), do: :long
  defp to_torch_type({:s, 8}, _), do: :char
  defp to_torch_type({:s, 16}, _), do: :short
  defp to_torch_type({:s, 32}, _), do: :int
  defp to_torch_type({:s, 64}, _), do: :long
  defp to_torch_type({:bf, 16}, _), do: :brain
  defp to_torch_type({:f, 16}, _), do: :half
  defp to_torch_type({:f, 32}, _), do: :float
  defp to_torch_type({:f, 64}, _), do: :double
  defp to_torch_type({:c, 64}, _), do: :complex
  defp to_torch_type({:c, 128}, _), do: :complex_double

  if Application.compile_env(:torchx, :check_shape_and_type, false) do
    defp check_shape_and_type!(device_ref, shape, type) do
      current_type = Torchx.scalar_type(device_ref) |> from_torch_type()

      case {current_type, type} do
        {{:s, 32}, {:u, 16}} ->
          :ok

        {{:s, 64}, {:u, 32}} ->
          :ok

        {{:s, 64}, {:u, 64}} ->
          :ok

        _ when current_type != type ->
          raise "type mismatch in Torchx: expected #{inspect(type)}, got: #{inspect(current_type)}. " <>
                  "Please report this bug"

        _ ->
          :ok
      end

      current_shape = Torchx.shape(device_ref)

      if current_shape != shape do
        raise "shape mismatch in Torchx: expected #{inspect(shape)}, got: #{inspect(current_shape)}. " <>
                "Please report this bug"
      end

      device_ref
    end
  else
    defp check_shape_and_type!(device_ref, _, _), do: device_ref
  end

  ## Helpers

  defp to_number(n) when is_number(n), do: n
  defp to_number(%T{} = t), do: t |> from_nx() |> Torchx.item()

  defp to_typed_ref(tensor, expected_type, expected_type),
    do: tensor

  defp to_typed_ref(tensor, _ref_type, expected_type),
    do: Torchx.to_type(tensor, to_torch_type(expected_type))

  defp device_option(nil), do: Torchx.default_device()
  defp device_option(backend_opts), do: backend_opts[:device] || Torchx.default_device()

  defp unsupported_option!(opts, key, acceptable_default) do
    if opts[key] != nil and opts[key] != acceptable_default do
      raise "#{inspect(key)} option with #{inspect(opts[key])} is not supported in #{caller()}"
    end
  end

  defp caller(depth \\ 3) do
    {module, func, arity, [file: _file, line: _line]} =
      Process.info(self(), :current_stacktrace) |> elem(1) |> Enum.fetch!(depth)

    "#{inspect(module)}.#{func}/#{arity - 1}"
  end

  ## Functionality we can't provide

  not_possible =
    [count_leading_zeros: 2, population_count: 2] ++
      [map: 4, reduce: 5, window_reduce: 6]

  for {fun, arity} <- not_possible do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(fun)(unquote_splicing(args)) do
      raise "operation #{unquote(fun)} is not supported on Torchx.Backend"
    end
  end
end

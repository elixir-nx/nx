defmodule Candlex.Backend do
  @moduledoc """
  An opaque Nx backend with bindings to candle.
  """

  defstruct [:device, :resource]

  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias Candlex.Native

  @device_cuda :cuda
  @device_cpu :cpu

  @impl true
  def init(opts) do
    Keyword.validate!(opts, [:device])
  end

  # Creation

  @impl true
  def constant(%T{} = tensor, scalar, backend_options) do
    tensor
    |> Nx.BinaryBackend.constant(scalar, [])
    |> Nx.BinaryBackend.backend_transfer(__MODULE__, backend_options)
  end

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, backend_options) do
    binary
    |> Native.from_binary(to_candle_dtype(type), shape, device_option(backend_options))
    |> unwrap!()
    |> to_nx(tensor)
  end

  @impl true
  def iota(%T{shape: {}} = out, nil, backend_options) do
    constant(out, 0, backend_options)
  end

  def iota(%T{shape: shape, type: type} = out, nil, backend_options) do
    Native.arange(0, Nx.size(shape), to_candle_dtype(type), shape, device_option(backend_options))
    |> unwrap!()
    |> to_nx(out)
  end

  def iota(%T{shape: shape, type: type} = out, axis, backend_options) do
    # Build in one dimension, then broadcast
    axis_size = elem(shape, axis)

    Native.arange(
      0,
      axis_size,
      to_candle_dtype(type),
      Tuple.duplicate(1, Nx.rank(shape)) |> put_elem(axis, axis_size),
      device_option(backend_options)
    )
    |> unwrap!()
    |> Native.broadcast_to(shape)
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def eye(%T{shape: shape, type: type} = _out, backend_options) do
    iota = Nx.iota(shape, backend: {__MODULE__, backend_options})

    Nx.equal(Nx.tril(iota), Nx.triu(iota))
    |> Nx.as_type(type)
  end

  # Backend

  @impl true
  def backend_transfer(tensor, backend, backend_options) do
    if backend == __MODULE__ && same_device?(tensor, device_option(backend_options)) do
      tensor
    else
      try do
        backend_copy(tensor, backend, backend_options)
      after
        backend_deallocate(tensor)
      end
    end
  end

  @impl true
  def backend_copy(%T{} = tensor, Candlex.Backend, backend_options) do
    tensor
    |> from_nx()
    |> Native.to_device(device_option(backend_options))
    |> unwrap!()
    |> to_nx(tensor)
  end

  def backend_copy(%T{} = tensor, backend, backend_options) do
    backend.from_binary(tensor, to_binary(tensor), backend_options)
  end

  @impl true
  def backend_deallocate(%T{} = _tensor) do
    true
  end

  # Conversion

  @impl true
  def to_binary(tensor, _limit \\ nil) do
    # TODO: don't ignore limit

    from_nx(tensor)
    |> Native.to_binary()
    |> unwrap!()
  end

  # Aggregates

  @impl true
  def all(%T{} = out, %T{} = tensor, opts) do
    case opts[:axes] do
      nil ->
        from_nx(tensor)
        |> Native.all()

      axes ->
        from_nx(tensor)
        |> Native.all_within_dims(axes, opts[:keep_axes])
    end
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def any(%T{} = out, %T{} = tensor, opts) do
    case opts[:axes] do
      nil ->
        from_nx(tensor)
        |> Native.any()

      axes ->
        from_nx(tensor)
        |> Native.any_within_dims(axes, opts[:keep_axes])
    end
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def sum(%T{type: out_type} = out, %T{} = t, opts) do
    axes = opts[:axes] || Nx.axes(t)
    keep_axes = opts[:keep_axes] || false

    t
    |> from_nx()
    |> Native.sum(axes, keep_axes)
    |> unwrap!()
    |> Native.to_type(to_candle_dtype(out_type))
    |> unwrap!()
    |> to_nx(out)
  end

  for op <- [:argmax, :argmin] do
    @impl true
    def unquote(op)(%T{} = out, %T{shape: {}} = _tensor, _opts) do
      out
      |> constant(0, [])
    end

    def unquote(op)(%T{type: type} = out, %T{} = tensor, opts) do
      axis = opts[:axis] || -1
      keep_axis = opts[:keep_axis] || false

      tensor
      |> from_nx()
      |> Native.unquote(op)(axis, keep_axis)
      |> unwrap!()
      # candle argmax/argmin changes to u32
      |> Native.to_type(to_candle_dtype(type))
      |> unwrap!()
      |> to_nx(out)
    end
  end

  @impl true
  def reduce_max(%T{} = out, %T{shape: {}} = tensor, _opts) do
    out
    |> from_binary(to_binary(tensor), [])
  end

  def reduce_max(%T{} = out, %T{} = tensor, opts) do
    axis =
      case opts[:axes] do
        nil -> 0
        [] -> 0
        [axis] -> axis
        axes -> raise "doesn't support axes option with more than 1 axis, '#{inspect(axes)}'"
      end

    keep_axis = opts[:keep_axes] || false

    tensor
    |> from_nx()
    |> Native.reduce_max(axis, keep_axis)
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def reduce_min(%T{} = out, %T{shape: {}} = tensor, _opts) do
    out
    |> from_binary(to_binary(tensor), [])
  end

  def reduce_min(%T{} = out, %T{} = tensor, opts) do
    axis =
      case opts[:axes] do
        nil -> 0
        [] -> 0
        [axis] -> axis
        axes -> raise "doesn't support axes option with more than 1 axis, '#{inspect(axes)}'"
      end

    keep_axis = opts[:keep_axes] || false

    tensor
    |> from_nx()
    |> Native.reduce_min(axis, keep_axis)
    |> unwrap!()
    |> to_nx(out)
  end

  # Element-wise

  @impl true
  def clip(%T{} = out, %T{} = t, %T{} = min, %T{} = max) do
    [t, min, max] = maybe_upcast([t, min, max])

    t
    |> from_nx()
    |> Native.clamp(from_nx(min), from_nx(max))
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def select(%T{shape: shape, type: type} = out, pred, on_true, on_false) do
    on_true =
      on_true
      |> from_nx()
      |> Native.to_type(to_candle_dtype(type))
      |> unwrap!()
      |> Native.broadcast_to(shape)
      |> unwrap!()

    on_false =
      on_false
      |> from_nx()
      |> Native.to_type(to_candle_dtype(type))
      |> unwrap!()
      |> Native.broadcast_to(shape)
      |> unwrap!()

    pred
    |> from_nx()
    |> Native.where_cond(on_true, on_false)
    |> unwrap!()
    |> to_nx(out)
  end

  # Binary ops

  for op <- [:add, :divide, :max, :min, :multiply, :subtract] do
    @impl true
    def unquote(op)(%T{} = out, %T{} = left, %T{} = right) do
      {left, right} = maybe_transfer_device(left, right)
      {left, right} = maybe_upcast(left, right)

      from_nx(left)
      |> Native.unquote(op)(from_nx(right))
      |> unwrap!()
      |> to_nx(out)
    end
  end

  for op <- [:atan2, :pow, :quotient, :remainder] do
    @impl true
    def unquote(op)(%T{} = out, %T{} = left, %T{} = right) do
      {left, right} = maybe_upcast(left, right)
      {left, right} = maybe_broadcast_bin_args(out.shape, left, right)

      left
      |> Native.unquote(op)(right)
      |> unwrap!()
      |> to_nx(out)
    end
  end

  for op <- [
        :bitwise_and,
        :bitwise_or,
        :bitwise_xor,
        :equal,
        :greater,
        :greater_equal,
        :left_shift,
        :less,
        :less_equal,
        :logical_and,
        :logical_or,
        :logical_xor,
        :not_equal,
        :right_shift
      ] do
    @impl true
    def unquote(op)(%T{} = out, %T{} = left, %T{} = right) do
      {left, right} = maybe_transfer_device(left, right)
      {left, right} = maybe_upcast(left, right)
      {left, right} = maybe_broadcast_bin_args(out.shape, left, right)

      left
      |> Native.unquote(op)(right)
      |> unwrap!()
      # TODO: Do this conditionally or as part of native op
      |> Native.to_type(to_candle_dtype(out.type))
      |> unwrap!()
      |> to_nx(out)
    end
  end

  # Unary ops

  for op <- [
        :abs,
        :acos,
        :acosh,
        :asin,
        :asinh,
        :atan,
        :atanh,
        :bitwise_not,
        :cbrt,
        :ceil,
        :cos,
        :cosh,
        :erf,
        :erfc,
        :erf_inv,
        :exp,
        :expm1,
        :floor,
        :is_infinity,
        :is_nan,
        :log,
        :log1p,
        :negate,
        :round,
        :rsqrt,
        :sigmoid,
        :sign,
        :sin,
        :sinh,
        :sqrt,
        :tan,
        :tanh
      ] do
    @impl true
    def unquote(op)(%T{} = out, %T{} = tensor) do
      tensor
      |> from_nx()
      |> Native.unquote(op)()
      |> unwrap!()
      |> to_nx(out)
    end
  end

  # Indexed

  @impl true
  def gather(%T{} = out, %T{shape: {_}} = tensor, %T{} = indices) do
    tensor
    |> from_nx()
    |> Native.gather(from_nx(Nx.flatten(indices)), 0)
    |> unwrap!()
    |> to_nx(out)
  end

  def gather(%T{} = _out, %T{} = _tensor, %T{} = _indices) do
    raise("unsupported gather for tensor of rank greater than 1")
  end

  @impl true
  def indexed_add(%T{} = out, %T{shape: {_}} = tensor, %T{} = indices, %T{} = updates) do
    {tensor, updates} = maybe_upcast(tensor, updates)

    tensor
    |> from_nx()
    |> Native.index_add(from_nx(Nx.flatten(indices)), from_nx(updates), 0)
    |> unwrap!()
    |> to_nx(out)
  end

  def indexed_add(%T{} = _out, %T{} = _tensor, %T{} = _indices, %T{} = _updates) do
    raise("unsupported indexed_add for tensor of rank greater than 1")
  end

  @impl true
  def put_slice(%T{} = out, %T{} = t, [_ | _] = start_indices, slice) do
    [last_start_index | leading_start_indices] = Enum.reverse(start_indices)

    if Enum.all?(leading_start_indices, fn i -> Nx.equal(i, 0) end) do
      t
      |> from_nx()
      |> Native.slice_scatter(
        from_nx(slice),
        length(start_indices) - 1,
        Nx.to_number(last_start_index)
      )
      |> unwrap!()
      |> to_nx(out)
    else
      raise "put_slice only supports last start index not to be 0 for now"
    end
  end

  @impl true
  def slice(
        %T{shape: _output_shape} = out,
        %T{shape: input_shape} = t,
        starts,
        lengths,
        _strides
      ) do
    t
    |> from_nx()
    |> narrow(starts, lengths, 0, input_shape)
    # TODO: Support strides
    # |> stride(output_shape, lengths, strides)
    |> to_nx(out)
  end

  @impl true
  def take(%T{} = out, %T{} = tensor, %T{} = indexes, axis) do
    if Nx.rank(indexes) > 1 do
      raise "only indexes of rank=1 supported for now"
    end

    tensor
    |> from_nx()
    |> Native.index_select(from_nx(indexes), axis)
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def take_along_axis(%T{} = out, %T{} = tensor, %T{} = indexes, axis) do
    tensor
    |> from_nx()
    |> Native.gather(from_nx(indexes), axis)
    |> unwrap!()
    |> to_nx(out)
  end

  # N-dim

  @impl true
  def concatenate(%T{} = out, tensors, axis) do
    tensors
    |> maybe_upcast()
    |> Enum.map(&from_nx/1)
    |> Native.concatenate(axis)
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def conv(%T{type: out_type} = out, %T{shape: shape} = tensor, %T{} = kernel, opts) do
    # TODO: Support more opts
    unsupported_option!(opts, :batch_group_size, 1)
    unsupported_option!(opts, :feature_group_size, 1)

    # For now we assume:
    # strides = opts[:strides] # [1, 1]
    # padding = opts[:padding] # [{0, 0}, {0, 0}]
    # input_dilation = opts[:input_dilation] # [1, 1]
    # kernel_dilation = opts[:kernel_dilation] # [1, 1]

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

    native_tensor =
      tensor
      |> from_nx()
      |> permute(input_permutation)
      |> Native.to_type(to_candle_dtype(out_type))
      |> unwrap!()

    native_kernel =
      kernel
      |> from_nx()
      |> permute(kernel_permutation)
      |> Native.to_type(to_candle_dtype(out_type))
      |> unwrap!()

    native_result =
      case Nx.rank(shape) do
        3 -> Native.conv1d(native_tensor, native_kernel)
        4 -> Native.conv2d(native_tensor, native_kernel)
        rank -> raise("unsupported conv for tensor of rank #{rank}, only 3 or 4 supported")
      end

    native_result
    |> unwrap!()
    |> permute(output_permutation)
    |> to_nx(out)
  end

  @impl true
  def dot(
        %T{type: _out_type} = out,
        %T{shape: left_shape, type: _left_type} = left,
        [left_axis] = _left_axes,
        [] = _left_batched_axes,
        %T{shape: right_shape, type: _right_type} = right,
        [0] = _right_axes,
        [] = _right_batched_axes
      )
      when tuple_size(left_shape) >= 1 and tuple_size(right_shape) == 1 and
             left_axis == tuple_size(left_shape) - 1 do
    {left, right} = maybe_upcast(left, right)

    from_nx(left)
    |> Native.dot(from_nx(right))
    |> unwrap!()
    |> to_nx(out)
  end

  def dot(
        %T{type: _out_type} = out,
        %T{shape: left_shape, type: _left_type} = left,
        [1] = _left_axes,
        [] = _left_batched_axes,
        %T{shape: right_shape, type: _right_type} = right,
        [0] = _right_axes,
        [] = _right_batched_axes
      )
      when tuple_size(left_shape) == 2 and tuple_size(right_shape) == 2 do
    {left, right} = maybe_upcast(left, right)

    Native.matmul(
      from_nx(left),
      from_nx(right)
    )
    |> unwrap!()
    |> to_nx(out)
  end

  def dot(
        out,
        %T{shape: left_shape} = left,
        [0],
        left_batched_axes,
        right,
        right_axes,
        right_batched_axes
      )
      when tuple_size(left_shape) == 2 do
    dot(
      out,
      left |> Nx.transpose(axes: [1, 0]),
      [1],
      left_batched_axes,
      right,
      right_axes,
      right_batched_axes
    )
  end

  def dot(
        out,
        left,
        left_axes,
        left_batched_axes,
        %T{shape: right_shape} = right,
        [1],
        right_batched_axes
      )
      when tuple_size(right_shape) == 2 do
    dot(
      out,
      left,
      left_axes,
      left_batched_axes,
      right |> Nx.transpose(axes: [1, 0]),
      [0],
      right_batched_axes
    )
  end

  # Shape

  @impl true
  def broadcast(out, %T{} = t, shape, axes) do
    t
    |> maybe_reshape(shape, axes)
    |> from_nx()
    |> Native.broadcast_to(shape)
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def pad(%T{} = out, %T{} = _t, _pad_value, []) do
    out
  end

  def pad(%T{} = out, %T{} = t, %T{shape: {}} = pad_value, [{low, high, 0 = _inner}]) do
    if !Nx.equal(pad_value, 0) do
      raise "only pad_value=0 supported for now"
    end

    t
    |> from_nx()
    |> Native.pad_with_zeros(low, high)
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def reshape(%T{shape: shape} = out, %T{} = t) do
    from_nx(t)
    |> Native.reshape(shape)
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def squeeze(%T{} = out, %T{} = t, axes) do
    # sort the axes desc so we don't have to decrease the axis numbers after each squeeze
    for axis <- Enum.sort(axes, :desc), reduce: from_nx(t) do
      ref ->
        ref
        |> Native.squeeze(axis)
        |> unwrap!()
    end
    |> to_nx(out)
  end

  @impl true
  def transpose(out, %T{} = t, axes) do
    from_nx(t)
    |> Native.permute(axes)
    |> unwrap!()
    |> to_nx(out)
  end

  # Type

  @impl true
  def as_type(%T{type: type} = out, %T{} = t) do
    from_nx(t)
    |> Native.to_type(to_candle_dtype(type))
    |> unwrap!()
    |> to_nx(out)
  end

  @impl true
  def bitcast(out, tensor) do
    out
    |> from_binary(to_binary(tensor), [])
  end

  # Inspect

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  defp maybe_add_signature(result, %T{data: %__MODULE__{device: device, resource: ref}})
       when is_reference(ref) do
    Inspect.Algebra.concat([
      "Candlex.Backend(#{device})",
      Inspect.Algebra.line(),
      result
    ])
  end

  defp narrow(t, [start | starts], [length | lengths], axis, shape) do
    dim = elem(shape, axis)
    start = min(start, dim - length)

    if start == 0 and length == dim do
      # Nothing to narrow at this step
      t
    else
      t
      |> Native.narrow(axis, start, length)
      |> unwrap!()
    end
    |> narrow(starts, lengths, axis + 1, shape)
  end

  defp narrow(t, [], [], _axis, _shape), do: t

  defp maybe_reshape(%T{shape: {}} = t, target_shape, _axes) do
    shape =
      1
      |> List.duplicate(tuple_size(target_shape))
      |> List.to_tuple()

    t
    |> Nx.reshape(shape)
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

    t
    |> Nx.reshape(new_shape)
  end

  defp maybe_upcast(%T{type: t} = left, %T{type: t} = right) do
    {left, right}
  end

  defp maybe_upcast(left, right) do
    type = Nx.Type.merge(left.type, right.type)

    {Nx.as_type(left, type), Nx.as_type(right, type)}
  end

  defp maybe_upcast([first | _] = tensors) do
    type =
      tensors
      |> Enum.reduce(
        first.type,
        fn tensor, type ->
          Nx.Type.merge(type, tensor.type)
        end
      )

    tensors
    |> Enum.map(fn tensor ->
      Nx.as_type(tensor, type)
    end)
  end

  defp maybe_broadcast_bin_args(out_shape, l, r) do
    {
      case l.shape do
        ^out_shape ->
          from_nx(l)

        _ ->
          l |> from_nx() |> Native.broadcast_to(out_shape) |> unwrap!()
      end,
      case r.shape do
        ^out_shape -> from_nx(r)
        _ -> r |> from_nx() |> Native.broadcast_to(out_shape) |> unwrap!()
      end
    }
  end

  defp maybe_transfer_device(
         %T{data: %__MODULE__{device: device}} = l,
         %T{data: %__MODULE__{device: device}} = r
       ) do
    {l, r}
  end

  defp maybe_transfer_device(
         %T{data: %__MODULE__{device: device}} = l,
         %T{data: %__MODULE__{device: _other_device}} = r
       ) do
    {
      l,
      r |> Nx.backend_transfer({__MODULE__, device: device})
    }
  end

  defp maybe_transfer_device(%T{} = l, %T{data: %__MODULE__{device: device}} = r) do
    {
      l |> Nx.backend_transfer({__MODULE__, device: device}),
      r
    }
  end

  defp maybe_transfer_device(%T{data: %__MODULE__{device: device}} = l, %T{} = r) do
    {
      l,
      r |> Nx.backend_transfer({__MODULE__, device: device})
    }
  end

  ## Conversions

  @impl true
  def to_batched(%T{shape: out_shape} = out, %T{shape: shape} = t, opts) do
    leftover = opts[:leftover]
    first_dimension = 0
    batch_size = elem(out_shape, first_dimension)
    axis_total = elem(shape, first_dimension)
    remainder = rem(axis_total, batch_size)
    num_batches = div(axis_total, batch_size)
    native_tensor = from_nx(t)

    cond do
      remainder == 0 ->
        native_tensor
        |> Native.chunk(num_batches)
        |> unwrap!()

      remainder > 0 && leftover == :repeat ->
        [
          native_tensor,
          Native.narrow(native_tensor, first_dimension, 0, batch_size - remainder)
          |> unwrap!()
        ]
        |> Native.concatenate(first_dimension)
        |> unwrap!()
        |> Native.chunk(num_batches + 1)
        |> unwrap!()

      true ->
        raise "not implemented"
    end
    |> Stream.map(&to_nx(&1, out))
  end

  for op <- [
        :cholesky,
        :conjugate,
        :count_leading_zeros,
        :imag,
        :population_count,
        :real
      ] do
    @impl true
    def unquote(op)(_out, _tensor) do
      raise "unsupported Candlex.Backend.#{unquote(op)} function"
    end
  end

  for op <- [
        :argsort,
        :eigh,
        :fft,
        :ifft,
        :lu,
        :product,
        :qr,
        :reverse,
        :sort
      ] do
    @impl true
    def unquote(op)(_out, _tensor, _) do
      raise "unsupported Candlex.Backend.#{unquote(op)} function"
    end
  end

  for op <- [
        :indexed_put,
        :map,
        :triangular_solve,
        :window_max,
        :window_min,
        :window_product,
        :window_sum
      ] do
    @impl true
    def unquote(op)(_out, _tensor, _, _) do
      raise "unsupported Candlex.Backend.#{unquote(op)} function"
    end
  end

  @impl true
  def reduce(_out, _tensor, _, _, _) do
    raise "unsupported Candlex.Backend.reduce function"
  end

  for op <- [
        :window_reduce,
        :window_scatter_max,
        :window_scatter_min
      ] do
    @impl true
    def unquote(op)(_out, _tensor, _, _, _, _) do
      raise "unsupported Candlex.Backend.#{unquote(op)} function"
    end
  end

  defp permute(native_tensor, permutation) do
    native_tensor
    |> Native.permute(permutation)
    |> unwrap!()
  end

  @doc false
  defp from_nx(%T{data: %__MODULE__{} = data}), do: data

  defp from_nx(%T{} = tensor) do
    tensor
    |> Nx.backend_transfer(__MODULE__)
    |> from_nx()
  end

  defp to_nx(%__MODULE__{resource: ref} = backend_tensor, %T{type: nx_type, shape: nx_shape} = t)
       when is_reference(ref) do
    {:ok, candle_dtype} = Native.dtype(backend_tensor)
    {:ok, candle_shape} = Native.t_shape(backend_tensor)

    case {nx_type, from_candle_dtype(candle_dtype)} do
      {{:u, 64}, {:s, 64}} ->
        :ok

      {type, type} ->
        :ok

      {type, other_type} ->
        raise "tensor type mismatch, Nx (#{inspect(type)}) and Candle (#{inspect(other_type)})"
    end

    if nx_shape != candle_shape do
      raise "tensor shape mismatch, Nx (#{inspect(nx_shape)}) and Candle (#{inspect(candle_shape)})"
    end

    %{t | data: backend_tensor}
  end

  defp to_candle_dtype({:s, 8} = t), do: unsupported_dtype(t)
  defp to_candle_dtype({:s, 16} = t), do: unsupported_dtype(t)
  defp to_candle_dtype({:s, 32} = t), do: unsupported_dtype(t)
  defp to_candle_dtype({:s, 64}), do: "i64"
  defp to_candle_dtype({:u, 8}), do: "u8"
  defp to_candle_dtype({:u, 16} = t), do: unsupported_dtype(t)
  defp to_candle_dtype({:u, 32}), do: "u32"
  defp to_candle_dtype({:u, 64}), do: "i64"
  defp to_candle_dtype({:f, 16}), do: "f16"
  defp to_candle_dtype({:f, 32}), do: "f32"
  defp to_candle_dtype({:f, 64}), do: "f64"
  defp to_candle_dtype({:bf, 16}), do: "bf16"
  defp to_candle_dtype({:c, 64} = t), do: unsupported_dtype(t)
  defp to_candle_dtype({:c, 128} = t), do: unsupported_dtype(t)

  defp from_candle_dtype("i64"), do: {:s, 64}
  defp from_candle_dtype("u8"), do: {:u, 8}
  defp from_candle_dtype("u32"), do: {:u, 32}
  defp from_candle_dtype("f16"), do: {:f, 16}
  defp from_candle_dtype("bf16"), do: {:bf, 16}
  defp from_candle_dtype("f32"), do: {:f, 32}
  defp from_candle_dtype("f64"), do: {:f, 64}

  defp device_option(nil) do
    default_device()
  end

  defp device_option(backend_options) do
    backend_options[:device] || default_device()
  end

  defp default_device do
    if cuda_available?() do
      @device_cuda
    else
      @device_cpu
    end
  end

  defp same_device?(%T{data: %__MODULE__{device: device}}, device) do
    true
  end

  defp same_device?(_t, _d) do
    false
  end

  def cuda_available? do
    Native.is_cuda_available()
  end

  defp unsupported_dtype(t) do
    raise("Unsupported candle dtype for #{inspect(t)}")
  end

  defp unsupported_option!(opts, key, acceptable_default) do
    if opts[key] != nil and opts[key] != acceptable_default do
      raise "#{inspect(key)} option with #{inspect(opts[key])} is not supported"
    end
  end

  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("Candlex: #{error}")
end

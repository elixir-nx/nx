defmodule Candlex.Backend do
  @moduledoc """
  An opaque Nx backend with bindings to candle.
  """

  defstruct [:resource]

  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias Candlex.Backend, as: CB
  alias Candlex.Native

  @device_cuda :cuda
  @device_cpu :cpu

  @impl true
  def init(opts) do
    if opts != [] do
      raise ArgumentError, "Candlex.Backend accepts no options"
    end

    opts
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

  @impl true
  def eye(%T{shape: shape, type: type} = out, backend_options) do
    iota = Nx.iota(shape, backend_options)

    Nx.equal(Nx.tril(iota), Nx.triu(iota))
    |> Nx.as_type(type)
  end

  # Backend

  @impl true
  def backend_copy(%T{} = tensor, backend, backend_options) do
    backend.from_binary(tensor, to_binary(tensor), backend_options)
  end

  @impl true
  def backend_transfer(tensor, backend, backend_options) do
    backend_copy(tensor, backend, backend_options)
  after
    backend_deallocate(tensor)
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
  def all(%T{} = out, %T{} = tensor, _opts) do
    from_nx(tensor)
    |> Native.all()
    |> unwrap!()
    |> to_nx(out)
  end

  # Element-wise

  for op <- [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] do
    @impl true
    def unquote(op)(out, l, r) do
      unsupported_op(unquote(op))
    end
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

  for op <- [:add, :max, :min, :multiply, :subtract] do
    @impl true
    def unquote(op)(%T{} = out, %T{} = left, %T{} = right) do
      {left, right} = maybe_upcast(left, right)

      from_nx(left)
      |> Native.unquote(op)(from_nx(right))
      |> unwrap!()
      |> to_nx(out)
    end
  end

  for op <- [:equal, :greater_equal, :less, :less_equal] do
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

  # Unary ops

  unary_ops = [:abs, :cos, :exp, :log, :negate, :sin, :sqrt, :tanh]

  for op <- unary_ops do
    @impl true
    def unquote(op)(%T{} = out, %T{} = tensor) do
      tensor
      |> from_nx()
      |> Native.unquote(op)()
      |> unwrap!()
      |> to_nx(out)
    end
  end

  for op <- [:bitwise_not, :erf_inv] do
    @impl true
    def unquote(op)(out, t) do
      unsupported_op(unquote(op))
    end
  end

  # Indexed

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
  def dot(
        %T{type: _out_type} = out,
        %T{shape: left_shape, type: _left_type} = left,
        _left_axes,
        [] = _left_batched_axes,
        %T{shape: right_shape, type: _right_type} = right,
        _right_axes,
        [] = _right_batched_axes
      )
      when tuple_size(left_shape) == 2 and tuple_size(right_shape) == 2 do
    Native.matmul(
      from_nx(left),
      from_nx(right)
    )
    |> unwrap!()
    |> to_nx(out)
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

  defp maybe_add_signature(result, %T{data: %CB{resource: ref}}) when is_reference(ref) do
    Inspect.Algebra.concat([
      "Candlex.Backend(#{:erlang.ref_to_list(ref)})",
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

  ## Conversions

  @doc false
  defp from_nx(%T{data: %CB{} = data}), do: data

  defp from_nx(%T{} = tensor) do
    tensor
    |> Nx.backend_transfer(CB)
    |> from_nx()
  end

  defp to_nx(%{resource: ref} = backend_tensor, %T{} = t) when is_reference(ref) do
    %{t | data: backend_tensor}
  end

  defp to_candle_dtype({:s, 8}), do: unsupported_dtype()
  defp to_candle_dtype({:s, 16}), do: unsupported_dtype()
  defp to_candle_dtype({:s, 32}), do: unsupported_dtype()
  defp to_candle_dtype({:s, 64}), do: "i64"
  defp to_candle_dtype({:u, 8}), do: "u8"
  defp to_candle_dtype({:u, 16}), do: unsupported_dtype()
  defp to_candle_dtype({:u, 32}), do: "u32"
  defp to_candle_dtype({:u, 64}), do: unsupported_dtype()
  defp to_candle_dtype({:f, 16}), do: "f16"
  defp to_candle_dtype({:f, 32}), do: "f32"
  defp to_candle_dtype({:f, 64}), do: "f64"
  defp to_candle_dtype({:bf, 16}), do: "bf16"
  defp to_candle_dtype({:c, 64}), do: unsupported_dtype()
  defp to_candle_dtype({:c, 128}), do: unsupported_dtype()

  defp device_option(nil) do
    default_device()
  end

  defp device_option(backend_options) do
    backend_options[:device] || default_device()
  end

  defp default_device do
    # TODO: Support CUDA
    # if cuda_available?() do
    #   @device_cuda
    # else
    #   @device_cpu
    # end
    @device_cpu
  end

  defp cuda_available? do
    Native.is_cuda_available()
  end

  defp unsupported_dtype do
    raise("Unsupported candle dtype")
  end

  defp unsupported_op(op_name) do
    raise("Unsupported candlex operation '#{op_name}'")
  end

  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("Candlex: #{error}")
end

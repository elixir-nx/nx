defmodule Candlex.Backend do
  @moduledoc """
  An opaque Nx backend with bindings to candle.
  """

  defstruct [:resource]

  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias Candlex.Backend, as: CB
  alias Candlex.Native

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
  def from_binary(%T{shape: shape, type: type} = tensor, binary, _backend_options) do
    # TODO: Don't ignore backend options

    binary
    |> Native.from_binary(to_candle_dtype(type), shape)
    |> unwrap!()
    |> to_nx(tensor)
  end

  @impl true
  def iota(%T{shape: {}} = out, nil, backend_options) do
    constant(out, 0, backend_options)
  end

  def iota(%T{shape: shape} = out, nil, _backend_options) do
    # TODO: Support different types
    Native.arange(0, Nx.size(shape), shape)
    |> unwrap!()
    |> to_nx(out)
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

  @impl true
  def select(%T{shape: shape} = out, pred, on_true, on_false) do
    on_true =
      on_true
      |> Nx.as_type(Nx.type(out))
      |> from_nx()
      |> Native.broadcast_to(shape)
      |> unwrap!()

    on_false =
      on_false
      |> Nx.as_type(Nx.type(out))
      |> from_nx()
      |> Native.broadcast_to(shape)
      |> unwrap!()

    pred
    |> from_nx()
    |> Native.where_cond(on_true, on_false)
    |> unwrap!()
    |> to_nx(out)
  end

  # Binary ops

  for op <- [:add, :equal, :greater_equal, :multiply, :subtract] do
    @impl true
    def unquote(op)(%T{} = out, %T{} = left, %T{} = right) do
      {left, right} = maybe_broadcast_bin_args(out.shape, left, right)

      left
      |> Native.unquote(op)(right)
      |> unwrap!()
      |> to_nx(out)
    end
  end

  # Unary ops

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

  # Shape

  @impl true
  def as_type(%T{type: type} = out, %T{} = t) do
    from_nx(t)
    |> Native.to_type(to_candle_dtype(type))
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

  # defp device_option(_backend_options) do
  #   # TODO: Support CUDA
  #   :cpu
  # end

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

  defp maybe_broadcast_bin_args(_out_shape, %{shape: {}} = l, r), do: {from_nx(l), from_nx(r)}
  defp maybe_broadcast_bin_args(_out_shape, l, %{shape: {}} = r), do: {from_nx(l), from_nx(r)}

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
  defp from_nx(%T{data: data}), do: data

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

  defp unsupported_dtype do
    raise("Unsupported candle dtype")
  end

  defp unsupported_op do
    raise("Unsupported candle op")
  end

  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("Candlex: #{error}")
end

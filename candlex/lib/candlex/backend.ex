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

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  @impl true
  def to_binary(tensor, _limit \\ nil) do
    # TODO: don't ignore limit

    from_nx(tensor)
    |> Native.to_binary()
    |> unwrap!()
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

  ## Conversions

  @doc false
  defp from_nx(%T{data: data}), do: data

  defp to_nx(%{resource: ref} = backend_tensor, %T{} = t) when is_reference(ref) do
    %{t | data: backend_tensor}
  end

  defp to_candle_dtype({:u, 8}), do: "u8"
  defp to_candle_dtype({:u, 32}), do: "u32"
  defp to_candle_dtype({:f, 32}), do: "f32"
  defp to_candle_dtype({:f, 64}), do: "f64"

  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("Candlex: " <> List.to_string(error))
end
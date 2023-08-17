defmodule Candlex.Backend do
  @moduledoc """
  An opaque Nx backend with bindings to candle.
  """

  defstruct [:resource]

  # TODO: enable behaviour
  # @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias Candlex.Backend, as: CB
  alias Candlex.Native

  # @impl true
  def init(opts) do
    if opts != [] do
      raise ArgumentError, "Candlex.Backend accepts no options"
    end

    opts
  end

  # @impl true
  def constant(%T{shape: {}, type: _type} = tensor, scalar, _backend_options) do
    # TODO: Don't ignore backend options

    scalar
    |> Native.scalar_tensor()
    |> to_nx(tensor)
  end

  # @impl true
  def from_binary(%T{shape: {2}, type: {:u, 32}} = tensor, binary, _backend_options) do
    # TODO: Don't ignore backend options

    binary
    |> Native.from_binary()
    |> to_nx(tensor)
  end

  # @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  def to_binary(tensor, _limit) do
    # TODO: don't ignore limit

    from_nx(tensor)
    |> Native.to_binary()
  end

  defp maybe_add_signature(result, %T{data: %CB{resource: ref}}) when is_reference(ref) do
    Inspect.Algebra.concat([
      "Candlex.Backend(#{:erlang.ref_to_list(ref)})",
      Inspect.Algebra.line(),
      result
    ])
  end

  # defp to_candle_type({:u, 32}), do: :u32

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
end

defmodule EXLA.DeviceBackend do
  @moduledoc """
  A Nx tensor backend for the data kept on the device.

  You can directly transfer to this backend by calling
  `Nx.backend_transfer/2` or `Nx.backend_copy/2`. It
  allows the following options:

    * `:client` - the client to store the data on.
      Defaults to the client configured in `Nx.Defn`,
      otherwise uses `:host`.

    * `:device_id` - which device to store it on

  To get the data out of the device backend into a regular
  tensor, call `Nx.backend_transfer/1` (with the device
  tensor as the single argument).
  """

  @behaviour Nx.Backend
  @enforce_keys [:buffer]
  defstruct [:buffer]

  alias Nx.Tensor, as: T
  alias EXLA.DeviceBackend, as: DB

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, opts) do
    {client, device_id} = client_and_device_id(opts)
    shape = EXLA.Shape.make_shape(type, shape)
    buffer = EXLA.Buffer.place_on_device(binary, shape, client, device_id)
    put_in(tensor.data, %DB{buffer: buffer})
  end

  @impl true
  def backend_copy(tensor, Nx.Tensor, opts) do
    backend_copy(tensor, Nx.BinaryBackend, opts)
  end

  # TODO: Support direct transfers without going through Elixir
  def backend_copy(%T{data: %DB{buffer: buffer}} = tensor, backend, opts) do
    backend.from_binary(tensor, EXLA.Buffer.read(buffer), opts)
  end

  @impl true
  def backend_transfer(%T{data: %DB{buffer: buffer}} = tensor, backend, opts) do
    if backend == __MODULE__ and same_client_device?(buffer, opts) do
      tensor
    else
      try do
        backend_copy(tensor, backend, opts)
      after
        EXLA.Buffer.deallocate(buffer)
      end
    end
  end

  @impl true
  def backend_deallocate(%T{data: %DB{buffer: buffer}}) do
    EXLA.Buffer.deallocate(buffer)
  end

  @impl true
  def to_binary(%T{data: %DB{buffer: buffer}, type: {_, size}}, limit) do
    EXLA.Buffer.read(buffer, limit * div(size, 8))
  end

  @impl true
  def inspect(%T{data: %DB{buffer: buffer}}, _opts) do
    %EXLA.Buffer{client_name: client_name, device_id: device_id, ref: ref} = buffer
    '#Ref<' ++ rest = :erlang.ref_to_list(ref)
    "EXLA.DeviceBackend<#{client_name}:#{device_id}, " <> List.to_string(rest)
  end

  ## Helpers

  defp default_client_name do
    opts = Nx.Defn.default_options()

    if opts[:compiler] == EXLA do
      opts[:client] || :host
    else
      :host
    end
  end

  defp client_and_device_id(opts) do
    client = EXLA.Client.fetch!(opts[:client] || default_client_name())
    device_id = opts[:device_id] || client.default_device_id
    {client, device_id}
  end

  defp same_client_device?(buffer, opts) do
    {client, device_id} = client_and_device_id(opts)
    buffer.client_name == client.name and buffer.device_id == device_id
  end

  ## All remaining callbacks

  funs = Nx.Backend.behaviour_info(:callbacks) -- Module.definitions_in(__MODULE__, :def)

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(fun)(unquote_splicing(args)) do
      raise "operation #{unquote(fun)} is not supported on EXLA.DeviceBackend. " <>
              "You must first transfer the tensor to Elixir by calling Nx.backend_transfer/1"
    end
  end
end

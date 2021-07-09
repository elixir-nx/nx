defmodule EXLA.DeviceBackend do
  @moduledoc """
  A Nx tensor backend for the data kept on the device.

  You can directly transfer to this backend by calling
  `Nx.backend_transfer/2` or `Nx.backend_copy/2`. It
  allows the following options:

    * `:client` - the client to store the data on
    * `:device_id` - which device to store it on

  To get the data out of the device backend into a regular
  tensor, call `Nx.backend_transfer/1` (with the device
  tensor as the single argument).
  """

  @behaviour Nx.Backend
  @enforce_keys [:state]
  defstruct [:state]

  alias Nx.Tensor, as: T
  alias EXLA.DeviceBackend, as: DB

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, opts) do
    client = EXLA.Client.fetch!(opts[:client] || :default)
    device_id = opts[:device_id] || client.default_device_id
    buffer = EXLA.Buffer.buffer(binary, EXLA.Shape.make_shape(type, shape))
    buffer = EXLA.Buffer.place_on_device(buffer, client, device_id)
    put_in(tensor.data, %DB{state: buffer.ref})
  end

  @impl true
  def backend_copy(tensor, Nx.Tensor, opts) do
    backend_copy(tensor, Nx.BinaryBackend, opts)
  end

  # TODO: Support direct transfers without going through Elixir
  def backend_copy(%T{data: %DB{state: state}} = tensor, backend, opts) do
    backend.from_binary(tensor, EXLA.Buffer.read(state), opts)
  end

  @impl true
  def backend_transfer(%T{data: %DB{state: state}} = tensor, backend, opts) do
    backend_copy(tensor, backend, opts)
  after
    EXLA.Buffer.deallocate(state)
  end

  @impl true
  def backend_deallocate(%T{data: %DB{state: state}}) do
    EXLA.Buffer.deallocate(state)
  end

  @impl true
  def inspect(%T{data: %DB{state: state}}, _opts) do
    "EXLA.DeviceBackend<#{inspect(state)}>"
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

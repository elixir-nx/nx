defmodule Exla.Executable do
  alias __MODULE__, as: Executable
  alias Exla.Buffer
  alias Exla.Client
  alias Exla.Shape

  @enforce_keys [:client, :ref]
  defstruct [:client, :ref, :device]

  def populate_input_buffers(client, arguments, device) do
    buffers =
      arguments
      |> Enum.map(&Buffer.to_shaped_buffer(client, &1, device))
    {:ok, buffers}
  end

  def run(
        %Executable{client: client, ref: exec},
        arguments,
        options \\ []
      ) do
    # A tuple of {platform, ordinal} representing a device
    {platform, ordinal} = Keyword.get(options, :device, {client.platform, Client.get_default_device_ordinal(client)})
    # Run ID of this logical execution
    run_id = Keyword.get(options, :run_id, System.unique_integer([:positive, :monotonic]))
    # TODO: Another bad default. Looking at the source, this is used only with TPU devices. In PjRt, this is generated
    # from a uniform distribution between the min and max value of a 32-bit integer. The
    # XLA default is 0, which will work for us for now.
    rng_seed = Keyword.get(options, :rng_seed, 0)
    # Launch ID used to coordinate multi-device launches.
    # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/pjrt_client.h#L752-L755
    launch_id = Keyword.get(options, :launch_id, 0)
    # Whether to keep result on device
    keep_on_device = Keyword.get(options, :keep_on_device, false)
    keep_on_device_int = if keep_on_device, do: 1, else: 0
    # This is the same as OneFlow's XLA Executable Context, but we do some work in Elixir
    with {:ok, input_buffers} <- populate_input_buffers(client, arguments, {platform, ordinal}),
         {:ok, data} <-
           Exla.NIF.run(
             client.ref,
             exec,
             input_buffers,
             ordinal,
             run_id,
             rng_seed,
             launch_id,
             keep_on_device_int
           ) do
      if keep_on_device do
        # We can expect a reference back
        %Buffer{data: nil, ref: data}
      else
        %Buffer{data: data, ref: nil}
      end
    end
  end
end

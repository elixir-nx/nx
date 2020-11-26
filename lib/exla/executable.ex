defmodule Exla.Executable do
  alias __MODULE__, as: Executable
  alias Exla.Buffer
  alias Exla.Client

  @enforce_keys [:client, :ref, :output_shape]
  defstruct [:client, :ref, :output_shape, :device]

  def run(
        %Executable{client: client, ref: exec, output_shape: output_shape},
        arguments,
        options \\ []
      ) do
    # A tuple of {platform, ordinal} representing a device
    {_platform, ordinal} =
      Keyword.get(options, :device, {client.platform, Client.get_default_device_ordinal(client)})

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

    {:ok, data} =
      if all_loaded?(arguments) do
        inputs =  arguments |> Enum.map(& &1.ref)
        Exla.NIF.run(client.ref, exec, inputs, ordinal, run_id, rng_seed, launch_id, keep_on_device_int)
      else
        {inputs, shapes} =
          arguments
          |> Enum.map(&{&1.data, &1.shape.ref})
          |> Enum.unzip()

        Exla.NIF.run(
          client.ref,
          exec,
          inputs,
          shapes,
          ordinal,
          run_id,
          rng_seed,
          launch_id,
          keep_on_device_int
        )
      end

    if keep_on_device do
      %Buffer{data: nil, ref: data, shape: output_shape}
    else
      %Buffer{data: data, ref: nil, shape: output_shape}
    end
  end

  # Helper to check if all buffers are already on the device
  defp all_loaded?(arguments) do
    arguments
    |> Enum.filter(& &1.ref == nil)
    |> Enum.count()
    |> Kernel.==(0)
  end


end

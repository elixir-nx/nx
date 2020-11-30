defmodule Exla.Executable do
  alias __MODULE__
  alias Exla.{Buffer, Shape}

  @enforce_keys [:client, :ref, :output_shape, :device_ordinal]
  defstruct [:client, :ref, :output_shape, :device_ordinal]

  def run(
        %Executable{} = executable,
        arguments,
        options \\ []
      ) do
    %{client: client, ref: exec, output_shape: output_shape, device_ordinal: device_ordinal} =
      executable

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

    inputs =
      Enum.map(arguments, fn
        %Buffer{ref: {ref, _}, data: nil} -> ref
        %Buffer{data: data, shape: shape, ref: nil} -> {data, shape.ref}
      end)

    {:ok, data} =
      Exla.NIF.run(
        client.ref,
        exec,
        inputs,
        device_ordinal,
        run_id,
        rng_seed,
        launch_id,
        keep_on_device_int
      )

    if keep_on_device,
      do: decompose_output(data, output_shape, client),
      else: decompose_output(data, output_shape)
  end

  defp decompose_output(data, shape, client) do
    case shape do
      %Shape{dtype: {:t, shapes}} ->
        tuple =
          data
          |> Enum.zip(shapes)
          |> Enum.map(fn {buf, subshape} ->
            decompose_output(buf, subshape, client)
          end)

        {:tuple, tuple}

      _ ->
        Buffer.buffer({data, client.name}, shape)
    end
  end

  defp decompose_output(data, shape) do
    case shape do
      %Shape{dtype: {:t, shapes}} ->
        tuple =
          data
          |> Enum.zip(shapes)
          |> Enum.map(fn {buf, subshape} -> decompose_output(buf, subshape) end)

        {:tuple, tuple}

      _ ->
        Buffer.buffer(data, shape)
    end
  end
end

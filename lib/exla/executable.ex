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
    keep_on_device_int = if keep_on_device || client.platform == :cuda, do: 1, else: 0

    inputs =
      Enum.map(arguments, fn
        %Buffer{ref: {ref, _}, data: nil} -> ref
        buffer = %Buffer{data: data, shape: shape, ref: nil} ->
          case client.platform do
            :cuda ->
              %Buffer{ref: {ref, _}} = Buffer.place_on_device(buffer, client, device_ordinal)
              ref
            _ -> {data, shape.ref}
          end
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

    decompose_output(data, output_shape, client, keep_on_device)
  end

  defp decompose_output(data, shape, client, keep_on_device) do
    case shape do
      %Shape{dtype: {:t, shapes}} ->
        tuple =
          data
          |> Enum.zip(shapes)
          |> Enum.map(fn {buf, subshape} ->
            decompose_output(buf, subshape, client, keep_on_device)
          end)

        {:tuple, tuple}

      _ ->
        case data do
          data when is_reference(data) ->
            if keep_on_device,
              do: Buffer.buffer({data, client.name}, shape),
              else: Buffer.read({data, client.name}) |> Buffer.buffer(shape)
          _ -> Buffer.buffer(data, shape)
        end
    end
  end
end

defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{Buffer, Shape, ShardedBuffer}

  @enforce_keys [:client, :ref, :output_shape, :device_ordinal, :num_replicas, :num_partitions]
  defstruct [:client, :ref, :output_shape, :device_ordinal, :num_replicas, :num_partitions]

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

    {replica, partition} = Keyword.get(options, :device_assignment, {1, 1})

    async_run = Keyword.get(options, :async_run, false)
    async_run_int = if async_run, do: 1, else: 0

    keep_on_device_int = if keep_on_device, do: 1, else: 0

    inputs =
      Enum.map(arguments, fn
        %Buffer{ref: {ref, _}, data: nil} ->
          ref

        %Buffer{data: data, shape: shape, ref: nil} ->
          {data, shape.ref}
      end)

    data =
      case client.platform do
        :host ->
          EXLA.NIF.run_cpu(
            client.ref,
            exec,
            inputs,
            output_shape.ref,
            device_ordinal,
            run_id,
            rng_seed,
            launch_id,
            replica,
            partition,
            async_run_int,
            keep_on_device_int
          )
        _ ->
          EXLA.NIF.run_io(
            client.ref,
            exec,
            inputs,
            output_shape.ref,
            device_ordinal,
            run_id,
            rng_seed,
            launch_id,
            replica,
            partition,
            async_run_int,
            keep_on_device_int
          )
      end

    result =
      if async_run,
        do: EXLA.Client.await_streams(client, unwrap!(data), keep_on_device_int),
        else: data

    case result do
      {:ok, output} ->
        decompose_output(output, output_shape, client)
      {:error, reason} ->
        {:error, reason}
    end
  end

  def device_assignment_to_device_id(%Executable{ref: exec}, replica, partition) do
    EXLA.NIF.device_assignment_to_device_id(exec, replica, partition) |> unwrap!()
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

      _ when is_reference(data) ->
        Buffer.buffer({data, client.name}, shape)

      _ ->
        Buffer.buffer(data, shape)
    end
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end

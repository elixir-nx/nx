defmodule EXLA.Executable do
  @moduledoc """
  Wrapper around XLA's executable.
  """

  alias __MODULE__
  alias EXLA.{Buffer, Shape, Client}

  @enforce_keys [:client, :ref, :output_shape, :num_replicas, :num_partitions]
  defstruct [:client, :ref, :output_shape, :num_replicas, :num_partitions, :async]

  @doc """
  Runs the given executable with arguments.

  ## Options

    *  `:run_id` - a positive integer identifier of this execution.
      One is generated automatically if none is given.

    * `:keep_on_device` - if the data should be kept on the device
      after the computation (defaults to `false`).

    * `:replica` - the replica to run the executable on

  Some options apply to TPU only and therefore are not currently supported:

    * `:launch_id` - the launch id used to coordinate multi-device launches

    * `:rng_seed` - the seed for random numbers

    * `:partition` - the partition within a replica to run the executable on

  """
  def run(%Executable{} = executable, arguments, options \\ []) do
    %{client: client, output_shape: output_shape} = executable
    {data, _} = run(client, executable, arguments, options, 0)
    decompose_output(data, output_shape, client)
  end

  @doc """
  Runs the given function async.
  """
  def async_run(%Executable{} = executable, arguments, options \\ []) do
    {data, _} = run(executable.client, executable, arguments, options, 1)
    keep_on_device = Keyword.get(options, :keep_on_device, false)
    %{executable | async: {data, keep_on_device}}
  end

  @doc """
  Awaits the given function run.
  """
  def await_run(%Executable{async: {data, keep_on_device}} = executable) do
    %{client: client, output_shape: output_shape} = executable

    client
    |> await_streams(data, keep_on_device)
    |> unwrap!()
    |> decompose_output(output_shape, client)
  end

  defp await_streams(%Client{ref: ref, platform: platform}, buffer, keep_on_device) do
    keep_on_device_int = if keep_on_device, do: 1, else: 0

    # See https://github.com/elixir-nx/exla/pull/124 for discussion on this
    case platform do
      :host -> EXLA.NIF.await_streams_cpu(ref, buffer, keep_on_device_int)
      _ -> EXLA.NIF.await_streams_io(ref, buffer, keep_on_device_int)
    end
  end

  defp run(client, executable, arguments, options, async_run_int) do
    %{ref: exec, output_shape: output_shape} = executable

    run_id = Keyword.get(options, :run_id, System.unique_integer([:positive, :monotonic]))
    replica = Keyword.get(options, :replica, 1)
    keep_on_device = Keyword.get(options, :keep_on_device, false)
    keep_on_device_int = if keep_on_device, do: 1, else: 0

    # Launch ID used to coordinate multi-device launches.
    # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/pjrt_client.h#L752-L755
    launch_id = Keyword.get(options, :launch_id, 0)
    rng_seed = Keyword.get(options, :rng_seed, 0)
    partition = Keyword.get(options, :partition, 1)

    # TODO: Validate replicas and partitions against the client
    # TODO: Raise if buffers belong to different clients/ordinals

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
            run_id,
            rng_seed,
            launch_id,
            replica,
            partition,
            async_run_int,
            keep_on_device_int
          )
      end

    unwrap!(data)
  end

  defp decompose_output(data, shape, client) do
    %Shape{dtype: {:t, shapes}} = shape

    # TODO: Use Enum.zip_with on Elixir v1.12
    data
    |> Enum.zip(shapes)
    |> Enum.map(fn
      {buf, subshape} when is_reference(buf) ->
        Buffer.buffer({buf, client.name}, subshape)

      {buf, subshape} ->
        Buffer.buffer(buf, subshape)
    end)
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end

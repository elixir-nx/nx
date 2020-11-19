defmodule Exla.LocalExecutable do
  alias __MODULE__, as: LocalExecutable
  alias Exla.Options.ExecutableRunOptions
  alias Exla.Tensor
  alias Exla.Client
  alias Exla.Shape

  @enforce_keys [:client, :ref]
  defstruct [:client, :ref, :device]

  # TODO: This might be a useful part of the public `Client` API.
  def populate_input_buffers(client, arguments, device) do
    # We can avoid a lot of this by enforcing constraints: either all tensors are loaded on run
    # or no tensors are loaded on run. This method presents possibly a pretty significant bottleneck
    # so it's worth taking a careful look at later on.
    with {:ok, inputs} <- {:ok, Enum.with_index(Tuple.to_list(arguments))},
         {:ok, bin_inps, ref_inps} <- _group_by_type(inputs),
         {:ok, new_ref_inps} <- _place_on_device(client, bin_inps, device) do
      inputs =
        new_ref_inps
        |> Kernel.++(ref_inps)
        |> Enum.sort_by(&elem(&1, 1))
        |> Enum.map(fn {%Tensor{data: {:ref, ref}}, _idx} -> ref end)

      {:ok, List.to_tuple(inputs)}
    end
  end

  defp _group_by_type(inputs) do
    groups =
      inputs
      |> Enum.group_by(fn {%Tensor{data: data}, _} -> elem(data, 0) end)

    bins = if groups[:binary], do: groups[:binary], else: []
    refs = if groups[:ref], do: groups[:ref], else: []
    {:ok, bins, refs}
  end

  defp _place_on_device(client = %Client{}, inputs, device) do
    new_ref_inps =
      inputs
      |> Enum.map(fn {tensor, idx} -> {Tensor.to_device(client, tensor, device), idx} end)

    {:ok, new_ref_inps}
  end

  def run(
        %LocalExecutable{client: client, ref: exec},
        arguments,
        options \\ []
  ) do
    # A tuple of {platform, ordinal} representing a device
    device = Keyword.get(options, :device, {client.platform, -1})
    # TODO: This is a bad default. It works for now, but the `RunId` in XLA allows logical
    # executions to be broken down into multiple HLO Modules and then communicate between
    # the modules using collective ops. The RunId MUST be unique for each logical execution
    # so we need to enforce that within Elixir. It's probably better to add a `run_id` field
    # to a local executable.
    run_id = Keyword.get(options, :run_id, 0)
    # TODO: Another bad default. Looking at the source, this is used only with TPU devices. In PjRt, this is generated
    # from a uniform distribution between the min and max value of a 32-bit integer. The
    # XLA default is 0, which will work for us for now.
    rng_seed = Keyword.get(options, :rng_seed, 0)
    # Launch ID used to coordinate multi-device launches.
    # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/pjrt_client.h#L752-L755
    launch_id = Keyword.get(options, :launch_id, 0)
    # This is the same as OneFlow's XLA Executable Context, but we do some work in Elixir
    with {:ok, {_platform, ordinal}} <- Client.check_device_compatibility(client, device),
         {:ok, inputs} <- populate_input_buffers(client, arguments, device),
         {:ok, ref} <- Exla.NIF.run(client.ref, exec, inputs, ordinal, run_id, rng_seed, launch_id),
         # TODO: Replace this with something similar to `populate_output_buffers`
         # TODO: We can set the result layout during compilation, maybe we can use that here.
         {:ok, shape} <- Exla.NIF.on_host_shape(ref) do
      %Tensor{data: {:ref, ref}, shape: %Shape{ref: shape}, device: device}
    end
  end
end

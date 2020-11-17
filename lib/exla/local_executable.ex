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
        |> Enum.map(fn {%Tensor{data: {:ref, ref}}, idx} -> ref end)

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

  # TODO: This might be a useful part of the public `Client` API.
  # Compatible only if the platforms match and ordinal within client device count
  def ensure_client_and_device_compatible(
        client = %Client{platform: platform},
        {platform, ordinal}
      ) do
    cond do
      ordinal < 0 ->
        {:ok, {platform, Client.get_default_device_ordinal(client)}}

      ordinal < Client.get_device_count(client) ->
        {:ok, {platform, ordinal}}

      true ->
        {:error, "Invalid device ordinal."}
    end
  end

  def run(
        %LocalExecutable{client: client, ref: exec},
        arguments,
        options \\ %ExecutableRunOptions{}
      ) do
    # This is the same as OneFlow's XLA Executable Context, but we do some work in Elixir
    with {:ok, device} <- ensure_client_and_device_compatible(client, options.device),
         {:ok, inputs} <- populate_input_buffers(client, arguments, device),
         {:ok, ref} <- Exla.NIF.run(client.ref, exec, inputs, options),
         # TODO: Replace this with something similar to `populate_output_buffers`
         {:ok, shape} <- Exla.NIF.on_host_shape(ref) do
      %Tensor{data: {:ref, ref}, shape: %Shape{ref: shape}, device: device}
    end
  end
end

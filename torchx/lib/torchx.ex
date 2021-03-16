defmodule Torchx do
  alias Torchx.NIF

  import Torchx.Backend, only: [torch_type: 1, torch_device: 1]

  @doc """
  Check if device of the given type is available for Torchx.
  Device atom can be any of
  [:cpu, :cuda, :mkldnn, :opengl, :opencl, :ideep, :hip, :fpga, :msnpu, :xla, :vulkan, :metal, :xpu].

  But only :cuda availability check is supported for now.
  """
  def device_available?(:cuda), do: NIF.cuda_is_available()
  def device_available?(:cpu), do: true
  def device_available?(_), do: raise("Only CUDA device availability check is supported for now.")

  @doc """
  Return devices quantity for the given device type. Only :cuda is supported for now.
  """
  def device_count(:cuda), do: NIF.cuda_device_count()
  def device_count(_), do: raise("Only CUDA devices can be counted for now.")

  # LibTorch API bindings

  ## Creation

  def arange(from, to, step \\ 1, opts \\ []) do
    type = opts[:type] || :float
    device = opts[:device] || :cpu

    NIF.call(:arange, device, [from, to, step, type, torch_device(device)])
    |> wrap_with_device(device)
  end

  ## Operations

  def tensordot(left, right, left_axes, right_axes) do
    {device, [left_ref, right_ref]} = to_refs([left, right])

    NIF.call(:tensordot, device, [left_ref, right_ref, left_axes, right_axes])
    |> wrap_with_device(device)
  end

  ## Utils

  @doc false
  def type_of(ref), do: NIF.scalar_type(ref) |> unwrap!()

  @doc false
  def device_of(ref),
    do: NIF.device_of(ref) |> unwrap!() |> List.to_string() |> parse_torch_device_str()

  defp parse_torch_device_str(str) when is_binary(str) do
    str
    |> String.split(":")
    |> case do
      [type, index] ->
        {String.to_existing_atom(type), String.to_integer(index)}

      [type] ->
        String.to_existing_atom(type)
    end
  end

  @doc false
  def shape_of(ref), do: NIF.shape(ref) |> unwrap!()

  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("Torchx: " <> List.to_string(error))

  defp to_refs([{device, ref} | t]) do
    refs =
      Enum.map(t, fn
        {^device, ref} ->
          ref

        {other_device, _ref} ->
          raise "cannot perform operations on across devices: #{inspect(device)} and #{
                  inspect(other_device)
                }"
      end)

    {device, [ref | refs]}
  end

  defp wrap_with_device(maybe_ref, device), do: {device, unwrap!(maybe_ref)}
end

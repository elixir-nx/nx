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
    type = opts[:type] || Nx.Type.infer([from, to])
    device = opts[:device] || :cpu

    NIF.arange(from, to, step, torch_type(type), torch_device(device))
    |> wrap_with_device()
  end

  ## Operations

  def tensordot(left_ref, right_ref, left_axes, right_axes) do
    NIF.tensordot(
      left_ref,
      right_ref,
      left_axes,
      right_axes
    )
    |> wrap_with_device()
  end

  ## Utils

  @doc false
  def type_of(ref), do: NIF.scalar_type(ref) |> unwrap!() |> from_torch_type()

  defp from_torch_type(:char), do: {:s, 8}
  defp from_torch_type(:byte), do: {:u, 8}
  defp from_torch_type(:bool), do: {:u, 8}
  defp from_torch_type(:short), do: {:s, 16}
  defp from_torch_type(:int), do: {:s, 32}
  defp from_torch_type(:long), do: {:s, 64}
  defp from_torch_type(:brain), do: {:bf, 16}
  defp from_torch_type(:half), do: {:f, 16}
  defp from_torch_type(:float), do: {:f, 32}
  defp from_torch_type(:double), do: {:f, 64}

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

  defp wrap_with_device(maybe_ref) do
    ref = unwrap!(maybe_ref)
    {device_of(ref), ref}
  end
end

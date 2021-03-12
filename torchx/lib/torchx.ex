defmodule Torchx do
  alias Torchx.NIF

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
end

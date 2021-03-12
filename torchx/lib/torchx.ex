defmodule Torchx do
  alias Torchx.NIF

  def device_available?(:cuda), do: NIF.cuda_is_available()
  def device_available?(_), do: raise("Only CUDA device availability check is supported for now.")

  def device_count(:cuda), do: NIF.cuda_device_count()
  def device_count(_), do: raise("Only CUDA devices can be counted for now.")
end

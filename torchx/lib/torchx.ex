defmodule Torchx do
  alias Torchx.NIF

  def cuda_is_available?(), do: NIF.cuda_is_available()

  def cuda_device_count(), do: NIF.cuda_device_count()
end

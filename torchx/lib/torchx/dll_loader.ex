defmodule Torchx.NIF.DLLLoader do
  @moduledoc false
  @on_load :__on_load__
  def __on_load__ do
    case :os.type() do
      {:win32, _} ->
        path = :filename.join(:code.priv_dir(:torchx), ~c"torchx_dll_loader")
        :erlang.load_nif(path, 0)
        add_dll_directory()

      _ ->
        :ok
    end
  end

  def add_dll_directory do
    case :os.type() do
      {:win32, _} ->
        :erlang.nif_error(:undef)

      _ ->
        :ok
    end
  end
end

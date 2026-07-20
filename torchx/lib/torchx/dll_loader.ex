defmodule Torchx.NIF.DLLLoader do
  @moduledoc false

  # Loads the loader NIF and adds libtorch to the DLL search path. Driven
  # explicitly by Torchx.NIF.__on_load__ rather than via @on_load, since the
  # ordering of @on_load handlers across modules is not guaranteed during
  # release boot and could call add_dll_directory/0 before this module's NIF
  # was loaded (resulting in an :undef crash).
  def setup do
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

defmodule Torchx.NIF.DLLLoader do
  @moduledoc false

  # Loads the loader NIF and adds libtorch to the DLL search path. Driven
  # explicitly by Torchx.NIF.__on_load__ rather than via @on_load, since the
  # ordering of @on_load handlers across modules is not guaranteed during
  # release boot and could call add_dll_directory/1 before this module's NIF
  # was loaded (resulting in an :undef crash).
  #
  # The DLL directory is chosen on the Elixir side (config / priv path) and
  # passed into the NIF — the C loader does not read environment variables.
  def setup do
    case :os.type() do
      {:win32, _} ->
        path = :filename.join(:code.priv_dir(:torchx), ~c"torchx_dll_loader")
        :erlang.load_nif(path, 0)
        add_dll_directory(libtorch_dll_dir())

      _ ->
        :ok
    end
  end

  def add_dll_directory(_dll_dir) do
    case :os.type() do
      {:win32, _} ->
        :erlang.nif_error(:undef)

      _ ->
        :ok
    end
  end

  defp libtorch_dll_dir do
    case Application.get_env(:torchx, :libtorch_dll_dir) do
      dir when is_binary(dir) and dir != "" ->
        dir

      nil ->
        Application.app_dir(:torchx, ["priv", "libtorch"])

      other ->
        raise ArgumentError,
              "config :torchx, :libtorch_dll_dir must be a non-empty binary, got: #{inspect(other)}"
    end
  end
end

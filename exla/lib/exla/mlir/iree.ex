defmodule EXLA.MLIR.IREE do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), ~c"libireecompiler")
    :erlang.load_nif(path, 0)
  end

  def compile(_module, _target), do: :erlang.nif_error(:undef)

  def global_initialize, do: :erlang.nif_error(:undef)

  def run_module(_module, _inputs), do: :erlang.nif_error(:undef)
end

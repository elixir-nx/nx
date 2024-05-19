defmodule EXLA.MLIR.IREE do
  @moduledoc false
  alias EXLA.MLIR.IREE.InstancePool

  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), ~c"libireecompiler")
    :erlang.load_nif(path, 0)
  end

  def run(module, inputs) do
    InstancePool.checkout(fn instance ->
      device = :persistent_term.get({EXLA.MLIR.IREE, :device})
      run_module(instance, device, module, inputs)
    end)
  end

  def read(buffer, size) do
    device = :persistent_term.get({EXLA.MLIR.IREE, :device})
    read_buffer(buffer, device, size)
  end

  def compile(_module, _flags), do: :erlang.nif_error(:undef)

  def global_initialize, do: :erlang.nif_error(:undef)

  def run_module(_instance, _device, _module, _inputs), do: :erlang.nif_error(:undef)

  def setup_runtime, do: :erlang.nif_error(:undef)

  def create_instance, do: :erlang.nif_error(:undef)

  def deallocate_buffer(_buffer), do: :erlang.nif_error(:undef)

  def read_buffer(_buffer, _device, _size), do: :erlang.nif_error(:undef)
end

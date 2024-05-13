defmodule EXLA.MLIR.IREE do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:exla), ~c"libireecompiler")
    :erlang.load_nif(path, 0)
  end

  def init do
    global_initialize()
    {:ok, instance} = runtime_create_instance()
    :persistent_term.put({__MODULE__, :instance}, instance)

    :ok
  end

  def run(module, inputs) do
    instance = :persistent_term.get({__MODULE__, :instance})
    run_module(instance, module, inputs)
  end

  def compile(_module, _target), do: :erlang.nif_error(:undef)

  def global_initialize, do: :erlang.nif_error(:undef)

  def runtime_create_instance, do: :erlang.nif_error(:undef)

  def run_module(_instance, _module, _inputs), do: :erlang.nif_error(:undef)
end

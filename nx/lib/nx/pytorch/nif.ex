defmodule Nx.Pytorch.NIF do
  @moduledoc false
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:nx), 'pytorch')
    :erlang.load_nif(path, 0)
  end

  def randint(_min, _max, _shape, _type), do: nif_error(__ENV__.function)
  def rand(_min, _max, _shape, _type), do: nif_error(__ENV__.function)
  def arange(_start, _end, _step, _type), do: nif_error(__ENV__.function)
  def arange(_start, _end, _step, _type, _shape), do: nif_error(__ENV__.function)

  def from_blob(_blob, _shape, _type),
    do: nif_error(__ENV__.function)

  def ones(_shape),
    do: nif_error(__ENV__.function)

  def add(_tensorA, _tensorB),
    do: nif_error(__ENV__.function)

  defp nif_error({name, arity}) do
    raise "failed to load implementation of #{inspect(__MODULE__)}.#{name}/#{arity}"
  end
end

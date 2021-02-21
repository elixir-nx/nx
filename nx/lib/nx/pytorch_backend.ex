defmodule Nx.PytorchBackend do
  @behaviour Nx.Tensor

  defstruct [:data]

  funs =
    Nx.Tensor.behaviour_info(:callbacks) -- [from_binary: 3, backend_deallocate: 1, inspect: 2]

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    def unquote(fun)(unquote_splicing(args)) do
      raise "#{unquote(fun)}() is not supported"
    end
  end

  def from_binary(tensor, binary, _opts) do
    put_in(tensor.data, %__MODULE__{data: binary})
  end

  def backend_deallocate(%Nx.Tensor{data: _data}) do
    :ok
  end

  def inspect(%{data: %{data: binary}}, opts) do
    Inspect.BitString.inspect(binary, opts)
  end
end

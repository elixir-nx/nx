defmodule ProcessBackend do
  @behaviour Nx.Backend
  defstruct [:key]

  def init(opts), do: opts

  def from_binary(tensor, binary, opts) do
    key = Keyword.fetch!(opts, :key)
    Process.put(key, binary)
    put_in(tensor.data, %__MODULE__{key: key})
  end

  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{key: key}}) do
    if Process.delete(key) do
      :ok
    else
      :already_deallocated
    end
  end

  funs =
    Nx.Backend.behaviour_info(:callbacks) --
      (Nx.Backend.behaviour_info(:optional_callbacks) ++ Module.definitions_in(__MODULE__, :def))

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    def unquote(fun)(unquote_splicing(args)) do
      raise "not supported"
    end
  end
end

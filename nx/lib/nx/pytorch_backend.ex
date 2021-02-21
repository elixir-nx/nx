defmodule Nx.PytorchBackend do
  @behaviour Nx.Tensor

  alias Nx.Pytorch.NIF

  defstruct [:data]

  funs =
    Nx.Tensor.behaviour_info(:callbacks) --
      [tensor: 1, random_uniform: 3, from_binary: 3, backend_deallocate: 1, inspect: 2]

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    def unquote(fun)(unquote_splicing(args)) do
      raise "#{unquote(fun)}() is not supported"
    end
  end

  ## Creation

  @impl true
  def tensor(tensor) do
    tensor
  end

  def torch_type({:u, 8}), do: :byte
  def torch_type({:s, 8}), do: :char
  def torch_type({:s, 16}), do: :short
  def torch_type({:s, 32}), do: :int
  def torch_type({:s, 64}), do: :long
  def torch_type({:f, 16}), do: :half
  def torch_type({:f, 32}), do: :float
  def torch_type({:f, 64}), do: :double

  @impl true
  def random_uniform(%{type: type, shape: shape} = out, min, max) do
    NIF.randint(min, max, shape, torch_type(type))
    # gen =
    #   case type do
    #     {:s, _} -> fn -> min + :rand.uniform(max - min) - 1 end
    #     {:u, _} -> fn -> min + :rand.uniform(max - min) - 1 end
    #     {_, _} -> fn -> (max - min) * :rand.uniform() + min end
    #   end

    # data = for _ <- 1..Nx.size(shape), into: "", do: number_to_binary(gen.(), type)
    # from_binary(out, data)
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

defmodule Nx.PytorchBackend do
  @behaviour Nx.Tensor

  alias Nx.Pytorch.NIF

  defstruct [:ref]

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
  def random_uniform(%{type: {s, _} = type, shape: shape} = out, min, max) when s in [:u, :s] do
    tensor_ref = NIF.randint(min, max, shape, torch_type(type))
    from_binary(out, tensor_ref)
  end

  def random_uniform(%{type: {:f, _} = type, shape: shape} = out, min, max) do
    tensor_ref = NIF.rand(min, max, shape, torch_type(type))
    from_binary(out, tensor_ref)
  end

  @impl true
  def from_binary(t, ref, _opts), do: from_binary(t, ref)
  defp from_binary(t, ref) when is_reference(ref), do: %{t | data: %__MODULE__{ref: ref}}

  @impl true
  def backend_deallocate(%Nx.Tensor{data: _data}) do
    :ok
  end

  @impl true
  def inspect(%{data: %{ref: ref}}, opts) do
    Inspect.inspect(ref, opts)
  end
end

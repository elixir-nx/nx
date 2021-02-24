defmodule Torchx.Backend do
  @behaviour Nx.Backend

  defstruct [:ref]

  alias Torchx.NIF

  def torch_type({:u, 8}), do: :byte

  def torch_type({:u, 16}),
    do: raise(ArgumentError, "PyTorch does not support unsigned 16 bit integer.")

  def torch_type({:u, 32}),
    do: raise(ArgumentError, "PyTorch does not support unsigned 32 bit integer.")

  def torch_type({:u, 64}),
    do: raise(ArgumentError, "PyTorch does not support unsigned 64 bit integer.")

  def torch_type({:s, 8}), do: :char
  def torch_type({:s, 16}), do: :short
  def torch_type({:s, 32}), do: :int
  def torch_type({:s, 64}), do: :long
  def torch_type({:bf, 16}), do: :brain
  def torch_type({:f, 16}), do: :half
  def torch_type({:f, 32}), do: :float
  def torch_type({:f, 64}), do: :double

  ## Creation

  @impl true
  def iota(out, axis \\ nil)

  def iota(%{shape: {}, type: type} = out, nil) do
    NIF.scalar_tensor(0, torch_type(type)) |> from_ref(out)
  end

  def iota(%{shape: shape, type: type} = out, nil) do
    NIF.arange(0, Nx.size(shape), 1, torch_type(type), shape) |> from_ref(out)
  end

  def iota(%{shape: {n}, type: type} = out, 0) do
    NIF.arange(0, n, 1, torch_type(type)) |> from_ref(out)
  end

  def iota(%{shape: shape, type: type} = out, axis) do
    # gets the size of iota
    dim = elem(shape, axis)

    # build the iota in one dimension
    aten = NIF.arange(0, dim, 1, torch_type(type)) |> unwrap!()

    # reshape the tensor above to be have shape where everything is 1, except for dim
    reshape = Tuple.duplicate(1, Nx.rank(shape)) |> put_elem(axis, dim)
    aten = NIF.reshape(aten, reshape) |> unwrap!()

    # Now broadcast the tensor using the original shape
    NIF.broadcast_to(aten, shape) |> from_ref(out)
  end

  @impl true
  def eye(%{shape: {n, n}, type: type} = out) do
    NIF.eye(n, torch_type(type)) |> from_ref(out)
  end

  @impl true
  def random_uniform(%{type: {s, _} = type, shape: shape} = out, min, max) when s in [:u, :s] do
    NIF.randint(min, max, shape, torch_type(type)) |> from_ref(out)
  end

  def random_uniform(%{type: {f, _} = type, shape: shape} = out, min, max) when f in [:f, :bf] do
    NIF.rand(min, max, shape, torch_type(type)) |> from_ref(out)
  end

  @impl true
  def random_normal(%{type: _type, shape: shape} = out, mu, sigma) do
    NIF.normal(mu, sigma, shape) |> from_ref(out)
  end

  @impl true
  def from_binary(%{type: type, shape: shape} = out, binary, _opts) do
    NIF.from_blob(binary, shape, torch_type(type)) |> from_ref(out)
  end

  @impl true
  def to_binary(%{data: %{ref: ref}}), do: NIF.to_blob(ref)
  def to_binary(%{data: %{ref: ref}}, limit), do: NIF.to_blob(ref, limit)

  @impl true
  def backend_transfer(tensor, Nx.Tensor, opts) do
    backend_transfer(tensor, Nx.BinaryBackend, opts)
  end

  def backend_transfer(tensor, Torchx.Backend, _opts) do
    tensor
  end

  def backend_transfer(tensor, backend, opts) do
    backend.from_binary(tensor, to_binary(tensor), opts)
  end

  ## Shape

  @impl true
  def reshape(out, %{data: %{ref: ref}} = tensor, shape),
    do: NIF.reshape(ref, shape) |> from_ref(tensor)

  @impl true
  def squeeze(out, tensor, _axes) do
    NIF.squeeze(tensor.data.ref) |> from_ref(out)
  end

  ## Ops

  @impl true
  def add(out, left, right) do
    NIF.add(left.data.ref, right.data.ref) |> from_ref(out)
  end

  @impl true
  def dot(
        out,
        %{type: t1, data: %{ref: left_ref}} = left,
        _axes1,
        %{type: t2, data: %{ref: right_ref}} = right,
        _axes2
      ) do
    NIF.dot(left_ref, right_ref) |> from_ref(out)
  end

  @impl true
  def cholesky(%{type: output_type, shape: {rows, cols}} = out, tensor) do
    NIF.cholesky(tensor.data.ref) |> from_ref(out)
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %{ref: ref}}), do: NIF.delete_tensor(ref)

  @impl true
  def inspect(tensor, inspect_opts) do
    limit = inspect_opts.limit
    binary = Nx.to_binary(tensor, if(limit == :infinity, do: [], else: [limit: limit + 1]))
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  defp unwrap!({:ok, tensor}), do: tensor

  defp unwrap!({:error, error}) when is_binary(error),
    do: raise(RuntimeError, error)

  defp from_ref(ref, t) when is_tuple(ref), do: unwrap!(ref) |> from_ref(t)
  defp from_ref(ref, t) when is_reference(ref), do: %{t | data: %__MODULE__{ref: ref}}

  ## All remaining callbacks

  funs = Nx.Backend.behaviour_info(:callbacks) -- Module.definitions_in(__MODULE__, :def)

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    @impl true
    def unquote(fun)(unquote_splicing(args)) do
      raise "operation #{unquote(fun)} is not supported on Torchx.Backend."
    end
  end
end

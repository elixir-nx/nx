defmodule Torchx.Backend do
  @behaviour Nx.Backend

  defstruct [:ref]

  alias Torchx.NIF

  def torch_type({:u, 8}), do: :byte
  def torch_type({:s, 8}), do: :char
  def torch_type({:s, 16}), do: :short
  def torch_type({:s, 32}), do: :int
  def torch_type({:s, 64}), do: :long
  def torch_type({:f, 16}), do: :half
  def torch_type({:f, 32}), do: :float
  def torch_type({:f, 64}), do: :double

  ## Creation

  @impl true
  def iota(out, axis \\ nil)

  def iota(%{shape: {}, type: type} = out, nil), do: iota(%{out | shape: {1}}, nil)

  def iota(%{shape: shape, type: type} = out, nil) do
    t = NIF.arange(0, Nx.size(shape), 1, torch_type(type), shape)
    from_ref(out, t)
  end

  def iota(%{shape: {n}, type: type} = out, 0) do
    t = NIF.arange(0, n, 1, torch_type(type))
    from_ref(out, t)
  end

  def iota(%{shape: shape, type: type} = out, axis) do
    # gets the size of iota
    dim = elem(shape, axis)

    # build the iota in one dimension
    aten = NIF.arange(0, dim, 1, torch_type(type))

    # reshape the tensor above to be have shape where everything is 1, except for dim
    reshape = Tuple.duplicate(1, Nx.rank(shape)) |> put_elem(axis, dim)
    aten = NIF.reshape(aten, reshape)

    # Now broadcast the tensor using the original shape
    aten = NIF.broadcast_to(aten, shape)

    from_ref(out, aten)
  end

  @impl true
  def eye(%{shape: {n, n}, type: type} = out) do
    t = NIF.eye(n, torch_type(type))
    from_ref(out, t)
  end

  @impl true
  def random_uniform(%{type: {s, _} = type, shape: shape} = out, min, max) when s in [:u, :s] do
    tensor_ref = NIF.randint(min, max, shape, torch_type(type))
    from_ref(out, tensor_ref)
  end

  def random_uniform(%{type: {:f, _} = type, shape: shape} = out, min, max) do
    tensor_ref = NIF.rand(min, max, shape, torch_type(type))
    from_ref(out, tensor_ref)
  end

  @impl true
  def random_normal(%{type: type, shape: shape} = out, mu, sigma) do
    tensor_ref = NIF.normal(mu, sigma, shape, torch_type(type))
    from_ref(out, tensor_ref)
  end

  @impl true
  def from_binary(%{type: type, shape: shape} = out, binary, _opts) do
    t = NIF.from_blob(binary, shape, torch_type(type))
    from_ref(out, t)
  end

  @impl true
  def to_binary(%{data: %{ref: ref}}), do: NIF.to_blob(ref)
  def to_binary(%{data: %{ref: ref}}, limit), do: NIF.to_blob(ref, limit)

  ## Shape

  @impl true
  def reshape(out, %{data: %{ref: ref}} = tensor, shape),
    do: %{tensor | data: %__MODULE__{ref: NIF.reshape(ref, shape)}}

  @impl true
  def squeeze(out, tensor, _axes) do
    t = NIF.squeeze(tensor.data.ref)
    from_ref(out, t)
  end

  ## Ops

  @impl true
  def add(out, left, right) do
    t = NIF.add(left.data.ref, right.data.ref)
    from_ref(out, t)
  end

  @impl true
  def dot(
        out,
        %{type: t1, data: %{ref: left_ref}} = left,
        _axes1,
        %{type: t2, data: %{ref: right_ref}} = right,
        _axes2
      ) do
    t = NIF.dot(left_ref, right_ref)
    from_ref(out, t)
  end

  @impl true
  def cholesky(%{type: output_type, shape: {rows, cols}} = out, tensor) do
    t = NIF.cholesky(tensor.data.ref)
    from_ref(out, t)
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %{ref: ref}}), do: NIF.delete_tensor(ref)

  @impl true
  def inspect(tensor, inspect_opts) do
    limit = inspect_opts.limit
    binary = Nx.to_binary(tensor, if(limit == :infinity, do: [], else: [limit: limit + 1]))
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  defp from_ref(t, ref) when is_reference(ref), do: %{t | data: %__MODULE__{ref: ref}}

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

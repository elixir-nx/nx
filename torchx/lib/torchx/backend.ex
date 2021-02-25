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
  def eye(%{shape: {n, n}, type: type} = out) do
    NIF.eye(n, torch_type(type)) |> from_ref(out)
  end

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

  ## Transfer

  @impl true
  def to_batched_list(%{shape: shape} = out, %{data: %{ref: ref}}),
    do: NIF.split(ref, elem(shape, 0)) |> from_ref(out)

  @big_tensor_threshold 10_000_000

  def to_binary(_tensor, _limit \\ nil) do
    raise "Operation to_binary is not supported on Torchx.Backend. " <>
            "You must first transfer the tensor to Elixir by calling Nx.backend_transfer/1"
  end

  defp to_blob(%{type: {_, elem_size}, data: %{ref: ref}} = tensor, limit \\ nil) do
    if(limit,
      do: NIF.to_blob(ref, limit),
      else: NIF.to_blob(ref)
    )
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %{ref: ref}}), do: NIF.delete_tensor(ref)

  @impl true
  def backend_transfer(tensor, Nx.Tensor, opts) do
    backend_transfer(tensor, Nx.BinaryBackend, opts)
  end

  def backend_transfer(tensor, Torchx.Backend, _opts) do
    tensor
  end

  def backend_transfer(tensor, backend, opts) do
    backend.from_binary(tensor, to_blob(tensor), opts)
  end

  @impl true
  def from_binary(%{type: type, shape: shape} = out, binary, _opts) do
    NIF.from_blob(binary, shape, torch_type(type)) |> from_ref(out)
  end

  ## Shape

  @impl true
  def reshape(out, %{data: %{ref: ref}} = tensor, shape),
    do: NIF.reshape(ref, shape) |> from_ref(tensor)

  @impl true
  def as_type(%{type: type} = out, %{data: %{ref: ref}} = tensor),
    do: NIF.to_type(ref, torch_type(type)) |> from_ref(out)

  @impl true
  def squeeze(out, tensor, _axes) do
    NIF.squeeze(tensor.data.ref) |> from_ref(out)
  end

  @impl true
  def transpose(out, tensor, axes), do: IO.inspect(axes)
  # do: NIF.transpose(tensor, dim0, dim1) |> from_ref(out)

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
  def qr(
        {q_holder, r_holder},
        tensor,
        opts
      ),
      do: NIF.qr(tensor.data.ref, opts[:mode] == :reduced) |> from_ref({q_holder, r_holder})

  @impl true
  def inspect(%{type: {_, elem_size}} = tensor, inspect_opts) do
    limit = if(inspect_opts.limit == :infinity, do: nil, else: inspect_opts.limit + 1)

    if on_cpu?(tensor) do
      byte_size = nbytes(tensor)
      byte_limit = limit && limit * div(elem_size, 8)

      if(min(byte_limit, byte_size) > @big_tensor_threshold) do
        raise "Tensor is too big (#{byte_size} bytes) for to_binary/1 operation." <>
                "You must first transfer the tensor to Elixir by calling Nx.backend_transfer/1"
      else
        binary = to_blob(tensor, limit)
        Nx.Backend.inspect(tensor, binary, inspect_opts)
      end
    else
      raise "Tensor is located on #{device(tensor)} device, so direct to_binary/1 operation is not supported. " <>
              "You must first transfer the tensor to Elixir by calling Nx.backend_transfer/1"
    end
  end

  defp unwrap!({:ok, result}), do: result

  defp unwrap!({:error, error}),
    do: raise(RuntimeError, "PyTorch: " <> to_string(error))

  defp from_ref({:ok, ref}, t), do: from_ref(ref, t)
  defp from_ref({:error, _error} = input, _t), do: unwrap!(input)
  defp from_ref({ref1, ref2}, {t1, t2}), do: {from_ref(ref1, t1), from_ref(ref2, t2)}
  defp from_ref([ref | list], t), do: [from_ref(ref, t) | from_ref(list, t)]
  defp from_ref([], _t), do: []
  defp from_ref(ref, t) when is_reference(ref), do: %{t | data: %__MODULE__{ref: ref}}

  defp device(%{data: %{ref: ref}}), do: NIF.device(ref) |> unwrap!() |> to_string()
  defp nbytes(%{data: %{ref: ref}}), do: NIF.nbytes(ref) |> unwrap!()
  defp on_cpu?(tensor), do: device(tensor) == "cpu"

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

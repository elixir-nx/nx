defmodule Torchx.Backend do
  @behaviour Nx.Backend

  defstruct [:ref]

  alias Torchx.NIF

  import Nx.Shared

  funs =
    Nx.Backend.behaviour_info(:callbacks) --
      [
        tensor: 1,
        iota: 2,
        eye: 1,
        random_uniform: 3,
        random_normal: 3,
        reshape: 3,
        squeeze: 3,
        add: 3,
        dot: 5,
        cholesky: 2,
        from_binary: 3,
        to_binary: 2,
        backend_deallocate: 1,
        inspect: 2
      ]

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
    {dims_before, [dim | dims_after]} =
      shape
      |> Tuple.to_list()
      |> Enum.split(axis)

    # Number of repetitions of an index in memory
    repeat_blocks =
      dims_after
      |> Enum.reduce(1, &*/2)

    # Number of cycles of the counting pattern
    cycles =
      dims_before
      |> Enum.reduce(1, &*/2)

    data =
      for _ <- 1..cycles,
          i <- 0..(dim - 1),
          _ <- 1..repeat_blocks,
          into: "",
          do: number_to_binary(i, type)

    t = NIF.from_blob(data, shape, torch_type(type))
    from_ref(out, t)
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
  def backend_deallocate(%Nx.Tensor{data: _data}) do
    :ok
  end

  @impl true
  def inspect(tensor, inspect_opts) do
    limit = inspect_opts.limit
    binary = Nx.to_binary(tensor, if(limit == :infinity, do: [], else: [limit: limit + 1]))
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  defp from_ref(t, ref) when is_reference(ref), do: %{t | data: %__MODULE__{ref: ref}}

  defp number_to_binary(number, type),
    do: match_types([type], do: <<write!(number, 0)>>)
end

defmodule Torchx.Backend do
  @behaviour Nx.Backend

  defstruct [:ref]

  alias Torchx.NIF
  alias Nx.Tensor, as: T
  alias Torchx.Backend, as: TB

  def torch_type({:u, 8}), do: :byte

  def torch_type({:u, 16}),
    do: raise(ArgumentError, "Torchx does not support unsigned 16 bit integer")

  def torch_type({:u, 32}),
    do: raise(ArgumentError, "Torchx does not support unsigned 32 bit integer")

  def torch_type({:u, 64}),
    do: raise(ArgumentError, "Torchx does not support unsigned 64 bit integer")

  def torch_type({:s, 8}), do: :char
  def torch_type({:s, 16}), do: :short
  def torch_type({:s, 32}), do: :int
  def torch_type({:s, 64}), do: :long
  def torch_type({:bf, 16}), do: :brain
  def torch_type({:f, 16}), do: :half
  def torch_type({:f, 32}), do: :float
  def torch_type({:f, 64}), do: :double

  def from_torch_type(:char), do: {:s, 8}
  def from_torch_type(:byte), do: {:u, 8}
  def from_torch_type(:bool), do: {:u, 8}
  def from_torch_type(:short), do: {:s, 16}
  def from_torch_type(:int), do: {:s, 32}
  def from_torch_type(:long), do: {:s, 64}
  def from_torch_type(:brain), do: {:bf, 16}
  def from_torch_type(:half), do: {:f, 16}
  def from_torch_type(:float), do: {:f, 32}
  def from_torch_type(:double), do: {:f, 64}

  ## Creation

  @impl true
  def eye(%T{shape: {n, n}, type: type} = out) do
    NIF.eye(n, torch_type(type)) |> from_ref(out)
  end

  @impl true
  def iota(out, axis \\ nil)

  def iota(%T{shape: {}, type: type} = out, nil) do
    NIF.scalar_tensor(0, torch_type(type)) |> from_ref(out)
  end

  def iota(%T{shape: shape, type: type} = out, nil) do
    NIF.arange(0, Nx.size(shape), 1, torch_type(type), shape) |> from_ref(out)
  end

  def iota(%T{shape: {n}, type: type} = out, 0) do
    NIF.arange(0, n, 1, torch_type(type)) |> from_ref(out)
  end

  def iota(%T{shape: shape, type: type} = out, axis) do
    NIF.iota(shape, axis, torch_type(type)) |> from_ref(out)
  end

  @impl true
  def random_uniform(%T{type: {s, _} = type, shape: shape} = out, min, max) when s in [:u, :s] do
    NIF.randint(min, max, shape, torch_type(type)) |> from_ref(out)
  end

  def random_uniform(%T{type: {f, _} = type, shape: shape} = out, min, max) when f in [:f, :bf] do
    NIF.rand(min, max, shape, torch_type(type)) |> from_ref(out)
  end

  @impl true
  def random_normal(%T{type: _type, shape: shape} = out, mu, sigma) do
    NIF.normal(mu, sigma, shape) |> from_ref(out)
  end

  ## Transfer

  @impl true
  def to_batched_list(%T{shape: shape} = out, %T{} = t),
    do: NIF.split(to_ref(t), elem(shape, 0)) |> from_list_ref(out)

  @impl true
  def to_binary(_tensor, _limit \\ nil) do
    raise "operation to_binary is not supported on Torchx.Backend. " <>
            "You must first transfer the tensor to Elixir by calling Nx.backend_transfer/1"
  end

  defp to_blob(tensor, limit \\ nil)
  defp to_blob(%T{} = t, nil), do: NIF.to_blob(to_ref(t))
  defp to_blob(%T{} = t, limit), do: NIF.to_blob(to_ref(t), limit)

  @impl true
  def backend_deallocate(%T{} = t), do: NIF.delete_tensor(to_ref(t))

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
  def from_binary(%T{type: type, shape: shape} = out, binary, _opts) do
    NIF.from_blob(binary, shape, torch_type(type)) |> from_ref(out)
  end

  ## Shape

  @impl true
  def reshape(out, %T{} = t, shape),
    do: NIF.reshape(to_ref(t), shape) |> from_ref(out)

  @impl true
  def as_type(%T{type: type} = out, %T{} = t),
    do: NIF.to_type(to_ref(t), torch_type(type)) |> from_ref(out)

  @impl true
  def squeeze(out, tensor, _axes) do
    NIF.squeeze(to_ref(tensor)) |> from_ref(out)
  end

  @impl true
  def transpose(_out, _tensor, axes), do: IO.inspect(axes)
  # do: NIF.transpose(tensor, dim0, dim1) |> from_ref(out)

  ## Ops

  binary_ops =
    [:add, :subtract, :multiply, :power, :remainder, :divide, :atan2, :min, :max, :quotient] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] ++
      [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] ++
      [:logical_and, :logical_or, :logical_xor] ++
      [:outer]

  for op <- binary_ops do
    @impl true
    def unquote(op)(out, left, right) do
      NIF.unquote(op)(to_ref(left), to_ref(right)) |> from_ref(out)
    end
  end

  unary_ops =
    Enum.map(Nx.Shared.unary_math_funs(), &elem(&1, 0)) ++
      [:abs, :bitwise_not, :ceil, :floor, :negate, :round, :sign]

  # [:count_leading_zeros, :population_count]

  for op <- unary_ops do
    if {op, 1} in NIF.__info__(:functions) do
      @impl true
      def unquote(op)(out, tensor) do
        NIF.unquote(op)(to_ref(tensor)) |> from_ref(out)
      end
    end
  end

  @impl true
  def dot(
        out,
        %T{} = left,
        _axes1,
        %T{} = right,
        _axes2
      ) do
    NIF.dot(to_ref(left), to_ref(right)) |> from_ref(out)
  end

  @impl true
  def cholesky(%T{} = out, %T{} = t) do
    NIF.cholesky(to_ref(t)) |> from_ref(out)
  end

  @impl true
  def qr(
        {q_holder, r_holder},
        tensor,
        opts
      ),
      do: NIF.qr(to_ref(tensor), opts[:mode] == :reduced) |> from_pair_ref({q_holder, r_holder})

  @big_tensor_threshold_bytes 10_000_000

  @impl true
  def inspect(%T{type: {_, elem_size}} = tensor, inspect_opts) do
    alias Inspect.Algebra, as: IA

    limit = if(inspect_opts.limit == :infinity, do: nil, else: inspect_opts.limit + 1)

    result =
      if on_cpu?(tensor) do
        byte_size = nbytes(tensor)
        byte_limit = limit && limit * div(elem_size, 8)

        if min(byte_limit, byte_size) > @big_tensor_threshold_bytes do
          "Torchx tensor is too large to inspect. Explicitly transfer the tensor by calling Nx.backend_transfer/1"
        else
          binary = to_blob(tensor, limit)
          Nx.Backend.inspect(tensor, binary, inspect_opts)
        end
      else
        "Tensors on the GPU cannot be inspected. Explicitly transfer the tensor by calling Nx.backend_transfer/1"
      end

    IA.concat(["Torchx.Backend(#{device(tensor)})", IA.line(), result])
  end

  ## Helpers

  defp unwrap!({:ok, result}), do: result

  defp unwrap!({:error, error}),
    do: raise(RuntimeError, "Torchx: " <> List.to_string(error))

  defp from_ref(maybe_ref, t), do: maybe_ref |> unwrap!() |> to_tensor(t)

  defp from_pair_ref(maybe_ref, {t1, t2}) do
    {left, right} = unwrap!(maybe_ref)
    {to_tensor(left, t1), to_tensor(right, t2)}
  end

  defp from_list_ref(maybe_ref, t),
    do:
      maybe_ref
      |> unwrap!()
      |> Enum.map(&to_tensor(&1, t))

  defp to_ref(%T{data: %TB{ref: ref}}), do: ref

  defp to_ref(%T{} = tensor),
    do: Nx.backend_transfer(tensor, TB) |> to_ref()

  # Update out tensor type here to mark the cases where our type policy mismatches with the libtorch's one.
  defp to_tensor(ref, %T{} = t),
    do: %T{t | data: %__MODULE__{ref: ref}, type: from_torch_type(unwrap!(NIF.type(ref)))}

  defp device(%T{data: %TB{ref: ref}}), do: NIF.device(ref) |> unwrap!() |> List.to_string()
  defp nbytes(%T{data: %TB{ref: ref}}), do: NIF.nbytes(ref) |> unwrap!()
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

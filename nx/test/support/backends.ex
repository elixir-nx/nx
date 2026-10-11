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

defmodule RecordingBackend do
  @moduledoc false
  # Wraps Nx.BinaryBackend and records the `donatable?` flag of every tensor
  # argument of each operation in the process dictionary.
  @behaviour Nx.Backend
  defstruct [:inner]

  def init(opts), do: opts

  def recorded do
    :recording_backend |> Process.get([]) |> Enum.reverse()
  end

  def reset, do: Process.put(:recording_backend, [])

  @doc false
  def wrap(%Nx.Tensor{data: %Nx.BinaryBackend{}} = tensor),
    do: %{tensor | data: %__MODULE__{inner: tensor.data}}

  def wrap(list) when is_list(list), do: Enum.map(list, &wrap/1)
  def wrap(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> wrap() |> List.to_tuple()
  def wrap(other), do: other

  @doc false
  def unwrap(%Nx.Tensor{data: %__MODULE__{inner: inner}} = tensor), do: %{tensor | data: inner}
  def unwrap(list) when is_list(list), do: Enum.map(list, &unwrap/1)

  def unwrap(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> unwrap() |> List.to_tuple()

  def unwrap(other), do: other

  @doc false
  def flags(args) do
    args
    |> List.flatten()
    |> Enum.flat_map(fn
      %Nx.Tensor{data: %__MODULE__{}, donatable?: flag} -> [flag]
      tuple when is_tuple(tuple) -> flags(Tuple.to_list(tuple))
      _ -> []
    end)
  end

  @doc false
  def record(name, args) do
    Process.put(:recording_backend, [{name, flags(args)} | Process.get(:recording_backend, [])])
  end

  funs =
    Nx.Backend.behaviour_info(:callbacks) --
      (Nx.Backend.behaviour_info(:optional_callbacks) ++ Module.definitions_in(__MODULE__, :def))

  for {fun, arity} <- funs do
    args = Macro.generate_arguments(arity, __MODULE__)

    def unquote(fun)(unquote_splicing(args)) do
      record(unquote(fun), [unquote_splicing(tl(args))])
      unwrapped = unwrap([unquote_splicing(args)])

      Nx.BinaryBackend |> apply(unquote(fun), unwrapped) |> wrap()
    end
  end
end

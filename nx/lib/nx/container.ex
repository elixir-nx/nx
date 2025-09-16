defprotocol Nx.Container do
  @moduledoc """
  A protocol that teaches `defn` how to traverse data structures.

  When you invoke a `defn`, its arguments must implement
  a `Nx.LazyContainer` and return a data structure that
  implements `Nx.Container`. Inside `defn`, you can work
  with any container data structure, such as:

    1. numbers/tensors
    2. tuples
    3. maps of any key
    4. any struct that implements `Nx.Container`

  In other words, LazyContainer is how you convert data
  structures that are not meant to work inside `defn` into
  a `Nx.Container`. And a `Nx.Container` is a data structure
  that can be manipulated inside `defn` itself.

  The easiest way to implement `Nx.Container` is by deriving
  it. For example:

      @derive {Nx.Container,
               containers: [:field_name, :other_field]}
      defstruct [:field_name, :other_fields, ...]

  The `:containers` option is required and it must specify a
  list of fields that contains tensors (or other containers).
  Inside `defn`, the container fields will be automatically
  converted to tensor expressions. All other fields will be
  reset to their default value, unless you explicitly declare
  them to be kept:

      @derive {Nx.Container,
               containers: [:field_name, :other_field],
               keep: [:another_field]}
      defstruct [:field_name, :other_fields, ...]

  Note `Nx.LazyContainer` is automatically provided for all
  data structures that implement `Nx.Container`.

  > **Careful!**: If you keep a field, its value will be part
  > of the `Nx.Defn` compiler cache key (i.e. therefore if you
  > give a struct with two different values for a kept field,
  > `Nx.Defn` will have to compile and cache it twice).
  > You must only keep fields that you are certain to be used
  > inside `defn` during compilation time.

  ## Serialization

  If you `@derive {Nx.Container, ...}`, it will automatically
  define a serialization function with the container and keep
  fields you declare. If you expect a struct to be serialized,
  then you must be careful to evolve its schema over time in
  a compatible way. In particular, removing fields will lead to
  crashes. If you change the type of a field value, previously
  serialized structs may still hold the old type. And if you add
  new fields, previously serialized structs won't have such fields
  and therefore be deserialized with its default value.
  """

  @doc """
  Traverses non-recursively tensors in a data structure with `acc` and `fun`.

  `fun` is invoked with each tensor or tensor container in the
  data structure plus an accumulator. It must return a two element
  tuple with the updated value and accumulator.

  This function returns the updated container and the accumulator.

  Given `fun` may receive containers, it is not recursive by default.
  See `Nx.Defn.Composite.traverse/3` for a recursive variant.
  """
  @spec traverse(t(), acc, (t(), acc -> {t(), acc})) :: {t(), acc} when acc: term()
  def traverse(data, acc, fun)

  @doc """
  Reduces non-recursively tensors in a data structure with `acc` and `fun`.

  `fun` is invoked with each tensor or tensor container in the
  data structure plus an accumulator. It must return the new
  accumulator.

  This function the final accumulator.

  Given `fun` may receive containers, it is not recursive by default.
  See `Nx.Defn.Composite.reduce/3` for a recursive variant.
  """
  @spec reduce(t(), acc, (t(), acc -> acc)) :: acc when acc: term()
  def reduce(data, acc, fun)

  @doc """
  Defines how this container must be serialized to disk.

  It receives the container and it must return a three element tuple
  of `{module, list_of_container_tuples, metadata}` where:

    * the `module` to deserialize the container
    * a list of tuples in the shape `{key, container}` with containers to be further serialized
    * additional metadata for serialization/deserialization

  On deserialization, `module.deserialize(list_of_container_tuples, metadata)`
  will be invoked.
  """
  @spec serialize(t()) :: {module(), [{term(), t()}], term()}
  def serialize(struct)
end

defimpl Nx.Container, for: Tuple do
  def traverse(tuple, acc, fun) do
    tuple
    |> Tuple.to_list()
    |> Enum.map_reduce(acc, fun)
    |> then(fn {list, acc} -> {List.to_tuple(list), acc} end)
  end

  def reduce(tuple, acc, fun) do
    tuple
    |> Tuple.to_list()
    |> Enum.reduce(acc, fun)
  end

  def serialize(tuple) do
    pairs = for v <- Tuple.to_list(tuple), do: {[], v}
    {__MODULE__, pairs, :ok}
  end

  def deserialize(pairs, :ok) do
    pairs |> Enum.map(&elem(&1, 1)) |> List.to_tuple()
  end
end

defimpl Nx.Container, for: Map do
  def traverse(map, acc, fun) do
    map
    |> Map.to_list()
    |> Enum.sort()
    |> Enum.map_reduce(acc, fn {k, v}, acc ->
      {v, acc} = fun.(v, acc)
      {{k, v}, acc}
    end)
    |> then(fn {list, acc} -> {Map.new(list), acc} end)
  end

  def reduce(map, acc, fun) do
    map
    |> Map.to_list()
    |> Enum.sort()
    |> Enum.reduce(acc, fn {_, v}, acc -> fun.(v, acc) end)
  end

  def serialize(map) do
    {__MODULE__, Map.to_list(map), :ok}
  end

  def deserialize(pairs, :ok) do
    Map.new(pairs)
  end
end

defimpl Nx.Container, for: [Integer, Float, Complex, Nx.Tensor] do
  def traverse(tensor, acc, fun), do: {tensor, fun.(tensor, acc)}
  def reduce(tensor, acc, fun), do: fun.(tensor, acc)
  def serialize(_), do: raise("cannot be serialized directly")
end

defimpl Nx.Container, for: Any do
  defmacro __deriving__(module, struct, options) do
    containers = Keyword.fetch!(options, :containers)
    keep = Keyword.get(options, :keep, [])

    container_pattern = Enum.map(containers, &field_var(struct, &1))
    keep_pattern = Enum.map(keep, &field_var(struct, &1))
    full_pattern = container_pattern ++ keep_pattern

    updates =
      for field <- containers do
        var = Macro.var(field, Nx.Container)

        quote do
          {unquote(var), acc} = fun.(unquote(var), acc)
        end
      end

    reduces =
      for field <- containers do
        var = Macro.var(field, Nx.Container)

        quote do
          acc = fun.(unquote(var), acc)
        end
      end

    return =
      struct
      |> Map.to_list()
      |> Keyword.drop(keep ++ containers)
      |> Macro.escape()
      |> Keyword.merge(full_pattern)

    quote do
      defimpl Nx.Container, for: unquote(module) do
        def traverse(%{unquote_splicing(full_pattern)} = struct, acc, fun) do
          unquote_splicing(updates)
          {%{unquote_splicing(return)}, acc}
        end

        def reduce(%{unquote_splicing(container_pattern)} = struct, acc, fun) do
          unquote_splicing(reduces)
          acc
        end

        def serialize(%{unquote_splicing(full_pattern)} = struct) do
          {__MODULE__, [unquote_splicing(container_pattern)], [unquote_splicing(keep_pattern)]}
        end

        def deserialize(containers, keep) do
          struct!(unquote(module), containers ++ keep)
        end
      end
    end
  end

  defp field_var(struct, field) do
    unless Map.has_key?(struct, field) do
      raise ArgumentError,
            "cannot derive Nx.Container for struct #{inspect(struct.__struct__)} " <>
              "because it does not have field #{inspect(field)}"
    end

    {field, Macro.var(field, Nx.Container)}
  end

  def traverse(data, _acc, _fun) do
    raise Protocol.UndefinedError, protocol: @protocol, value: data
  end

  def reduce(data, _acc, _fun) do
    raise Protocol.UndefinedError, protocol: @protocol, value: data
  end

  def serialize(data) do
    raise Protocol.UndefinedError, protocol: @protocol, value: data
  end
end

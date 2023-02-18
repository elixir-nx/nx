defmodule Container do
  @derive {Nx.Container, containers: [:a, :b], keep: [:d]}
  defstruct [:a, :b, c: %{}, d: %{}]
end

# Assert empty container emits no warnings
defmodule EmptyContainer do
  @derive {Nx.Container, containers: []}
  defstruct [:var, :fun, :acc]
end

# Assert underscored fields emit no warnings
defmodule UnderscoredContainer do
  @derive {Nx.Container, containers: [], keep: [:__special__]}
  defstruct [:__special__]
end

# The result of lazy container traversal
defmodule LazyWrapped do
  @derive {Nx.Container, containers: [:a, :b, :c]}
  defstruct [:a, :b, :c]
end

# The lazy container itself (which is not a container)
defmodule LazyOnly do
  defstruct [:a, :b, :c]

  defimpl Nx.LazyContainer do
    def traverse(%LazyOnly{a: a, b: b, c: c}, acc, fun) do
      {a, acc} = fun.(Nx.to_template(a), fn -> Nx.tensor(a) end, acc)

      {b, acc} =
        if b do
          fun.(Nx.to_template(b), fn -> raise "don't call b" end, acc)
        else
          {b, acc}
        end

      {c, acc} = Nx.LazyContainer.traverse(c, acc, fun)
      {%LazyWrapped{a: a, b: b, c: c}, acc}
    end
  end
end

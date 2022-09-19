defmodule Container do
  @derive {Nx.Container, containers: [:a, :b], keep: [:d]}
  defstruct [:a, :b, :c, :d]
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
      {b, acc} = fun.(Nx.to_template(b), fn -> raise "don't call b" end, acc)
      {c, acc} = fun.(Nx.to_template(c), fn -> Nx.tensor(c) end, acc)
      {%LazyWrapped{a: a, b: b, c: c}, acc}
    end
  end
end

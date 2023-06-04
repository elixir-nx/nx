defmodule Nx.Defn.CompilationDiff do
  defstruct [:left, :right, :compatible, :nesting]

  defp is_valid_container?(t) do
    impl = Nx.Container.impl_for(t)
    not is_nil(impl) and impl != Nx.Container.Any
  end

  def build(left, right) do
    l = is_valid_container?(left)
    r = is_valid_container?(right)

    cond do
      not l and not r ->
        %__MODULE__{left: left, right: right, compatible: left == right}

      not l or not r ->
        %__MODULE__{left: left, right: right, compatible: false}

      Nx.Container.impl_for(left) != Nx.Container.impl_for(right) ->
        %__MODULE__{left: left, right: right, compatible: false}

      l and r ->
        {diff, acc} =
          Nx.Defn.Composite.traverse(left, Nx.Defn.Composite.flatten_list([right]), fn
            left, [] ->
              {%__MODULE__{left: left}, :incompatible_sizes}

            left, [right | acc] ->
              {
                %__MODULE__{
                  left: left,
                  right: right,
                  compatible: Nx.compatible?(left, right)
                },
                acc
              }
          end)

        if acc == :incompatible_sizes do
          %__MODULE__{left: left, right: right, compatible: false}
        else
          diff
        end
    end
  end

  def build_and_inspect(left, right) do
    left
    |> build(right)
    |> inspect()
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(%Nx.Defn.CompilationDiff{left: left, right: nil}, _opts) do
      inspect_as_template(left)
    end

    def inspect(%Nx.Defn.CompilationDiff{left: left, compatible: true}, _opts) do
      inspect_as_template(left)
    end

    def inspect(%Nx.Defn.CompilationDiff{left: left, right: right}, _opts) do
      concat([
        IO.ANSI.light_black_background(),
        IO.ANSI.green(),
        line(),
        "<<<<<<<<<<",
        line(),
        inspect_as_template(left),
        line(),
        "==========",
        line(),
        IO.ANSI.red(),
        inspect_as_template(right),
        line(),
        ">>>>>>>>>>",
        line(),
        IO.ANSI.reset()
      ])
    end

    defp inspect_as_template(data) do
      if is_number(data) or is_tuple(data) or
           (is_map(data) and Nx.Container.impl_for(data) != Nx.Container.Any) do
        data
        |> Nx.to_template()
        |> Kernel.inspect(custom_options: [skip_template_backend_header: true])
      else
        inspect(data)
      end
    end
  end
end

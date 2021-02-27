defmodule Nx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.1.0-dev"

  def project do
    [
      app: :nx,
      name: "Nx",
      version: @version,
      elixir: "~> 1.11",
      deps: deps(),
      docs: docs()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:ex_doc, "~> 0.23", only: :dev}
    ]
  end

  defp docs do
    [
      main: "Nx",
      source_ref: "v#{@version}",
      source_url: @source_url,
      groups_for_functions: [
        "Functions: Aggregates": &(&1[:type] == :aggregation),
        "Functions: Conversion": &(&1[:type] == :conversion),
        "Functions: Creation": &(&1[:type] in [:creation, :random]),
        "Functions: Element-wise": &(&1[:type] == :element),
        "Functions: Linalg": &(&1[:type] == :linalg),
        "Functions: N-dim": &(&1[:type] == :ndim),
        "Functions: Shape": &(&1[:type] == :shape),
        "Functions: Type": &(&1[:type] == :type)
      ],
      groups_for_modules: [
        # Nx,
        # Nx.Async,
        # Nx.Defn,
        # Nx.Defn.Kernel

        "Backends": [
          Nx.Backend,
          Nx.BinaryBackend,
          Nx.TemplateBackend,
          Nx.Type
        ],
        "Compilers": [
          Nx.Defn.Compiler,
          Nx.Defn.Evaluator,
          Nx.Defn.Expr
        ],
        "Structs": [
          Nx.Heatmap,
          Nx.Tensor
        ],
      ]
    ]
  end
end

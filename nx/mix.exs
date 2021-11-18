if :erlang.system_info(:otp_release) < '24' do
  Mix.raise("Nx requires Erlang/OTP 24+")
end

defmodule Nx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.1.0-dev"

  def project do
    [
      app: :nx,
      name: "Nx",
      version: @version,
      elixir: "~> 1.12-dev",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      env: [default_backend: {Nx.BinaryBackend, []}]
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

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
        "Functions: Backend": &(&1[:type] == :backend),
        "Functions: Conversion": &(&1[:type] == :conversion),
        "Functions: Creation": &(&1[:type] in [:creation, :random]),
        "Functions: Element-wise": &(&1[:type] == :element),
        "Functions: N-dim": &(&1[:type] == :ndim),
        "Functions: Shape": &(&1[:type] == :shape),
        "Functions: Type": &(&1[:type] == :type)
      ],
      groups_for_modules: [
        # Nx,
        # Nx.Defn,
        # Nx.Defn.Kernel,
        # Nx.LinAlg,

        Backends: [
          Nx.Backend,
          Nx.BinaryBackend,
          Nx.TemplateBackend,
          Nx.Type
        ],
        Protocols: [
          Nx.Container,
          Nx.Stream
        ],
        Structs: [
          Nx.Heatmap,
          Nx.Tensor
        ],
        Compilers: [
          Nx.Defn.Compiler,
          Nx.Defn.Evaluator,
          Nx.Defn.Expr,
          Nx.Defn.Tree
        ]
      ]
    ]
  end
end

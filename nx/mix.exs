if :erlang.system_info(:otp_release) < '24' do
  Mix.raise("Nx requires Erlang/OTP 24+")
end

defmodule Nx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.1.0"

  def project do
    [
      app: :nx,
      version: @version,
      elixir: "~> 1.12",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),
      name: "Nx",
      description: "Multi-dimensional arrays (tensors) and numerical definitions for Elixir",
      package: package(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      env: [default_backend: {Nx.BinaryBackend, []}, default_defn_options: []]
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  defp deps do
    [
      {:ex_doc, "~> 0.27", only: :docs}
    ]
  end

  defp package do
    [
      maintainers: ["Sean Moriarity", "José Valim", "Paulo Valente"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "Nx",
      logo: "numbat.png",
      source_url_pattern: "#{@source_url}/blob/v#{@version}/nx/%{path}#L%{line}",
      groups_for_functions: [
        "Functions: Aggregates": &(&1[:type] == :aggregation),
        "Functions: Backend": &(&1[:type] == :backend),
        "Functions: Conversion": &(&1[:type] == :conversion),
        "Functions: Creation": &(&1[:type] in [:creation, :random]),
        "Functions: Element-wise": &(&1[:type] == :element),
        "Functions: Indexed": &(&1[:type] == :indexed),
        "Functions: N-dim": &(&1[:type] == :ndim),
        "Functions: Shape": &(&1[:type] == :shape),
        "Functions: Type": &(&1[:type] == :type),
        "Functions: Window": &(&1[:type] == :window)
      ],
      groups_for_modules: [
        # Nx,
        # Nx.Constants,
        # Nx.Defn,
        # Nx.Defn.Kernel,
        # Nx.LinAlg,

        Protocols: [
          Nx.Container,
          Nx.Stream
        ],
        Structs: [
          Nx.Heatmap,
          Nx.Tensor
        ],
        Backends: [
          Nx.Backend,
          Nx.BinaryBackend,
          Nx.TemplateBackend,
          Nx.Type
        ],
        Compilers: [
          Nx.Defn.Compiler,
          Nx.Defn.Composite,
          Nx.Defn.Evaluator,
          Nx.Defn.Expr,
          Nx.Defn.Token,
          Nx.Defn.Tree
        ]
      ]
    ]
  end
end

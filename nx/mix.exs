defmodule Nx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.9.2"

  def project do
    [
      app: :nx,
      version: @version,
      elixir: "~> 1.15",
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
      mod: {Nx.Application, []},
      env: [default_backend: {Nx.BinaryBackend, []}, default_defn_options: []]
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  defp deps do
    [
      {:complex, "~> 0.5"},
      {:telemetry, "~> 0.4.0 or ~> 1.0"},
      {:ex_doc, "~> 0.29", only: :docs}
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
      before_closing_body_tag: &before_closing_body_tag/1,
      extras: [
        "CHANGELOG.md",
        "guides/intro-to-nx.livemd",
        "guides/advanced/vectorization.livemd",
        "guides/advanced/aggregation.livemd",
        "guides/exercises/exercises-1-20.livemd"
      ],
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"],
      groups_for_docs: [
        Guards: &(&1[:type] in [:guards]),
        "Functions: Aggregates": &(&1[:type] == :aggregation),
        "Functions: Backend": &(&1[:type] == :backend),
        "Functions: Conversion": &(&1[:type] == :conversion),
        "Functions: Creation": &(&1[:type] in [:creation, :random]),
        "Functions: Cumulative": &(&1[:type] == :cumulative),
        "Functions: Element-wise": &(&1[:type] == :element),
        "Functions: Indexed": &(&1[:type] == :indexed),
        "Functions: N-dim": &(&1[:type] == :ndim),
        "Functions: Shape": &(&1[:type] == :shape),
        "Functions: Vectorization": &(&1[:type] == :vectorization),
        "Functions: Type": &(&1[:type] == :type),
        "Functions: Window": &(&1[:type] == :window)
      ],
      groups_for_modules: [
        # Nx,
        # Nx.Constants,
        # Nx.Defn,
        # Nx.Defn.Kernel,
        # Nx.LinAlg,
        # Nx.Serving,

        Protocols: [
          Nx.Container,
          Nx.LazyContainer,
          Nx.Stream
        ],
        Structs: [
          Nx.Batch,
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
      ],
      groups_for_extras: [
        Exercises: ~r"^guides/exercises/",
        Advanced: ~r"^guides/advanced/"
      ]
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/katex.min.css" integrity="sha384-beuqjL2bw+6DBM2eOpr5+Xlw+jiH44vMdVQwKxV28xxpoInPHTVmSvvvoPq9RdSh" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/katex.min.js" integrity="sha384-aaNb715UK1HuP4rjZxyzph+dVss/5Nx3mLImBe9b0EW4vMUkc1Guw4VRyQKBC0eG" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.19/dist/contrib/auto-render.min.js" integrity="sha384-+XBljXPPiv+OzfbB3cVmLHf4hdUFHlWNZN5spNQ7rmHTXpd7WvJum6fIACpNNfIR" crossorigin="anonymous"
            onload="renderMathInElement(document.body);"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ]
        });
      });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end

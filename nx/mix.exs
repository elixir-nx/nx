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
      elixir: "~> 1.13",
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
      {:complex, "~> 0.4.0"},
      {:ex_doc, "~> 0.28.3", only: :docs}
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

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css" integrity="sha384-t5CR+zwDAROtph0PXGte6ia8heboACF9R5l/DiY+WZ3P2lxNgvJkQk5n7GPvLMYw" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js" integrity="sha384-FaFLTlohFghEIZkw6VGwmf9ISTubWAVYW8tG8+w2LAIftJEULZABrF9PPFv+tVkH" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/contrib/auto-render.min.js" integrity="sha384-bHBqxz8fokvgoJ/sc17HODNxa42TlaEhB+w8ZJXTc2nZf1VgEaFZeZvT4Mznfz0v" crossorigin="anonymous"
        onload="renderMathInElement(document.body, {delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '\\[', right: '\\]', display: true},
          {left: '$', right: '$', display: false}
        ]});"></script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end

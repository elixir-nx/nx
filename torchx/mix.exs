defmodule Torchx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.1.0-dev"

  def project do
    [
      app: :torchx,
      name: "Torchx",
      version: @version,
      elixir: "~> 1.12-dev",
      deps: deps(),
      docs: docs(),
      compilers: [:elixir_make] ++ Mix.compilers(),
      elixirc_paths: elixirc_paths(Mix.env())
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, path: "../nx"},
      {:elixir_make, "~> 0.6"},
      {:ex_doc, "~> 0.23", only: :dev}
    ]
  end

  defp docs do
    [
      main: "Torchx",
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end
end

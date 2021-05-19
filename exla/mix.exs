defmodule EXLA.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/exla"
  @version "0.1.0-dev"

  def project do
    [
      app: :exla,
      name: "EXLA",
      version: @version,
      elixir: "~> 1.12-dev",
      deps: deps(),
      docs: docs(),
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: %{
        "MIX_CURRENT_PATH" => File.cwd!(),
        "ERTS_VERSION" => List.to_string(:erlang.system_info(:version))
      }
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {EXLA.Application, []},
      env: [clients: [default: []]]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, path: "../nx"},
      {:elixir_make, "~> 0.6"},
      {:benchee, "~> 1.0", only: :dev},
      {:ex_doc, "~> 0.23", only: :dev}
    ]
  end

  defp docs do
    [
      main: "EXLA",
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end
end

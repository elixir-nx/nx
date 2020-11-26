defmodule Exla.MixProject do
  use Mix.Project

  def project do
    [
      app: :exla,
      version: "0.0.6-dev",
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: [:elixir_make] ++ Mix.compilers()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Exla.Application, []},
      env: [clients: [default: []]]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:elixir_make, "~> 0.6"},
      {:benchee, "~> 1.0", only: :dev}
    ]
  end
end

# This mix.exs is only used avaialble so "mix format" works from root.
defmodule NxRoot do
  use Mix.Project

  def project do
    [
      app: :nx_root,
      version: "0.1",
      deps: [{:nx, path: "nx"}]
    ]
  end
end

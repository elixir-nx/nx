# Provide aliases for running common tasks
defmodule NxRoot do
  use Mix.Project

  def project do
    [
      app: :nx_root,
      version: "0.1.0",
      deps: [{:nx, path: "nx"}],
      aliases: [
        setup: cmd("deps.get"),
        compile: cmd("compile"),
        test: cmd("test")
      ]
    ]
  end

  defp cmd(command) do
    ansi = IO.ANSI.enabled?()
    base = ["--erl", "-elixir ansi_enabled #{ansi}", "-S", "mix", command]

    for app <- ~w(nx exla torchx) do
      fn args ->
        {_, res} = System.cmd("elixir", base ++ args, into: IO.binstream(:stdio, :line), cd: app)

        if res > 0 do
          System.at_exit(fn _ -> exit({:shutdown, 1}) end)
        end
      end
    end
  end
end

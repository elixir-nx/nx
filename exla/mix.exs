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
      # We want to always trigger XLA compilation when XLA_BUILD is set,
      # otherwise its Makefile will run only upon the initial compilation
      compilers:
        if(xla_build?(), do: [:xla], else: []) ++ [:exla, :elixir_make] ++ Mix.compilers(),
      aliases: aliases()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {EXLA.Application, []},
      env: [
        clients: [
          default: [],
          cuda: [platform: :cuda],
          rocm: [platform: :rocm],
          tpu: [platform: :tpu]
        ]
      ]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, path: "../nx"},
      {:xla, "~> 0.2.0", runtime: false},
      {:elixir_make, "~> 0.6", runtime: false},
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

  defp aliases do
    [
      "compile.xla": "deps.compile xla",
      "compile.exla": &compile/1
    ]
  end

  # We keep track of the current XLA archive path in xla_snapshot.txt.
  # Whenever the path changes, we extract it again and Makefile picks
  # up this change
  defp compile(_) do
    xla_archive_path = XLA.archive_path!()

    cache_dir = Path.join(__DIR__, "cache")
    xla_snapshot_path = Path.join(cache_dir, "xla_snapshot.txt")
    xla_extension_path = Path.join(cache_dir, "xla_extension")

    case File.read(xla_snapshot_path) do
      {:ok, ^xla_archive_path} ->
        :ok

      _ ->
        File.rm_rf!(xla_extension_path)

        Mix.shell().info("Unpacking #{xla_archive_path} into #{cache_dir}")

        case :erl_tar.extract(xla_archive_path, [:compressed, cwd: cache_dir]) do
          :ok -> :ok
          {:error, term} -> Mix.raise("failed to extract xla archive, reason: #{inspect(term)}")
        end

        File.write!(xla_snapshot_path, xla_archive_path)
    end

    {:ok, []}
  end

  defp xla_build?() do
    System.get_env("XLA_BUILD") == "true"
  end
end

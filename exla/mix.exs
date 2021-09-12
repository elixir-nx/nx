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
      compilers: [:exla, :elixir_make] ++ Mix.compilers(),
      aliases: aliases(),
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
      # TODO: adjust
      {:xla, "~> 0.1.1-dev", runtime: false, github: "jonatanklosko/xla", branch: "jk-build"},
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
      "compile.exla": &compile/1
    ]
  end

  # Custom compiler, extracts the archive from xla priv directory
  # into exla priv directory

  defp compile(_) do
    exla_priv_path = Path.join(__DIR__, "priv")
    unpacked_archive_path = Path.join(exla_priv_path, "xla_extension")

    unless File.exists?(unpacked_archive_path) do
      case :code.priv_dir(:xla) do
        {:error, :bad_name} ->
          Mix.raise("could not find xla priv directory")

        xla_priv_path ->
          archive_path = Path.join(xla_priv_path, "xla_extension.tar.gz")

          case :erl_tar.extract(archive_path, [:compressed, cwd: exla_priv_path]) do
            :ok ->
              :ok

            {:error, term} ->
              Mix.raise("failed to extract xla archive, reason: #{inspect(term)}")
          end
      end
    end

    # Symlink the priv directory
    Mix.Project.build_structure()

    {:ok, []}
  end
end

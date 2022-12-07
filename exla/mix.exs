defmodule EXLA.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.4.1"

  def project do
    [
      app: :exla,
      version: @version,
      elixir: "~> 1.13",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),

      # Package
      name: "EXLA",
      description: "Google's XLA (Accelerated Linear Algebra) compiler/backend for Nx",
      package: package(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ],
      compilers: [:exla, :elixir_make] ++ Mix.compilers(),
      aliases: [
        "compile.exla": &compile/1
      ],
      make_env: %{
        "MIX_BUILD_EMBEDDED" => "#{Mix.Project.config()[:build_embedded]}"
      }
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {EXLA.Application, []},
      env: [
        clients: [
          cuda: [platform: :cuda],
          rocm: [platform: :rocm],
          tpu: [platform: :tpu],
          host: [platform: :host]
        ],
        preferred_clients: [:cuda, :rocm, :tpu, :host]
      ]
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  defp deps do
    [
      {:nx, "~> 0.4.1"},
      # {:nx, path: "../nx"},
      {:telemetry, "~> 0.4.0 or ~> 1.0"},
      {:xla, "~> 0.4.0", runtime: false},
      {:elixir_make, "~> 0.6", runtime: false},
      {:benchee, "~> 1.0", only: :dev},
      {:ex_doc, "~> 0.29.0", only: :docs}
    ]
  end

  defp docs do
    [
      main: "EXLA",
      source_url_pattern: "#{@source_url}/blob/v#{@version}/exla/%{path}#L%{line}",
      extras: [
        "guides/rotating-image.livemd",
        "CHANGELOG.md"
      ],
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"],
      groups_for_modules: [
        # EXLA,
        # EXLA.Backend,

        Bindings: [
          EXLA.BinaryBuffer,
          EXLA.Builder,
          EXLA.Client,
          EXLA.Computation,
          EXLA.DeviceBuffer,
          EXLA.Executable,
          EXLA.Lib,
          EXLA.Op,
          EXLA.Shape
        ]
      ]
    ]
  end

  defp package do
    [
      maintainers: ["Sean Moriarity", "José Valim"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
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
end

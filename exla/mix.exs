defmodule EXLA.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.7.0-dev"

  def project do
    [
      app: :exla,
      version: @version,
      elixir: "~> 1.14",
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
      compilers: [:extract_xla, :cached_make] ++ Mix.compilers(),
      aliases: [
        "compile.extract_xla": &extract_xla/1,
        "compile.cached_make": &cached_make/1
      ],
      make_env: fn ->
        priv_path = Path.join(Mix.Project.app_path(), "priv")
        cwd_relative_to_priv = relative_to(File.cwd!(), priv_path)

        %{
          "MIX_BUILD_EMBEDDED" => "#{Mix.Project.config()[:build_embedded]}",
          "CWD_RELATIVE_TO_PRIV_PATH" => cwd_relative_to_priv
        }
      end
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
        preferred_clients: [:cuda, :rocm, :tpu, :host],
        automatic_device_transfer_platforms: [:host]
      ]
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  defp deps do
    [
      # {:nx, "~> 0.6.0"},
      {:nx, path: "../nx"},
      {:telemetry, "~> 0.4.0 or ~> 1.0"},
      {:xla, "~> 0.6.0", runtime: false},
      {:elixir_make, "~> 0.6", runtime: false},
      {:benchee, "~> 1.0", only: :dev},
      {:ex_doc, "~> 0.29", only: :docs}
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
  defp extract_xla(_) do
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
          :ok -> File.write!(xla_snapshot_path, xla_archive_path)
          {:error, term} -> Mix.raise("failed to extract xla archive, reason: #{inspect(term)}")
        end
    end

    {:ok, []}
  end

  defp cached_make(_) do
    contents =
      for path <- Path.wildcard("c_src/**/*"),
          {:ok, contents} <- [File.read(path)],
          do: contents

    md5 =
      [XLA.archive_path!() | contents]
      |> :erlang.md5()
      |> Base.encode32(padding: false, case: :lower)

    cache_key =
      "elixir-#{System.version()}-erts-#{:erlang.system_info(:version)}-xla-#{Application.spec(:xla, :vsn)}-exla-#{@version}-#{md5}"

    cached_so = Path.join([xla_cache_dir(), "exla", cache_key, "libexla.so"])
    cached? = File.exists?(cached_so)

    if cached? do
      Mix.shell().info("Using libexla.so from #{cached_so}")
      File.cp!(cached_so, "cache/libexla.so")
    end

    result = Mix.Tasks.Compile.ElixirMake.run([])

    if not cached? and match?({:ok, _}, result) do
      Mix.shell().info("Caching libexla.so at #{cached_so}")
      File.mkdir_p!(Path.dirname(cached_so))
      File.cp!("cache/libexla.so", cached_so)
    end

    result
  end

  defp xla_cache_dir() do
    # The directory where we store all the archives
    if dir = System.get_env("XLA_CACHE_DIR") do
      Path.expand(dir)
    else
      :filename.basedir(:user_cache, "xla")
    end
  end

  # Returns `path` relative to the `from` directory.
  defp relative_to(path, from) do
    path_parts = path |> Path.expand() |> Path.split()
    from_parts = from |> Path.expand() |> Path.split()
    {path_parts, from_parts} = drop_common_prefix(path_parts, from_parts)
    root_relative = for _ <- from_parts, do: ".."
    Path.join(root_relative ++ path_parts)
  end

  defp drop_common_prefix([h | left], [h | right]), do: drop_common_prefix(left, right)
  defp drop_common_prefix(left, right), do: {left, right}
end

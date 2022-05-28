defmodule Torchx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.3.0-dev"

  @valid_targets ["cpu", "cu102", "cu111"]

  @libtorch_version System.get_env("LIBTORCH_VERSION", "1.10.2")
  @libtorch_target System.get_env("LIBTORCH_TARGET", "cpu")

  @libtorch_base "libtorch"
  @libtorch_dir System.get_env(
                  "LIBTORCH_DIR",
                  Path.join(__DIR__, "cache/libtorch-#{@libtorch_version}-#{@libtorch_target}")
                )
  @libtorch_compilers [:torchx, :elixir_make]

  def project do
    [
      app: :torchx,
      version: @version,
      elixir: "~> 1.13",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),

      # Package
      name: "Torchx",
      description: "LibTorch bindings and backend for Nx",
      package: package(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ],

      # Compilers
      compilers: @libtorch_compilers ++ Mix.compilers(),
      aliases: aliases(),
      make_env: %{
        "LIBTORCH_DIR" => @libtorch_dir,
        "LIBTORCH_BASE" => @libtorch_base,
        "MIX_BUILD_EMBEDDED" => "#{Mix.Project.config()[:build_embedded]}"
      }
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, path: "../nx"},
      # {:nx, "~> 0.2.0"},
      {:dll_loader_helper, "~> 0.1.0"},
      {:elixir_make, "~> 0.6"},
      {:ex_doc, "~> 0.28.3", only: :docs}
    ]
  end

  defp docs do
    [
      main: "Torchx",
      source_url_pattern: "#{@source_url}/blob/v#{@version}/torchx/%{path}#L%{line}",
      extras: [
        "CHANGELOG.md"
      ]
    ]
  end

  defp package do
    [
      maintainers: ["Paulo Valente", "José Valim"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: [
        "lib",
        "mix.exs",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "c_src",
        "CMakeLists.txt",
        "Makefile",
        "Makefile.win"
      ]
    ]
  end

  defp aliases do
    [
      "compile.torchx": &download_and_unzip/1
    ]
  end

  defp download_and_unzip(args) do
    libtorch_cache = @libtorch_dir
    cache_dir = Path.dirname(libtorch_cache)

    if "--force" in args do
      File.rm_rf!(cache_dir)
    end

    if File.dir?(libtorch_cache) do
      {:ok, []}
    else
      download_and_unzip(cache_dir, libtorch_cache)
    end
  end

  defp download_and_unzip(cache_dir, libtorch_cache) do
    File.mkdir_p!(cache_dir)
    libtorch_zip = libtorch_cache <> ".zip"

    unless File.exists?(libtorch_zip) do
      # Download libtorch

      # This is so we don't forget to update the URLs below when we want to update libtorch
      if @libtorch_target != "cpu" and {:unix, :darwin} == :os.type() do
        Mix.raise("No CUDA support on OSX")
      end

      # Check if target is valid
      unless Enum.member?(@valid_targets, @libtorch_target) do
        Mix.raise("Invalid target, please use one of #{inspect(@valid_targets)}")
      end

      url =
        case :os.type() do
          {:unix, :linux} ->
            "https://download.pytorch.org/libtorch/#{@libtorch_target}/libtorch-cxx11-abi-shared-with-deps-#{@libtorch_version}%2B#{@libtorch_target}.zip"

          {:unix, :darwin} ->
            # MacOS
            # pytorch only provides official pre-built binaries for x86_64
            case List.to_string(:erlang.system_info(:system_architecture)) do
              "x86_64" <> _ ->
                "https://download.pytorch.org/libtorch/#{@libtorch_target}/libtorch-macos-#{@libtorch_version}.zip"

              _ ->
                Mix.raise(
                  "Please download pre-built/compile LibTorch and set environment variable LIBTORCH_DIR"
                )
            end

          {:win32, :nt} ->
            # Windows
            "https://download.pytorch.org/libtorch/#{@libtorch_target}/libtorch-win-shared-with-deps-#{@libtorch_version}%2B#{@libtorch_target}.zip"

          os ->
            Mix.raise("OS #{inspect(os)} is not supported")
        end

      download!(url, libtorch_zip)
    end

    # Unpack libtorch and move to the target cache dir

    {:ok, _} =
      libtorch_zip |> String.to_charlist() |> :zip.unzip(cwd: String.to_charlist(cache_dir))

    # Keep libtorch cache scoped by version and target
    File.rename!(Path.join(cache_dir, "libtorch"), libtorch_cache)

    :ok
  end

  defp assert_network_tool!() do
    unless network_tool() do
      raise "expected either curl or wget to be available in your system, but neither was found"
    end
  end

  defp download!(url, dest) do
    assert_network_tool!()

    case download(url, dest) do
      :ok ->
        :ok

      _ ->
        raise "unable to download libtorch from #{url}"
    end
  end

  defp download(url, dest) do
    {command, args} =
      case network_tool() do
        :curl -> {"curl", ["--fail", "-L", url, "-o", dest]}
        :wget -> {"wget", ["-O", dest, url]}
      end

    case System.cmd(command, args) do
      {_, 0} -> :ok
      _ -> :error
    end
  end

  defp network_tool() do
    cond do
      executable_exists?("curl") -> :curl
      executable_exists?("wget") -> :wget
      true -> nil
    end
  end

  defp executable_exists?(name), do: not is_nil(System.find_executable(name))
end

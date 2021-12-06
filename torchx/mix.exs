defmodule Torchx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.1.0-dev"

  @libtorch_version "1.9.1"
  @libtorch_target "cpu"
  @libtorch_nerves_url_base "https://github.com/cocoa-xu/elixir-nx-libtorch/releases/download/v#{@libtorch_version}"

  def project do
    [
      app: :torchx,
      name: "Torchx",
      version: @version,
      elixir: "~> 1.12-dev",
      deps: deps(),
      docs: docs(),
      compilers: compilers() ++ Mix.compilers(),
      elixirc_paths: elixirc_paths(Mix.env()),
      aliases: aliases(),
      make_env: %{
        "LIBTORCH_DIR" => System.get_env("LIBTORCH_DIR", get_libtorch_cache()),
        "NERVES_BUILD" => is_nerves_build()
      }
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

  defp compilers do
    if System.get_env("LIBTORCH_DIR"), do: [:elixir_make], else: [:torchx, :elixir_make]
  end

  defp aliases do
    [
      "compile.torchx": &compile/1
    ]
  end

  defp get_libtorch_cache do
    Path.join(__DIR__, "cache/libtorch-#{@libtorch_version}-#{@libtorch_target}-#{libtorch_build_target_name()}")
  end

  defp compile(args) do
    libtorch_cache = get_libtorch_cache()
    cache_dir = Path.dirname(libtorch_cache)

    if "--force" in args do
      File.rm_rf!(cache_dir)
    end

    if File.dir?(libtorch_cache) do
      {:ok, []}
    else
      install_libtorch(cache_dir, libtorch_cache)
    end
  end

  defp libtorch_build_target do
    case Kernel.function_exported?(Mix, :target, 0) do
      true -> {:nerves, Mix.target()}
      _ -> :os.type()
    end
  end

  defp libtorch_build_target_name do
    case libtorch_build_target() do
      {:unix, _} -> "host"
      {:nerves, target} -> Atom.to_charlist(target)
    end
  end

  defp is_nerves_build do
    case libtorch_build_target() do
      {:nerves, _} -> "nerves_build_yes"
      _ -> "nerves_build_no"
    end
  end

  defp install_libtorch(cache_dir, libtorch_cache) do
    File.mkdir_p!(cache_dir)
    libtorch_zip = libtorch_cache <> ".zip"

    unless File.exists?(libtorch_zip) do
      # Download libtorch

      # This is so we don't forget to update the URLs below when we want to update libtorch
      unless @libtorch_version == "1.9.1" and @libtorch_target == "cpu" do
        raise "ensure the download URLs match the default libtorch version"
      end

      url =
        # allow users to specify their own build of libtorch
        # export LIBTORCH_DOWNLOAD_URL=https://custom.domain/libtorch-version.zip
        case System.get_env("LIBTORCH_DOWNLOAD_URL") do
          :nil ->
            case libtorch_build_target() do
              {:unix, :linux} ->
                "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcpu.zip"

              {:unix, :darwin} ->
                # MacOS
                "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.1.zip"

              {:nerves, target} ->
                case target do
                  :rpi4 ->
                    "#{@libtorch_nerves_url_base}/libtorch-1.9.1-aarch64.zip"
                  :x86_64 ->
                    "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcpu.zip"
                  target when target in [:rpi3, :rpi3a, :rpi2, :bbb, :osd32mp1] ->
                    "#{@libtorch_nerves_url_base}/libtorch-1.9.1-armv7.zip"
                  target when target in [:rpi0, :rpi] ->
                    "#{@libtorch_nerves_url_base}/libtorch-1.9.1-armv6.zip"
                end
            end
          url -> url
        end

      # skip download if it is on local filesystem
      case File.exists?(url) do
        true -> File.cp!(url, libtorch_zip)
        false -> download!(url, libtorch_zip)
      end
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

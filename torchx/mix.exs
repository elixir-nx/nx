defmodule Torchx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.1.0-dev"
  @default_libtorch_version File.read!(".default_libtorch_version")

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
      aliases: aliases()
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

  defp compile(args) do
    cache_dir = Path.join(__DIR__, "cache")
    # TODO: This directory name must also include the libtorch version.
    # Otherwise if someone depends on torchx as a git dependency, they
    # can upgrade the version and we will think it is cached but it is actually old.
    libtorch_dir = Path.join(cache_dir, "libtorch_#{@default_libtorch_version}")

    if "--force" in args do
      File.rm_rf!(cache_dir)
    end

    if File.dir?(libtorch_dir) do
      {:ok, []}
    else
      install_libtorch(cache_dir, libtorch_dir)
    end
  end

  defp install_libtorch(cache_dir, libtorch_dir) do
    File.mkdir_p!(cache_dir)

    libtorch_zip =
      System.get_env(
        "LIBTORCH_ZIP",
        Path.join(cache_dir, "libtorch_#{@default_libtorch_version}.zip")
      )

    unless File.exists?(libtorch_zip) do
      # Download libtorch

      # Sanity check for future changes or updates on the default libtorch version
      if @default_libtorch_version != "1_9_1_cpu",
        do: raise("ensure the download URLs match the default libtorch version")

      url =
        case :os.type() do
          {:unix, :linux} ->
            "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcpu.zip"

          {:unix, :darwin} ->
            # MacOS
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.1.zip"

          os ->
            raise "OS #{os} is not supported"
        end

      download!(url, libtorch_zip)
    end

    # Unpack libtorch to libtorch_dir

    {:ok, _} =
      libtorch_zip |> String.to_charlist() |> :zip.unzip(cwd: String.to_charlist(cache_dir))

    unpack_dir = Path.join(cache_dir, "libtorch")

    unless unpack_dir == libtorch_dir do
      File.rename!(unpack_dir, libtorch_dir)
    end

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

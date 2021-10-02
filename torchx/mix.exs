defmodule Torchx.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/nx"
  @version "0.1.0-dev"

  def project do
    [
      app: :torchx,
      name: "Torchx",
      version: @version,
      elixir: "~> 1.12-dev",
      deps: deps(),
      docs: docs(),
      compilers: [:torchx, :elixir_make] ++ Mix.compilers(),
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

  defp aliases do
    [
      "compile.torchx": &compile/1
    ]
  end

  defp compile(args) do
    cache_dir = Path.join(__DIR__, "cache")

    force_download = "--force" in args

    libtorch_dir =
      case System.get_env("LIBTORCH_DIR") do
        nil ->
          Path.join(cache_dir, "libtorch")

        val ->
          val
      end

    with {:force_download, false} <- {:force_download, force_download},
         {:libtorch_dir_exists, true} <- {:libtorch_dir_exists, File.dir?(libtorch_dir)},
         {:ok, content} <- File.read(Path.join(libtorch_dir, "build-hash")),
         true <- String.contains?(content, "pytorch") do
      :noop
    else
      {:force_download, true} ->
        File.rm_rf!(cache_dir)
        install_libtorch(cache_dir, libtorch_dir)

      {:libtorch_dir_exists, false} ->
        install_libtorch(cache_dir, libtorch_dir)

      _ ->
        msg = "directory #{libtorch_dir} set in LIBTORCH_DIR does not contain libtorch"
        IO.puts(msg)

        {:error,
         [
           %Mix.Task.Compiler.Diagnostic{
             compiler_name: "torchx",
             details: msg,
             file: "mix.exs",
             message: msg,
             position: nil,
             severity: :error
           }
         ]}
    end
  end

  defp install_libtorch(cache_dir, libtorch_dir) do
    File.mkdir_p!(cache_dir)

    libtorch_zip = System.get_env("LIBTORCH_ZIP", Path.join(cache_dir, "libtorch.zip"))

    os =
      case System.cmd("uname", ~w(-s)) do
        {"Linux" <> _, 0} ->
          :linux

        {"Darwin" <> _, 0} ->
          :mac

        {os, 0} ->
          raise "OS #{os} is not supported"

        _ ->
          raise "unable to fetch current OS"
      end

    unless File.exists?(libtorch_zip) do
      # Download libtorch

      url =
        case os do
          :linux ->
            "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcpu.zip"

          :mac ->
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.1.zip"
        end

      download!(url, libtorch_zip)
    end

    # Unpack libtorch to libtorch_dir

    {:ok, _} = libtorch_zip |> String.to_charlist() |> :zip.unzip(cwd: String.to_charlist(cache_dir))

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

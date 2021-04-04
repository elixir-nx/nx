defmodule EXLA.Computation do
  @moduledoc """
  Wrapper around XLA's computation.
  """

  @enforce_keys [:ref, :output_shape]
  defstruct [:ref, :output_shape]

  alias EXLA.{Client, Computation, Executable}

  @doc """
  Compiles a computation into an executable.

  ## Options

    * `:num_replicas` - the number of replicas this computation will run on
    * `:use_spmd` - enable single-program multiple data

  Currently those options do not have an effect as they related to running the
  same compiled executabled on multiple replicas.

  Some options apply to TPU only and therefore are not currently supported:

    * `:num_partitions` - the number of partitions this computation will run on

  """
  def compile(computation = %Computation{}, client = %Client{}, argument_shapes, options \\ []) do
    num_replicas = Keyword.get(options, :num_replicas, 1)
    num_partitions = Keyword.get(options, :num_partitions, 1)

    use_spmd = Keyword.get(options, :use_spmd, false)
    use_spmd_int = if use_spmd, do: 1, else: 0

    output_shape = assert_output_shape!(computation)

    # TODO: Validate replicas and partitions against the client

    ref =
      EXLA.NIF.compile(
        client.ref,
        computation.ref,
        Enum.map(argument_shapes, & &1.ref),
        num_replicas,
        num_partitions,
        use_spmd_int
      )
      |> unwrap!()

    %Executable{
      client: client,
      ref: ref,
      output_shape: output_shape,
      num_replicas: num_replicas,
      num_partitions: num_partitions
    }
  end

  @doc """
  Performs AOT compilation of the given computation.

  It expects the following arguments:

    * the `computation`
    * the path for the input protobuf text file with
      a description of the inputs and outputs
    * the path to the output header file
    * the path to the output object file
    * the generated function name as a string
    * the generated class name as a string

  It writes to the output paths and returns `:ok`.

  ## Options

    * `:target_triple` - the target triple to compile to.
      It defaults to the current target triple but one
      can be set for cross-compilation. A list is available
      here: https://github.com/tensorflow/tensorflow/blob/e687cab61615a95b8aea79120c9f168d4cc30955/tensorflow/compiler/aot/tfcompile.bzl

          target_triple: "x86_64-pc-linux"

    * `:target_features` - the default executable makes
      no assumption about the target runtime, so special
      instructions such as SIMD are not leveraged. But you
      can specify those flags if desired:

          target_features: "+sse4.1 +sse4.2 +avx +avx2 +fma"

  """
  def compile_aot(
        %Computation{ref: ref} = comp,
        pbtext_path,
        header_path,
        object_path,
        function_name,
        class_name,
        options \\ []
      ) do
    assert_output_shape!(comp)

    EXLA.NIF.compile_aot(
      ref,
      pbtext_path,
      header_path,
      object_path,
      function_name,
      class_name,
      options[:target_triple] || target_triple(),
      options[:target_features] || ""
    )
  end

  @doc """
  Returns the default target triplet for computations.
  """
  def target_triple() do
    case :os.type() do
      {:win32, _} ->
        "x86_64-none-windows"

      {:unix, :linux} ->
        "x86_64-pc-linux"

      {:unix, osname} ->
        arch_str = :erlang.system_info(:system_architecture)
        [arch, vendor | _] = arch_str |> List.to_string() |> String.split("-")
        arch <> "-" <> vendor <> "-" <> Atom.to_string(osname)
    end
  end

  defp assert_output_shape!(%{output_shape: output_shape}) do
    if root_tuple_only?(output_shape) do
      output_shape
    else
      raise ArgumentError,
            "can only compile computations with a tuple at the root (and only at the root), " <>
              "got: #{inspect(output_shape)}"
    end
  end

  defp root_tuple_only?(shape) do
    case shape do
      %{dtype: {:t, inner}} -> Enum.all?(inner, &(not match?({:t, _}, &1.dtype)))
      %{} -> false
    end
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end

defmodule Nx.Defn.CompilerTest do
  use ExUnit.Case, async: true

  defmodule SomeInvalidServing do
    def init(_, _, _) do
      :ok
    end
  end

  test "raises an error if the __compile__ callback is missing" do
    msg =
      "the expected compiler callback __compile__/4 is missing. Please check that the module SomeInvalidCompiler is an Nx.Defn.Compiler."

    assert_raise ArgumentError, msg, fn ->
      Nx.Defn.compile(&Function.identity/1, [Nx.template({}, :f32)],
        compiler: SomeInvalidCompiler
      )
    end
  end

  test "raises an error if the __jit__ callback is missing" do
    msg =
      "the expected compiler callback __jit__/5 is missing. Please check that the module SomeInvalidCompiler is an Nx.Defn.Compiler."

    assert_raise ArgumentError, msg, fn ->
      Nx.Defn.jit(&Function.identity/1, compiler: SomeInvalidCompiler).(1)
    end
  end

  test "raises an error if the __partitions_options__ callback is missing" do
    msg =
      "the expected compiler callback __partitions_options__/1 is missing. Please check that the module SomeInvalidCompiler is an Nx.Defn.Compiler."

    serving = Nx.Serving.new(SomeInvalidServing, [], compiler: SomeInvalidCompiler)

    assert_raise ArgumentError, msg, fn ->
      Nx.Serving.init({MyName, serving, true, [1], 10, 1000, nil, 1})
    end
  end

  test "raises an error if the __to_backend__ callback is missing" do
    msg =
      "the expected compiler callback __to_backend__/1 is missing. Please check that the module SomeInvalidCompiler is an Nx.Defn.Compiler."

    assert_raise ArgumentError, msg, fn ->
      Nx.Defn.to_backend(compiler: SomeInvalidCompiler)
    end
  end
end

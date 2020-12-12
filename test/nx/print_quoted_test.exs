defmodule Nx.PrintQuotedTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureIO

  test "prints the quoted expression" do
    assert capture_io(fn ->
             defmodule Example1 do
               import Nx.Defn
               defn print_grad(t), do: print_quoted(grad(t, Nx.power(t, 3)))
             end
           end) =~ """
           (
             a = Nx.power(b, 3)
             Nx.assert_shape(a, {}, "grad expects the numerical expression to return a scalar tensor")
             (
               c = 3
               Nx.multiply(c, Nx.power(b, Nx.subtract(c, 1)))
             )
           )
           """
  end

  test "prints the reshape expression" do
    assert capture_io(fn ->
             defmodule Example2 do
               import Nx.Defn
               defn print_grad(t), do: print_quoted(grad(t, Nx.reshape(t, {4, 4})))
             end
           end) =~ """
           (
             a = Nx.reshape(b, {4, 4})
             Nx.assert_shape(a, {}, "grad expects the numerical expression to return a scalar tensor")
             c = Nx.broadcast(1.0, {4, 4})
             Nx.reshape(c, b)
           )
           """
  end
end

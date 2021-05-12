defmodule EXLATest do
  use ExUnit.Case, async: true

  @moduletag :capture_log
  @compile {:no_warn_undefined, ExpAotDemo}
  @compile {:no_warn_undefined, ExpExportDemo}
  @compile {:no_warn_undefined, ComplexAotDemo}

  doctest EXLA

  import Nx.Defn
  defn tuple_return({a, b}, c, d), do: {{a + c, b + d}, a + b}
  defn tuple_unused({a, _b}, c, _d), do: a + c

  @tag :aot
  test "complex aot example" do
    a = Nx.tensor([1, 2, 3])
    b = Nx.tensor([1.0, 2.0, 3.0])
    c = Nx.tensor(5)

    dot1 = Nx.iota({3, 3}, type: {:f, 32})
    dot2 = Nx.eye({3, 3}, type: {:f, 32})

    functions = [
      {:tuple_return1, &tuple_return(&1, &2, -5), [{a, b}, c]},
      {:tuple_return2, &tuple_return(&1, 5, &2), [{a, b}, -5]},
      {:tuple_unused, &tuple_unused(&1, 5, &2), [{a, b}, c]},
      {:dot, &Nx.dot/2, [dot1, dot2]}
    ]

    {:module, _, _, _} = EXLA.aot(ComplexAotDemo, functions)

    assert ComplexAotDemo.tuple_return1({a, b}, c) ==
             {{Nx.tensor([6, 7, 8]), Nx.tensor([-4.0, -3.0, -2.0])}, Nx.tensor([2.0, 4.0, 6.0])}

    assert ComplexAotDemo.tuple_return2({a, b}, -5) ==
             {{Nx.tensor([6, 7, 8]), Nx.tensor([-4.0, -3.0, -2.0])}, Nx.tensor([2.0, 4.0, 6.0])}

    assert ComplexAotDemo.tuple_unused({a, b}, c) == Nx.tensor([6, 7, 8])

    assert ComplexAotDemo.dot(dot1, dot2) == dot1
    assert ComplexAotDemo.dot(dot2, dot1) == dot1
  end
end

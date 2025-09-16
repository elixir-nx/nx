defmodule Nx.Defn.TemplateDiffTest do
  use ExUnit.Case, async: true

  defp build(left, right) do
    Nx.Defn.TemplateDiff.build(left, right, "left", "right", &Nx.compatible?/2)
  end

  test "compatible" do
    assert build(1, 2) == %Nx.Defn.TemplateDiff{
             left: 1,
             right: 2,
             compatible: true,
             left_title: "left",
             right_title: "right"
           }
  end

  test "incompatible" do
    assert build(1, Nx.tensor([2])) == %Nx.Defn.TemplateDiff{
             left: 1,
             right: Nx.tensor([2]),
             compatible: false,
             left_title: "left",
             right_title: "right"
           }
  end

  test "implementation incompatible" do
    assert build(%{foo: 1}, Nx.tensor([2])) == %Nx.Defn.TemplateDiff{
             left: %{foo: 1},
             right: Nx.tensor([2]),
             compatible: false,
             left_title: "left",
             right_title: "right"
           }
  end

  test "size incompatible" do
    assert build(%{foo: 1}, %{}) == %Nx.Defn.TemplateDiff{
             left: %{foo: 1},
             right: %{},
             compatible: false,
             left_title: "left",
             right_title: "right"
           }

    assert build(%{}, %{foo: 1}) == %Nx.Defn.TemplateDiff{
             left: %{},
             right: %{foo: 1},
             compatible: false,
             left_title: "left",
             right_title: "right"
           }
  end

  test "child incompatible" do
    assert build(%{foo: 1}, %{foo: Nx.tensor([2])}) ==
             %{
               foo: %Nx.Defn.TemplateDiff{
                 left: 1,
                 right: Nx.tensor([2]),
                 compatible: false,
                 left_title: "left",
                 right_title: "right"
               }
             }
  end

  test "keys incompatible" do
    assert build(%{foo: 1}, %{bar: 1}) ==
             %Nx.Defn.TemplateDiff{
               left: %{foo: 1},
               right: %{bar: 1},
               compatible: false,
               left_title: "left",
               right_title: "right"
             }
  end
end

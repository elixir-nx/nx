defmodule Nx.HeatmapTest do
  use ExUnit.Case, async: true

  @tensor0 Nx.tensor(0)
  @tensor1 Nx.tensor([1, 2, 3, 4, 5])
  @tensor2 Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
  @tensor3 Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
  @non_finite Nx.tensor([:neg_infinity, -1, :nan, 1, :infinity])

  test "rank 0" do
    assert_raise ArgumentError, fn -> Nx.to_heatmap(@tensor0) end
  end

  describe "without ANSI" do
    test "equal" do
      assert Nx.tensor([1, 1, 1]) |> Nx.to_heatmap(ansi_enabled: false) |> inspect() == """
             #Nx.Heatmap<
               s64[3]
               000
             >\
             """
    end

    test "rank 1" do
      assert @tensor1 |> Nx.to_heatmap(ansi_enabled: false) |> inspect() == """
             #Nx.Heatmap<
               s64[5]
               02579
             >\
             """
    end

    test "rank 2" do
      assert @tensor2 |> Nx.to_heatmap(ansi_enabled: false) |> inspect() == """
             #Nx.Heatmap<
               s64[4][3]
             \s\s
               012
               234
               567
               789
             >\
             """
    end

    test "rank 3" do
      assert @tensor3 |> Nx.to_heatmap(ansi_enabled: false) |> inspect() == """
             #Nx.Heatmap<
               s64[2][2][3]
               [
                 024
                 579,
             \s\s\s\s
                 024
                 579
               ]
             >\
             """
    end

    test "non-finite" do
      assert @non_finite |> Nx.to_heatmap(ansi_enabled: false) |> inspect() == """
             #Nx.Heatmap<
               f32[5]
               -0x9+
             >\
             """

      assert Nx.stack([@non_finite, @non_finite])
             |> Nx.to_heatmap(ansi_enabled: false)
             |> inspect() == """
             #Nx.Heatmap<
               f32[2][5]
             \s\s
               -0x9+
               -0x9+
             >\
             """
    end
  end

  describe "with ANSI" do
    test "equal" do
      assert Nx.tensor([1, 1, 1]) |> Nx.to_heatmap(ansi_enabled: true) |> inspect() == """
             #Nx.Heatmap<
               s64[3]
               \e[48;5;232m　\e[48;5;232m　\e[48;5;232m　\e[0m
             >\
             """
    end

    test "rank 1" do
      assert @tensor1 |> Nx.to_heatmap(ansi_enabled: true) |> inspect() == """
             #Nx.Heatmap<
               s64[5]
               \e[48;5;232m\u3000\e[48;5;238m\u3000\e[48;5;244m\u3000\e[48;5;249m\u3000\e[48;5;255m\u3000\e[0m
             >\
             """
    end

    test "rank 2" do
      assert @tensor2 |> Nx.to_heatmap(ansi_enabled: true) |> inspect() == """
             #Nx.Heatmap<
               s64[4][3]
             \s\s
               \e[48;5;232m\u3000\e[48;5;234m\u3000\e[48;5;236m\u3000\e[0m
               \e[48;5;238m\u3000\e[48;5;240m\u3000\e[48;5;242m\u3000\e[0m
               \e[48;5;245m\u3000\e[48;5;247m\u3000\e[48;5;249m\u3000\e[0m
               \e[48;5;251m\u3000\e[48;5;253m\u3000\e[48;5;255m\u3000\e[0m
             >\
             """
    end

    test "rank 3" do
      assert @tensor3 |> Nx.to_heatmap(ansi_enabled: true) |> inspect() == """
             #Nx.Heatmap<
               s64[2][2][3]
               [
                 \e[48;5;232m\u3000\e[48;5;237m\u3000\e[48;5;241m\u3000\e[0m
                 \e[48;5;246m\u3000\e[48;5;250m\u3000\e[48;5;255m\u3000\e[0m,
             \s\s\s\s
                 \e[48;5;232m\u3000\e[48;5;237m\u3000\e[48;5;241m\u3000\e[0m
                 \e[48;5;246m\u3000\e[48;5;250m\u3000\e[48;5;255m\u3000\e[0m
               ]
             >\
             """
    end

    test "non-finite" do
      assert @non_finite |> Nx.to_heatmap(ansi_enabled: true) |> inspect() == """
             #Nx.Heatmap<
               f32[5]
               \e[48;5;232m∞\e[48;5;232m　\e[41m　\e[48;5;255m　\e[48;5;255m∞\e[0m
             >\
             """
    end
  end

  describe "access" do
    test "scalar" do
      assert Nx.to_heatmap(Nx.tensor([[1, 2], [3, 4]]))[[0, 0]] == Nx.tensor(1)
    end

    test "non-scalar" do
      assert Nx.to_heatmap(Nx.tensor([[1, 2], [3, 4]]))[0] ==
               Nx.to_heatmap(Nx.tensor([1, 2]))

      assert Nx.to_heatmap(Nx.tensor([[1, 2], [3, 4]]))[0..0] ==
               Nx.to_heatmap(Nx.tensor([[1, 2]]))
    end

    test "preserves options" do
      assert Nx.to_heatmap(Nx.tensor([[1, 2], [3, 4]]), ansi_whitespace: "\s")[0] ==
               Nx.to_heatmap(Nx.tensor([1, 2]), ansi_whitespace: "\s")
    end
  end
end

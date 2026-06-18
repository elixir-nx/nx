defmodule Nx.Defn.KernelTest do
  use ExUnit.Case, async: true

  require Nx.Defn.Kernel
  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  defp zero(), do: Nx.tensor(0, type: {:u, 8})
  defp one(), do: Nx.tensor(1, type: {:u, 8})

  describe "doctests" do
    use Nx.Defn.Kernel
    doctest Nx.Defn.Kernel
  end

  describe "with numbers" do
    test "+" do
      assert Nx.Defn.Kernel.+(1, 2) == 3
    end

    test "-" do
      assert Nx.Defn.Kernel.-(1, 2) == -1
    end

    test "*" do
      assert Nx.Defn.Kernel.*(1, 2) == 2
    end

    test "**" do
      assert Nx.Defn.Kernel.**(1, 2) == 1
    end

    test "/" do
      assert Nx.Defn.Kernel./(1, 2) == 0.5
    end

    test "comparison" do
      assert Nx.Defn.Kernel.==(0, 0) == one()
      assert Nx.Defn.Kernel.!=(0, 0) == zero()
      assert Nx.Defn.Kernel.>(0, 0) == zero()
      assert Nx.Defn.Kernel.>=(0, 0) == one()
      assert Nx.Defn.Kernel.<(0, 0) == zero()
      assert Nx.Defn.Kernel.<=(0, 0) == one()
    end

    test "and" do
      assert Nx.Defn.Kernel.and(0, 0) == zero()
      assert Nx.Defn.Kernel.and(1, 0) == zero()
      assert Nx.Defn.Kernel.and(0, 2) == zero()
      assert Nx.Defn.Kernel.and(1, 1) == one()

      assert Nx.Defn.Kernel.and(0, 0.0) == zero()
      assert Nx.Defn.Kernel.and(1, 0.0) == zero()
      assert Nx.Defn.Kernel.and(0, 2.0) == zero()
      assert Nx.Defn.Kernel.and(1, 1.0) == one()
    end

    test "or" do
      assert Nx.Defn.Kernel.or(0, 0) == zero()
      assert Nx.Defn.Kernel.or(1, 0) == one()
      assert Nx.Defn.Kernel.or(0, 2) == one()
      assert Nx.Defn.Kernel.or(1, 1) == one()

      assert Nx.Defn.Kernel.or(0, 0.0) == zero()
      assert Nx.Defn.Kernel.or(1, 0.0) == one()
      assert Nx.Defn.Kernel.or(0, 2.0) == one()
      assert Nx.Defn.Kernel.or(1, 1.0) == one()
    end

    test "not" do
      assert Nx.Defn.Kernel.not(0) == one()
      assert Nx.Defn.Kernel.not(1) == zero()
      assert Nx.Defn.Kernel.not(2) == zero()

      assert Nx.Defn.Kernel.not(0.0) == one()
      assert Nx.Defn.Kernel.not(1.0) == zero()
      assert Nx.Defn.Kernel.not(2.0) == zero()
    end

    test "&&&" do
      assert Nx.Defn.Kernel.&&&(1, 2) == 0
    end

    test "|||" do
      assert Nx.Defn.Kernel.|||(1, 2) == 3
    end

    test "<<<" do
      assert Nx.Defn.Kernel.<<<(1, 2) == 4
    end

    test ">>>" do
      assert Nx.Defn.Kernel.>>>(1, 2) == 0
    end

    test "unary +/-" do
      assert Nx.Defn.Kernel.+(1) == 1
      assert Nx.Defn.Kernel.-(1) == -1
    end

    test "min/max" do
      assert Nx.Defn.Kernel.min(0, 1) == 0
      assert Nx.Defn.Kernel.max(0, 1) == 1
    end

    test "div" do
      assert Nx.Defn.Kernel.div(11, 5) == 2
    end

    test "rem" do
      assert Nx.Defn.Kernel.rem(1, 5) == 1
    end

    test ".." do
      assert Nx.Defn.Kernel.".."(1, 2) == 1..2
    end
  end

  describe "inside defn" do
    import Nx.Defn

    defn tap_and_then(a, b, c) do
      a
      |> Nx.add(b)
      |> tap(&send_up/1)
      |> then(&Nx.subtract(c, &1))
    end

    defp send_up(expr) do
      send(self(), {:expr, expr})
    end

    test "tap and then" do
      assert tap_and_then(1, 2, 3) == Nx.tensor(0)
      assert_received {:expr, %Nx.Tensor{data: %Nx.Defn.Expr{}}}
    end

    import Nx.Defn.Kernel, only: [keyword!: 2]
    defn defn_after_import(tensor), do: -tensor

    test "defn after import works" do
      assert defn_after_import(1) == Nx.tensor(-1)
    end
  end

  describe "tokens" do
    defp zero_expr(), do: Nx.tensor(0, type: {:u, 8}, backend: Nx.Defn.Expr)

    test "io_call/2,3" do
      result = Nx.Defn.Kernel.io_call(zero_expr(), :a)
      assert %T{data: %Expr{op: :io_call}} = result
      assert inspect(result, safe: false) =~ "io_call"
      assert inspect(result, safe: false) =~ "a:"

      result = Nx.Defn.Kernel.io_call(zero_expr(), &Function.identity/1)
      assert %T{data: %Expr{op: :io_call}} = result
      assert inspect(result, safe: false) =~ "io_call"

      result = Nx.Defn.Kernel.io_call(zero_expr(), :a, &Function.identity/1)
      assert %T{data: %Expr{op: :io_call}} = result
      assert inspect(result, safe: false) =~ "a:"
    end
  end

  describe "macros" do
    test "raise outside of defn" do
      assert_raise RuntimeError,
                   "cannot invoke Nx.Defn.Kernel.if/2 because you are not inside a defn",
                   fn ->
                     Code.eval_quoted(
                       quote do
                         require Nx.Defn.Kernel
                         Nx.Defn.Kernel.if(1, do: 0)
                       end
                     )
                   end
    end
  end
end

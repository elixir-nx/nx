defmodule NEXTTest do
  use ExUnit.Case, async: true

  import NEXT

  defn add_two(a, b) do
    a + b
  end

  test "basic example" do
    {_, _} = meta = add_two(1, 2)
    assert eval(meta) == 3
  end

  defp eval({ast, binding}) do
    {result, _} = Code.eval_quoted(ast, binding)
    result
  end
end

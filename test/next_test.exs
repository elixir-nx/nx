defmodule NEXTTest do
  use ExUnit.Case, async: true

  import NEXT

  defmodule Macros do
    defmacro add(a, b) do
      quote do
        add(unquote(a), unquote(b))
      end
    end
  end

  defn add_two_from_macro(a, b) do
    require Macros
    Macros.add(a, b)
  end

  defn add_two(a, b) do
    a + b
  end

  test "supports NEXT.Kernel function" do
    {_, _} = meta = add_two(1, 2)
    assert eval(meta) == 3

    {_, _} = meta = add_two([1, 2, 3], 2)
    assert eval(meta) == [3, 4, 5]
  end

  test "supports external macro" do
    {_, _} = meta = add_two_from_macro(1, 2)
    assert eval(meta) == 3

    {_, _} = meta = add_two_from_macro([1, 2, 3], 2)
    assert eval(meta) == [3, 4, 5]
  end

  defp eval({ast, binding}) do
    {result, _} = Code.eval_quoted(ast, binding)
    result
  end
end

defmodule NEXTTest do
  use ExUnit.Case, async: true

  alias NEXTTest.Sample
  import NEXT

  describe "operators" do
    defn add_two(a, b) do
      a + b
    end

    test "+" do
      {_, _} = meta = add_two(1, 2)
      assert eval(meta) == 3

      {_, _} = meta = add_two([1, 2, 3], 2)
      assert eval(meta) == [3, 4, 5]
    end
  end

  describe "macros" do
    defmodule Macros do
      defmacro add(a, b) do
        import Kernel, only: []
        import NEXT.Kernel

        quote do
          unquote(a) + unquote(b)
        end
      end
    end

    defn add_two_from_external_macro(a, b) do
      require Macros
      Macros.add(a, b)
    end

    test "external" do
      {_, _} = meta = add_two_from_external_macro(1, 2)
      assert eval(meta) == 3

      {_, _} = meta = add_two_from_external_macro([1, 2, 3], 2)
      assert eval(meta) == [3, 4, 5]
    end

    defmacrop add_internal(a, b) do
      import Kernel, only: []
      import NEXT.Kernel

      quote do
        unquote(a) + unquote(b)
      end
    end

    defn add_two_from_internal_macro(a, b) do
      add_internal(a, b)
    end

    test "internal" do
      {_, _} = meta = add_two_from_external_macro(1, 2)
      assert eval(meta) == 3

      {_, _} = meta = add_two_from_external_macro([1, 2, 3], 2)
      assert eval(meta) == [3, 4, 5]
    end

    defn add_two_from_alias(a, b) do
      alias Nx, as: N
      N.add(a, b)
    end

    test "aliases" do
      {_, _} = meta = add_two_from_alias(1, 2)
      assert eval(meta) == 3

      {_, _} = meta = add_two_from_alias([1, 2, 3], 2)
      assert eval(meta) == [3, 4, 5]
    end
  end

  describe "warnings" do
    import ExUnit.CaptureIO

    test "unused private functions" do
      assert capture_io(:stderr, fn ->
               defmodule Sample do
                 import NEXT
                 defnp lonely(a, b), do: a + b
               end
             end) =~ "function lonely/2 is unused"
    after
      purge(Sample)
    end

    test "empty blocks" do
      assert capture_io(:stderr, fn ->
               defmodule Sample do
                 import NEXT

                 defn empty(_a, _b) do
                 end
               end
             end) =~ "body has nil return type, -1 will be returned instead"
    after
      purge(Sample)
    end

    test "does not emit used underscore vars" do
      assert capture_io(:stderr, fn ->
               defmodule Sample do
                 import NEXT
                 defn empty(a, _b), do: a
               end
             end) == ""
    after
      purge(Sample)
    end
  end

  defp purge(module) do
    :code.purge(module)
    :code.delete(module)
  end

  defp eval({ast, binding}) do
    {result, _} = Code.eval_quoted(ast, binding)
    result
  end
end

defmodule Nx.DefnTest do
  use ExUnit.Case, async: true

  alias Nx.DefnTest.Sample
  import Nx.Defn

  defmacrop location(plus) do
    file = Path.relative_to_cwd(__CALLER__.file)
    quote do: "#{unquote(file)}:#{unquote(__CALLER__.line) + unquote(plus)}"
  end

  describe "operators" do
    defn add_two(a, b) do
      a + b
    end

    test "+" do
      assert add_two(1, 2) == 3
      assert add_two(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_with_pipe(a, b) do
      a |> Nx.add(b)
    end

    test "|>" do
      assert add_two_with_pipe(1, 2) == 3
      assert add_two_with_pipe(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end
  end

  describe "macros" do
    defmodule Macros do
      defmacro add(a, b) do
        use Nx.Defn.Kernel

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
      assert add_two_from_external_macro(1, 2) == 3
      assert add_two_from_external_macro(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defmacrop add_internal(a, b) do
      use Nx.Defn.Kernel

      quote do
        unquote(a) + unquote(b)
      end
    end

    defn add_two_from_internal_macro(a, b) do
      add_internal(a, b)
    end

    test "internal" do
      assert add_two_from_external_macro(1, 2) == 3
      assert add_two_from_external_macro(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_from_alias(a, b) do
      alias Nx, as: N
      N.add(a, b)
    end

    test "aliases" do
      assert add_two_from_alias(1, 2) == 3
      assert add_two_from_alias(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end
  end

  describe "local functions" do
    defn add_two_from_public(a, b) do
      add_two_from_public_impl(a, b)
    end

    defn add_two_from_public_impl(a, b) do
      a + b
    end

    test "public" do
      assert add_two_from_public(1, 2) == 3
      assert add_two_from_public(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_from_private(a, b) do
      add_two_from_private_impl(a, b)
    end

    defn add_two_from_private_impl(a, b) do
      a + b
    end

    test "private" do
      assert add_two_from_private(1, 2) == 3
      assert add_two_from_private(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_var_conflict(a, b) do
      c = 1
      b = add_two_var_conflict_impl(a, b)
      c + b
    end

    defn add_two_var_conflict_impl(c, d) do
      c + d
    end

    test "var conflict" do
      assert add_two_var_conflict(2, 3) == 6
    end
  end

  describe "warnings" do
    import ExUnit.CaptureIO

    test "unused private functions" do
      assert capture_io(:stderr, fn ->
               defmodule Sample do
                 import Nx.Defn
                 defnp lonely(a, b), do: a + b
               end
             end) =~ "function lonely/2 is unused"
    after
      purge(Sample)
    end

    test "empty blocks" do
      assert capture_io(:stderr, fn ->
               defmodule Sample do
                 import Nx.Defn

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
                 import Nx.Defn
                 defn empty(a, _b), do: a
               end
             end) == ""
    after
      purge(Sample)
    end
  end

  describe "errors" do
    test "invalid numerical expression" do
      assert_raise CompileError, ~r"#{location(+4)}: invalid numerical expression", fn ->
        defmodule Sample do
          import Nx.Defn

          defn add(_a, _b) do
            receive do
              :ok -> :ok
            end
          end
        end
      end
    end

    test "recursive definitions" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: add/2 is being called recursively by add/2",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, b), do: add(a, b)
                     end
                   end

      assert_raise CompileError, ~r"add/2 is being called recursively by add1/2", fn ->
        defmodule Sample do
          import Nx.Defn
          defn add(a, b), do: add1(a, b)
          defn add1(a, b), do: add(a, b)
        end
      end
    end

    test "non variables used as arguments" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: only variables are allowed as arguments in defn",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(1, 2), do: 3
                     end
                   end
    end

    test "defaults" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: default arguments are not supported by defn",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, b \\ 2), do: a + b
                     end
                   end
    end
  end

  defp purge(module) do
    :code.purge(module)
    :code.delete(module)
  end
end

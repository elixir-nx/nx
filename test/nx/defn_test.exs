defmodule Nx.DefnTest do
  use ExUnit.Case, async: true

  alias Nx.DefnTest.Sample
  import Nx.Defn

  defmacrop location(plus) do
    file = Path.relative_to_cwd(__CALLER__.file)
    quote do: "#{unquote(file)}:#{unquote(__CALLER__.line) + unquote(plus)}"
  end

  describe "scalar" do
    defn just_two_int, do: 2
    defn just_two_float, do: 2.0

    test "returns the tensor for the scalar" do
      assert just_two_int() == Nx.tensor(2)
      assert just_two_float() == Nx.tensor(2.0)
    end
  end

  describe "tuples" do
    defn two_constant_tuples, do: {-1, 1.0}
    defn three_constant_tuples, do: {1, 2.0, 3}

    test "returns tuples with constants" do
      assert two_constant_tuples() == {Nx.tensor(-1), Nx.tensor(1.0)}
      assert three_constant_tuples() == {Nx.tensor(1), Nx.tensor(2.0), Nx.tensor(3)}
    end

    defn add_subtract_tuple(a, b), do: {a + b, a - b}

    test "returns tuples with operation results" do
      assert add_subtract_tuple(2, 3) == {Nx.tensor(5), Nx.tensor(-1)}

      assert add_subtract_tuple(Nx.tensor([-1, 0, 1]), 10) ==
               {Nx.tensor([9, 10, 11]), Nx.tensor([-11, -10, -9])}
    end

    defn pattern_tuple({a, b}), do: a + b

    test "matches on tuples" do
      assert pattern_tuple({2, 3}) == Nx.tensor(5)

      assert pattern_tuple({Nx.tensor([1, 2]), Nx.tensor([[3], [4]])}) ==
               Nx.tensor([[4, 5], [5, 6]])
    end

    defn calls_pattern_tuple(a, b), do: pattern_tuple({a, b})

    test "matches on inlined tuples" do
      assert calls_pattern_tuple(2, 3) == Nx.tensor(5)

      assert calls_pattern_tuple(Nx.tensor([1, 2]), Nx.tensor([[3], [4]])) ==
               Nx.tensor([[4, 5], [5, 6]])
    end
  end

  describe "tensor constants" do
    @two 2
    defn add_two_attribute(t), do: t + @two

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn add_2x2_attribute(t), do: t + @two_per_two

    test "expands module attributes to scalars" do
      assert add_two_attribute(1) == Nx.tensor(3)
      assert add_two_attribute(Nx.tensor([1, 2, 3])) == Nx.tensor([3, 4, 5])
    end

    test "expands module attributes to tensors" do
      assert add_2x2_attribute(1) == Nx.tensor([[2, 3], [4, 5]])
      assert add_2x2_attribute(Nx.tensor([1, 2])) == Nx.tensor([[2, 4], [4, 6]])
    end
  end

  describe "pattern matching" do
    defn complex_pattern_matching(expr) do
      ({a, b} = c) = {d, e} = f = expr
      {a, b, c, d, e, f}
    end

    test "normalizes to one pattern per expression" do
      assert ast_to_string(:complex_pattern_matching, 1) == """
             (
               (
                 c = expr
                 f = c
                 {a, b} = c
                 {d, e} = c
               )
               {a, b, c, d, e, f}
             )\
             """
    end

    defn nested_pattern_matching do
      {{a, _} = c, {d, e} = f} = _ = {{1, 2}, {3, 4}}
      _ = {a, c, d, e, f}
    end

    test "unnests nested patterns" do
      assert ast_to_string(:nested_pattern_matching, 0) == """
             (
               (
                 nvar = {{1, 2}, {3, 4}}
                 {nvar, nvar} = nvar
                 (
                   f = nvar
                   {d, e} = f
                 )
                 (
                   c = nvar
                   {a, _} = c
                 )
               )
               nvar = {a, c, d, e, f}
             )\
             """

      a = Nx.tensor(1)
      b = Nx.tensor(2)
      c = Nx.tensor(3)
      d = Nx.tensor(4)

      assert nested_pattern_matching() == {a, {a, b}, c, d, {c, d}}
    end
  end

  describe "operators" do
    defn add_two(a, b), do: a + b

    test "+" do
      assert add_two(1, 2) == Nx.tensor(3)
      assert add_two(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn subtract_two(a, b), do: a - b

    test "-" do
      assert subtract_two(Nx.tensor([1, 2, 3]), Nx.tensor(2)) == Nx.tensor([-1, 0, 1])
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      assert multiply_two(Nx.tensor([1, 2, 3]), Nx.tensor(2)) == Nx.tensor([2, 4, 6])
    end

    defn divide_two(a, b), do: a / b

    test "/" do
      assert divide_two(Nx.tensor([1, 2, 3]), Nx.tensor(2)) == Nx.tensor([0.5, 1.0, 1.5])
    end

    defn band_two(a, b), do: a &&& b

    test "&&&" do
      assert band_two(Nx.tensor([-1, 0, 1]), Nx.tensor(1)) == Nx.tensor([1, 0, 1])
    end

    defn bor_two(a, b), do: a ||| b

    test "|||" do
      assert bor_two(Nx.tensor([-1, 0, 1]), Nx.tensor(1)) == Nx.tensor([-1, 1, 1])
    end

    defn bxor_two(a, b), do: a ^^^ b

    test "^^^" do
      assert bxor_two(Nx.tensor([-1, 0, 1]), Nx.tensor(1)) == Nx.tensor([-2, 1, 0])
    end

    defn bsl_two(a, b), do: a <<< b

    test "<<<" do
      assert bsl_two(Nx.tensor([-1, 0, 1]), Nx.tensor(1)) == Nx.tensor([-2, 0, 2])
    end

    defn bsr_two(a, b), do: a >>> b

    test ">>>" do
      assert bsr_two(Nx.tensor([-2, 1, 2]), Nx.tensor(1)) == Nx.tensor([-1, 0, 1])
    end

    defn add_two_with_pipe(a, b), do: a |> Nx.add(b)

    test "|>" do
      assert add_two_with_pipe(1, 2) == Nx.tensor(3)
      assert add_two_with_pipe(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn unary_plus(a), do: +a
    defn unary_minus(a), do: -a

    test "unary plus and minus" do
      assert unary_plus(1) == Nx.tensor(1)
      assert unary_plus(Nx.tensor([-1, 0, 1])) == Nx.tensor([-1, 0, 1])

      assert unary_minus(1) == Nx.tensor(-1)
      assert unary_minus(Nx.tensor([-1, 0, 1])) == Nx.tensor([1, 0, -1])
    end

    defn unary_bnot(a), do: ~~~a

    test "~~~" do
      assert unary_bnot(Nx.tensor([-1, 0, 1])) == Nx.tensor([0, -1, -2])
    end
  end

  describe "kernel functions" do
    defn max_two(a, b) do
      max(a, b)
    end

    test "max/2" do
      assert max_two(Nx.tensor([1, 2, 3]), Nx.tensor(2)) == Nx.tensor([2, 2, 3])
    end

    defn min_two(a, b) do
      min(a, b)
    end

    test "min/2" do
      assert min_two(Nx.tensor([1, 2, 3]), Nx.tensor(2)) == Nx.tensor([1, 2, 2])
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
      assert add_two_from_external_macro(1, 2) == Nx.tensor(3)
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
      assert add_two_from_external_macro(1, 2) == Nx.tensor(3)
      assert add_two_from_external_macro(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_from_alias(a, b) do
      alias Nx, as: N
      N.add(a, b)
    end

    test "aliases" do
      assert add_two_from_alias(1, 2) == Nx.tensor(3)
      assert add_two_from_alias(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    dynamic_name = String.to_atom(Enum.join(~w(dynamic name add two), "_"))
    operator = :add
    defp unquote(dynamic_name)(left, right), do: Nx.unquote(operator)(left, right)

    test "dynamic name" do
      assert dynamic_name_add_two(1, 2) == Nx.tensor(3)
      assert dynamic_name_add_two(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
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
      assert add_two_from_public(1, 2) == Nx.tensor(3)
      assert add_two_from_public(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_from_private(a, b) do
      add_two_from_private_impl(a, b)
    end

    defn add_two_from_private_impl(a, b) do
      a + b
    end

    test "private" do
      assert add_two_from_private(1, 2) == Nx.tensor(3)
      assert add_two_from_private(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_var_conflict(a, b) do
      c = 1
      b = add_two_var_conflict_impl(a, b)
      b + c
    end

    defn add_two_var_conflict_impl(c, d) do
      c + d
    end

    test "var conflict" do
      assert add_two_var_conflict(2, 3) == Nx.tensor(6)
    end

    test "expansion" do
      assert ast_to_string(:add_two_from_public, 2) == "Nx.add(a, b)"
    end

    defn add_two_with_underscore(a, b), do: add_two_with_underscore_impl(a, b)

    defn add_two_with_underscore_impl(_, b) do
      _ = 2
      b
    end

    test "handles underscores" do
      assert ast_to_string(:add_two_with_underscore, 2) == """
             (
               nvar = 2
               b
             )\
             """
           end
  end

  describe "remote functions" do
    defmodule Remote do
      defn add_two(c, d), do: c + d
    end

    defn add_two_remote(a, b), do: Remote.add_two(a, b)

    test "public" do
      assert add_two_remote(1, 2) == Nx.tensor(3)
      assert add_two_remote(Nx.tensor([1, 2, 3]), 2) == Nx.tensor([3, 4, 5])
    end

    defn add_two_remote_var_conflict(a, b) do
      c = 1
      b = Remote.add_two(a, b)
      b + c
    end

    test "var conflict" do
      assert add_two_remote_var_conflict(2, 3) == Nx.tensor(6)
    end

    test "expansion" do
      assert ast_to_string(:add_two_remote, 2) == "Nx.DefnTest.Remote.add_two(a, b)"
    end
  end

  describe "module attributes" do
    test "overrides default compiler with custom" do
      defmodule Sample do
        import Nx.Defn
        @default_defn_compiler "unknown"
        @defn_compiler Nx.Defn
        defn add(a, b), do: a + b
        assert Module.get_attribute(__MODULE__, :defn_compiler) == nil
        assert @default_defn_compiler == "unknown"
      end

      assert Sample.add(1, 2) == Nx.tensor(3)
    after
      purge(Sample)
    end
  end

  describe "warnings" do
    import ExUnit.CaptureIO

    test "unused private functions" do
      assert capture_io(:stderr, fn ->
               defmodule Sample do
                 import Nx.Defn
                 defnp will_be_unused(a, b), do: a + b
               end
             end) =~ "function will_be_unused/2 is unused"
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
             end) =~ "body has nil return type, 0 will be returned instead"
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
                   ~r"#{location(+4)}: only variables and tuples are allowed as arguments in defn",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(1, 2), do: 3
                     end
                   end
    end

    test "dup vars used as arguments" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: variable \"a\" appears twice in pattern \[a, a\]",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, a), do: 3
                     end
                   end
    end

    test "dup vars used as patterns" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: variable \"b\" appears twice in pattern \{b, b\}",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a), do: {b, b} = a
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

    test "unknown defn compiler" do
      assert_raise UndefinedFunctionError,
                   ~r"Unknown.__compile__/6",
                   fn ->
                     defmodule Sample do
                       @defn_compiler Unknown
                       import Nx.Defn
                       defn add(a, b), do: a + b
                     end
                   end
    end

    test "unknown module" do
      assert_raise CompileError,
                   ~r"cannot invoke Unknown.foo/2 because Unknown does not exist",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, b), do: Unknown.foo(a, b)
                     end
                   end
    end

    test "unknown defn" do
      assert_raise CompileError,
                   ~r"undefined numerical function Nx.DefnTest.unknown/2",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, b), do: Nx.DefnTest.unknown(a, b)
                     end
                   end
    end

    test "invalid defn compiler" do
      assert_raise ArgumentError,
                   ~r"expected @defn_compiler/@default_defn_compiler to be an atom or",
                   fn ->
                     defmodule Sample do
                       @defn_compiler "unknown"
                       import Nx.Defn
                       defn add(a, b), do: a + b
                     end
                   end
    end

    test "invalid default defn compiler" do
      assert_raise ArgumentError,
                   ~r"expected @defn_compiler/@default_defn_compiler to be an atom or",
                   fn ->
                     defmodule Sample do
                       @default_defn_compiler "unknown"
                       import Nx.Defn
                       defn add(a, b), do: a + b
                     end
                   end
    end

    test "invalid list" do
      assert_raise CompileError,
                   ~r"invalid numerical expression: \[a, b\] \(only keyword lists are allowed\)",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, b), do: [a, b]
                     end
                   end
    end

    test "invalid keyword list" do
      assert_raise CompileError,
                   ~r"invalid numerical expression: \[a: a, b: b\] \(the only allowed keys",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(a, b), do: [a: a, b: b]
                     end
                   end
    end

    test "invalid tensor constant" do
      assert_raise CompileError,
                   ~r"defn expects a tensor allocated on Nx.BitStringDevice as a constant",
                   fn ->
                     defmodule Sample do
                       @nx_tensor Nx.tensor(1) |> Map.replace!(:data, {SomethingBad, :another})
                       import Nx.Defn
                       defn default, do: @nx_tensor
                     end
                   end
    end
  end

  defp purge(module) do
    :code.purge(module)
    :code.delete(module)
  end

  defp ast_to_string(name, arity) do
    {_, _, ast} = __defn__(name, arity)
    Macro.to_string(ast)
  end
end

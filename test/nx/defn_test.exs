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
      t = just_two_int()
      assert Nx.to_bitstring(t) == <<2::64-native>>
      assert Nx.type(t) == {:s, 64}
      assert Nx.shape(t) == {}

      t = just_two_float()
      assert Nx.to_bitstring(t) == <<2.0::float-64-native>>
      assert Nx.type(t) == {:f, 64}
      assert Nx.shape(t) == {}
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

    test "invalid tensor constant" do
      assert_raise CompileError,
                   ~r"defn expects a tensor allocated on Nx.BitStringDevice as a constant",
                   fn ->
                     defmodule Sample do
                       @nx_tensor Nx.tensor([]) |> Map.replace!(:data, {SomethingBad, :another})
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
end

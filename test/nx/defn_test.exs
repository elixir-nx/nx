defmodule Nx.DefnTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr
  alias Nx.DefnTest.Sample
  import Nx.Defn

  defmacrop location(plus) do
    file = Path.relative_to_cwd(__CALLER__.file)
    quote do: "#{unquote(file)}:#{unquote(__CALLER__.line) + unquote(plus)}"
  end

  defmodule Identity do
    @behaviour Nx.Defn.Compiler

    def __compile__(_env, _kind, vars, fun, _opts) do
      params =
        for var <- vars do
          unless is_struct(var, Nx.Tensor) or is_number(var) do
            raise "invalid argument"
          end

          tensor = Nx.tensor(var)
          Nx.Defn.Expr.parameter(Nx.shape(tensor), tensor)
        end

      fun.(params)
    end
  end

  @default_defn_compiler Identity

  describe "unary ops" do
    defn exp(t), do: Nx.exp(t)

    test "to expr" do
      assert %Expr{op: :exp, args: [_], shape: {3}} = exp(Nx.tensor([1, 2, 3]))
    end
  end

  describe "binary ops" do
    defn add(t1, t2), do: Nx.add(t1, t2)

    test "to expr" do
      assert %Expr{op: :add, args: [_, _], shape: {3}} = add(Nx.tensor([1, 2, 3]), Nx.tensor(1))

      assert %Expr{op: :add, args: [_, _], shape: {2, 2}} =
               add(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1, 2]))
    end
  end

  describe "aggregate axes ops" do
    defn sum_all(t), do: Nx.sum(t)
    defn sum_pos(t), do: Nx.sum(t, axes: [0, 1])
    defn sum_neg(t), do: Nx.sum(t, axes: [-1, -2])

    test "to expr" do
      assert %Expr{op: :sum, args: [_, []], shape: {}} = sum_all(Nx.tensor([1, 2, 3]))

      assert %Expr{op: :sum, args: [_, [axes: [0, 1]]], shape: {}} =
               sum_pos(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %Expr{op: :sum, args: [_, [axes: [0, 1]]], shape: {3}} =
               sum_pos(Nx.tensor([[[1, 2, 3], [1, 2, 3]]]))

      assert %Expr{op: :sum, args: [_, [axes: [1, 0]]], shape: {}} =
               sum_neg(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %Expr{op: :sum, args: [_, [axes: [2, 1]]], shape: {1}} =
               sum_neg(Nx.tensor([[[1, 2, 3], [1, 2, 3]]]))
    end
  end

  describe "aggregate axis ops" do
    defn arg_all(t), do: Nx.argmin(t)
    defn arg_pos(t), do: Nx.argmin(t, axis: 0)
    defn arg_neg(t), do: Nx.argmin(t, axis: -1)

    test "to expr" do
      assert %Expr{op: :argmin, args: [_, []], shape: {}} = arg_all(Nx.tensor([1, 2, 3]))

      assert %Expr{op: :argmin, args: [_, [axis: 0]], shape: {3}} =
               arg_pos(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %Expr{op: :argmin, args: [_, [axis: 1]], shape: {2}} =
               arg_neg(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end
  end

  describe "rank, shape, size" do
    defn rank(t), do: Nx.rank(t)
    defn shape(t), do: Nx.shape(t)
    defn size(t), do: Nx.size(t)

    test "rank" do
      assert %Expr{shape: {}, op: :constant, args: [2]} = rank(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "shape" do
      assert {%Expr{shape: {}, op: :constant, args: [2]},
              %Expr{shape: {}, op: :constant, args: [3]}} =
               shape(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "size" do
      assert %Expr{shape: {}, op: :constant, args: [6]} = size(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end
  end

  describe "creation ops" do
    defn iota(t), do: Nx.iota(t)
    defn random_uniform(t), do: Nx.random_uniform(t, 0.0, 2.0)
    defn random_normal(t), do: Nx.random_normal(t, 0.0, 1.0)

    test "iota" do
      assert %Expr{op: :iota, args: [{3}, []], shape: {3}} = iota(Nx.tensor([1, 2, 3]))
    end

    test "random uniform" do
      assert %Expr{op: :random_uniform, args: [{3}, 0.0, 2.0, []], shape: {3}} =
               random_uniform(Nx.tensor([1, 2, 3]))
    end

    test "random normal" do
      assert %Expr{op: :random_normal, args: [{3}, 0.0, 1.0, []], shape: {3}} =
               random_normal(Nx.tensor([1, 2, 3]))
    end
  end

  describe "tensor ops" do
    defn dot(t1, t2), do: Nx.dot(t1, t2)
    defn outer(t1, t2), do: Nx.outer(t1, t2)
    defn transpose_1(t), do: Nx.transpose(t)
    defn transpose_2(t), do: Nx.transpose(t, [-1, -2])
    defn reshape(t), do: Nx.reshape(t, {2, 3})
    defn broadcast(t), do: Nx.broadcast(t, {3, 3, 3})

    test "dot product" do
      assert %Expr{op: :dot, args: [_, _], shape: {2, 2}} =
               dot(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end

    test "outer product" do
      assert %Expr{op: :outer, args: [_, _], shape: {3, 3}} =
               outer(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3]))
    end

    test "transpose" do
      assert %Expr{op: :transpose, args: [_, [1, 0]], shape: {3, 2}} =
               transpose_1(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %Expr{op: :transpose, args: [_, [1, 0]], shape: {3, 2}} =
               transpose_2(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "reshape" do
      assert %Expr{op: :reshape, args: [_, _], shape: {2, 3}} =
               reshape(Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end

    test "broadcast" do
      assert %Expr{op: :broadcast, args: [_, _], shape: {3, 3, 3}} =
               broadcast(Nx.tensor([1, 2, 3]))
    end
  end

  describe "conditional ops" do
    defn select(t1, t2, t3), do: Nx.select(t1, t2, t3)

    test "select with tensor predicate" do
      assert %Expr{op: :select, args: [_, _, _], shape: {2, 2}} =
               select(Nx.tensor([[1, 1], [0, 0]]), Nx.tensor(1), Nx.tensor(0))
    end

    test "select with scalar predicate" do
      assert %Expr{op: :select, args: [_, _, _], shape: {5}} = select(Nx.tensor(1), Nx.tensor([1, 2, 3, 4, 5]), Nx.tensor(0))
    end
  end

  describe "scalar" do
    defn just_two_int, do: 2
    defn just_two_float, do: 2.0

    test "returns a constant for the scalar" do
      assert %Expr{op: :constant, args: [2], shape: {}} = just_two_int()
      assert %Expr{op: :constant, args: [2.0], shape: {}} = just_two_float()
    end
  end

  describe "parameter" do
    defn parameter_var(a, b, c), do: {a, b, c}

    test "as vars" do
      assert {%Expr{op: :parameter}, %Expr{op: :parameter}, %Expr{op: :parameter}} =
               parameter_var(1, 2, 3)
    end

    defn parameter_tuple({a, b}, c), do: {a, b, c}

    test "as tuples" do
      assert {%Expr{op: :parameter}, %Expr{op: :parameter}, %Expr{op: :parameter}} =
               parameter_tuple({1, 2}, 3)
    end
  end

  describe "tensor constants" do
    @two 2
    defn two_attribute(), do: @two

    test "expands module attributes to scalars" do
      assert %Expr{op: :constant, args: [2], shape: {}} = two_attribute()
    end

    @two_per_two Nx.tensor([[1, 2], [3, 4]])
    defn two_per_two_attribute(), do: @two_per_two

    test "expands module attributes to tensors" do
      assert %Expr{op: :tensor, args: [_], shape: {2, 2}} = two_per_two_attribute()
    end

    defn two_per_two_nx_tensor(), do: Nx.tensor([[1, 2], [3, 4]])

    test "supports Nx.tensor calls" do
      assert %Expr{op: :tensor, args: [_], shape: {2, 2}} = two_per_two_attribute()
    end

    @invalid_tensor Nx.tensor(1) |> Map.replace!(:data, {SomethingBad, :another})
    defn invalid_tensor, do: @invalid_tensor

    test "raises on invalid tensor" do
      assert_raise ArgumentError,
                   "tensors inside defn must be allocated on Nx.BitStringDevice",
                   fn -> invalid_tensor() end
    end
  end

  describe "operators" do
    defn add_two(a, b), do: a + b

    test "+" do
      assert %Expr{op: :add, args: [_, _]} = add_two(1, 2)
    end

    defn subtract_two(a, b), do: a - b

    test "-" do
      assert %Expr{op: :subtract, args: [_, _]} = subtract_two(1, 2)
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      assert %Expr{op: :multiply, args: [_, _]} = multiply_two(1, 2)
    end

    defn divide_two(a, b), do: a / b

    test "/" do
      assert %Expr{op: :divide, args: [_, _]} = divide_two(1, 2)
    end

    defn band_two(a, b), do: a &&& b

    test "&&&" do
      assert %Expr{op: :bitwise_and, args: [_, _]} = band_two(1, 2)
    end

    defn bor_two(a, b), do: a ||| b

    test "|||" do
      assert %Expr{op: :bitwise_or, args: [_, _]} = bor_two(1, 2)
    end

    defn bxor_two(a, b), do: a ^^^ b

    test "^^^" do
      assert %Expr{op: :bitwise_xor, args: [_, _]} = bxor_two(1, 2)
    end

    defn bsl_two(a, b), do: a <<< b

    test "<<<" do
      assert %Expr{op: :left_shift, args: [_, _]} = bsl_two(1, 2)
    end

    defn bsr_two(a, b), do: a >>> b

    test ">>>" do
      assert %Expr{op: :right_shift, args: [_, _]} = bsr_two(1, 2)
    end

    defn add_two_with_pipe(a, b), do: a |> Nx.add(b)

    test "|>" do
      assert %Expr{op: :add, args: [_, _]} = add_two_with_pipe(1, 2)
    end

    defn unary_plus(a), do: +a
    defn unary_minus(a), do: -a

    test "unary plus and minus" do
      assert %Expr{op: :parameter, args: [_]} = unary_plus(1)
      assert %Expr{op: :negate, args: [_]} = unary_minus(1)
    end

    defn unary_bnot(a), do: ~~~a

    test "~~~" do
      assert %Expr{op: :bitwise_not, args: [_]} = unary_bnot(1)
    end
  end

  describe "kernel functions" do
    defn max_two(a, b) do
      max(a, b)
    end

    test "max/2" do
      assert %Expr{op: :max, args: [_, _]} = max_two(1, 2)
    end

    defn min_two(a, b) do
      min(a, b)
    end

    test "min/2" do
      assert %Expr{op: :min, args: [_, _]} = min_two(1, 2)
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
      assert %Expr{op: :add, args: [_, _]} = add_two_from_external_macro(1, 2)
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
      assert %Expr{op: :add, args: [_, _]} = add_two_from_external_macro(1, 2)
    end

    defn add_two_from_alias(a, b) do
      alias Nx, as: N
      N.add(a, b)
    end

    test "aliases" do
      assert %Expr{op: :add, args: [_, _]} = add_two_from_alias(1, 2)
    end

    dynamic_name = String.to_atom(Enum.join(~w(dynamic name add two), "_"))
    operator = :add
    defnp unquote(dynamic_name)(left, right), do: Nx.unquote(operator)(left, right)

    test "dynamic name" do
      assert %Expr{op: :add, args: [_, _]} = dynamic_name_add_two(1, 2)
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
      assert %Expr{op: :add, args: [_, _]} = add_two_from_public(1, 2)
    end

    defn add_two_from_private(a, b) do
      add_two_from_private_impl(a, b)
    end

    defn add_two_from_private_impl(a, b) do
      a + b
    end

    test "private" do
      assert %Expr{op: :add, args: [_, _]} = add_two_from_private(1, 2)
    end
  end

  describe "remote functions" do
    defmodule Remote do
      defn add_two(c, d), do: c + d
    end

    defn add_two_remote(a, b), do: Remote.add_two(a, b)

    test "public" do
      assert %Expr{op: :add, args: [_, _]} = add_two_remote(1, 2)
    end

    defn add_two_unknown(a, b), do: Nx.DefnTest.unknown(a, b)

    test "invalid remote" do
      assert_raise UndefinedFunctionError,
                   "function Nx.DefnTest.unknown/2 is undefined or private",
                   fn -> add_two_unknown(1, 2) end
    end
  end

  describe "Nx.Defn" do
    @defn_compiler Nx.Defn
    defn add_default(a, b), do: {a + b, a - b, 5}

    # Check the attribute has been reset
    nil = Module.get_attribute(__MODULE__, :defn_compiler)

    test "can be set explicitly set" do
      assert add_default(1, 2) == {Nx.tensor(3), Nx.tensor(-1), Nx.tensor(5)}
    end

    test "is the default compiler" do
      defmodule DefaultCompiler do
        import Nx.Defn
        defn add(a, b), do: a + b
      end

      assert DefaultCompiler.add(1, 2) == Nx.tensor(3)
    end
  end

  describe "compilation errors" do
    test "invalid numerical expression" do
      assert_raise CompileError, ~r"#{location(+5)}: invalid numerical expression", fn ->
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
  end
end

defmodule Nx.DefnTest do
  use ExUnit.Case, async: true

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr
  alias Nx.DefnTest.Sample
  import Nx.Defn

  defmacrop location(plus) do
    file = Path.relative_to_cwd(__CALLER__.file)
    quote do: "#{unquote(file)}:#{unquote(__CALLER__.line) + unquote(plus)}"
  end

  defmodule Identity do
    @behaviour Nx.Defn.Compiler

    def __jit__(key, vars, fun, _opts) do
      Process.put(__MODULE__, key)
      fun.(vars)
    end
  end

  describe "kernel doctests" do
    use Nx.Defn.Kernel
    doctest Nx.Defn.Kernel
  end

  @default_defn_compiler Identity

  describe "constants" do
    @tensor Nx.tensor([1, 2, 3])
    defn tensor_constant, do: Nx.tensor(@tensor)

    test "from tensor" do
      assert %T{data: %Expr{op: :tensor}} = tensor_constant()
    end

    @tensor [1, 2, 3]
    defn list_constant, do: Nx.tensor(@tensor)

    test "from list" do
      assert %T{data: %Expr{op: :tensor}} = list_constant()
    end
  end

  describe "unary ops" do
    defn exp(t), do: Nx.exp(t)

    test "to expr" do
      assert %T{shape: {3}, type: {:f, 64}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3]))

      assert %T{shape: {3}, type: {:f, 32}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3], type: {:f, 32}))

      assert %T{shape: {3}, type: {:bf, 16}, data: %Expr{op: :exp, args: [_]}} =
               exp(Nx.tensor([1, 2, 3], type: {:bf, 16}))
    end
  end

  describe "binary ops" do
    defn add(t1, t2), do: Nx.add(t1, t2)
    defn add_two_int(t), do: Nx.add(t, 2)
    defn add_two_float(t), do: Nx.add(t, 2)

    test "to expr" do
      assert %T{shape: {3}, type: {:s, 64}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([1, 2, 3]), Nx.tensor(1))

      assert %T{shape: {2, 2}, type: {:s, 64}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1, 2]))

      assert %T{shape: {2, 2}, type: {:f, 64}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([[1, 2], [3, 4]], type: {:f, 32}), Nx.tensor([1, 2]))

      assert %T{shape: {2, 2}, type: {:f, 64}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([[1, 2], [3, 4]], type: {:f, 32}), Nx.tensor([1, 2], type: {:s, 32}))
    end

    test "constant" do
      assert %T{shape: {3}, type: {:u, 8}, data: %Expr{op: :add, args: [_, _]}} =
               add_two_int(Nx.tensor([1, 2, 3], type: {:u, 8}))

      assert %T{shape: {3}, type: {:bf, 16}, data: %Expr{op: :add, args: [_, _]}} =
               add_two_float(Nx.tensor([1, 2, 3], type: {:bf, 16}))
    end
  end

  describe "aggregate axes ops" do
    defn sum_all(t), do: Nx.sum(t)
    defn sum_pos(t), do: Nx.sum(t, axes: [0, 1])
    defn sum_neg(t), do: Nx.sum(t, axes: [-1, -2])

    test "to expr" do
      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :sum, args: [_, [axes: nil]]}} =
               sum_all(Nx.tensor([1, 2, 3]))

      assert %T{shape: {}, type: {:s, 64}, data: %Expr{op: :sum, args: [_, [axes: [0, 1]]]}} =
               sum_pos(Nx.tensor([[1, 2, 3], [1, 2, 3]], type: {:s, 8}))

      assert %T{shape: {3}, type: {:f, 32}, data: %Expr{op: :sum, args: [_, [axes: [0, 1]]]}} =
               sum_pos(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:f, 32}))

      assert %T{shape: {}, type: {:u, 64}, data: %Expr{op: :sum, args: [_, [axes: [1, 0]]]}} =
               sum_neg(Nx.tensor([[1, 2, 3], [1, 2, 3]], type: {:u, 8}))

      assert %T{shape: {1}, type: {:bf, 16}, data: %Expr{op: :sum, args: [_, [axes: [2, 1]]]}} =
               sum_neg(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:bf, 16}))
    end
  end

  describe "creation ops" do
    defn iota(t), do: Nx.iota(t)
    defn random_uniform(t), do: Nx.random_uniform(t, 0.0, 2.0)
    defn random_normal(t), do: Nx.random_normal(t, 0.0, 1.0)

    test "iota" do
      assert %T{shape: {3}, data: %Expr{op: :iota, args: [nil]}} = iota(Nx.tensor([1, 2, 3]))
    end

    test "random uniform" do
      assert %T{shape: {3}, data: %Expr{op: :random_uniform, args: [0.0, 2.0]}} =
               random_uniform(Nx.tensor([1, 2, 3]))
    end

    test "random normal" do
      assert %T{shape: {3}, data: %Expr{op: :random_normal, args: [0.0, 1.0]}} =
               random_normal(Nx.tensor([1, 2, 3]))
    end
  end

  describe "tensor ops" do
    defn dot2(t1, t2), do: Nx.dot(t1, t2)
    defn dot4(t1, t2), do: Nx.dot(t1, [-2], t2, [-1])
    defn outer(t1, t2), do: Nx.outer(t1, t2)
    defn transpose_1(t), do: Nx.transpose(t)
    defn transpose_2(t), do: Nx.transpose(t, axes: [-1, -2])
    defn reshape(t), do: Nx.reshape(t, {2, 3})
    defn broadcast(t), do: Nx.broadcast(t, {3, 3, 3})
    defn broadcast_axes(t), do: Nx.broadcast(t, {3, 2}, axes: [-2])
    defn squeeze(t), do: Nx.squeeze(t)

    test "dot product" do
      assert %T{data: %Expr{op: :dot, args: [_, [0], _, [0]]}, shape: {2}} =
               dot2(Nx.tensor([1, 2, 3]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))

      assert %T{data: %Expr{op: :dot, args: [_, [1], _, [0]]}, shape: {2}} =
               dot2(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :dot, args: [_, [1], _, [0]]}, shape: {2, 2}} =
               dot2(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))

      assert %T{data: %Expr{op: :dot, args: [_, [0], _, [1]]}, shape: {3, 3}} =
               dot4(Nx.tensor([[1, 2, 3], [1, 2, 3]]), Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end

    test "squeeze" do
      assert %T{data: %Expr{op: :squeeze, args: [_, [axes: [0, 2, 4]]]}, shape: {3, 2}} =
               squeeze(Nx.iota({1, 3, 1, 2, 1}))
    end

    test "outer product" do
      assert %T{data: %Expr{op: :outer, args: [_, _]}, shape: {3, 3}} =
               outer(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3]))
    end

    test "transpose" do
      assert %T{data: %Expr{op: :transpose, args: [_, [axes: [1, 0]]]}, shape: {3, 2}} =
               transpose_1(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %T{data: %Expr{op: :transpose, args: [_, [axes: [1, 0]]]}, shape: {3, 2}} =
               transpose_2(Nx.tensor([[1, 2, 3], [1, 2, 3]]))
    end

    test "reshape" do
      assert %T{data: %Expr{op: :reshape, args: [_, _]}, shape: {2, 3}} =
               reshape(Nx.tensor([[1, 2], [3, 4], [5, 6]]))
    end

    test "broadcast" do
      assert %T{data: %Expr{op: :broadcast, args: [_, _, [2]]}, shape: {3, 3, 3}} =
               broadcast(Nx.tensor([1, 2, 3]))

      assert %T{data: %Expr{op: :broadcast, args: [_, _, [0]]}, shape: {3, 2}} =
               broadcast_axes(Nx.tensor([1, 2, 3]))
    end
  end

  describe "conditional ops" do
    defn select(t1, t2, t3), do: Nx.select(t1, t2, t3)

    test "select with tensor predicate" do
      assert %{data: %Expr{op: :select, args: [_, _, _]}, shape: {2, 2}} =
               select(Nx.tensor([[1, 1], [0, 0]]), Nx.tensor(1), Nx.tensor(0))
    end

    test "select with scalar predicate" do
      assert %{data: %Expr{op: :select, args: [_, _, _]}, shape: {5}} =
               select(Nx.tensor(1), Nx.tensor([1, 2, 3, 4, 5]), Nx.tensor(0))

      assert %{data: %Expr{op: :select, args: [_, _, _]}, shape: {2, 2}} =
               select(Nx.tensor(1), Nx.tensor([[1], [2]]), Nx.tensor([[1, 2]]))
    end
  end

  describe "reduce ops" do
    defn reduce(t1, acc), do: Nx.reduce(t1, acc, fn x, y -> x + y end)

    defn reduce_invalid(t1, amplifier), do: Nx.reduce(t1, 0, fn x, y -> x * amplifier + y end)

    defn reduce_non_scalar(t1), do: Nx.reduce(t1, 0, fn x, y -> Nx.broadcast(x * y, {1, 1}) end)

    defn reduce_with_opts(t1, acc),
      do: Nx.reduce(t1, acc, [type: {:f, 64}, axes: [-1]], fn x, y -> x + y end)

    test "reduces with function" do
      assert %{
               data: %Expr{op: :reduce, args: [_, _, [axes: nil], fun]},
               type: {:s, 64},
               shape: {}
             } = reduce(Nx.tensor([1, 2, 3]), 0)

      assert %T{data: %Expr{op: :fun}} = fun

      assert %{
               data: %Expr{op: :reduce, args: [_, _, [axes: [1]], fun]},
               type: {:f, 64},
               shape: {3}
             } = reduce_with_opts(Nx.tensor([[1], [2], [3]]), 0)

      assert %T{data: %Expr{op: :fun}} = fun
    end

    test "reduces raises on invalid expression" do
      assert_raise RuntimeError,
                   ~r"cannot build defn because expressions come from different contexts",
                   fn -> reduce_invalid(Nx.tensor([1, 2, 3]), 0) end
    end

    test "reduces raises on non scalar functions" do
      assert_raise RuntimeError,
                   "reduce function must return a scalar tensor, got: {1, 1}",
                   fn -> reduce_non_scalar(Nx.tensor([1, 2, 3])) end
    end
  end

  describe "operators" do
    defn add_two(a, b), do: a + b

    test "+" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two(1, 2)
      assert Nx.Defn.Kernel.+(1, 2) == 3
    end

    defn subtract_two(a, b), do: a - b

    test "-" do
      assert %T{data: %Expr{op: :subtract, args: [_, _]}} = subtract_two(1, 2)
      assert Nx.Defn.Kernel.-(1, 2) == -1
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      assert %T{data: %Expr{op: :multiply, args: [_, _]}} = multiply_two(1, 2)
      assert Nx.Defn.Kernel.*(1, 2) == 2
    end

    defn divide_two(a, b), do: a / b

    test "/" do
      assert %T{data: %Expr{op: :divide, args: [_, _]}} = divide_two(1, 2)
      assert Nx.Defn.Kernel./(1, 2) == 0.5
    end

    defn land_two(a, b), do: a and b

    test "and" do
      assert %T{data: %Expr{op: :logical_and, args: [_, _]}} = land_two(1, 2)

      assert Nx.Defn.Kernel.and(0, 0) == 0
      assert Nx.Defn.Kernel.and(1, 0) == 0
      assert Nx.Defn.Kernel.and(0, 2) == 0
      assert Nx.Defn.Kernel.and(1, 1) == 1

      assert Nx.Defn.Kernel.and(0, 0.0) == 0.0
      assert Nx.Defn.Kernel.and(1, 0.0) == 0.0
      assert Nx.Defn.Kernel.and(0, 2.0) == 0.0
      assert Nx.Defn.Kernel.and(1, 1.0) == 1.0
    end

    defn lor_two(a, b), do: a or b

    test "or" do
      assert %T{data: %Expr{op: :logical_or, args: [_, _]}} = lor_two(1, 2)

      assert Nx.Defn.Kernel.or(0, 0) == 0
      assert Nx.Defn.Kernel.or(1, 0) == 1
      assert Nx.Defn.Kernel.or(0, 2) == 1
      assert Nx.Defn.Kernel.or(1, 1) == 1

      assert Nx.Defn.Kernel.or(0, 0.0) == 0.0
      assert Nx.Defn.Kernel.or(1, 0.0) == 1.0
      assert Nx.Defn.Kernel.or(0, 2.0) == 1.0
      assert Nx.Defn.Kernel.or(1, 1.0) == 1.0
    end

    defn lnot(a), do: not a

    test "not" do
      assert %T{data: %Expr{op: :equal, args: [_, _]}} = lnot(1)

      assert Nx.Defn.Kernel.not(0) == 1
      assert Nx.Defn.Kernel.not(1) == 0
      assert Nx.Defn.Kernel.not(2) == 0

      assert Nx.Defn.Kernel.not(0.0) == 1.0
      assert Nx.Defn.Kernel.not(1.0) == 0.0
      assert Nx.Defn.Kernel.not(2.0) == 0.0
    end

    defn band_two(a, b), do: a &&& b

    test "&&&" do
      assert %T{data: %Expr{op: :bitwise_and, args: [_, _]}} = band_two(1, 2)
      assert Nx.Defn.Kernel.&&&(1, 2) == 0
    end

    defn bor_two(a, b), do: a ||| b

    test "|||" do
      assert %T{data: %Expr{op: :bitwise_or, args: [_, _]}} = bor_two(1, 2)
      assert Nx.Defn.Kernel.|||(1, 2) == 3
    end

    defn bxor_two(a, b), do: a ^^^ b

    test "^^^" do
      assert %T{data: %Expr{op: :bitwise_xor, args: [_, _]}} = bxor_two(1, 2)
      assert Nx.Defn.Kernel.^^^(1, 2) == 3
    end

    defn bsl_two(a, b), do: a <<< b

    test "<<<" do
      assert %T{data: %Expr{op: :left_shift, args: [_, _]}} = bsl_two(1, 2)
      assert Nx.Defn.Kernel.<<<(1, 2) == 4
    end

    defn bsr_two(a, b), do: a >>> b

    test ">>>" do
      assert %T{data: %Expr{op: :right_shift, args: [_, _]}} = bsr_two(1, 2)
      assert Nx.Defn.Kernel.>>>(1, 2) == 0
    end

    defn add_two_with_pipe(a, b), do: a |> Nx.add(b)

    test "|>" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_with_pipe(1, 2)
    end

    defn unary_plus(a), do: +a
    defn unary_minus(a), do: -a

    test "unary plus and minus" do
      assert %T{data: %Expr{op: :parameter, args: [_]}} = unary_plus(1)
      assert %T{data: %Expr{op: :negate, args: [_]}} = unary_minus(1)

      assert Nx.Defn.Kernel.+(1) == 1
      assert Nx.Defn.Kernel.-(1) == -1
    end

    defn unary_bnot(a), do: ~~~a

    test "~~~" do
      assert %T{data: %Expr{op: :bitwise_not, args: [_]}} = unary_bnot(1)
      assert Nx.Defn.Kernel.~~~(1) == -2
    end

    defn access_sum(a, opts \\ []), do: Nx.sum(a, axes: opts[:axes])

    test "access" do
      assert %T{data: %Expr{op: :sum, args: [_, [axes: [1]]]}} =
               access_sum(Nx.iota({2, 2}), axes: [1])
    end
  end

  describe "kernel functions" do
    defn max_two(a, b) do
      max(a, b)
    end

    test "max/2" do
      assert %T{data: %Expr{op: :max, args: [_, _]}} = max_two(1, 2)
    end

    defn min_two(a, b) do
      min(a, b)
    end

    test "min/2" do
      assert %T{data: %Expr{op: :min, args: [_, _]}} = min_two(1, 2)
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
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_external_macro(1, 2)
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
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_external_macro(1, 2)
    end

    defn add_two_from_alias(a, b) do
      alias Nx, as: N
      N.add(a, b)
    end

    test "aliases" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_alias(1, 2)
    end

    defn add_two_from_kernel_alias(a, b) do
      a |> Kernel.+(b)
    end

    test "kernel alias" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_kernel_alias(1, 2)
    end

    dynamic_name = String.to_atom(Enum.join(~w(dynamic name add two), "_"))
    operator = :add
    defn unquote(dynamic_name)(left, right), do: Nx.unquote(operator)(left, right)

    test "dynamic name" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = dynamic_name_add_two(1, 2)
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
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_public(1, 2)
    end

    defn add_two_from_private(a, b) do
      add_two_from_private_impl(a, b)
    end

    defn add_two_from_private_impl(a, b) do
      a + b
    end

    test "private" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_from_private(1, 2)
    end
  end

  describe "remote functions" do
    defmodule Remote do
      defn add_two(c, d), do: c + d
    end

    defn add_two_remote(a, b), do: Remote.add_two(a, b)

    test "public" do
      assert %T{data: %Expr{op: :add, args: [_, _]}} = add_two_remote(1, 2)
    end

    defn add_two_unknown(a, b), do: Nx.DefnTest.unknown(a, b)

    test "invalid remote" do
      assert_raise UndefinedFunctionError,
                   "function Nx.DefnTest.unknown/2 is undefined or private",
                   fn -> add_two_unknown(1, 2) end
    end
  end

  describe "if" do
    defn if3(a, b, c), do: if(a, do: b, else: c)
    defn if2(a, b), do: if(a, do: b)

    test "converges types" do
      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:f, 32}} =
               if3(Nx.tensor(0), Nx.tensor(0, type: {:s, 16}), Nx.tensor(0, type: {:f, 32}))

      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:f, 64}} =
               if3(Nx.tensor(0), Nx.tensor(0, type: {:s, 32}), Nx.tensor(0, type: {:f, 32}))

      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:u, 16}} =
               if2(Nx.tensor(0), Nx.tensor(0, type: {:u, 16}))

      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:f, 32}} =
               if2(Nx.tensor(0), Nx.tensor(0, type: {:f, 32}))
    end

    test "converges shapes and names" do
      assert %T{data: %Expr{op: :cond}, shape: {2, 2}, names: [:x, :y]} =
               if3(
                 Nx.tensor(0),
                 Nx.tensor([1, 2], names: [:y]),
                 Nx.tensor([[3], [4]], names: [:x, nil])
               )
    end
  end

  describe "cond" do
    defn cond4(a, b, c, d) do
      cond do
        Nx.greater(a, 0) -> b + 1
        Nx.less(a, 0) -> c - 1
        true -> d * 1
      end
    end

    test "supports multiple clauses" do
      assert %T{data: %Expr{op: :cond, args: [clauses, last]}, shape: {}, type: {:s, 64}} =
               cond4(Nx.tensor(0), Nx.tensor(1), Nx.tensor(2), Nx.tensor(3))

      [{first_head, first_body}, {second_head, second_body}] = clauses
      assert %T{data: %Expr{op: :greater}} = first_head
      assert %T{data: %Expr{op: :add}} = first_body
      assert %T{data: %Expr{op: :less}} = second_head
      assert %T{data: %Expr{op: :subtract}} = second_body
      assert %T{data: %Expr{op: :multiply}} = last
    end

    test "converges types" do
      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:s, 32}} =
               cond4(
                 Nx.tensor(0),
                 Nx.tensor(1, type: {:s, 8}),
                 Nx.tensor(2, type: {:s, 16}),
                 Nx.tensor(3, type: {:s, 32})
               )
    end

    test "converges shapes and names" do
      assert %T{data: %Expr{op: :cond}, shape: {2, 2}, names: [:x, :y]} =
               cond4(
                 Nx.tensor(0),
                 Nx.tensor([1, 2], names: [:y]),
                 Nx.tensor([[3], [4]], names: [:x, nil]),
                 Nx.tensor(5)
               )
    end

    test "raises if cond is missing last atom clause" do
      assert_raise CompileError, ~r"expected the last clause of cond to match on an atom", fn ->
        defmodule InvalidCond do
          defn badcond(a) do
            cond do
              Nx.any?(a) -> +a
              Nx.all?(a) -> -a
            end
          end
        end
      end
    end
  end

  describe "Nx.Defn" do
    @defn_compiler Nx.Defn
    defn add_default(a, b), do: {a + b, a - b}

    # Check the attribute has been reset
    nil = Module.get_attribute(__MODULE__, :defn_compiler)

    test "can be set explicitly set" do
      assert add_default(1, 2) == {Nx.tensor(3), Nx.tensor(-1)}
    end

    test "is the default compiler" do
      defmodule DefaultCompiler do
        import Nx.Defn
        defn add(a, b), do: a + b
      end

      assert DefaultCompiler.add(1, 2) == Nx.tensor(3)
    end

    @defn_compiler Nx.Defn
    defn default_add_two_int(t), do: Nx.add(t, 2)

    @defn_compiler Nx.Defn
    defn default_add_two_float(t), do: Nx.add(t, 2)

    test "constant" do
      assert %T{shape: {3}, type: {:u, 8}} =
               default_add_two_int(Nx.tensor([1, 2, 3], type: {:u, 8}))

      assert %T{shape: {3}, type: {:bf, 16}} =
               default_add_two_float(Nx.tensor([1, 2, 3], type: {:bf, 16}))
    end

    @defn_compiler Nx.Defn
    defn default_iota(), do: Nx.iota({2, 2})

    test "iota" do
      assert %T{shape: {2, 2}, type: {:s, 64}} = default_iota()
    end

    @defn_compiler Nx.Defn
    defn default_reshape(t), do: Nx.reshape(t, {3, 2})

    test "reshape" do
      assert %T{shape: {3, 2}, type: {:s, 64}} = default_reshape(Nx.iota({2, 3}))
    end

    @defn_compiler Nx.Defn
    defn default_if3(a, b, c), do: if(a, do: b, else: c)

    test "if" do
      assert default_if3(Nx.tensor(0), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})) ==
               Nx.tensor(2, type: {:f, 32})

      assert default_if3(Nx.tensor(1), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})) ==
               Nx.tensor(1, type: {:f, 32})

      assert default_if3(Nx.tensor(2), Nx.tensor(1, type: {:s, 16}), Nx.tensor(2, type: {:f, 32})) ==
               Nx.tensor(1, type: {:f, 32})

      assert default_if3(Nx.tensor(0), Nx.tensor([1, 2]), Nx.tensor([[3], [4]])) ==
               Nx.tensor([[3, 3], [4, 4]])

      assert default_if3(Nx.tensor(1), Nx.tensor([1, 2]), Nx.tensor([[3], [4]])) ==
               Nx.tensor([[1, 2], [1, 2]])
    end

    @defn_compiler Nx.Defn
    defn default_if_tuple(a, b, c), do: if(a, do: {{a, b}, c}, else: {{c, b}, a})

    test "if with tuples" do
      assert default_if_tuple(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) ==
               {{Nx.tensor(20), Nx.tensor(10)}, Nx.tensor(0)}

      assert default_if_tuple(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) ==
               {{Nx.tensor(1), Nx.tensor(10)}, Nx.tensor(20)}

      assert default_if_tuple(Nx.tensor(0), Nx.tensor(10), Nx.tensor([20, 30])) ==
               {{Nx.tensor([20, 30]), Nx.tensor(10)}, Nx.tensor([0, 0])}

      assert default_if_tuple(Nx.tensor(1), Nx.tensor(10), Nx.tensor([20, 30])) ==
               {{Nx.tensor([1, 1]), Nx.tensor(10)}, Nx.tensor([20, 30])}
    end

    @defn_compiler Nx.Defn
    defn default_if_tuple_match(a, b, c) do
      {{x, y}, z} = if(a, do: {{a, b}, c}, else: {{c, b}, a})
      x * y - z
    end

    test "if with matched tuples" do
      assert default_if_tuple_match(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(200)
      assert default_if_tuple_match(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) == Nx.tensor(-10)
    end

    @defn_compiler Nx.Defn
    defn default_if_tuple_return(a, b, c) do
      {xy, _} = if(a, do: {{a, b}, c}, else: {{c, b}, a})
      xy
    end

    test "if with return tuple" do
      assert default_if_tuple_return(Nx.tensor(0), Nx.tensor(10), Nx.tensor(20)) ==
               {Nx.tensor(20), Nx.tensor(10)}

      assert default_if_tuple_return(Nx.tensor(1), Nx.tensor(10), Nx.tensor(20)) ==
               {Nx.tensor(1), Nx.tensor(10)}
    end
  end

  describe "transform" do
    defn transform_inspect(a, b) do
      (Nx.tanh(a) + Nx.power(b, 2)) |> print_expr()
    end

    test "executes the transformation" do
      assert ExUnit.CaptureIO.capture_io(fn -> transform_inspect(1, 2) end) == """
             #Nx.Tensor<
               Nx.Defn.Expr
               parameter a         s64
               parameter c         s64
               b = tanh [ a ]      f64
               d = power [ c, 2 ]  s64
               e = add [ b, d ]    f64
             >
             """
    end

    @defn_compiler Nx.Defn
    defn transform_back_and_forth(a) do
      Nx.exp(transform(Nx.negate(a), &private_back_and_forth/1))
    end

    defp private_back_and_forth(a) do
      Nx.Defn = Process.get(Nx.Defn.Compiler)
      final_back_and_forth(a)
    end

    defn final_back_and_forth(a), do: Nx.tanh(a)

    test "back and forth between Elixir and defn" do
      assert transform_back_and_forth(Nx.tensor(1)) ==
               Nx.tensor(1) |> Nx.negate() |> Nx.tanh() |> Nx.exp()
    end
  end

  describe "jit" do
    defn defn_jit({a, b}, c), do: a + b - c

    def elixir_jit({a, b}, c) do
      true = Process.get(Nx.Defn.Compiler) in [Nx.Defn, Identity]
      a |> Nx.add(b) |> Nx.subtract(c)
    end

    test "compiles defn function" do
      assert Nx.Defn.jit(&defn_jit/2, Nx.Defn).({4, 5}, 3) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit/2, Nx.Defn).({4, 5}, Nx.tensor(3)) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit(&1, 3), Nx.Defn).({4, 5}) == Nx.tensor(6)

      assert %T{data: %Expr{op: :subtract}} = Nx.Defn.jit(&defn_jit/2, Identity).({1, 2}, 3)
    end

    test "compiles elixir function" do
      assert Nx.Defn.jit(&elixir_jit/2, Nx.Defn).({4, 5}, 3) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit/2, Nx.Defn).({4, 5}, Nx.tensor(3)) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit(&1, 3), Nx.Defn).({4, 5}) == Nx.tensor(6)

      assert %T{data: %Expr{op: :subtract}} = Nx.Defn.jit(&elixir_jit/2, Identity).({4, 5}, 3)
    end

    test "raises if it doesn't return an expression" do
      assert_raise ArgumentError,
                   "defn must return an expression tensor or a tuple, got: :ok",
                   fn ->
                     Nx.Defn.jit(fn -> :ok end, Nx.Defn).()
                   end
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

  @default_defn_compiler Nx.Defn

  describe "default arguments" do
    defn sum_axis_opts(a, opts \\ []), do: Nx.sum(a, opts)

    test "are supported" do
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]])) == Nx.tensor(10)
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]]), axes: [0]) == Nx.tensor([4, 6])
      assert sum_axis_opts(Nx.tensor([[1, 2], [3, 4]]), axes: [1]) == Nx.tensor([3, 7])
    end

    defn random_opts(opts \\ []), do: Nx.random_uniform({}, 0, 1, opts)

    test "exclusively" do
      assert random_opts([]).type == {:s, 64}
      assert random_opts(type: {:f, 64}).type == {:f, 64}
    end

    @defn_compiler Identity
    defn sum_axis_expr(a, opts \\ []), do: Nx.sum(a, opts)

    test "have their own cache key" do
      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      key0 = Process.get(Identity)
      assert is_function(key0, 1)

      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [1])
      key1 = Process.get(Identity)
      assert is_function(key1, 1)

      sum_axis_expr(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      assert Process.get(Identity) == key0
    end
  end

  describe "private definitions" do
    defnp private(a, b), do: a + b
    defn calls_private(a, b), do: private(a, b)

    test "work" do
      assert private(1, 2) == Nx.tensor(3)
    end

    test "are not exported" do
      refute function_exported?(__MODULE__, :private, 2)
    end

    test "are callable from defn" do
      assert calls_private(1, 2) == Nx.tensor(3)
    end
  end
end

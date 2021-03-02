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

    def __async__(_, _, _, _), do: raise("not implemented")

    def __jit__(key, vars, fun, _opts) do
      Process.put(__MODULE__, key)
      fun.(vars)
    end
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

    @tensor Nx.to_binary(Nx.tensor([1, 2, 3]))
    defn binary_constant, do: Nx.from_binary(@tensor, {:s, 64})

    test "from binary" do
      assert %T{data: %Expr{op: :tensor}} = binary_constant()
    end
  end

  describe "tuple" do
    defn tuple_shape_match({_, _} = var) do
      {a, b} = var
      a + b
    end

    test "allows pattern matching on the tuple shape with underscores" do
      assert %T{shape: {}, type: {:f, 32}, data: %Expr{op: :add, args: [left, right]}} =
               tuple_shape_match({1, 2.0})

      assert %T{data: %Expr{op: :parameter, args: [0]}, type: {:s, 64}} = left
      assert %T{data: %Expr{op: :parameter, args: [1]}, type: {:f, 32}} = right
    end
  end

  describe "unary ops" do
    defn exp(t), do: Nx.exp(t)

    test "to expr" do
      assert %T{shape: {3}, type: {:f, 32}, data: %Expr{op: :exp, args: [_]}} =
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

      assert %T{shape: {2, 2}, type: {:f, 32}, data: %Expr{op: :add, args: [_, _]}} =
               add(Nx.tensor([[1, 2], [3, 4]], type: {:f, 32}), Nx.tensor([1, 2]))

      assert %T{shape: {2, 2}, type: {:f, 32}, data: %Expr{op: :add, args: [_, _]}} =
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
    defn sum_keep(t), do: Nx.sum(t, axes: [0, 1], keep_axes: true)

    test "to expr" do
      assert %T{
               shape: {},
               type: {:s, 64},
               data: %Expr{op: :sum, args: [_, [axes: nil, keep_axes: false]]}
             } = sum_all(Nx.tensor([1, 2, 3]))

      assert %T{
               shape: {},
               type: {:s, 64},
               data: %Expr{op: :sum, args: [_, [axes: [0, 1], keep_axes: false]]}
             } = sum_pos(Nx.tensor([[1, 2, 3], [1, 2, 3]], type: {:s, 8}))

      assert %T{
               shape: {3},
               type: {:f, 32},
               data: %Expr{op: :sum, args: [_, [axes: [0, 1], keep_axes: false]]}
             } = sum_pos(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:f, 32}))

      assert %T{
               shape: {},
               type: {:u, 64},
               data: %Expr{op: :sum, args: [_, [axes: [1, 0], keep_axes: false]]}
             } = sum_neg(Nx.tensor([[1, 2, 3], [1, 2, 3]], type: {:u, 8}))

      assert %T{
               shape: {1},
               type: {:bf, 16},
               data: %Expr{op: :sum, args: [_, [axes: [2, 1], keep_axes: false]]}
             } = sum_neg(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:bf, 16}))

      assert %T{
               shape: {1, 1, 3},
               type: {:f, 32},
               data: %Expr{op: :sum, args: [_, [axes: [0, 1], keep_axes: true]]}
             } = sum_keep(Nx.tensor([[[1, 2, 3], [1, 2, 3]]], type: {:f, 32}))
    end
  end

  describe "creation ops" do
    defn iota(t), do: Nx.iota(t)
    defn eye, do: Nx.eye(2)
    defn random_uniform(t), do: Nx.random_uniform(t, 0.0, 2.0)
    defn random_normal(t), do: Nx.random_normal(t, 0.0, 1.0)

    test "iota" do
      assert %T{shape: {3}, data: %Expr{op: :iota, args: [nil]}} = iota(Nx.tensor([1, 2, 3]))
    end

    test "eye" do
      assert %T{shape: {2, 2}, data: %Expr{op: :eye, args: []}} = eye()
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

    test "outer product" do
      assert %T{data: %Expr{op: :outer, args: [_, _]}, shape: {3, 3}} =
               outer(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 3]))
    end

    test "transpose" do
      assert %T{data: %Expr{op: :transpose, args: [_, [1, 0]]}, shape: {3, 2}} =
               transpose_1(Nx.tensor([[1, 2, 3], [1, 2, 3]]))

      assert %T{data: %Expr{op: :transpose, args: [_, [1, 0]]}, shape: {3, 2}} =
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

  describe "squeeze" do
    defn squeeze(t), do: Nx.squeeze(t)

    test "sized one dimensions" do
      assert %T{data: %Expr{op: :squeeze, args: [_, [0, 2, 4]]}, shape: {3, 2}} =
               squeeze(Nx.iota({1, 3, 1, 2, 1}))
    end

    defn squeeze_collapse1(t), do: t |> Nx.squeeze(axes: [0, 2]) |> Nx.squeeze(axes: [0, 2])
    defn squeeze_collapse2(t), do: t |> Nx.squeeze(axes: [3, 1]) |> Nx.squeeze(axes: [2])
    defn squeeze_collapse3(t), do: t |> Nx.squeeze(axes: [2]) |> Nx.squeeze(axes: [3, 1])

    test "with explicit dimensions are collapsed" do
      assert %T{data: %Expr{op: :squeeze, args: [_, [0, 1, 2, 4]]}, shape: {1}, names: [:d]} =
               squeeze_collapse1(Nx.iota({1, 1, 1, 1, 1}, names: [:a, :b, :c, :d, :e]))

      assert %T{data: %Expr{op: :squeeze, args: [_, [1, 3, 4]]}, shape: {1, 1}, names: [:a, :c]} =
               squeeze_collapse2(Nx.iota({1, 1, 1, 1, 1}, names: [:a, :b, :c, :d, :e]))

      assert %T{data: %Expr{op: :squeeze, args: [_, [1, 2, 4]]}, shape: {1, 1}, names: [:a, :d]} =
               squeeze_collapse3(Nx.iota({1, 1, 1, 1, 1}, names: [:a, :b, :c, :d, :e]))
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
               data: %Expr{op: :reduce, args: [_, _, [axes: nil, keep_axes: false], fun]},
               type: {:s, 64},
               shape: {}
             } = reduce(Nx.tensor([1, 2, 3]), 0)

      assert %T{data: %Expr{op: :fun}} = fun

      assert %{
               data: %Expr{op: :reduce, args: [_, _, [axes: [1], keep_axes: false], fun]},
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
    end

    defn subtract_two(a, b), do: a - b

    test "-" do
      assert %T{data: %Expr{op: :subtract, args: [_, _]}} = subtract_two(1, 2)
    end

    defn multiply_two(a, b), do: a * b

    test "*" do
      assert %T{data: %Expr{op: :multiply, args: [_, _]}} = multiply_two(1, 2)
    end

    defn divide_two(a, b), do: a / b

    test "/" do
      assert %T{data: %Expr{op: :divide, args: [_, _]}} = divide_two(1, 2)
    end

    defn land_two(a, b), do: a and b

    test "and" do
      assert %T{data: %Expr{op: :logical_and, args: [_, _]}} = land_two(1, 2)
    end

    defn lor_two(a, b), do: a or b

    test "or" do
      assert %T{data: %Expr{op: :logical_or, args: [_, _]}} = lor_two(1, 2)
    end

    defn lnot(a), do: not a

    test "not" do
      assert %T{data: %Expr{op: :equal, args: [_, _]}} = lnot(1)
    end

    defn band_two(a, b), do: a &&& b

    test "&&&" do
      assert %T{data: %Expr{op: :bitwise_and, args: [_, _]}} = band_two(1, 2)
    end

    defn bor_two(a, b), do: a ||| b

    test "|||" do
      assert %T{data: %Expr{op: :bitwise_or, args: [_, _]}} = bor_two(1, 2)
    end

    defn bxor_two(a, b), do: a ^^^ b

    test "^^^" do
      assert %T{data: %Expr{op: :bitwise_xor, args: [_, _]}} = bxor_two(1, 2)
    end

    defn bsl_two(a, b), do: a <<< b

    test "<<<" do
      assert %T{data: %Expr{op: :left_shift, args: [_, _]}} = bsl_two(1, 2)
    end

    defn bsr_two(a, b), do: a >>> b

    test ">>>" do
      assert %T{data: %Expr{op: :right_shift, args: [_, _]}} = bsr_two(1, 2)
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
    end

    defn unary_bnot(a), do: ~~~a

    test "~~~" do
      assert %T{data: %Expr{op: :bitwise_not, args: [_]}} = unary_bnot(1)
    end
  end

  describe "kernel functions/macros" do
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

    defn maxu(a), do: rewrite_types(a, max_unsigned_type: {:u, 32})
    defn maxs(a), do: rewrite_types(a, max_signed_type: {:s, 32})
    defn maxf(a), do: rewrite_types(a, max_float_type: {:f, 32})

    test "max_*_type/2" do
      assert %T{data: %Expr{op: :as_type, args: [_]}} = maxu(Nx.tensor(1, type: {:u, 64}))
      assert %T{data: %Expr{op: :as_type, args: [_]}} = maxs(Nx.tensor(1, type: {:s, 64}))
      assert %T{data: %Expr{op: :as_type, args: [_]}} = maxf(Nx.tensor(1, type: {:f, 64}))
    end
  end

  describe "access" do
    defn single_access(t), do: {t[0], t[-1]}

    test "single dimensional single access" do
      {zero, minus_one} = single_access(Nx.tensor([1, 2, 3, 4, 5]))
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {}} = zero
      assert %T{data: %Expr{op: :slice, args: [_, [0], [1], [1]]}, shape: {1}} = slice

      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {}} = minus_one
      assert %T{data: %Expr{op: :slice, args: [_, [4], [1], [1]]}, shape: {1}} = slice
    end

    test "multi dimensional single access" do
      {zero, minus_one} = single_access(Nx.iota({3, 4, 5}))
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {4, 5}} = zero

      assert %T{
               data: %Expr{op: :slice, args: [_, [0, 0, 0], [1, 4, 5], [1, 1, 1]]},
               shape: {1, 4, 5}
             } = slice

      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {4, 5}} = minus_one

      assert %T{
               data: %Expr{op: :slice, args: [_, [2, 0, 0], [1, 4, 5], [1, 1, 1]]},
               shape: {1, 4, 5}
             } = slice
    end

    defn multi_access(t), do: t[1][2][3]

    test "multi dimensional multi-access with integers is collapsed" do
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0, 1, 2]]}, shape: {}} =
               multi_access(Nx.iota({3, 4, 5}))

      assert %T{
               data: %Expr{op: :slice, args: [_, [1, 2, 3], [1, 1, 1], [1, 1, 1]]},
               shape: {1, 1, 1}
             } = slice
    end

    defn range_access(t), do: t[1][1..2]

    test "multi dimensional multi-access with ranges is collapsed" do
      assert %T{data: %Expr{op: :squeeze, args: [slice, [0]]}, shape: {2, 5}} =
               range_access(Nx.iota({3, 4, 5}))

      assert %T{
               data: %Expr{op: :slice, args: [_, [1, 1, 0], [1, 2, 5], [1, 1, 1]]},
               shape: {1, 2, 5}
             } = slice
    end

    defn keyword_access(t), do: t[[z: 1..-2]][[y: 1..2]]

    test "multi dimensional multi-access with keywords is collapsed" do
      assert %T{
               data: %Expr{op: :slice, args: [_, [0, 1, 1], [3, 2, 3], [1, 1, 1]]},
               shape: {3, 2, 3}
             } = keyword_access(Nx.iota({3, 4, 5}, names: [:x, :y, :z]))
    end

    defn elixir_access(a, opts \\ []), do: Nx.sum(a, axes: opts[:axes])

    test "also works for other Elixir data structures" do
      assert %T{data: %Expr{op: :sum, args: [_, [axes: [1], keep_axes: false]]}} =
               elixir_access(Nx.iota({2, 2}), axes: [1])
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

      assert %T{data: %Expr{op: :cond}, shape: {}, type: {:f, 32}} =
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

  describe "qr" do
    defn qr(t), do: Nx.qr(t)

    test "returns tuples" do
      assert {left, right} = qr(Nx.iota({3, 2}))

      assert %T{data: %Expr{op: :elem, args: [qr_expr, 0, 2]}, shape: {3, 2}} = left
      assert %T{data: %Expr{op: :elem, args: [^qr_expr, 1, 2]}, shape: {2, 2}} = right
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

    defn cond_lit(a) do
      if Nx.any?(a), do: 1, else: -1
    end

    test "supports literals" do
      assert cond_lit(Nx.tensor(0)), do: Nx.tensor(-1)
      assert cond_lit(Nx.tensor(1)), do: Nx.tensor(1)
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

  describe "transform" do
    defn transform_inspect(a, b) do
      (Nx.tanh(a) + Nx.power(b, 2)) |> inspect_expr()
    end

    test "executes the transformation" do
      assert ExUnit.CaptureIO.capture_io(fn -> transform_inspect(1, 2) end) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a         s64
               parameter c         s64
               b = tanh [ a ]      f32
               d = power [ c, 2 ]  s64
               e = add [ b, d ]    f32
             >
             """
    end

    @defn_compiler Nx.Defn.Evaluator
    defn transform_back_and_forth(a) do
      Nx.exp(transform(Nx.negate(a), &private_back_and_forth/1))
    end

    defp private_back_and_forth(a) do
      Nx.Defn.Evaluator = Process.get(Nx.Defn.Compiler)
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
      true = Process.get(Nx.Defn.Compiler) in [Nx.Defn.Evaluator, Identity]
      a |> Nx.add(b) |> Nx.subtract(c)
    end

    test "compiles defn function" do
      assert Nx.Defn.jit(&defn_jit/2, [{4, 5}, 3]) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit/2, [{4, 5}, Nx.tensor(3)]) == Nx.tensor(6)
      assert Nx.Defn.jit(&defn_jit(&1, 3), [{4, 5}]) == Nx.tensor(6)

      assert %T{data: %Expr{op: :subtract}} = Nx.Defn.jit(&defn_jit/2, [{1, 2}, 3], Identity)
    end

    test "compiles elixir function" do
      assert Nx.Defn.jit(&elixir_jit/2, [{4, 5}, 3]) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit/2, [{4, 5}, Nx.tensor(3)]) == Nx.tensor(6)
      assert Nx.Defn.jit(&elixir_jit(&1, 3), [{4, 5}]) == Nx.tensor(6)

      assert %T{data: %Expr{op: :subtract}} = Nx.Defn.jit(&elixir_jit/2, [{4, 5}, 3], Identity)
    end

    test "raises if it doesn't return an expression" do
      assert_raise ArgumentError,
                   "defn must return a tensor expression or a tuple, got: :ok",
                   fn ->
                     Nx.Defn.jit(fn -> :ok end, [], Nx.Defn.Evaluator).()
                   end
    end

    defn jit_iota(), do: Nx.iota({3, 3})

    @tag :capture_log
    test "uses the default backend on iota" do
      Nx.default_backend(UnknownBackend)
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(&jit_iota/0, []) end
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(fn -> Nx.iota({3, 3}) end, []) end
    end

    defn jit_tensor(), do: Nx.tensor([1, 2, 3])

    @tag :capture_log
    test "uses the default backend on tensor" do
      Nx.default_backend(UnknownBackend)
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(&jit_tensor/0, []) end
      assert_raise UndefinedFunctionError, fn -> Nx.Defn.jit(fn -> Nx.tensor(13) end, []) end
    end
  end

  describe "async" do
    defn defn_async({a, b}, c), do: a + b - c

    def elixir_async({a, b}, c) do
      true = Process.get(Nx.Defn.Compiler) in [Nx.Defn.Evaluator, Identity]
      a |> Nx.add(b) |> Nx.subtract(c)
    end

    test "runs defn function async" do
      assert %_{} = async = Nx.Defn.async(&defn_async/2, [{4, 5}, 3])
      assert Nx.Async.await!(async) == Nx.tensor(6)

      assert %_{} = async = Nx.Defn.async(&defn_async/2, [{4, 5}, Nx.tensor(3)])
      assert Nx.Async.await!(async) == Nx.tensor(6)
      assert %_{} = async = Nx.Defn.async(&defn_async(&1, 3), [{4, 5}])
      assert Nx.Async.await!(async) == Nx.tensor(6)
    end

    test "runs elixir function async" do
      assert %_{} = async = Nx.Defn.async(&elixir_async/2, [{4, 5}, 3])
      assert Nx.Async.await!(async) == Nx.tensor(6)

      assert %_{} = async = Nx.Defn.async(&elixir_async/2, [{4, 5}, Nx.tensor(3)])
      assert Nx.Async.await!(async) == Nx.tensor(6)

      assert %_{} = async = Nx.Defn.async(&elixir_async(&1, 3), [{4, 5}])
      assert Nx.Async.await!(async) == Nx.tensor(6)
    end

    @tag :capture_log
    test "raises on errors" do
      Process.flag(:trap_exit, true)
      assert %_{} = async = Nx.Defn.async(fn -> :ok end, [])

      ref = Process.monitor(async.pid)
      assert_receive {:DOWN, ^ref, _, _, _}

      assert catch_exit(Nx.Async.await!(async)) == {:noproc, {Nx.Async, :await!, [async]}}
      assert_receive {:EXIT, _, _}
    end

    test "raises if already awaited" do
      assert %_{} = async = Nx.Defn.async(&defn_async/2, [{4, 5}, 3])
      assert Nx.Async.await!(async) == Nx.tensor(6)
      assert catch_exit(Nx.Async.await!(async)) == {:noproc, {Nx.Async, :await!, [async]}}
    end

    defn async_iota(), do: Nx.iota({3, 3})

    @tag :capture_log
    test "uses the default backend on iota" do
      Process.flag(:trap_exit, true)
      Nx.default_backend(UnknownBackend)
      assert %_{} = Nx.Defn.async(&async_iota/0, [])
      assert_receive {:EXIT, _, {:undef, _}}
    end

    defn async_tensor(), do: Nx.tensor([1, 2, 3])

    @tag :capture_log
    test "uses the default backend on tensor" do
      Process.flag(:trap_exit, true)
      Nx.default_backend(UnknownBackend)
      assert %_{} = Nx.Defn.async(&async_tensor/0, [])
      assert_receive {:EXIT, _, {:undef, _}}
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

    test "non-variables used as arguments" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: only variables and tuples are allowed as arguments in defn",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add(1, 2), do: 3
                     end
                   end
    end

    test "non-variables matching as arguments" do
      assert_raise CompileError,
                   ~r"#{location(+4)}: using = in arguments expects at least one of the sides to be a variable, got: {arg, arg} = {arg, arg}",
                   fn ->
                     defmodule Sample do
                       import Nx.Defn
                       defn add({_, _} = {_, _}, x), do: x
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

  @default_defn_compiler Nx.Defn.Evaluator

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

    test "are supported" do
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

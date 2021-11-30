defmodule Nx.Defn.Kernel do
  @moduledoc """
  All imported functionality available inside `defn` blocks.
  """

  @special_forms [alias: 1, alias: 2, import: 1, import: 2, require: 1, require: 2, cond: 1]

  @doc false
  defmacro __using__(_opts) do
    quote do
      import Kernel, only: []
      import Nx.Defn.Kernel, except: unquote(Kernel.@(special_forms))
      alias Nx.Defn.Kernel, as: Kernel
    end
  end

  @doc """
  Defines an alias, as in `Kernel.SpecialForms.alias/2`.

  An alias allows you to refer to a module using its aliased
  name. For example:

      defn some_fun(t) do
        alias Math.Helpers, as: MH
        MH.fft(t)
      end

  If the `:as` option is not given, the alias defaults to
  the last part of the given alias. For example,

      alias Math.Helpers

  is equivalent to:

      alias Math.Helpers, as: Helpers

  Finally, note that aliases define outside of a function also
  apply to the function, as they have lexical scope:

      alias Math.Helpers, as: MH

      defn some_fun(t) do
        MH.fft(t)
      end

  """
  defmacro alias(module, opts \\ []), do: special_form!([module, opts])

  @doc """
  Imports functions and macros into the current scope,
  as in `Kernel.SpecialForms.import/2`.

  Imports are typically discouraged in favor of `alias/2`.

  ## Examples

      defn some_fun(t) do
        import Math.Helpers
        fft(t)
      end

  """
  defmacro import(module, opts \\ []), do: special_form!([module, opts])

  @doc """
  Requires a module in order to use its macros, as in `Kernel.SpecialForms.require/2`.

  ## Examples

      defn some_fun(t) do
        require NumericalMacros

        NumericalMacros.some_macro t do
          ...
        end
      end

  """
  defmacro require(module, opts \\ []), do: special_form!([module, opts])

  @doc """
  Evaluates the expression corresponding to the first
  clause that evaluates to a truthy value.

  It has the format of:

      cond do
        condition1 ->
          expr1

        condition2 ->
          expr2

        :otherwise ->
          expr3
      end

  The conditions must be a scalar. Zero is considered false,
  any other number is considered true.

  All clauses are normalized to the same type and are broadcast
  to the same shape. The last condition must always evaluate to
  an atom, typically `:otherwise`.

  ## Examples

      cond do
        Nx.all?(Nx.greater(a, 0)) -> b *
        Nx.all?(Nx.less(a, 0)) -> b + c
        true -> b - c
      end

  """
  defmacro cond(opts), do: special_form!([opts])

  defp special_form!(_args),
    do: raise("special forms must not be imported and exist for documentation purposes")

  @doc """
  Defines a transform that executes the given `fun` with `arg`
  when building `defn` expressions.

  ## Example

  Take the following defn expression:

      defn tanh_power(a, b) do
        Nx.tanh(a) + Nx.power(b, 2)
      end

  Let's see a trivial example, which is to use `IO.inspect/1` to
  print a tensor expression at definition time:

      defn tanh_power(a, b) do
        Nx.tanh(a) + Nx.power(b, 2) |> transform(&IO.inspect/1)
      end

  Or:

      defn tanh_power(a, b) do
        res = Nx.tanh(a) + Nx.power(b, 2)
        transform(res, &IO.inspect/1)
        res
      end

  When invoked in both cases, it will print the expression being built
  by `defn`:

      #Nx.Defn.Expr<
        parameter a
        parameter c
        b = tanh [ a ] ()
        d = power [ c, 2 ] ()
        e = add [ b, d ] ()
      >

  Although, for convenience, you might use `inspect_expr/2` instead.

  ## Pitfalls

  Because `transform/2` is invoked inside `defn`, its scope is tied
  to `defn`. For example, if you do this:

      transform(tensor, fn tensor ->
        if Nx.type(tensor) != {:f, 32} do
          raise "bad"
        end
      end)

  it won't work because it will use the `!=` operator defined in
  this module, which only works with tensors, instead of the operator
  defined in Elixir's `Kernel`. Therefore, we recommend all `transform/2`
  calls to simply dispatch to a separate function. The example above
  could be rewritten as:

      transform(tensor, &assert_2x2_shape(&1))

  where:

      defp assert_2x2_shape(tensor) do
        if Nx.shape(tensor) != {2, 2} do
          raise "bad"
        end
      end

  """
  def transform(arg, fun) when is_function(fun, 1) do
    fun.(arg)
  end

  @doc """
  Inspects the given expression to the terminal.

  It returns the given expressions.

  ### Examples

      defn tanh_grad(t) do
        grad(t, &Nx.tanh/1) |> inspect_expr()
      end

  When invoked, it will print the expression being built by `defn`:

      #Nx.Tensor<
        Nx.Defn.Expr
        parameter a s64
        parameter c s64
        b = tanh [ a ] f64
        d = power [ c, 2 ] s64
        e = add [ b, d ] f64
      >

  """
  def inspect_expr(expr, opts \\ []) do
    IO.inspect(expr, opts)
  end

  @doc """
  Inspects the value at runtime to the terminal.

  This function is implemented on top of `hook/3` and therefore
  has the following restrictions:

    * It can only inspect tensors and `Nx.Container`
    * The return value of this function must be part of the output

  All options are passed to `IO.inspect/2`.

  ## Examples

      defn tanh_grad(t) do
        grad(t, fn t ->
          t
          |> Nx.tanh()
          |> inspect_value()
        end)
      end

      defn tanh_grad(t) do
        grad(t, fn t ->
          t
          |> Nx.tanh()
          |> inspect_value(label: "tanh")
        end)
      end

  """
  def inspect_value(expr, opts \\ []) do
    hook(expr, &IO.inspect(&1, opts))
  end

  @doc """
  Rewrites the types of `expr` recursively according to `opts`

  ## Options

    * `:max_unsigned_type` - replaces all signed tensors with size
      equal to or greater then the given type by the given type

    * `:max_signed_type` - replaces all signed tensors with size
      equal to or greater then the given type by the given type

    * `:max_float_type` - replaces all float tensors with size
      equal to or greater then the given type by the given type

  ## Examples

      rewrite_types(expr, max_float_type: {:f, 32})

  """
  def rewrite_types(expr, opts) do
    Nx.Defn.Tree.rewrite_types(expr, opts)
  end

  @doc """
  Stops computing the gradient for the given expression.

  It effectively annotates the gradient for the given
  expression is 1.0.

  ## Examples

      expr = stop_grad(expr)

  """
  def stop_grad(expr) do
    Nx.Defn.Expr.metadata(expr, %{stop_grad: true, inspect: :stop_grad})
  end

  @doc """
  Defines a custom gradient for the given expression.

  It expects a `fun` to compute the gradient. The function
  will be called with the expression itself and the current
  gradient. It must return a list of arguments and their
  updated gradient to continue applying `grad` on.

  ## Examples

  For example, if the gradient of `cos(t)` were to be
  implemented by hand:

      def cos(t) do
        custom_grad(Nx.cos(t), fn _ans, g ->
          [{t, -g * Nx.sin(t)}]
        end)
      end

  """
  def custom_grad(expr, fun) when is_function(fun, 2) do
    Nx.Defn.Expr.metadata(expr, %{custom_grad: fun, inspect: :custom_grad})
  end

  @doc """
  Element-wise unary plus operator.

  Simply returns the given argument.

  ## Examples

      defn plus_and_minus(a) do
        {+a, -a}
      end

  """
  def +tensor, do: tensor

  @doc """
  Element-wise unary plus operator.

  It delegates to `Nx.negate/1`.

  ## Examples

      defn plus_and_minus(a) do
        {+a, -a}
      end

  """
  def -tensor when is_number(tensor), do: Kernel.-(tensor)
  def -tensor, do: Nx.negate(tensor)

  @doc """
  Builds a range.

  Ranges are inclusive and both sides must be integers.

  The step of the range is computed based on the first
  and last values of the range.

  ## Examples

      iex> t = Nx.tensor([1, 2, 3])
      iex> t[1..2]
      #Nx.Tensor<
        s64[2]
        [2, 3]
      >

  """
  def first..last, do: Range.new(first, last)

  @doc """
  Builds a range with step.

  Ranges are inclusive and both sides must be integers.

  ## Examples

      iex> t = Nx.tensor([1, 2, 3])
      iex> t[1..2//1]
      #Nx.Tensor<
        s64[2]
        [2, 3]
      >

  """
  def first..last//step, do: Range.new(first, last, step)

  @doc """
  Element-wise addition operator.

  It delegates to `Nx.add/2` (supports broadcasting).

  ## Examples

      defn add(a, b) do
        a + b
      end

  """
  def left + right when Kernel.and(is_number(left), is_number(right)), do: Kernel.+(left, right)
  def left + right, do: Nx.add(left, right)

  @doc """
  Element-wise substraction operator.

  It delegates to `Nx.subtract/2` (supports broadcasting).

  ## Examples

      defn subtract(a, b) do
        a - b
      end

  """
  def left - right when Kernel.and(is_number(left), is_number(right)), do: Kernel.-(left, right)
  def left - right, do: Nx.subtract(left, right)

  @doc """
  Element-wise multiplication operator.

  It delegates to `Nx.multiply/2` (supports broadcasting).

  ## Examples

      defn multiply(a, b) do
        a * b
      end

  """
  def left * right when Kernel.and(is_number(left), is_number(right)), do: Kernel.*(left, right)
  def left * right, do: Nx.multiply(left, right)

  @doc """
  Element-wise division operator.

  It delegates to `Nx.divide/2` (supports broadcasting).

  ## Examples

      defn divide(a, b) do
        a / b
      end

  """
  def left / right when Kernel.and(is_number(left), is_number(right)), do: Kernel./(left, right)
  def left / right, do: Nx.divide(left, right)

  @doc """
  Element-wise remainder operation.

  It delegates to `Nx.remainder/2` (supports broadcasting).

  ## Examples

      defn divides_by_5?(a) do
        rem(a, 5)
        |> Nx.any?
        |> Nx.equal(Nx.tensor(1))
      end

  """
  def rem(left, right) when Kernel.and(is_number(left), is_number(right)),
    do: Kernel.rem(left, right)

  def rem(left, right), do: Nx.remainder(left, right)

  @doc """
  Element-wise maximum operation.

  It delegates to `Nx.max/2` (supports broadcasting).

  ## Examples

      defn min_max(a, b) do
        {min(a, b), max(a, b)}
      end

  """
  def max(left, right) when Kernel.and(is_number(left), is_number(right)),
    do: Kernel.max(left, right)

  def max(left, right), do: Nx.max(left, right)

  @doc """
  Element-wise minimum operation.

  It delegates to `Nx.min/2` (supports broadcasting).

  ## Examples

      defn min_max(a, b) do
        {min(a, b), max(a, b)}
      end

  """
  def min(left, right) when Kernel.and(is_number(left), is_number(right)),
    do: Kernel.min(left, right)

  def min(left, right), do: Nx.min(left, right)

  @doc """
  Element-wise logical AND operation.

  Zero is considered false, all other numbers
  are considered true.

  It delegates to `Nx.logical_and/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a and b, a or b}
      end

  """
  def left and right when Kernel.or(is_boolean(left), is_boolean(right)) do
    raise ArgumentError,
          "boolean value passed to Nx.Defn.Kernel.and/2, " <>
            "values passed to Nx.Defn.Kernel.and/2 must be " <>
            "tensors or numbers, consider using 1 for true " <>
            "and 0 for false as an alternative"
  end

  def left and right when Kernel.and(is_number(left), is_number(right)),
    do: logical_and(left, right)

  def left and right, do: Nx.logical_and(left, right)

  @doc """
  Element-wise logical OR operation.

  Zero is considered false, all other numbers
  are considered true.

  It delegates to `Nx.logical_or/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a and b, a or b}
      end

  """
  def left or right when Kernel.or(is_boolean(left), is_boolean(right)) do
    raise ArgumentError,
          "boolean value passed to Nx.Defn.Kernel.or/2, " <>
            "values passed to Nx.Defn.Kernel.or/2 must be " <>
            "tensors or numbers, consider using 1 for true " <>
            "and 0 for false as an alternative"
  end

  def left or right when Kernel.and(is_number(left), is_number(right)),
    do: logical_or(left, right)

  def left or right, do: Nx.logical_or(left, right)

  @doc """
  Element-wise logical NOT operation.

  Zero is considered false, all other numbers
  are considered true.

  It delegates to `Nx.logical_not/1`.

  ## Examples

      defn logical_not(a), do: not a

  """
  def not tensor when is_boolean(tensor) do
    raise ArgumentError,
          "boolean value passed to Nx.Defn.Kernel.not/1, " <>
            "values passed to Nx.Defn.Kernel.not/1 must be " <>
            "tensors or numbers, consider using 1 for true " <>
            "and 0 for false as an alternative"
  end

  def not tensor when is_number(tensor), do: logical_not(tensor)
  def not tensor, do: Nx.logical_not(tensor)

  defp logical_and(l, _) when Kernel.==(l, 0), do: zero()
  defp logical_and(_, r) when Kernel.==(r, 0), do: zero()
  defp logical_and(_, _), do: one()

  defp logical_or(l, r) when Kernel.and(Kernel.==(l, 0), Kernel.==(r, 0)), do: zero()
  defp logical_or(_, _), do: one()

  defp logical_not(0), do: one()
  defp logical_not(0.0), do: one()
  defp logical_not(_), do: zero()

  defp zero(), do: Nx.tensor(0, type: {:u, 8})
  defp one(), do: Nx.tensor(1, type: {:u, 8})

  @doc """
  Element-wise bitwise AND operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_and/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a &&& b, a ||| b}
      end

  """
  def left &&& right when Kernel.and(is_number(left), is_number(right)),
    do: Bitwise.&&&(left, right)

  def left &&& right, do: Nx.bitwise_and(left, right)

  @doc """
  Element-wise bitwise OR operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_or/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a &&& b, a ||| b}
      end

  """
  def left ||| right when Kernel.and(is_number(left), is_number(right)),
    do: Bitwise.|||(left, right)

  def left ||| right, do: Nx.bitwise_or(left, right)

  @doc """
  Element-wise bitwise not operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_not/1`.

  ## Examples

      defn bnot(a), do: ~~~a

  """
  def ~~~tensor when is_number(tensor), do: Bitwise.~~~(tensor)
  def ~~~tensor, do: Nx.bitwise_not(tensor)

  @doc """
  Element-wise left shift operation.

  Only integer tensors are supported.
  It delegates to `Nx.left_shift/2` (supports broadcasting).

  ## Examples

      defn shift_left_and_right(a, b) do
        {a <<< b, a >>> b}
      end

  """
  def left <<< right when Kernel.and(is_number(left), is_number(right)),
    do: Bitwise.<<<(left, right)

  def left <<< right, do: Nx.left_shift(left, right)

  @doc """
  Element-wise right shift operation.

  Only integer tensors are supported.
  It delegates to `Nx.right_shift/2` (supports broadcasting).

  ## Examples

      defn shift_left_and_right(a, b) do
        {a <<< b, a >>> b}
      end

  """
  def left >>> right when Kernel.and(is_number(left), is_number(right)),
    do: Bitwise.>>>(left, right)

  def left >>> right, do: Nx.right_shift(left, right)

  @doc """
  Element-wise equality operation.

  ## Examples

      defn check_equality(a, b) do
        a == b
      end

  """
  def left == right when Kernel.and(is_number(left), is_number(right)),
    do: to_constant(Kernel.==(left, right))

  def left == right, do: Nx.equal(left, right)

  @doc """
  Element-wise inequality operation.

  ## Examples

      defn check_inequality(a, b) do
        a != b
      end
  """
  def left != right when Kernel.and(is_number(left), is_number(right)),
    do: to_constant(Kernel.!=(left, right))

  def left != right, do: Nx.not_equal(left, right)

  @doc """
  Element-wise less than operation.

  ## Examples

      defn check_less_than(a, b) do
        a < b
      end
  """
  def left < right when Kernel.and(is_number(left), is_number(right)),
    do: to_constant(Kernel.<(left, right))

  def left < right, do: Nx.less(left, right)

  @doc """
  Element-wise greater than operation.

  ## Examples

      defn check_greater_than(a, b) do
        a > b
      end
  """
  def left > right when Kernel.and(is_number(left), is_number(right)),
    do: to_constant(Kernel.>(left, right))

  def left > right, do: Nx.greater(left, right)

  @doc """
  Element-wise less-equal operation.

  ## Examples

      defn check_less_equal(a, b) do
        a <= b
      end
  """
  def left <= right when Kernel.and(is_number(left), is_number(right)),
    do: to_constant(Kernel.<=(left, right))

  def left <= right, do: Nx.less_equal(left, right)

  @doc """
  Element-wise greater-equal operation.

  ## Examples

      defn check_greater_equal(a, b) do
        a >= b
      end
  """
  def left >= right when Kernel.and(is_number(left), is_number(right)),
    do: to_constant(Kernel.>=(left, right))

  def left >= right, do: Nx.greater_equal(left, right)

  defp to_constant(true), do: one()
  defp to_constant(false), do: zero()

  @doc """
  Ensures the first argument is a `keyword` with the given
  keys and default values.

  The second argument must be a list of atoms, specifying
  a given key, or tuples specifying a key and a default value.
  If any of the keys in the `keyword` is not defined on
  `values`, it raises an error.

  ## Examples

      iex> keyword!([], [one: 1, two: 2]) |> Enum.sort()
      [one: 1, two: 2]

      iex> keyword!([two: 3], [one: 1, two: 2]) |> Enum.sort()
      [one: 1, two: 3]

  If atoms are given, they are supported as keys but do not
  provide a default value:

      iex> keyword!([], [:one, two: 2]) |> Enum.sort()
      [two: 2]

      iex> keyword!([one: 1], [:one, two: 2]) |> Enum.sort()
      [one: 1, two: 2]

  Passing an unknown key raises:

      iex> keyword!([three: 3], [one: 1, two: 2])
      ** (ArgumentError) unknown key :three in [three: 3], expected one of [:one, :two]

  """
  def keyword!(keyword, values) when Kernel.and(is_list(keyword), is_list(values)) do
    # We use two lists to avoid reversing/concatenating
    # lists in the middle of traversals.
    case keyword!(keyword, values, [], []) do
      {:ok, keyword} ->
        keyword

      error ->
        keys =
          for value <- values,
              do: Kernel.if(is_atom(value), do: value, else: Kernel.elem(value, 0))

        case error do
          {:badkey, key} ->
            raise ArgumentError,
                  "unknown key #{inspect(key)} in #{inspect(keyword)}, " <>
                    "expected one of #{inspect(keys)}"

          :badkey ->
            raise ArgumentError,
                  "expected a keyword list with keys #{inspect(keys)}, got: #{inspect(keyword)}"
        end
    end
  end

  defp keyword!([{key, _} = pair | keyword], values1, values2, acc) when is_atom(key) do
    case find_key!(key, values1, values2) do
      {values1, values2} ->
        keyword!(keyword, values1, values2, [pair | acc])

      :error ->
        case find_key!(key, values2, values1) do
          {values1, values2} ->
            keyword!(keyword, values1, values2, [pair | acc])

          :error ->
            {:badkey, key}
        end
    end
  end

  defp keyword!([], values1, values2, acc) do
    {:ok, move_pairs!(values1, move_pairs!(values2, acc))}
  end

  defp keyword!(_keyword, _values1, _values2, _acc) do
    :badkey
  end

  defp find_key!(key, [key | rest], acc), do: {rest, acc}
  defp find_key!(key, [{key, _} | rest], acc), do: {rest, acc}
  defp find_key!(key, [head | tail], acc), do: find_key!(key, tail, [head | acc])
  defp find_key!(_key, [], _acc), do: :error

  defp move_pairs!([key | rest], acc) when is_atom(key),
    do: move_pairs!(rest, acc)

  defp move_pairs!([{key, _} = pair | rest], acc) when is_atom(key),
    do: move_pairs!(rest, [pair | acc])

  defp move_pairs!([], acc),
    do: acc

  defp move_pairs!([other | _], _) do
    raise ArgumentError,
          "keyword!/2 expects the second argument to be a list of atoms or tuples, " <>
            "got: #{inspect(other)}"
  end

  @doc """
  Pipes the argument on the left to the function call on the right.

  It delegates to `Kernel.|>/2`.

  ## Examples

      defn exp_sum(t) do
        t
        |> Nx.exp()
        |> Nx.sum()
      end

  """
  defmacro left |> right do
    quote do: Kernel.|>(unquote(left), unquote(right))
  end

  @doc """
  Provides if/else expressions.

  The first argument must be a scalar. Zero is considered false,
  any other number is considered true.

  The second argument is a keyword list with `do` and `else`
  blocks. The sides are broadcast to return the same shape
  and normalized to return the same type.

  ## Examples

      if Nx.any?(Nx.equal(t, 0)) do
        0.0
      else
        1 / t
      end

  In case else is not given, it is assumed to be 0 with the
  same as the do clause. If you want to nest multiple conditionals,
  see `cond/1` instead.
  """
  defmacro if(pred, do_else)

  defmacro if(pred, do: on_true) do
    quote do
      cond do
        unquote(pred) -> unquote(on_true)
        :otherwise -> 0
      end
    end
  end

  defmacro if(pred, do: on_true, else: on_false) do
    quote do
      cond do
        unquote(pred) -> unquote(on_true)
        :otherwise -> unquote(on_false)
      end
    end
  end

  defmacro if(_pred, other) do
    raise ArgumentError,
          "expected second argument to \"if\" to be a do/else block, " <>
            "got: #{Macro.to_string(other)}"
  end

  @doc """
  Defines a `while` loop.

  It expects the `initial` arguments, a `condition` expression, and
  a `block`:

      while initial, condition do
        block
      end

  `condition` must return a scalar tensor where 0 is false and any
  other number is true. The given `block` will be executed while
  `condition` is true. Each invocation of `block` must return a
  value in the same shape as `initial` arguments.

  `while` will return the value of the last execution of `block`.
  If `block` is never executed because the initial `condition` is
  false, it returns `initial`.

  ## Examples

  A simple loop that increments `x` until it is `10` can be written as:

        while x = 0, Nx.less(x, 10) do
          x + 1
        end

  Similarly, to compute the factorial of `x` using `while`:

        defn factorial(x) do
          {factorial, _} =
            while {factorial = 1, x}, Nx.greater(x, 1) do
              {factorial * x, x - 1}
            end

          factorial
        end

  Note `while/3` does not behave as a closure. Therefore, all
  variables used inside the `while` must be explicitly given
  as an `initial` value to `while`.
  """
  defmacro while(initial, condition, do: block) do
    {pattern, {vars, values}} = while_arg(initial, {[], []})

    quote do
      {unquote_splicing(vars)} = {unquote_splicing(values)}

      Nx.Defn.Kernel.__while__(
        __ENV__.file,
        __ENV__.line,
        unquote(pattern),
        fn unquote(pattern) -> unquote(condition) end,
        fn unquote(pattern) -> unquote(block) end
      )
    end
  end

  defmacro while(_var, _cond, other) do
    raise ArgumentError,
          "expected third argument to \"while\" to be a do-block, " <>
            "got: #{Macro.to_string(other)}"
  end

  @doc false
  defdelegate __while__(file, line, pattern, condition, block), to: Nx.Defn.Expr, as: :defn_while

  defp while_arg({left, right}, prelude) do
    {left, prelude} = while_arg(left, prelude)
    {right, prelude} = while_arg(right, prelude)
    {{left, right}, prelude}
  end

  defp while_arg({:{}, meta, args}, prelude) do
    {args, prelude} = Enum.map_reduce(args, prelude, &while_arg/2)
    {{:{}, meta, args}, prelude}
  end

  defp while_arg({:=, _meta, [{name, meta, ctx} = var, value]}, {vars, values})
       when Kernel.and(is_atom(name), is_atom(ctx)) do
    {{name, [generated: true] ++ meta, ctx}, {[var | vars], [value | values]}}
  end

  defp while_arg({name, meta, ctx}, prelude)
       when Kernel.and(is_atom(name), is_atom(ctx)) do
    {{name, [generated: true] ++ meta, ctx}, prelude}
  end

  defp while_arg(other, _prelude) do
    raise ArgumentError, """
    invalid initial argument for \"while\". Expected a variable, a variable assignment, \
    or a tuple of the same. For example:

        while x = 0, Nx.less(x, 10) do
          x + 1
        end

    Or when using tuples:

        x = 0

        {x, y} =
          while {x, y = 10}, Nx.not_equal(x, y) do
            {x + 1, y - 1}
          end

    Got: #{Macro.to_string(other)}
    """
  end

  @doc """
  Pipes `value` to the given `fun` and returns the `value` itself.

  Useful for running synchronous side effects in a pipeline.

  ## Examples

  Let's suppose you want to inspect an expression in the middle of
  a pipeline. You could write:

      a
      |> Nx.add(b)
      |> tap(&inspect_expr/1)
      |> Nx.multiply(c)

  """
  defmacro tap(value, fun) do
    quote bind_quoted: [fun: fun, value: value] do
      _ = fun.(value)
      value
    end
  end

  @doc """
  Pipes `value` into the given `fun`.

  In other words, it invokes `fun` with `value` as argument.
  This is most commonly used in pipelines, allowing you
  to pipe a value to a function outside of its first argument.

  ### Examples

      a
      |> Nx.add(b)
      |> then(&Nx.subtract(c, &1))

  """
  defmacro then(value, fun) do
    quote do
      unquote(fun).(unquote(value))
    end
  end

  @doc """
  Gets the element at the zero-based index in tuple.

  It raises ArgumentError when index is negative or it
  is out of range of the tuple elements.

  ## Examples

      iex> tuple = {1, 2, 3}
      iex> elem(tuple, 0)
      1

  """
  def elem(tuple, index), do: :erlang.element(Kernel.+(index, 1), tuple)

  @doc """
  Reads a module attribute at compilation time.

  It is useful to inject code constants into `defn`.
  It delegates to `Kernel.@/1`.

  ## Examples

      @two_per_two Nx.tensor([[1, 2], [3, 4]])
      defn add_2x2_attribute(t), do: t + @two_per_two

  """
  defmacro @expr do
    quote do: Kernel.@(unquote(expr))
  end

  @doc """
  Shortcut for `hook/3`.
  """
  def hook(expr, name_or_function)

  def hook(expr, name) when is_atom(name),
    do: unguarded_hook(expr, name, nil)

  def hook(expr, function) when is_function(function, 1),
    do: unguarded_hook(expr, random_hook_name(), function)

  @doc """
  Defines a hook.

  Hooks are a mechanism to execute an anonymous function for
  side-effects with runtime tensor values.

  Let's see an example:

      defmodule Hooks do
        import Nx.Defn

        defn add_and_mult(a, b) do
          add = hook(a + b, fn tensor -> IO.inspect({:add, tensor}) end)
          mult = hook(a * b, fn tensor -> IO.inspect({:mult, tensor}) end)
          {add, mult}
        end
      end

  The `defn` above defines two hooks, one is called with the
  value of `a + b` and another with `a * b`. Once you invoke
  the function above, you should see this printed:

      Hooks.add_and_mult(2, 3)
      {:add, #Nx.Tensor<
         s64
         5
      >}
      {:mult, #Nx.Tensor<
         s64
         6
      >}

  In other words, the `hook` function accepts a tensor
  expression as argument and it will invoke a custom
  function with a tensor value at runtime. `hook` returns
  the result of the given expression. The expression can
  be any tensor or a `Nx.Container`.

  Note **you must return the result of the `hook` call**.
  For example, the code below won't inspect the `:add`
  tuple, because the hook is not returned from `defn`:

      defn add_and_mult(a, b) do
        _add = hook(a + b, fn tensor -> IO.inspect({:add, tensor}) end)
        mult = hook(a * b, fn tensor -> IO.inspect({:mult, tensor}) end)
        mult
      end

  We will learn how to hook into a value that is not part
  of the result in the "Hooks and tokens" section.

  ## Named hooks

  It is possible to give names to the hooks. This allows them
  to be defined or overridden by calling `Nx.Defn.jit/3` or
  `Nx.Defn.stream/3`. Let's see an example:

      defmodule Hooks do
        import Nx.Defn

        defn add_and_mult(a, b) do
          add = hook(a + b, :hooks_add)
          mult = hook(a * b, :hooks_mult)
          {add, mult}
        end
      end

  Now you can pass the hook as argument as follows:
      
      hooks = %{
        hooks_add: fn tensor ->
          IO.inspect {:add, tensor}
        end
      }

      args = [Nx.tensor(2), Nx.tensor(3)]
      Nx.Defn.jit(&Hooks.add_and_mult/2, args, hooks: hooks)

  > **Important!** We recommend to prefix your hook names
  > by the name of your project to avoid conflicts.

  If a named hook is not given, compilers can optimize
  that away and not transfer the tensor from the device
  in the first place.

  You can also mix named hooks with callbacks:

      defn add_and_mult(a, b) do
        add = hook(a + b, :hooks_add, fn tensor -> IO.inspect({:add, tensor}) end)
        mult = hook(a * b, :hooks_mult, fn tensor -> IO.inspect({:mult, tensor}) end)
        {add, mult}
      end

  If a hook with the same name is given to `Nx.Defn.jit/3`
  or `Nx.Defn.stream/3`, then it will override the default
  callback.

  ## Hooks and tokens

  So far, we have always returned the result of the `hook`
  call. However, what happens if the values we want to
  hook are not part of the return value, such as below?

      defn add_and_mult(a, b) do
        _add = hook(a + b, :hooks_add, &IO.inspect({:add, &1}))
        mult = hook(a * b, :hooks_mult, &IO.inspect({:mult, &1}))
        mult
      end

  In such cases, you must use tokens. Tokens are used to
  create an ordering over hooks, ensuring hooks execute
  in a certain sequence:

      defn add_and_mult(a, b) do
        token = create_token()
        {token, _add} = hook_token(token, a + b, :hooks_add, &IO.inspect({:add, &1}))
        {token, mult} = hook_token(token, a * b, :hooks_mult, &IO.inspect({:mult, &1}))
        attach_token(token, mult)
      end    

  The example above creates a token and uses `hook_token/4`
  to create hooks attached to their respective tokens. By using a token,
  we guarantee that those hooks will be invoked in the order
  in which they were defined. Then, at the end of the function, 
  we attach the token (and its associated hooks) to the result `mult`.

  In fact, the `hook/3` function is implemented roughly like this:

      def hook(tensor_expr, name, function) do
        {token, result} = hook_token(create_token(), tensor_expr, name, function)
        attach_token(token, result)
      end

  Note you must attach the token at the end, otherwise the hooks
  will be "lost", as if they were not defined. This also applies
  to conditionals and loops. The token must be attached within
  the branch they are used. For example, this won't work:

      token = create_token()

      {token, result} =
        if Nx.any?(value) do
          hook_token(token, some_value)
        else
          hook_token(token, another_value)
        end

      attach_token(result)

  Instead, you must write:

      token = create_token()

      if Nx.any?(value) do
        {token, result} = hook_token(token, some_value)
        attach_token(token, result)
      else
        {token, result} = hook_token(token, another_value)
        attach_token(token, result)
      end

  """
  def hook(expr, name, function) when Kernel.and(is_atom(name), is_function(function, 1)),
    do: unguarded_hook(expr, name, function)

  defp unguarded_hook(expr, name, function) do
    {token, result} = Nx.Defn.Expr.add_hook(create_token(), expr, name, function)
    attach_token(token, result)
  end

  @doc """
  Shortcut for `hook_token/4`.
  """
  def hook_token(token, expr, name_or_function)

  def hook_token(%Nx.Defn.Token{} = token, expr, name) when is_atom(name),
    do: Nx.Defn.Expr.add_hook(token, expr, name, nil)

  def hook_token(%Nx.Defn.Token{} = token, expr, function) when is_function(function, 1),
    do: Nx.Defn.Expr.add_hook(token, expr, random_hook_name(), function)

  @doc """
  Defines a hook with an existing token. See `hook/3`.
  """
  def hook_token(%Nx.Defn.Token{} = token, expr, name, function)
      when Kernel.and(is_atom(name), is_function(function, 1)),
      do: Nx.Defn.Expr.add_hook(token, expr, name, function)

  defp random_hook_name(), do: :"hook_#{System.unique_integer([:positive])}"

  @doc """
  Creates a token for hooks. See `hook/3`.
  """
  def create_token do
    Nx.Defn.Token.new()
  end

  @doc """
  Attaches a token to an expression. See `hook/3`.
  """
  def attach_token(%Nx.Defn.Token{} = token, expr) do
    Nx.Defn.Expr.attach_token(token, expr)
  end

  @doc """
  Asserts the `tensor` has a certain `shape`.

  If it succeeds, it returns the given tensor. Raises
  an error otherwise.

  ## Examples

  To assert the tensor is a scalar, you can pass the empty tuple `shape`:

      iex> assert_shape Nx.tensor(13), {}
      #Nx.Tensor<
        s64
        13
      >

  If the shapes do not match, an error is raised:

      iex> assert_shape Nx.tensor([1, 2, 3]), {}
      ** (ArgumentError) expected tensor to be a scalar, got tensor with shape {3}

      iex> assert_shape Nx.tensor([1, 2, 3]), {4}
      ** (ArgumentError) expected tensor to have shape {4}, got tensor with shape {3}

  If you want to assert on the rank or shape patterns, use
  `assert_shape_pattern/2` instead.
  """
  def assert_shape(tensor, shape) when is_tuple(shape) do
    case Nx.shape(tensor) do
      ^shape ->
        tensor

      other ->
        raise ArgumentError,
              "expected tensor to #{shape_to_string(shape)}, got tensor with shape #{inspect(other)}"
    end
  end

  defp shape_to_string({}), do: "be a scalar"
  defp shape_to_string(shape), do: "have shape " <> inspect(shape)

  @doc """
  Asserts the `tensor` has a certain `shape` pattern.

  If it succeeds, it returns the given tensor. Raises
  an error otherwise.

  ## Examples

  Opposite to `assert_shape/2`, where the given shape is a value,
  `assert_shape_pattern` allows the shape to be any Elixir pattern.
  We can use this to match on ranks:

      iex> assert_shape_pattern Nx.tensor([[1, 2], [3, 4]]), {_, _}
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >

      iex> assert_shape_pattern Nx.tensor([1, 2, 3]), {_, _}
      ** (ArgumentError) expected tensor to match shape {_, _}, got tensor with shape {3}

  Or even use variables to assert on properties such as square matrices:

      iex> assert_shape_pattern Nx.tensor([[1, 2], [3, 4]]), {x, x}
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >

      iex> assert_shape_pattern Nx.tensor([1, 2, 3]), {x, x}
      ** (ArgumentError) expected tensor to match shape {x, x}, got tensor with shape {3}

  You can also use guards to specify tall matrices and so forth:

      iex> assert_shape_pattern Nx.tensor([[1], [2]]), {x, y} when x > y
      #Nx.Tensor<
        s64[2][1]
        [
          [1],
          [2]
        ]
      >

      iex> assert_shape_pattern Nx.tensor([1, 2]), {x, y} when x > y
      ** (ArgumentError) expected tensor to match shape {x, y} when x > y, got tensor with shape {2}

  """
  defmacro assert_shape_pattern(tensor, shape) do
    shape_pattern_string = shape_pattern_to_string(shape)

    quote do
      Nx.Defn.Kernel.transform(unquote(tensor), fn tensor ->
        # Revert scoping so guards work
        import unquote(__MODULE__), only: []
        import Kernel

        case Nx.shape(tensor) do
          unquote(shape) ->
            tensor

          shape ->
            unquote(__MODULE__).__assert_shape_pattern__!(unquote(shape_pattern_string), shape)
        end
      end)
    end
  end

  @doc false
  def __assert_shape_pattern__!(shape_pattern_string, shape) do
    raise ArgumentError,
          "expected tensor to #{shape_pattern_string}, got tensor with shape #{inspect(shape)}"
  end

  defp shape_pattern_to_string({:{}, _, []}), do: "be a scalar"
  defp shape_pattern_to_string(pattern), do: "match shape " <> Macro.to_string(pattern)
end

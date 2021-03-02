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

  Let's see a trivial example, which is `inspect_expr/1`. `inspect_expr/1`
  can be used to debug the current expression during compilation.
  It is implemented by using `transform/2` to invoke `IO.inspect/1` at
  definition time:

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

  """
  def transform(arg, fun) when is_function(fun, 1) do
    fun.(arg)
  end

  @doc """
  Inspects the given expression to the terminal.

  It returns the given expressions.

  ### Examples

      defn tanh_grad(t) do
        grad(t, Nx.tanh(t)) |> inspect_expr()
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
  defmacro inspect_expr(expr) do
    quote do
      Nx.Defn.Kernel.transform(
        unquote(expr),
        &IO.inspect/1
      )
    end
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
  defmacro rewrite_types(expr, opts) do
    quote do
      Nx.Defn.Kernel.transform(
        unquote(expr),
        &Nx.Defn.Tree.rewrite_types(&1, unquote(opts))
      )
    end
  end

  @doc """
  Computes the gradient of the given `var` on `expr`.

  ### Examples

      defn tanh_grad(t) do
        grad(t, Nx.tanh(t))
      end

  To differentiate on multiple vars, pass a tuple as first argument:

      defn tanh_power_grad(a, b) do
        grad({a, b}, Nx.tanh(a) + Nx.power(b, 2))
      end

  When a tuple is given, a tuple will be returned.

  Note you can also pass an already built expression to grad. For
  example, if you want to return the result of an expression and its
  gradient, you can do:

      defn tanh_power_grad(a, b) do
        expr = Nx.tanh(a) + Nx.power(b, 2)
        {expr, grad({a, b}, expr)}
      end

  """
  def grad(var_or_vars, expr) do
    Nx.Defn.Grad.transform(var_or_vars, expr)
  end

  @doc """
  Stops computing the gradient for the given expression.

  It effectively annotates the gradient for the given
  expression is 1.0.

  ## Examples

      expr = stop_grad(expr)

  """
  def stop_grad(expr) do
    Nx.Defn.Expr.metadata(expr, %{stop_grad: true})
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
    Nx.Defn.Expr.metadata(expr, %{custom_grad: fun})
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
  def not tensor when is_number(tensor), do: logical_not(tensor)
  def not tensor, do: Nx.logical_not(tensor)

  defp logical_and(l, r) when l == 0, do: zero(l, r)
  defp logical_and(l, r) when r == 0, do: zero(l, r)
  defp logical_and(l, r), do: one(l, r)

  defp logical_or(l, r) when Kernel.and(l == 0, r == 0), do: zero(l, r)
  defp logical_or(l, r), do: one(l, r)

  defp logical_not(0), do: 1
  defp logical_not(0.0), do: 1.0
  defp logical_not(n) when is_float(n), do: 0.0
  defp logical_not(n) when is_integer(n), do: 0

  defp zero(l, r) when Kernel.or(is_float(l), is_float(r)), do: 0.0
  defp zero(_, _), do: 0

  defp one(l, r) when Kernel.or(is_float(l), is_float(r)), do: 1.0
  defp one(_, _), do: 1

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
  Element-wise bitwise XOR operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_xor/2` (supports broadcasting).

  ## Examples

      defn and_or_xor(a, b) do
        {a &&& b, a ||| b, a ^^^ b}
      end

  """
  def left ^^^ right when Kernel.and(is_number(left), is_number(right)),
    do: Bitwise.^^^(left, right)

  def left ^^^ right, do: Nx.bitwise_xor(left, right)

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
      ** (ArgumentError) unknown key :three in [three: 3], expected to be one of [one: 1, two: 2]

  """
  def keyword!(keyword, values) when Kernel.and(is_list(keyword), is_list(values)) do
    # We use two lists to avoid reversing/concatenating
    # lists in the middle of traversals.
    case keyword!(keyword, values, [], []) do
      {:ok, keyword} ->
        keyword

      {:badkey, key} ->
        raise ArgumentError,
              "unknown key #{inspect(key)} in #{inspect(keyword)}, " <>
                "expected to be one of #{inspect(values)}"

      :badkey ->
        raise ArgumentError,
              "keyword!/2 expects the first argument to be a list, got: #{inspect(keyword)}"
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
            "got: #{inspect(Macro.to_string(other))}"
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
end

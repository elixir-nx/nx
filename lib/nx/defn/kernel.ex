defmodule Nx.Defn.Kernel do
  @moduledoc """
  All imported functionality available inside `defn` blocks.
  """

  @special_forms [alias: 1, alias: 2, import: 1, import: 2, require: 1, require: 2]

  @doc """
  Bring in `Nx.Defn.Kernel` functionality.

  Most times, you won't have to `use Nx.Defn.Kernel` directly,
  but, if you have to, it will remove Kernel functions and
  import all macros in this module, except the special forms
  which only exist for documentation purposes.
  """
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

  Let's see a trivial example, which is `print_expr/1`. `print_expr/1`
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
  Prints the given expression to the terminal.

  It returns the given expressions.

  ### Examples

      defn tanh_grad(t) do
        grad(t, Nx.tanh(t)) |> print_expr()
      end

  When invoked, it will print the expression being built by `defn`:

      #Nx.Defn.Expr<
        parameter a
        parameter c
        b = tanh [ a ] ()
        d = power [ c, 2 ] ()
        e = add [ b, d ] ()
      >

  """
  defmacro print_expr(expr) do
    quote do
      Nx.Defn.Kernel.transform(
        unquote(expr),
        &IO.inspect/1
      )
    end
  end

  @doc """
  Computes the gradient of the given `var` on `expr`.

  ### Examples

      defn tanh_grad(t) do
        grad(t, Nx.tanh(t))
      end

  To differenciate on multiple vars, pass a tuple as first argument:

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
  defmacro grad(var_or_vars, expr) do
    var_or_vars =
      case var_or_vars do
        {:{}, meta, vars} -> {:{}, meta, Enum.map(vars, &grad_var!/1)}
        {left, right} -> {grad_var!(left), grad_var!(right)}
        var -> grad_var!(var)
      end

    quote do
      Nx.Defn.Kernel.transform(
        {unquote(var_or_vars), unquote(expr)},
        &Nx.Defn.Grad.transform/1
      )
    end
  end

  defp grad_var!({name, _, ctx} = var) when is_atom(name) and is_atom(ctx), do: var

  defp grad_var!(expr) do
    raise ArgumentError,
          "first argument of grad/3 must be a variable or a tuple of variables, got: " <>
            Macro.to_string(expr)
  end

  @doc """
  Element-wise unary plus operator.

  Simply returns the given argument.

  ## Examples

      defn plus_and_minus(a) do
        {+a, -a}
      end

  """
  defmacro +tensor do
    tensor
  end

  @doc """
  Element-wise unary plus operator.

  It delegates to `Nx.negate/2`.

  ## Examples

      defn plus_and_minus(a) do
        {+a, -a}
      end

  """
  defmacro -tensor do
    if is_number(tensor) do
      Kernel.-(tensor)
    else
      quote do: Nx.negate(unquote(tensor))
    end
  end

  @doc """
  Element-wise addition operator.

  It delegates to `Nx.add/2` (supports broadcasting).

  ## Examples

      defn add(a, b) do
        a + b
      end

  """
  # TODO: Implement all functions using this style once we merge Nx and Nx.Defn.Expr.
  def left + right when is_number(left) and is_number(right), do: Kernel.+(left, right)
  def left + right, do: Nx.Defn.Expr.add(left, right)

  @doc """
  Element-wise substraction operator.

  It delegates to `Nx.subtract/2` (supports broadcasting).

  ## Examples

      defn subtract(a, b) do
        a - b
      end

  """
  defmacro left - right do
    quote do: Nx.subtract(unquote(left), unquote(right))
  end

  @doc """
  Element-wise multiplication operator.

  It delegates to `Nx.multiply/2` (supports broadcasting).

  ## Examples

      defn subtract(a, b) do
        a * b
      end

  """
  defmacro left * right do
    quote do: Nx.multiply(unquote(left), unquote(right))
  end

  @doc """
  Element-wise division operator.

  It delegates to `Nx.divide/2` (supports broadcasting).

  ## Examples

      defn divide(a, b) do
        a / b
      end

  """
  defmacro left / right do
    quote do: Nx.divide(unquote(left), unquote(right))
  end

  @doc """
  Element-wise maximum operation.

  It delegates to `Nx.max/2` (supports broadcasting).

  ## Examples

      defn min_max(a, b) do
        {min(a, b), max(a, b)}
      end

  """
  defmacro max(left, right) do
    quote do: Nx.max(unquote(left), unquote(right))
  end

  @doc """
  Element-wise minimum operation.

  It delegates to `Nx.min/2` (supports broadcasting).

  ## Examples

      defn min_max(a, b) do
        {min(a, b), max(a, b)}
      end

  """
  defmacro min(left, right) do
    quote do: Nx.min(unquote(left), unquote(right))
  end

  @doc """
  Element-wise bitwise AND operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_and/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a &&& b, a ||| b}
      end

  """
  defmacro left &&& right do
    quote do: Nx.bitwise_and(unquote(left), unquote(right))
  end

  @doc """
  Element-wise bitwise OR operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_or/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a &&& b, a ||| b}
      end

  """
  defmacro left ||| right do
    quote do: Nx.bitwise_or(unquote(left), unquote(right))
  end

  @doc """
  Element-wise bitwise XOR operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_xor/2` (supports broadcasting).

  ## Examples

      defn and_or_xor(a, b) do
        {a &&& b, a ||| b, a ^^^ b}
      end

  """
  defmacro left ^^^ right do
    quote do: Nx.bitwise_xor(unquote(left), unquote(right))
  end

  @doc """
  Element-wise bitwise not operation.

  Only integer tensors are supported.
  It delegates to `Nx.bitwise_not/1`.

  ## Examples

      defn bnot(a), do: ~~~a

  """
  defmacro ~~~tensor do
    quote do: Nx.bitwise_not(unquote(tensor))
  end

  @doc """
  Element-wise left shift operation.

  Only integer tensors are supported.
  It delegates to `Nx.left_shift/2` (supports broadcasting).

  ## Examples

      defn shift_left_and_right(a, b) do
        {a <<< b, a >>> b}
      end

  """
  defmacro left <<< right do
    quote do: Nx.left_shift(unquote(left), unquote(right))
  end

  @doc """
  Element-wise right shift operation.

  Only integer tensors are supported.
  It delegates to `Nx.right_shift/2` (supports broadcasting).

  ## Examples

      defn shift_left_and_right(a, b) do
        {a <<< b, a >>> b}
      end

  """
  defmacro left >>> right do
    quote do: Nx.right_shift(unquote(left), unquote(right))
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

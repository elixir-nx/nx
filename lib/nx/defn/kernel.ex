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
  Element-wise addition operator.

  It delegates to `Nx.add/2` (supports broadcasting).

  ## Examples

      defn add(a, b) do
        a + b
      end

  """
  defmacro left + right do
    quote do: Nx.add(unquote(left), unquote(right))
  end

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
  Element-wise logical `and` operation.

  Only integer tensors are supported. Zero is false,
  all other numbers are true.
  
  It delegates to `Nx.logical_and/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a and b, a or b}
      end

  """
  defmacro left and right do
    quote do: Nx.logical_and(unquote(left), unquote(right))
  end

  @doc """
  Element-wise logical `or` operation.

  Only integer tensors are supported. Zero is false,
  all other numbers are true.
  
  It delegates to `Nx.logical_or/2` (supports broadcasting).

  ## Examples

      defn and_or(a, b) do
        {a and b, a or b}
      end

  """
  defmacro left or right do
    quote do: Nx.logical_or(unquote(left), unquote(right))
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

defmodule Nx.Shared do
  # A collection of **private** helpers and macros shared in Nx.
  @moduledoc false

  alias Nx.Tensor, as: T

  ## Macros

  @doc """
  Match the cartesian product of all given types.

  A macro that allows us to writes all possibles match types
  in the most efficient format. This is done by looking at @0,
  @1, etc and replacing them by currently matched type at the
  given position. In other words, this:

     match_types [input_type, output_type] do
       for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(read!(seg, 0) + right, 1)>>
     end

  Is compiled into:

     for <<seg::float-native-size(...) <- data>>, into: <<>>, do: <<seg+right::float-native-size(...)>>

  for all possible valid types between input and input types.

  `match!` is used in matches and must be always followed by a `read!`.
  `write!` is used to write to the binary.

  The implementation unfolds the loops at the top level. In particular,
  note that a rolled out case such as:

      for <<seg::size(size)-signed-integer <- data>>, into: <<>> do
        <<seg+number::signed-integer-size(size)>>
      end

  is twice as fast and uses twice less memory than:

      for <<seg::size(size)-signed-integer <- data>>, into: <<>> do
        case output_type do
          {:s, size} ->
            <<seg+number::signed-integer-size(size)>>
          {:f, size} ->
            <<seg+number::float-native-size(size)>>
          {:u, size} ->
            <<seg+number::unsigned-integer-size(size)>>
        end
      end
  """
  defmacro match_types([_ | _] = args, do: block) do
    sizes = Macro.generate_arguments(length(args), __MODULE__)
    matches = match_types(sizes)

    clauses =
      Enum.flat_map(matches, fn match ->
        block =
          Macro.prewalk(block, fn
            {:match!, _, [var, pos]} when is_integer(pos) ->
              {type, size} = Enum.fetch!(match, pos)
              match_bin_modifier(var, type, size)

            {:read!, _, [var, pos]} when is_integer(pos) ->
              {type, size} = Enum.fetch!(match, pos)
              read_bin_modifier(var, type, size)

            {:write!, _, [var, pos]} when is_integer(pos) ->
              {type, size} = Enum.fetch!(match, pos)
              write_bin_modifier(var, type, size)

            other ->
              other
          end)

        quote do
          {unquote_splicing(match)} -> unquote(block)
        end
      end)

    quote do
      case {unquote_splicing(args)}, do: unquote(clauses)
    end
  end

  @all_types [:s, :f, :bf, :u]

  defp match_types([h | t]) do
    for type <- @all_types, t <- match_types(t) do
      [{type, h} | t]
    end
  end

  defp match_types([]), do: [[]]

  defp match_bin_modifier(var, :bf, _),
    do: quote(do: unquote(var) :: binary - size(2))

  defp match_bin_modifier(var, type, size),
    do: shared_bin_modifier(var, type, size)

  defp read_bin_modifier(var, :bf, _) do
    if System.endianness() == :little do
      quote do
        <<x::float-little-32>> = <<0::16, unquote(var)::binary>>
        x
      end
    else
      quote do
        <<x::float-big-32>> = <<unquote(var)::binary, 0::16>>
        x
      end
    end
  end

  defp read_bin_modifier(var, _, _),
    do: var

  defp write_bin_modifier(var, :bf, _) do
    if System.endianness() == :little do
      quote(do: binary_part(<<unquote(var)::float-native-32>>, 2, 2) :: binary)
    else
      quote(do: binary_part(<<unquote(var)::float-native-32>>, 0, 2) :: binary)
    end
  end

  defp write_bin_modifier(var, type, size),
    do: shared_bin_modifier(var, type, size)

  defp shared_bin_modifier(var, :s, size),
    do: quote(do: unquote(var) :: signed - integer - native - size(unquote(size)))

  defp shared_bin_modifier(var, :u, size),
    do: quote(do: unquote(var) :: unsigned - integer - native - size(unquote(size)))

  defp shared_bin_modifier(var, :f, size),
    do: quote(do: unquote(var) :: float - native - size(unquote(size)))

  ## Reflection

  @doc """
  Returns the definition of mathemtical unary funs.
  """
  def unary_math_funs,
    do: [
      exp: {"exponential", quote(do: :math.exp(var!(x)))},
      expm1: {"exponential minus one", quote(do: :math.exp(var!(x)) - 1)},
      log: {"natural log", quote(do: :math.log(var!(x)))},
      log1p: {"natural log plus one", quote(do: :math.log(var!(x) + 1))},
      logistic: {"standard logistic (a sigmoid)", quote(do: 1 / (1 + :math.exp(-var!(x))))},
      cos: {"cosine", quote(do: :math.cos(var!(x)))},
      sin: {"sine", quote(do: :math.sin(var!(x)))},
      tan: {"tangent", quote(do: :math.tan(var!(x)))},
      cosh: {"hyperbolic cosine", quote(do: :math.cosh(var!(x)))},
      sinh: {"hyperbolic sine", quote(do: :math.sinh(var!(x)))},
      tanh: {"hyperbolic tangent", quote(do: :math.tanh(var!(x)))},
      arccos: {"inverse cosine", quote(do: :math.acos(var!(x)))},
      arcsin: {"inverse sine", quote(do: :math.asin(var!(x)))},
      arctan: {"inverse tangent", quote(do: :math.atan(var!(x)))},
      arccosh: {"inverse hyperbolic cosine", acosh_formula()},
      arcsinh: {"inverse hyperbolic sine", asinh_formula()},
      arctanh: {"inverse hyperbolic tangent", atanh_formula()},
      sqrt: {"square root", quote(do: :math.sqrt(var!(x)))},
      rsqrt: {"reverse square root", quote(do: 1 / :math.sqrt(var!(x)))},
      cbrt: {"cube root", quote(do: :math.pow(var!(x), 1 / 3))},
      erf: {"error function", erf_formula()},
      erfc: {"one minus error function", erfc_formula()}
    ]

  defp atanh_formula do
    if Code.ensure_loaded?(:math) and math_func_supported?(:atanh, 1) do
      quote(do: :math.atanh(var!(x)))
    else
      quote(do: :math.log((1 + var!(x)) / (1 - var!(x))) / 2)
    end
  end

  defp asinh_formula do
    if Code.ensure_loaded?(:math) and math_func_supported?(:asinh, 1) do
      quote(do: :math.asinh(var!(x)))
    else
      quote(do: :math.log(var!(x) + :math.sqrt(1 + var!(x) * var!(x))))
    end
  end

  defp acosh_formula do
    if Code.ensure_loaded?(:math) and math_func_supported?(:acosh, 1) do
      quote(do: :math.acosh(var!(x)))
    else
      quote(do: :math.log(var!(x) + :math.sqrt(var!(x) + 1) * :math.sqrt(var!(x) - 1)))
    end
  end

  defp erf_formula do
    if Code.ensure_loaded?(:math) and math_func_supported?(:erf, 1) do
      quote(do: :math.erf(var!(x)))
    else
      quote(do: Nx.Shared.erf_fallback(var!(x)))
    end
  end

  defp erfc_formula do
    if Code.ensure_loaded?(:math) and math_func_supported?(:erfc, 1) do
      quote(do: :math.erfc(var!(x)))
    else
      quote(do: 1.0 - Nx.Shared.erf_fallback(var!(x)))
    end
  end

  @doc false
  def erf_fallback(x) do
    # https://introcs.cs.princeton.edu/java/21function/ErrorFunction.java.html
    # the Chebyshev fitting estimate below is accurate to 7 significant digits

    t = 1.0 / (1.0 + 0.5 * abs(x))

    a =
      0.17087277
      |> muladd(t, -0.82215223)
      |> muladd(t, 1.48851587)
      |> muladd(t, -1.13520398)
      |> muladd(t, 0.27886807)
      |> muladd(t, -0.18628806)
      |> muladd(t, 0.09678418)
      |> muladd(t, 0.37409196)
      |> muladd(t, 1.00002368)
      |> muladd(t, 1.26551223)

    ans = 1 - t * :math.exp(-x * x - a)
    if x >= 0.0, do: ans, else: -ans
  end

  defp muladd(acc, t, n) do
    acc * t + n
  end

  @doc false
  def math_func_supported?(func, arity) do
    args =
      case {func, arity} do
        {:atan, 1} -> [3.14]
        {:atanh, 1} -> [0.9]
        {_, 1} -> [1.0]
        {_, 2} -> [1.0, 1.0]
      end

    _ = apply(:math, func, args)
    true
  rescue
    UndefinedFunctionError ->
      false
  end

  ## Types

  @doc """
  Builds the type of an element-wise binary operation.
  """
  def binary_type(a, b) when is_number(a) and is_number(b), do: Nx.Type.infer(a + b)
  def binary_type(a, b) when is_number(a), do: Nx.Type.merge_scalar(type(b), a)
  def binary_type(a, b) when is_number(b), do: Nx.Type.merge_scalar(type(a), b)
  def binary_type(a, b), do: Nx.Type.merge(type(a), type(b))

  defp type(%T{type: type}), do: type
  defp type(type), do: type

  ## Helpers

  @doc """
  Gets the implementation of a tensor.
  """
  def impl!(%T{data: %struct{}}), do: struct

  def impl!(%T{data: %struct1{}}, %T{data: %struct2{}}),
    do: pick_struct(struct1, struct2)

  def impl!(%T{data: %struct1{}}, %T{data: %struct2{}}, %T{data: %struct3{}}),
    do: struct1 |> pick_struct(struct2) |> pick_struct(struct3)

  @doc """
  Gets the implementation of a list of maybe tensors.
  """
  def find_impl!(list) do
    Enum.reduce(list, Nx.BinaryBackend, fn
      %T{data: %struct{}}, acc -> pick_struct(struct, acc)
      _, acc -> acc
    end)
  end

  defp pick_struct(Nx.BinaryBackend, struct), do: struct
  defp pick_struct(struct, Nx.BinaryBackend), do: struct
  defp pick_struct(struct, struct), do: struct

  defp pick_struct(struct1, struct2) do
    raise "cannot invoke Nx function because it relies on two incompatible tensor implementations: " <>
            "#{inspect(struct1)} and #{inspect(struct2)}"
  end
end

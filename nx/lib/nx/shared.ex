defmodule Nx.Shared do
  # A collection of **private** helpers and macros shared in Nx.
  @moduledoc false

  alias Nx.Tensor, as: T

  ## Macros

  @doc """
  Match the cartesian product of all given types.

  A macro that allows us to writes all possibles match types
  in the most efficient format. This is done by looking at @0,
  @1, etc., and replacing them by currently matched type at the
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

  @doc """
  Converts an Erlang float (float64) to float32 precision.
  """
  def to_float32(float64) when is_float(float64) do
    <<float32::float-32>> = <<float64::float-32>>
    float32
  end

  ## Reflection

  @doc """
  Returns the definition of mathematical unary funs.
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
      acos: {"inverse cosine", quote(do: :math.acos(var!(x)))},
      asin: {"inverse sine", quote(do: :math.asin(var!(x)))},
      atan: {"inverse tangent", quote(do: :math.atan(var!(x)))},
      acosh: {"inverse hyperbolic cosine", acosh_formula()},
      asinh: {"inverse hyperbolic sine", asinh_formula()},
      atanh: {"inverse hyperbolic tangent", atanh_formula()},
      sqrt: {"square root", quote(do: :math.sqrt(var!(x)))},
      rsqrt: {"reverse square root", quote(do: 1 / :math.sqrt(var!(x)))},
      cbrt: {"cube root", quote(do: :math.pow(var!(x), 1 / 3))},
      erf: {"error function", erf_formula()},
      erfc: {"one minus error function", erfc_formula()},
      erf_inv: {"inverse error function", quote(do: Nx.Shared.erf_inv(var!(x)))}
    ]

  defp atanh_formula do
    if Code.ensure_loaded?(:math) and math_fun_supported?(:atanh, 1) do
      quote(do: :math.atanh(var!(x)))
    else
      quote(do: :math.log((1 + var!(x)) / (1 - var!(x))) / 2)
    end
  end

  defp asinh_formula do
    if Code.ensure_loaded?(:math) and math_fun_supported?(:asinh, 1) do
      quote(do: :math.asinh(var!(x)))
    else
      quote(do: :math.log(var!(x) + :math.sqrt(1 + var!(x) * var!(x))))
    end
  end

  defp acosh_formula do
    if Code.ensure_loaded?(:math) and math_fun_supported?(:acosh, 1) do
      quote(do: :math.acosh(var!(x)))
    else
      quote(do: :math.log(var!(x) + :math.sqrt(var!(x) + 1) * :math.sqrt(var!(x) - 1)))
    end
  end

  defp erf_formula do
    if Code.ensure_loaded?(:math) and math_fun_supported?(:erf, 1) do
      quote(do: :math.erf(var!(x)))
    else
      quote(do: Nx.Shared.erf(var!(x)))
    end
  end

  defp erfc_formula do
    if Code.ensure_loaded?(:math) and math_fun_supported?(:erfc, 1) do
      quote(do: :math.erfc(var!(x)))
    else
      quote(do: 1.0 - Nx.Shared.erf(var!(x)))
    end
  end

  @doc """
  Checks if a given function is supported in the `:math` module.
  """
  def math_fun_supported?(fun, arity) do
    args =
      case {fun, arity} do
        {:atan, 1} -> [3.14]
        {:atanh, 1} -> [0.9]
        {_, 1} -> [1.0]
        {_, 2} -> [1.0, 1.0]
      end

    _ = apply(:math, fun, args)
    true
  rescue
    UndefinedFunctionError ->
      false
  end

  @doc """
  Approximation for the error function.

  ## Examples

      iex> Nx.Shared.erf(0.999)
      0.8422852791811658

      iex> Nx.Shared.erf(0.01)
      0.011283414826762329

  """
  def erf(x) do
    x = x |> max(-4.0) |> min(4.0)
    x2 = x * x

    alpha =
      0.0
      |> muladd(x2, -2.72614225801306e-10)
      |> muladd(x2, 2.77068142495902e-08)
      |> muladd(x2, -2.10102402082508e-06)
      |> muladd(x2, -5.69250639462346e-05)
      |> muladd(x2, -7.34990630326855e-04)
      |> muladd(x2, -2.95459980854025e-03)
      |> muladd(x2, -1.60960333262415e-02)

    beta =
      0.0
      |> muladd(x2, -1.45660718464996e-05)
      |> muladd(x2, -2.13374055278905e-04)
      |> muladd(x2, -1.68282697438203e-03)
      |> muladd(x2, -7.37332916720468e-03)
      |> muladd(x2, -1.42647390514189e-02)

    min(x * alpha / beta, 1.0)
  end

  defp muladd(acc, t, n) do
    acc * t + n
  end

  @doc """
  Approximation for the inverse error function.

  ## Examples

      iex> Nx.Shared.erf_inv(0.999)
      2.326753756865462

      iex> Nx.Shared.erf_inv(0.01)
      0.008862500728738846

  """
  def erf_inv(x) do
    w = -:math.log((1 - x) * (1 + x))
    erf_inv_p(w) * x
  end

  defp erf_inv_p(w) when w < 5 do
    w = w - 2.5

    2.81022636e-08
    |> muladd(w, 3.43273939e-07)
    |> muladd(w, -3.5233877e-06)
    |> muladd(w, -4.39150654e-06)
    |> muladd(w, 0.00021858087)
    |> muladd(w, -0.00125372503)
    |> muladd(w, -0.00417768164)
    |> muladd(w, 0.246640727)
    |> muladd(w, 1.50140941)
  end

  defp erf_inv_p(w) do
    w = :math.sqrt(w) - 3

    -0.000200214257
    |> muladd(w, 0.000100950558)
    |> muladd(w, 0.00134934322)
    |> muladd(w, -0.00367342844)
    |> muladd(w, 0.00573950773)
    |> muladd(w, -0.0076224613)
    |> muladd(w, 0.00943887047)
    |> muladd(w, 1.00167406)
    |> muladd(w, 2.83297682)
  end

  ## Types

  @doc """
  Builds the type of an element-wise binary operation.
  """
  def binary_type(a, b) when is_number(a) and is_number(b), do: Nx.Type.infer(a + b)
  def binary_type(a, b) when is_number(a), do: Nx.Type.merge_scalar(type(b), a)
  def binary_type(a, b) when is_number(b), do: Nx.Type.merge_scalar(type(a), b)
  def binary_type(a, b), do: Nx.Type.merge(type(a), type(b))

  # For unknown types, return {:f, 32} as the caller
  # should validate the input in a later pass.
  defp type(%T{type: type}), do: type
  defp type({_, _} = type), do: type
  defp type(_other), do: {:f, 32}

  ## Helpers

  @doc """
  Asserts on the given keys.
  """
  def assert_keys!(keyword, valid) do
    for kv <- keyword do
      case kv do
        {k, _} ->
          if k not in valid do
            raise ArgumentError,
                  "unknown key #{inspect(k)} in #{inspect(keyword)}, " <>
                    "expected one of #{inspect(valid)}"
          end

        _ ->
          raise ArgumentError,
                "expected a keyword list with keys #{inspect(valid)}, got: #{inspect(keyword)}"
      end
    end
  end

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
            "#{inspect(struct1)} and #{inspect(struct2)}. You may need to call Nx.backend_transfer/1 " <>
            "(or Nx.backend_copy/1) on one or both of them to transfer them to a common implementation"
  end
end

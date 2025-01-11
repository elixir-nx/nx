defmodule Nx.Shared do
  # A collection of **private** helpers and macros shared in Nx.
  @moduledoc false

  alias Nx.Tensor, as: T

  ## Type macros

  defmacro generated_case(expr, do: clauses) do
    clauses =
      Enum.map(clauses, fn {:->, meta, args} -> {:->, [generated: true] ++ meta, args} end)

    {:case, [generated: true], [expr, [do: clauses]]}
  end

  @doc """
  Match the cartesian product of all given types.

  A macro that allows us to write all possible match types
  in the most efficient format. This is done by looking at @0,
  @1, etc., and replacing them with the currently matched type at the
  given position. In other words, this:

     match_types [input_type, output_type] do
       for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(read!(seg, 0) + right, 1)>>
     end

  Is compiled into:

     for <<seg::float-native-size(...) <- data>>, into: <<>>, do: <<seg+right::float-native-size(...)>>

  for all possible valid types between input and output types.

  `match!` is used in matches and must always be followed by a `read!`.
  `write!` is used to write to the binary.

  The implementation unfolds the loops at the top level. In particular,
  note that a rolled out case such as:

      for <<seg::size(size)-signed-integer <- data>>, into: <<>> do
        <<seg+number::signed-integer-size(size)>>
      end

  is twice as fast and uses half the memory compared to:

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

  @all_types [:s, :f, :bf, :u, :c]

  defp match_types([h | t]) do
    for type <- @all_types, t <- match_types(t) do
      [{type, h} | t]
    end
  end

  defp match_types([]), do: [[]]

  defp match_bin_modifier(var, type, size) when type in [:f, :bf, :c],
    do: quote(do: unquote(var) :: bitstring - size(unquote(size)))

  defp match_bin_modifier(var, type, size),
    do: shared_bin_modifier(var, type, size)

  defp read_bin_modifier(var, :c, size) do
    quote do: Nx.Shared.read_complex(unquote(var), unquote(size))
  end

  defp read_bin_modifier(var, :bf, _) do
    quote do: Nx.Shared.read_bf16(unquote(var))
  end

  defp read_bin_modifier(var, :f, 8) do
    quote do: Nx.Shared.read_f8(unquote(var))
  end

  defp read_bin_modifier(var, :f, size) do
    quote do
      case unquote(var) do
        _ when unquote(size) == 8 -> Nx.Shared.read_f8(unquote(var))
        <<var::float-native-size(unquote(size))>> -> var
        var -> Nx.Shared.read_non_finite(var, unquote(size))
      end
    end
  end

  defp read_bin_modifier(var, _, _),
    do: var

  defp write_bin_modifier(var, :bf, _) do
    if System.endianness() == :little do
      quote do
        case unquote(var) do
          x when is_number(x) -> binary_part(<<x::float-native-32>>, 2, 2)
          x -> Nx.Shared.write_non_finite_bf16(x)
        end :: binary
      end
    else
      quote do
        case unquote(var) do
          x when is_number(x) -> binary_part(<<x::float-native-32>>, 0, 2)
          x -> Nx.Shared.write_non_finite_bf16(x)
        end :: binary
      end
    end
  end

  defp write_bin_modifier(var, :c, size) do
    quote do
      case unquote(var) do
        x when is_number(x) ->
          elem_size = div(unquote(size), 2)
          <<x::float-native-size(elem_size), 0::float-native-size(elem_size)>>

        %Complex{re: re, im: im} ->
          Nx.Shared.write_complex(re, im, div(unquote(size), 2))

        x ->
          elem_size = div(unquote(size), 2)
          Nx.Shared.write_non_finite(x, elem_size) <> <<0::float-native-size(elem_size)>>
      end :: binary
    end
  end

  defp write_bin_modifier(var, :f, size) do
    quote do
      case unquote(var) do
        x when is_number(x) and unquote(size) != 8 -> <<x::float-native-size(unquote(size))>>
        x when is_number(x) -> Nx.Shared.write_finite_f8(unquote(var))
        x -> Nx.Shared.write_non_finite(x, unquote(size))
      end :: binary
    end
  end

  defp write_bin_modifier(var, type, size),
    do: shared_bin_modifier(var, type, size)

  defp shared_bin_modifier(var, :s, size),
    do: quote(do: unquote(var) :: signed - integer - native - size(unquote(size)))

  defp shared_bin_modifier(var, :u, size),
    do: quote(do: unquote(var) :: unsigned - integer - native - size(unquote(size)))

  @doc """
  BF16 read callback.
  """
  def read_bf16(<<0xFF80::16-native>>), do: :neg_infinity
  def read_bf16(<<0x7F80::16-native>>), do: :infinity

  if System.endianness() == :little do
    def read_bf16(<<1::1, _::7, _sign::1, 127::7>>), do: :nan

    def read_bf16(bf16) do
      <<x::float-little-32>> = <<0::16, bf16::binary>>
      x
    end
  else
    def read_bf16(<<_sign::1, 255::8, _::7>>), do: :nan

    def read_bf16(bf16) do
      <<x::float-big-32>> = <<bf16::binary, 0::16>>
      x
    end
  end

  @doc """
  F8 read callback.
  """
  def read_f8(<<0xFC::8-native>>), do: :neg_infinity
  def read_f8(<<0x7C::8-native>>), do: :infinity
  def read_f8(<<_sign::1, 31::5, mantissa::2>>) when mantissa != 0, do: :nan

  def read_f8(<<sign::1, exp::5, mantissa::2>>) do
    float = :math.pow(2, exp - 15) * (1 + mantissa / 4)

    case sign do
      0 -> float
      _ -> -float
    end
  end

  @doc """
  C64 and C128 callback.
  """
  def read_complex(val, size) do
    elem_size = div(size, 2)
    <<real_part::bitstring-size(elem_size), imag_part::bitstring-size(elem_size)>> = val

    re =
      case real_part do
        <<x::float-native-size(elem_size)>> -> x
        _ -> read_non_finite(real_part, elem_size)
      end

    im =
      case imag_part do
        <<x::float-native-size(elem_size)>> -> x
        _ -> read_non_finite(imag_part, elem_size)
      end

    Complex.new(re, im)
  end

  @doc """
  BF16 write callback.
  """
  def write_non_finite_bf16(data) do
    case data do
      :infinity -> unquote(Nx.Type.infinity_binary({:bf, 16}))
      :neg_infinity -> unquote(Nx.Type.neg_infinity_binary({:bf, 16}))
      :nan -> unquote(Nx.Type.nan_binary({:bf, 16}))
    end
  end

  if System.endianness() == :little do
    def write_finite_f8(x) do
      binary_part(<<x::float-native-16>>, 1, 1)
    end
  else
    def write_finite_f8(x) do
      binary_part(<<x::float-native-16>>, 0, 1)
    end
  end

  @doc """
  Complex write callback.
  """
  def write_complex(re, im, size) when is_number(re) and is_number(im) do
    <<re::float-native-size(size), im::float-native-size(size)>>
  end

  def write_complex(re, im, size) when is_number(re) and not is_number(im) do
    <<re::float-native-size(size)>> <> write_non_finite(im, size)
  end

  def write_complex(re, im, size) when not is_number(re) and is_number(im) do
    write_non_finite(re, size) <> <<im::float-native-size(size)>>
  end

  def write_complex(re, im, size) when not is_number(re) and not is_number(im) do
    write_non_finite(re, size) <> write_non_finite(im, size)
  end

  @doc """
  Non-finite read callback.
  """
  def read_non_finite(data, 8) do
    case data do
      <<0xFC::8-native>> -> :neg_infinity
      <<0x7C::8-native>> -> :infinity
      _ -> :nan
    end
  end

  def read_non_finite(data, 16) do
    case data do
      <<0xFC00::16-native>> -> :neg_infinity
      <<0x7C00::16-native>> -> :infinity
      _ -> :nan
    end
  end

  def read_non_finite(data, 32) do
    case data do
      <<0xFF800000::32-native>> -> :neg_infinity
      <<0x7F800000::32-native>> -> :infinity
      _ -> :nan
    end
  end

  def read_non_finite(data, 64) do
    case data do
      <<0xFFF0000000000000::64-native>> -> :neg_infinity
      <<0x7FF0000000000000::64-native>> -> :infinity
      _ -> :nan
    end
  end

  @doc """
  Non-finite write callback.
  """
  for size <- [8, 16, 32, 64] do
    def write_non_finite(data, unquote(size)) do
      case data do
        :infinity -> unquote(Nx.Type.infinity_binary({:f, size}))
        :neg_infinity -> unquote(Nx.Type.neg_infinity_binary({:f, size}))
        :nan -> unquote(Nx.Type.nan_binary({:f, size}))
      end
    end
  end

  ## Kernel helper macros

  @doc """
  Defines a macro that delegates to Elixir.Kernel when inside a guard.
  """
  defmacro defnguard(call, fallback) do
    {name, args} = Macro.decompose_call(call)

    quote do
      defmacro unquote(name)(unquote_splicing(args)) do
        {module, name} =
          case __CALLER__.context do
            ctx when ctx in [:guard, :match] -> {Kernel, unquote(name)}
            _ -> {__MODULE__, unquote(fallback)}
          end

        {{:., [], [module, name]}, [], unquote(args)}
      end
    end
  end

  ## Reflection

  @doc """
  Returns the definition of mathematical unary funs.
  """
  def unary_math_funs,
    do: [
      exp: {"exponential", quote(do: Complex.exp(var!(x))), "$$exp(z) = e^z$$"},
      expm1:
        {"exponential minus one",
         quote do
           var!(x)
           |> Complex.exp()
           |> Complex.subtract(1)
         end, "$$expm1(z) = e^z - 1$$"},
      log:
        {"natural log", quote(do: Complex.log(var!(x))),
         ~S"""
         $$log(z) = ln(z),\quad \text{if z} \in \Reals$$

         $$log(z) = ln(r) + i\theta,\quad\text{if }z = re^{i\theta} \in \Complex$$
         """},
      log1p:
        {"natural log plus one", quote(do: Complex.log(Complex.add(var!(x), 1))),
         "$$log1p(z) = log(z + 1)$$"},
      sigmoid:
        {"sigmoid",
         quote do
           var!(x)
           |> Complex.negate()
           |> Complex.exp()
           |> Complex.add(1)
           |> then(&Complex.divide(1, &1))
         end, "$$sigmoid(z) = \\frac{1}{1 + e^{-z}}$$"},
      cos:
        {"cosine", quote(do: Complex.cos(var!(x))), "$$cos(z) = \\frac{e^{iz} + e^{-iz}}{2}$$"},
      sin: {"sine", quote(do: Complex.sin(var!(x))), "$$sin(z) = \\frac{e^{iz} - e^{-iz}}{2i}$$"},
      tan: {"tangent", quote(do: Complex.tan(var!(x))), "$$tan(z) = \\frac{sin(z)}{cos(z)}$$"},
      cosh:
        {"hyperbolic cosine", quote(do: Complex.cosh(var!(x))),
         "$$cosh(z) = \\frac{e^z + e^{-z}}{2}$$"},
      sinh:
        {"hyperbolic sine", quote(do: Complex.sinh(var!(x))),
         "$$sinh(z) = \\frac{e^z - e^{-z}}{2}$$"},
      tanh:
        {"hyperbolic tangent", quote(do: Complex.tanh(var!(x))),
         "$$sinh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$"},
      acos: {"inverse cosine", quote(do: Complex.acos(var!(x))), "$$acos(cos(z)) = z$$"},
      asin: {"inverse sine", quote(do: Complex.asin(var!(x))), "$$asin(sin(z)) = z$$"},
      atan: {"inverse tangent", quote(do: Complex.atan(var!(x))), "$$atan(tan(z)) = z$$"},
      acosh:
        {"inverse hyperbolic cosine", quote(do: Complex.acosh(var!(x))), "$$acosh(cosh(z)) = z$$"},
      asinh:
        {"inverse hyperbolic sine", quote(do: Complex.asinh(var!(x))), "$$asinh(sinh(z)) = z$$"},
      atanh:
        {"inverse hyperbolic tangent", quote(do: Complex.atanh(var!(x))),
         "$$atanh(tanh(z)) = z$$"},
      sqrt: {"square root", quote(do: Complex.sqrt(var!(x))), "$$sqrt(z) = \\sqrt{z}$$"},
      rsqrt:
        {"reverse square root", quote(do: Complex.divide(1, Complex.sqrt(var!(x)))),
         "$$rsqrt(z) = \\frac{1}{\\sqrt{z}}$$"},
      cbrt: {"cube root", quote(do: Complex.cbrt(var!(x))), "$$cbrt(z) = z^{\\frac{1}{3}}$$"},
      erf:
        {"error function", quote(do: Complex.erf(var!(x))),
         "$$erf(z) = \\frac{2}{\\sqrt{\\pi}} \\int_{0}^{z} e^{-t^2}dt$$"},
      erfc:
        {"one minus error function", quote(do: Complex.erfc(var!(x))), "$$erfc(z) = 1 - erf(z)$$"},
      erf_inv:
        {"inverse error function", quote(do: Complex.erf_inv(var!(x))),
         "$$erf\\text{\\textunderscore}inv(erf(z)) = z$$"}
    ]

  ## Types

  @doc """
  Builds the type of an element-wise binary operation.
  """
  def binary_type(a, b) when is_number(a) and is_number(b),
    do: Nx.Type.infer(a + b)

  def binary_type(a, b) when is_number(a), do: Nx.Type.merge_number(type(b), a)
  def binary_type(a, b) when is_number(b), do: Nx.Type.merge_number(type(a), b)
  def binary_type(a, b), do: Nx.Type.merge(type(a), type(b))

  # For unknown types, return {:f, 32} as the caller
  # should validate the input in a later pass.
  defp type(%T{type: type}), do: type
  defp type({_, _} = type), do: type
  defp type(%Complex{}), do: {:c, 64}
  defp type(_other), do: {:f, 32}

  ## Helpers

  @doc """
  Appends an element to a tuple.
  """
  def tuple_append(tuple, elem) do
    Tuple.insert_at(tuple, tuple_size(tuple), elem)
  end

  @doc """
  Extracts the backend from the given options.
  """
  def backend_from_options!(opts) do
    case Keyword.fetch(opts, :backend) do
      {:ok, backend} when is_atom(backend) ->
        {backend, []}

      {:ok, {backend, options}} when is_atom(backend) and is_list(options) ->
        {backend, options}

      {:ok, other} ->
        raise ArgumentError,
              ":backend must be an atom or a tuple {backend, options}, got: #{inspect(other)}"

      :error ->
        nil
    end
  end

  @doc """
  Converts an Erlang float (float64) to float32 precision.
  """
  def to_float32(float64) when is_float(float64) do
    <<float32::float-32>> = <<float64::float-32>>
    float32
  end

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
  def list_impl!(list) do
    case for(%T{data: %struct{}} <- list, do: struct) do
      [] -> raise ArgumentError, "expected at least one tensor in list_impl!"
      [head | tail] -> Enum.reduce(tail, head, &pick_struct/2)
    end
  end

  defp pick_struct(Nx.BinaryBackend, struct), do: struct
  defp pick_struct(struct, Nx.BinaryBackend), do: struct
  defp pick_struct(struct, struct), do: struct

  defp pick_struct(struct1, struct2) do
    raise "cannot invoke Nx function because it relies on two incompatible tensor implementations: " <>
            "#{inspect(struct1)} and #{inspect(struct2)}. " <>
            (if struct1 == Nx.Defn.Expr or struct2 == Nx.Defn.Expr do
               "This may mean you are passing a tensor to defn/jit as an optional argument " <>
                 "or as closure in an anonymous function. For efficiency, it is preferred " <>
                 "to always pass tensors as required arguments instead. Alternatively, you " <>
                 "could call Nx.backend_copy/1 on the tensor, however this will copy its " <>
                 "value and inline it inside the defn expression"
             else
               "You may need to call Nx.backend_transfer/2 (or Nx.backend_copy/2) " <>
                 "on one or both of them to transfer them to a common implementation"
             end)
  end

  @doc """
  Used to define an Nx callback with an optional implementation.

  The given body is used as the default implementation otherwise.
  """
  def optional(function_name, args, output, default_impl)
      when is_atom(function_name) and is_list(args) and is_function(default_impl) do
    arity = length(args) + 1
    backend = list_impl!(args)

    cond do
      function_exported?(backend, function_name, arity) ->
        apply(backend, function_name, [output | args])

      function_exported?(backend, :optional, 3) ->
        backend.optional(function_name, args, default_impl)
        |> ensure_optional_compatible!(output)

      true ->
        default_impl
        |> apply(args)
        |> ensure_optional_compatible!(output)
    end
  end

  defp ensure_optional_compatible!(left, right) when tuple_size(left) == tuple_size(right) do
    [Tuple.to_list(left), Tuple.to_list(right)]
    |> Enum.zip_with(fn [l, r] -> ensure_optional_compatible!(l, r) end)

    left
  end

  defp ensure_optional_compatible!(
         %{shape: shape, type: type, names: names} = left,
         %{shape: shape, type: type, names: names}
       ),
       do: left

  defp ensure_optional_compatible!(left, right) do
    raise ArgumentError,
          "expected default implementation to match template #{inspect(right)}, got: #{inspect(left)}"
  end

  @doc false
  def raise_complex_not_supported(function, arity) do
    raise ArgumentError, "Nx.#{function}/#{arity} does not support complex inputs"
  end

  @doc false
  def raise_complex_not_supported({:c, _}, function, arity),
    do: raise_complex_not_supported(function, arity)

  def raise_complex_not_supported(_, _, _), do: nil

  @doc false
  def raise_complex_not_implemented_yet(function, arity) do
    raise ArgumentError, "Nx.#{function}/#{arity} is not yet implemented for complex inputs"
  end

  @doc false
  def raise_complex_not_implemented_yet({:c, _}, function, arity),
    do: raise_complex_not_implemented_yet(function, arity)

  def raise_complex_not_implemented_yet(_, _, _), do: nil

  @doc false
  def raise_vectorized_not_implemented_yet(%T{vectorized_axes: [_ | _]}, {function, arity}) do
    raise ArgumentError, "#{function}/#{arity} is not yet implemented for vectorized inputs"
  end

  def raise_vectorized_not_implemented_yet(_, _), do: nil

  @doc false
  def raise_vectorization_not_supported(%T{vectorized_axes: [_ | _]}, {function, arity}) do
    raise ArgumentError, "#{function}/#{arity} does not support vectorized inputs"
  end

  def raise_vectorization_not_supported(_, _), do: nil

  @doc """
  The process dictionary key to store default backend under.
  """
  def backend_pdict_key, do: {Nx, :default_backend}
end

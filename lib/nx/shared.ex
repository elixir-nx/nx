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
      tanh: {"hyperbolic tangent", quote(do: :math.tanh(var!(x)))},
      sqrt: {"square root", quote(do: :math.sqrt(var!(x)))},
      rsqrt: {"reverse square root", quote(do: 1 / :math.sqrt(var!(x)))},
      cbrt: {"cube root", quote(do: :math.pow(var!(x), 1 / 3))}
    ]

  ## Creation ops

  def iota_out(tensor_or_shape, opts) do
    assert_keys!(opts, [:type, :axis, :names])
    shape = Nx.shape(tensor_or_shape)
    names = opts[:names] || Nx.Shape.named_axes!(names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type] || {:s, 64})

    if axis = opts[:axis] do
      axis = Nx.Shape.normalize_axis(shape, axis, names)
      {%T{type: type, shape: shape, names: names}, axis}
    else
      {%T{type: type, shape: shape, names: names}, nil}
    end
  end

  def random_uniform_out(tensor_or_shape, min, max, opts) do
    assert_keys!(opts, [:type, :names])
    shape = Nx.shape(tensor_or_shape)
    names = opts[:names] || Nx.Shape.named_axes!(names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type] || Nx.Type.infer(max - min))
    %T{shape: shape, type: type, names: names}
  end

  def random_normal_out(tensor_or_shape, opts \\ []) do
    assert_keys!(opts, [:type, :names])
    shape = Nx.shape(tensor_or_shape)
    names = opts[:names] || Nx.Shape.named_axes!(names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type] || {:f, 64})
    %T{shape: shape, type: type, names: names}
  end

  ## Helpers

  @doc """
  Returns the names of a tensor, if any.
  """
  def names!(%T{names: names}), do: names
  def names!(_), do: nil

  @doc """
  Asserts the given keys.
  """
  def assert_keys!(keyword, valid) do
    for {k, _} <- keyword, k not in valid do
      raise "unknown key #{inspect(k)} in #{inspect(keyword)}, expected one of #{inspect(valid)}"
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
    Enum.reduce(list, Nx.BinaryTensor, fn
      %T{data: %struct{}}, acc -> pick_struct(struct, acc)
      _, acc -> acc
    end)
  end

  defp pick_struct(Nx.BinaryTensor, struct), do: struct
  defp pick_struct(struct, Nx.BinaryTensor), do: struct
  defp pick_struct(struct, struct), do: struct

  defp pick_struct(struct1, struct2) do
    raise "cannot invoke Nx function because it relies on two incompatible tensor implementations: " <>
            "#{inspect(struct1)} and #{inspect(struct2)}"
  end
end

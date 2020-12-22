defmodule Nx.Shared do
  # A collection of **private** helpers and macros shared between Nx and Nx.Util.
  @moduledoc false

  @doc """
  Match the cartesian product of all given types.

  A macro that allows us to writes all possibles match types
  in the most efficient format. This is done by looking at @0,
  @1, etc and replacing them by currently matched type at the
  given position. In other words, this:

     combine_types [input_type, output_type] do
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

  @all_types [:s, :f, :bf, :u, :pred]

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

  defp shared_bin_modifier(var, :pred, size),
    do: quote(do: unquote(var) :: unsigned - integer - native - size(unquote(size)))

  defp shared_bin_modifier(var, :f, size),
    do: quote(do: unquote(var) :: float - native - size(unquote(size)))

  @doc """
  Converts a scalar to a binary, according to the type.
  """
  def scalar_to_bin(value, type) do
    match_types([type], do: <<write!(value, 0)>>)
  end

  @doc """
  Converts the shape to a weight shape list.

  A weighted shape is a list of tuples where the first
  element is the number of elements in the dimension
  and the second element is the size to be traversed in
  the binary to fetch the next element.

  This is often given to `weighted_traverse/3` as a general
  mechanism to traverse binaries.
  """
  def weighted_shape(shape, size) do
    Enum.reverse(weighted_shape(shape, tuple_size(shape), size))
  end

  defp weighted_shape(_shape, 0, _weight) do
    []
  end

  defp weighted_shape(shape, pos, weight) do
    element = :erlang.element(pos, shape)
    [{element, weight} | weighted_shape(shape, pos - 1, weight * element)]
  end

  def weighted_shape_limits(shape, size, lengths) do
    Enum.reverse(weighted_shape_limits(shape, tuple_size(shape), size, lengths))
  end

  defp weighted_shape_limits(_shape, 0, _weight, _lengths) do
    []
  end

  defp weighted_shape_limits(shape, pos, weight, lengths) do
    element = :erlang.element(pos, lengths)
    shape_elem = :erlang.element(pos, shape)
    [{element, weight} | weighted_shape_limits(shape, pos - 1, weight * shape_elem, lengths)]
  end

  @doc """
  Reads the chunk size from a weighted list at the given position.
  """
  def weighted_chunk(list, at, size) do
    {element, size} = Enum.at(list, at, {1, size})
    element * size
  end

  @doc """
  Traverses a binary using the elements and shape given by `weighted_shape`.

  When all dimensions are traversed, we read `read_size`.

  The `weighted_shape` can also contain functions, which are applied to the
  result of the remaining of the weighted shape.
  """
  def weighted_traverse(weighted_shape, binary, read_size)

  def weighted_traverse([], data, read_size) do
    <<chunk::size(read_size)-bitstring, _::bitstring>> = data
    chunk
  end

  def weighted_traverse([{dim, size} | dims], data, read_size) do
    weighted_traverse(dim, size, dims, data, read_size)
  end

  def weighted_traverse([fun | dims], data, read_size) do
    fun.(weighted_traverse(dims, data, read_size))
  end

  defp weighted_traverse(dim, dim_size, dims, data, read_size) do
    head = weighted_traverse(dims, data, read_size)

    case dim do
      1 ->
        [head]

      _ ->
        <<_::size(dim_size)-bitstring, data::bitstring>> = data
        [head | weighted_traverse(dim - 1, dim_size, dims, data, read_size)]
    end
  end

  @doc """
  Converts a shape to a padded list by rank where the
  first elements are the lower dimensions.
  """
  def shape_to_lower_ranked_list(_tuple, 0, 0),
    do: []

  def shape_to_lower_ranked_list(tuple, 0, rank),
    do: [1 | shape_to_lower_ranked_list(tuple, 0, rank - 1)]

  def shape_to_lower_ranked_list(tuple, size, rank),
    do: [:erlang.element(size, tuple) | shape_to_lower_ranked_list(tuple, size - 1, rank - 1)]

  @doc """
  Traverses a binary at the given anchor point using `weighted_shape`.
  """
  def anchored_weighted_traverse(weighted_shape, binary, read_size, offset)

  def anchored_weighted_traverse([], data, read_size, offset) do
    <<_::size(offset), chunk::size(read_size)-bitstring, _::bitstring>> = data
    chunk
  end

  def anchored_weighted_traverse([{dim, size} | dims], data, read_size, offset) do
    anchored_weighted_traverse(dim, size, dims, data, read_size, offset)
  end

  def anchored_weighted_traverse([fun | dims], data, read_size, offset) do
    fun.(anchored_weighted_traverse(dims, data, read_size, offset))
  end

  def anchored_weighted_traverse(dim, dim_size, dims, data, read_size, offset) do
    head = anchored_weighted_traverse(dims, data, read_size, offset)
    case dim do
      1 ->
        [head]

      _ ->
        <<_::size(dim_size)-bitstring, data::bitstring>> = data
        [head | anchored_weighted_traverse(dim - 1, dim_size, dims, data, read_size, offset)]
    end
  end
end

defmodule Nx do
  @moduledoc """
  Numerical Elixir.

  The `Nx` library is collection of functions and data
  types to work with Numerical Elixir. This module defines
  the main entry point for building and working with said
  data-structures. For example, to create a n-dimensional
  tensor, do:

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> Nx.shape(t)
      {2, 2}

  To implement [the Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
  using this library:

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t = Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t)))
      iex> Nx.to_bitstring(t)
      <<0.03205860328008499::float-64-native, 0.08714431874203257::float-64-native,
        0.23688281808991013::float-64-native, 0.6439142598879722::float-64-native>>

  The `Nx` library also provides the `Nx.Defn` functionality,
  which provides a subset of Elixir tailored for numerical
  computations. For example, it overrides Elixir's default
  operators so they are tensor-aware:

      defn softmax(t) do
        Nx.exp(t) / Nx.sum(Nx.exp(t))
      end

  Code inside `defn` functions can also be given to custom compilers,
  which can compile said functions to use either just-in-time (JIT)
  or ahead-of-time (AOT) compilers, and run on the CPU or in the GPU.
  For example, using the `Exla` library:

      @defn_compiler {Exla, platform: :host} # or platform: :cuda
      defn softmax(t) do
        Nx.exp(t) / Nx.sum(Nx.exp(t))
      end

  This complements Erlang's JIT compiler as it compiles direct to
  native code with numerical compilation and performance in mind.
  """

  ## Private macros

  # A macro that allows us to writes all possibles match types
  # in the most efficient format. This is done by looking at @0,
  # @1, etc and replacing them by currently matched type at the
  # given position. In other words, this:
  #
  #    combine_types [input_type, output_type] do
  #      for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(read!(seg, 0) + right, 1)>>
  #    end
  #
  # Is compiled into:
  #
  #    for <<seg::float-native-size(...) <- data>>, into: <<>>, do: <<seg+right::float-native-size(...)>>
  #
  # for all possible valid types between input and input types.
  #
  # `match!` is used in matches and must be always followed by a `read!`.
  # `write!` is used to write to the binary.
  #
  # The implementation unfolds the loops at the top level. In particular,
  # note that a rolled out case such as:
  #
  #     for <<seg::size(size)-signed-integer <- data>>, into: <<>> do
  #       <<seg+number::signed-integer-size(size)>>
  #     end
  #
  # is twice as fast and uses twice less memory than:
  #
  #     for <<seg::size(size)-signed-integer <- data>>, into: <<>> do
  #       case output_type do
  #         {:s, size} ->
  #           <<seg+number::signed-integer-size(size)>>
  #         {:f, size} ->
  #           <<seg+number::float-native-size(size)>>
  #         {:u, size} ->
  #           <<seg+number::unsigned-integer-size(size)>>
  #       end
  #     end
  #
  defmacrop match_types([_ | _] = args, do: block) do
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
      quote(do: read_bf16(unquote(var)))
    else
      quote(do: read_bf16(unquote(var)))
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

  @compile {:inline, read_bf16: 1}
  if System.endianness() == :little do
    defp read_bf16(bf16) do
      <<x::float-little-32>> = <<0::16, bf16::binary>>
      x
    end
  else
    defp read_bf16(bf16) do
      <<x::float-big-32>> = <<bf16::binary, 0::16>>
      x
    end
  end

  defp shared_bin_modifier(var, :s, size),
    do: quote(do: unquote(var) :: signed - integer - native - size(unquote(size)))

  defp shared_bin_modifier(var, :u, size),
    do: quote(do: unquote(var) :: unsigned - integer - native - size(unquote(size)))

  defp shared_bin_modifier(var, :f, size),
    do: quote(do: unquote(var) :: float - native - size(unquote(size)))

  ## API

  alias Nx.Tensor, as: T

  @doc """
  Builds a tensor.

  The argument is either a number or a boolean, which means the tensor is
  a scalar (zero-dimentions), a list of those (the tensor is a vector) or
  a list of n-lists of those, leading to n-dimensional tensors.

  ## Examples

  A number or a boolean returns a tensor of zero dimensions:

      iex> t = Nx.tensor(0)
      iex> Nx.to_bitstring(t)
      <<0::64-native>>
      iex> Nx.type(t)
      {:s, 64}
      iex> Nx.shape(t)
      {}

      iex> t = Nx.tensor(true)
      iex> Nx.to_bitstring(t)
      <<1::1>>
      iex> Nx.type(t)
      {:u, 1}
      iex> Nx.shape(t)
      {}

  Giving a list returns a vector (an one-dimensional tensor):

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.to_bitstring(t)
      <<1::64-native, 2::64-native, 3::64-native>>
      iex> Nx.type(t)
      {:s, 64}
      iex> Nx.shape(t)
      {3}

      iex> t = Nx.tensor([1.2, 2.3, 3.4, 4.5])
      iex> Nx.to_bitstring(t)
      <<1.2::float-64-native, 2.3::float-64-native, 3.4::float-64-native, 4.5::float-64-native>>
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.shape(t)
      {4}

  The type can be explicitly given. Integers and floats
  bigger than the given size overlap:

      iex> t = Nx.tensor([300, 301, 302], type: {:s, 8})
      iex> Nx.to_bitstring(t)
      <<44::8, 45::8, 46::8>>
      iex> Nx.type(t)
      {:s, 8}

      iex> t = Nx.tensor([1.2, 2.3, 3.4], type: {:f, 32})
      iex> Nx.to_bitstring(t)
      <<1.2::float-native-32, 2.3::float-native-32, 3.4::float-native-32>>
      iex> Nx.type(t)
      {:f, 32}

  An empty list defaults to floats:

      iex> t = Nx.tensor([])
      iex> Nx.to_bitstring(t)
      <<>>
      iex> Nx.type(t)
      {:f, 64}

  Mixed types get the highest precision type:

      iex> t = Nx.tensor([true, 2, 3.0])
      iex> Nx.to_bitstring(t)
      <<1.0::float-64-native, 2.0::float-64-native, 3::float-64-native>>
      iex> Nx.type(t)
      {:f, 64}

  Multi-dimensional tensors are also possible:

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.to_bitstring(t)
      <<1::64-native, 2::64-native, 3::64-native, 4::64-native, 5::64-native, 6::64-native>>
      iex> Nx.type(t)
      {:s, 64}
      iex> Nx.shape(t)
      {2, 3}

      iex> t = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      iex> Nx.to_bitstring(t)
      <<1::64-native, 2::64-native, 3::64-native, 4::64-native, 5::64-native, 6::64-native>>
      iex> Nx.type(t)
      {:s, 64}
      iex> Nx.shape(t)
      {3, 2}

      iex> t = Nx.tensor([[[1, 2], [3, 4], [5, 6]], [[-1, -2], [-3, -4], [-5, -6]]])
      iex> Nx.to_bitstring(t)
      <<1::64-native, 2::64-native, 3::64-native, 4::64-native, 5::64-native, 6::64-native,
        -1::64-native, -2::64-native, -3::64-native, -4::64-native, -5::64-native, -6::64-native>>
      iex> Nx.type(t)
      {:s, 64}
      iex> Nx.shape(t)
      {2, 3, 2}

  Brain-floating points are also supported, although they are
  emulated in Elixir and therefore perform slower without a
  compilation backend:

      iex> t = Nx.tensor([1, 2, 3], type: {:bf, 16})
      iex> Nx.to_bitstring(t)
      <<16256::16-native, 16384::16-native, 16448::16-native>>
      iex> Nx.type(t)
      {:bf, 16}

  ## Options

    * `:type` - sets the type of the tensor. If one is not given,
      one is automatically inferred based on the input. See `Nx.Type`
      and `Nx.Type.infer/1` for information.

  """
  def tensor(arg, opts \\ []) do
    type = opts[:type] || Nx.Type.infer(arg)
    Nx.Type.validate!(type)
    {dimensions, data} = flatten(arg, type)
    %T{shape: dimensions, type: type, data: data}
  end

  defp flatten(list, type) when is_list(list) do
    {dimensions, acc} = flatten_list(list, type, [], [])

    {dimensions |> Enum.reverse() |> List.to_tuple(),
     acc |> Enum.reverse() |> :erlang.list_to_bitstring()}
  end

  defp flatten(other, type), do: {{}, scalar_to_binary(other, type)}

  defp flatten_list([], _type, dimensions, acc) do
    {[0 | dimensions], acc}
  end

  defp flatten_list([head | rest], type, parent_dimensions, acc) when is_list(head) do
    {child_dimensions, acc} = flatten_list(head, type, [], acc)

    {n, acc} =
      Enum.reduce(rest, {1, acc}, fn list, {count, acc} ->
        case flatten_list(list, type, [], acc) do
          {^child_dimensions, acc} ->
            {count + 1, acc}

          {other_dimensions, _acc} ->
            raise ArgumentError,
                  "cannot build tensor because lists have different shapes, got " <>
                    inspect(List.to_tuple(child_dimensions)) <>
                    " at position 0 and " <>
                    inspect(List.to_tuple(other_dimensions)) <> " at position #{count + 1}"
        end
      end)

    {child_dimensions ++ [n | parent_dimensions], acc}
  end

  defp flatten_list(list, type, dimensions, acc) do
    {[length(list) | dimensions], Enum.reduce(list, acc, &[scalar_to_binary(&1, type) | &2])}
  end

  defp scalar_to_binary(true, type), do: scalar_to_binary(1, type)
  defp scalar_to_binary(false, type), do: scalar_to_binary(0, type)
  defp scalar_to_binary(value, type), do: match_types([type], do: <<write!(value, 0)>>)

  @doc """
  Returns the type of the tensor.

  See `Nx.Type` for more information.
  """
  def type(%T{type: type}), do: type

  @doc """
  Returns the shape of the tensor as a tuple.

  The size of this tuple gives the rank of the tensor.
  """
  def shape(%T{shape: shape}), do: shape

  @doc """
  Returns the rank of a tensor.
  """
  def rank(%T{shape: shape}), do: tuple_size(shape)

  @doc """
  Returns the underlying tensor as a bitstring.

  The bitstring is returned as is (which is row-major).
  """
  # TODO: What happens if the data is in the device?
  def to_bitstring(%T{data: data}), do: data

  @doc """
  Adds two tensors together.

  If a number is given, it is converted to a tensor on the fly.

  ## Examples

  ### Adding scalars

      iex> t = Nx.add(1, 2)
      iex> Nx.to_bitstring(t)
      <<3::64-native>>

      iex> t = Nx.add(1, 2.2)
      iex> Nx.to_bitstring(t)
      <<3.2::float-64-native>>

  ### Adding a scalar to a tensor

      iex> t = Nx.add(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<2::64-native, 3::64-native, 4::64-native>>

      iex> t = Nx.add(1, Nx.tensor([1, 2, 3]))
      iex> Nx.to_bitstring(t)
      <<2::64-native, 3::64-native, 4::64-native>>

  Given a float scalar converts the tensor to a float:

      iex> t = Nx.add(Nx.tensor([1, 2, 3]), 1.0)
      iex> Nx.to_bitstring(t)
      <<2.0::float-64-native, 3.0::float-64-native, 4.0::float-64-native>>

      iex> t = Nx.add(Nx.tensor([1.0, 2.0, 3.0]), 1)
      iex> Nx.to_bitstring(t)
      <<2.0::float-64-native, 3.0::float-64-native, 4.0::float-64-native>>

      iex> t = Nx.add(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}), 1)
      iex> Nx.to_bitstring(t)
      <<2.0::float-native-32, 3.0::float-native-32, 4.0::float-native-32>>

  The size of the tensor will grow if the scalar is bigger than
  the tensor size. But, if smaller, it overflows:

      iex> t = Nx.add(Nx.tensor([true, false, true]), 1)
      iex> Nx.to_bitstring(t)
      <<0::1, 1::1, 0::1>>

      iex> t = Nx.add(Nx.tensor([true, false, true]), 10)
      iex> Nx.to_bitstring(t)
      <<11::8, 10::8, 11::8>>

  Unsigned tensors become signed and double their size if a
  negative number is given:

      iex> t = Nx.add(Nx.tensor([0, 1, 2], type: {:u, 8}), -1)
      iex> Nx.to_bitstring(t)
      <<-1::16-native, 0::16-native, 1::16-native>>

  ### Adding tensors of the same shape

      iex> t = Nx.add(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[10, 20], [30, 40]]))
      iex> Nx.to_bitstring(t)
      <<11::64-native, 22::64-native, 33::64-native, 44::64-native>>

  ### Adding tensors with broadcasting

      iex> t = Nx.add(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      iex> Nx.to_bitstring(t)
      <<11::64-native, 21::64-native, 12::64-native, 22::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.add(Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]]))
      iex> Nx.to_bitstring(t)
      <<11::64-native, 21::64-native, 12::64-native, 22::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.add(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20], [30, 40]]))
      iex> Nx.to_bitstring(t)
      <<11::64-native, 21::64-native, 32::64-native, 42::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.add(Nx.tensor([[1, 2]]), Nx.tensor([[10, 20], [30, 40]]))
      iex> Nx.to_bitstring(t)
      <<11::64-native, 22::64-native, 31::64-native, 42::64-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def add(left, right)

  def add(left, right) when is_number(left) and is_number(right),
    do: tensor(left + right)

  def add(scalar, %T{data: data, type: input_type} = t) when is_number(scalar) do
    output_type = Nx.Type.merge_scalar(input_type, scalar)

    data =
      match_types [input_type, output_type] do
        for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(read!(seg, 0) + scalar, 1)>>
      end

    %{t | data: data, type: output_type}
  end

  def add(%T{data: data, type: input_type} = t, scalar) when is_number(scalar) do
    output_type = Nx.Type.merge_scalar(input_type, scalar)

    data =
      match_types [input_type, output_type] do
        for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(read!(seg, 0) + scalar, 1)>>
      end

    %{t | data: data, type: output_type}
  end

  def add(%T{type: left_type} = left, %T{type: right_type} = right) do
    output_type = Nx.Type.merge(left_type, right_type)

    {data, shape} =
      match_types [left_type, right_type, output_type] do
        broadcast(left, right, fn left_dimension, right_dimension ->
          for <<match!(left_seg, 0) <- left_dimension>>,
              <<match!(right_seg, 1) <- right_dimension>>,
              into: <<>> do
            <<write!(read!(left_seg, 0) + read!(right_seg, 1), 2)>>
          end
        end)
      end

    %T{data: data, type: output_type, shape: shape}
  end

  # TODO: Properly implement me
  def divide(
        %T{data: left_data, type: left_type} = left,
        %T{data: right_data, type: right_type, shape: {}}
      ) do
    output_type = Nx.Type.merge(left_type, right_type) |> Nx.Type.to_float()

    data =
      match_types [left_type, right_type, output_type] do
        <<match!(c, 1)>> = right_data
        c = read!(c, 1)
        for <<match!(seg, 0) <- left_data>>, into: <<>>, do: <<write!(read!(seg, 0) / c, 2)>>
      end

    %{left | data: data, type: output_type}
  end

  @doc """
  Calculates the exponential of the given tensor.

  If a scalar is given, a scalar is returned.
  Otherwise, returns an updated tensor. In both
  cases, the return type is float.

  ## Examples

      iex> t = Nx.exp(1)
      iex> Nx.to_bitstring(t)
      <<2.718281828459045::float-64-native>>

      iex> t = Nx.exp(Nx.tensor([1, 2, 3]))
      iex> Nx.to_bitstring(t)
      <<2.718281828459045::float-64-native, 7.38905609893065::float-64-native, 20.085536923187668::float-64-native>>

      iex> t = Nx.exp(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<2.718281828459045::float-native-32, 7.38905609893065::float-native-32, 20.085536923187668::float-native-32>>

      iex> t = Nx.exp(Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16}))
      iex> Nx.to_bitstring(t)
      <<16429::16-native, 16620::16-native, 16800::16-native>>

  """
  def exp(number)

  def exp(number) when is_number(number), do: tensor(:math.exp(number))

  def exp(%T{data: data, type: input_type} = t) do
    output_type = Nx.Type.to_float(input_type)

    data =
      match_types [input_type, output_type] do
        for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(:math.exp(read!(seg, 0)), 1)>>
      end

    %{t | data: data, type: output_type}
  end

  @doc """
  Returns the sum across all dimensions.

  ## Examples

      iex> t = Nx.sum(Nx.tensor([1, 2, 3]))
      iex> Nx.to_bitstring(t)
      <<6::64-native>>
      iex> Nx.shape(t)
      {}

      iex> t = Nx.sum(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      iex> Nx.to_bitstring(t)
      <<10.0::float-64-native>>
      iex> Nx.shape(t)
      {}

  """
  def sum(%T{data: data, type: type} = t) do
    data =
      match_types [type] do
        value =
          bin_reduce_all(data, 0, fn <<match!(var, 0), rest::bitstring>>, acc ->
            {read!(var, 0) + acc, rest}
          end)

        <<write!(value, 0)>>
      end

    %{t | data: data, shape: {}}
  end

  ## Broadcast helpers

  defp broadcast(
         %T{data: left_data, type: {_, left_size}, shape: shape},
         %T{data: right_data, type: {_, right_size}, shape: shape},
         fun
       ) do
    data = bin_zip_map_all(left_data, left_size, right_data, right_size, fun)
    {IO.iodata_to_binary(data), shape}
  end

  defp broadcast(
         %T{data: left_data, type: {_, left_size}, shape: left_shape},
         %T{data: right_data, type: {_, right_size}, shape: right_shape},
         fun
       ) do
    left_rank = tuple_size(left_shape)
    right_rank = tuple_size(right_shape)
    rank = max(left_rank, right_rank)
    left_ordered = shape_to_ranked_ordered_list(left_shape, left_rank, rank)
    right_ordered = shape_to_ranked_ordered_list(right_shape, right_rank, rank)

    {chunks, shape} =
      broadcast_chunks(left_ordered, right_ordered, left_size, right_size, [fun], [])

    {broadcast_recur(left_data, right_data, chunks), shape}
  end

  defp broadcast_recur(left_data, right_data, [fun]) do
    fun.(left_data, right_data)
  end

  defp broadcast_recur(left_data, right_data, [{:cross, left_chunk, right_chunk} | chunks]) do
    for <<left_part::bitstring-size(left_chunk) <- left_data>>,
        <<right_part::bitstring-size(right_chunk) <- right_data>>,
        into: <<>>,
        do: broadcast_recur(left_part, right_part, chunks)
  end

  defp broadcast_recur(left_data, right_data, [{:zip, left_chunk, right_chunk} | chunks]) do
    left_data
    |> bin_zip_map_all(left_chunk, right_data, right_chunk, &broadcast_recur(&1, &2, chunks))
    |> IO.iodata_to_binary()
  end

  defp shape_to_ranked_ordered_list(_tuple, 0, 0),
    do: []

  defp shape_to_ranked_ordered_list(tuple, 0, rank),
    do: [1 | shape_to_ranked_ordered_list(tuple, 0, rank - 1)]

  defp shape_to_ranked_ordered_list(tuple, size, rank),
    do: [:erlang.element(size, tuple) | shape_to_ranked_ordered_list(tuple, size - 1, rank - 1)]

  defp broadcast_chunks([], [], _, _, chunks, shape) do
    {chunks, List.to_tuple(shape)}
  end

  defp broadcast_chunks(left_ordered, right_ordered, left_size, right_size, chunks, shape)
       when hd(left_ordered) == 1 or hd(right_ordered) == 1 do
    left_ones = count_ones(left_ordered)
    right_ones = count_ones(right_ordered)
    {dir, size} = if left_ones <= right_ones, do: {:left, right_ones}, else: {:right, left_ones}

    {left_ordered, right_ordered, left_chunk, right_chunk, shape} =
      broadcast_split_chunks(left_ordered, right_ordered, left_size, right_size, size, shape)

    chunks = if dir == :left, do: chunks, else: [{:cross, left_size, right_size} | chunks]
    broadcast_chunks(left_ordered, right_ordered, left_chunk, right_chunk, chunks, shape)
  end

  defp broadcast_chunks(left_ordered, right_ordered, left_size, right_size, chunks, shape)
       when hd(left_ordered) == hd(right_ordered) do
    {left_ordered, right_ordered, left_chunk, right_chunk, shape} =
      broadcast_shared_chunks(left_ordered, right_ordered, left_size, right_size, shape)

    chunks = [{:zip, left_size, right_size} | chunks]
    broadcast_chunks(left_ordered, right_ordered, left_chunk, right_chunk, chunks, shape)
  end

  defp broadcast_chunks(_left_ordered, _right_ordered, _left_size, _right_size, _chunks, _shape),
    do: :error

  defp count_ones([1 | shape]), do: 1 + count_ones(shape)
  defp count_ones(_), do: 0

  defp broadcast_split_chunks([lh | lt], [rh | rt], ls, rs, n, shape) when n > 0,
    do: broadcast_split_chunks(lt, rt, ls * lh, rh * rs, n - 1, [max(lh, rh) | shape])

  defp broadcast_split_chunks(l, r, ls, rs, _n, shape),
    do: {l, r, ls, rs, shape}

  defp broadcast_shared_chunks([x | left], [x | right], ls, rs, shape),
    do: broadcast_shared_chunks(left, right, ls * x, rs * x, [x | shape])

  defp broadcast_shared_chunks(left, right, ls, rs, shape),
    do: {left, right, ls, rs, shape}

  ## Binary helpers

  @compile {:inline, bin_reduce_all: 3}

  defp bin_reduce_all(<<>>, acc, _fun) do
    acc
  end

  defp bin_reduce_all(binary, acc, fun) do
    {acc, rest} = fun.(binary, acc)
    bin_reduce_all(rest, acc, fun)
  end

  defp bin_zip_map_all(<<>>, _left_size, <<>>, _right_size, _fun), do: []

  defp bin_zip_map_all(left_data, left_size, right_data, right_size, fun) do
    <<left_head::bitstring-size(left_size), left_rest::bitstring>> = left_data
    <<right_head::bitstring-size(right_size), right_rest::bitstring>> = right_data

    [
      fun.(left_head, right_head)
      | bin_zip_map_all(left_rest, left_size, right_rest, right_size, fun)
    ]
  end
end

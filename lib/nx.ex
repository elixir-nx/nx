defmodule Nx do
  @moduledoc """
  Numerical Elixir.

  A collection of functions and data types to work
  with Numerical Elixir.
  """

  defmodule Tensor do
    defstruct [:data, :type, :shape]
  end

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
      <<1.2::float-native-64, 2.3::float-native-64, 3.4::float-native-64, 4.5::float-native-64>>
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
      <<1.0::float-native-64, 2.0::float-native-64, 3::float-native-64>>
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
    %Tensor{shape: dimensions, type: type, data: data}
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
  def type(%Tensor{type: type}), do: type

  @doc """
  Returns the shape of the tensor as a tuple.

  The size of this tuple gives the rank of the tensor.
  """
  def shape(%Tensor{shape: shape}), do: shape

  @doc """
  Returns the rank of a tensor.
  """
  def rank(%Tensor{shape: shape}), do: tuple_size(shape)

  @doc """
  Returns the underlying tensor as a bitstring.

  The bitstring is returned as is (which is row-major).

  # TODO: What happens if the data is in the device?
  """
  def to_bitstring(%Tensor{data: data}), do: data

  @doc """
  Adds two tensors together.

  If a scalar is given, they are kept as scalars.
  If a tensor and a scalar are given, it adds the
  scalar to all entries in the tensor.

  ## Examples

  ### Adding scalars

      iex> Nx.add(1, 2)
      3
      iex> Nx.add(1, 2.2)
      3.2

  ### Adding a scalar to a tensor

      iex> t = Nx.add(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<2::64-native, 3::64-native, 4::64-native>>

  Given a float scalar converts the tensor to a float:

      iex> t = Nx.add(Nx.tensor([1, 2, 3]), 1.0)
      iex> Nx.to_bitstring(t)
      <<2.0::float-native-64, 3.0::float-native-64, 4.0::float-native-64>>

      iex> t = Nx.add(Nx.tensor([1.0, 2.0, 3.0]), 1)
      iex> Nx.to_bitstring(t)
      <<2.0::float-native-64, 3.0::float-native-64, 4.0::float-native-64>>

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

  """
  # TODO: implement addition between tensors with broadcasting
  def add(left, right)

  def add(left, right) when is_number(left) and is_number(right), do: :erlang.+(left, right)

  def add(%Tensor{data: data, type: input_type} = left, right) when is_number(right) do
    output_type = Nx.Type.merge_scalar(input_type, right)

    data =
      match_types [input_type, output_type] do
        for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(read!(seg, 0) + right, 1)>>
      end

    %{left | data: data, type: output_type}
  end

  # TODO: Properly implement me
  def divide(
        %Tensor{data: left_data, type: left_type} = left,
        %Tensor{data: right_data, type: right_type, shape: {}}
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

      iex> Nx.exp(1)
      2.718281828459045

      iex> t = Nx.exp(Nx.tensor([1, 2, 3]))
      iex> Nx.to_bitstring(t)
      <<2.718281828459045::float-native-64, 7.38905609893065::float-native-64, 20.085536923187668::float-native-64>>

      iex> t = Nx.exp(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<2.718281828459045::float-native-32, 7.38905609893065::float-native-32, 20.085536923187668::float-native-32>>

      iex> t = Nx.exp(Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16}))
      iex> Nx.to_bitstring(t)
      <<16429::16-native, 16620::16-native, 16800::16-native>>

  """
  def exp(number)

  def exp(number) when is_number(number), do: :math.exp(number)

  def exp(%Tensor{data: data, type: input_type} = t) do
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
      <<10.0::float-native-64>>
      iex> Nx.shape(t)
      {}

  """
  def sum(%Tensor{data: data, type: type} = t) do
    data =
      match_types [type] do
        value =
          until_empty(data, 0, fn <<match!(var, 0), rest::bitstring>>, acc ->
            {read!(var, 0) + acc, rest}
          end)

        <<write!(value, 0)>>
      end

    %{t | data: data, shape: {}}
  end

  @compile {:inline, until_empty: 3}
  defp until_empty(<<>>, acc, _fun) do
    acc
  end

  defp until_empty(binary, acc, fun) do
    {acc, rest} = fun.(binary, acc)
    until_empty(rest, acc, fun)
  end
end

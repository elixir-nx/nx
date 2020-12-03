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
  For example, using the `Exla` compiler:

      @defn_compiler {Exla, platform: :host}
      defn softmax(t) do
        Nx.exp(t) / Nx.sum(Nx.exp(t))
      end

  This complements Erlang's JIT compiler as it compiles direct to
  native code with numerical compilation and performance in mind.

  ## Broadcasting

  TODO

  ## Devices

  The `Nx` library has built-in support for devices. A tensor is
  always allocated in a device, the default device being the
  `Nx.BitStringDevice`, which means the tensor is allocated as a
  bitstring within the Erlang VM.

  Most operations in the `Nx` module require the tensor to be
  allocated within the VM but, most often, when running `defn`
  functions that on the GPU, you want to keep the data on the
  GPU as much as possible. For example:

      @defn_compiler {Exla, platform: :host, keep_on_device: true}
      defn softmax(t) do
        Nx.exp(t) / Nx.sum(Nx.exp(t))
      end

  You can explicitly transfer data to a certain device or transfer
  it back as a binary by calling `device_transfer/3`. You can also
  call `device_read/1` to read the data from the binary, without
  deallocating it, and then explicitly call `device_deallocate/1`
  to deallocate it.

  To implement your own device, check the `Nx.Device` behaviour.
  """

  alias Nx.Tensor, as: T
  import Kernel, except: [max: 2, min: 2]

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

  ## Creation API

  @doc """
  Builds a tensor.

  The argument is either a number, which means the tensor is a scalar
  (zero-dimentions), a list of those (the tensor is a vector) or
  a list of n-lists of those, leading to n-dimensional tensors.

  You can also give a tensor as argument, which is just returned as
  is.

  ## Examples

  A number returns a tensor of zero dimensions:

      iex> t = Nx.tensor(0)
      iex> Nx.to_bitstring(t)
      <<0::64-native>>
      iex> Nx.type(t)
      {:s, 64}
      iex> Nx.shape(t)
      {}

      iex> t = Nx.tensor(1.0)
      iex> Nx.to_bitstring(t)
      <<1::float-64-native>>
      iex> Nx.type(t)
      {:f, 64}
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

      iex> t = Nx.tensor([1, 2, 3.0])
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

  Given a tensor to `tensor/2` returns the tensor itself:

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.tensor(t) == t
      true

  However, if a :type is given and they don't match, an error is
  raised:

      iex> Nx.tensor(Nx.tensor([1, 2, 3]), type: {:f, 64})
      ** (ArgumentError) got a tensor with type {:f, 64} but tensor has type {:s, 64}

  ## Options

    * `:type` - sets the type of the tensor. If one is not given,
      one is automatically inferred based on the input. See `Nx.Type`
      and `Nx.Type.infer/1` for more information on types. If a
      tensor is given alongside this option, then it verifies the
      tensor matches the given `:type`

  """
  def tensor(arg, opts \\ [])

  def tensor(%T{} = t, opts) do
    type = opts[:type]

    if type && type != t.type do
      raise ArgumentError,
            "got a tensor with type #{inspect(type)} but tensor has type #{inspect(t.type)}"
    end

    t
  end

  def tensor(arg, opts) do
    type = opts[:type] || Nx.Type.infer(arg)
    Nx.Type.validate!(type)
    {dimensions, data} = flatten(arg, type)
    %T{shape: dimensions, type: type, data: {Nx.BitStringDevice, data}}
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

  defp scalar_to_binary(value, type) do
    match_types([type], do: <<write!(value, 0)>>)
  end

  @doc """
  Creates a tensor from a `bitstring`, its `type`, and
  its `shape`.

  If the bitstring size does not match its type and shape,
  an error is raised.

  ## Examples

      iex> Nx.from_bitstring(<<1, 2, 3, 4>>, {:s, 8}, {4})
      Nx.tensor([1, 2, 3, 4], type: {:s, 8})

      iex> Nx.from_bitstring(<<1, 2, 3, 4>>, {:s, 8}, {2, 2})
      Nx.tensor([[1, 2], [3, 4]], type: {:s, 8})

      iex> Nx.from_bitstring(<<>>, {:s, 8}, {0})
      Nx.tensor([], type: {:s, 8})

      iex> Nx.from_bitstring(<<12.3::float-64-native>>, {:f, 64}, {})
      Nx.tensor(12.3)

      iex> Nx.from_bitstring(<<1, 2, 3, 4>>, {:f, 64}, {4})
      ** (ArgumentError) bitstring does not match the given type and dimensions

  """
  def from_bitstring(bitstring, type, shape) when is_bitstring(bitstring) and is_tuple(shape) do
    {_, size} = Nx.Type.validate!(type)

    if bit_size(bitstring) != size * tuple_product(shape) do
      raise ArgumentError, "bitstring does not match the given type and dimensions"
    end

    %T{data: {Nx.BitStringDevice, bitstring}, type: type, shape: shape}
  end

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  ## Reflection

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
  def to_bitstring(%T{} = t), do: data!(t)

  ## Device API

  @doc """
  Transfers data to the given device.

  If a device is not given, `Nx.BitStringDevice` is used, which means
  the data is read into an Elixir bitstring. If the device is already
  `Nx.BitStringDevice`, it returns the tensor as is.

  If a separate device is given, the data will be moved to the new
  device. Once transfer is done, the data is deallocated from the
  current tensor device. If the device has already been deallocated,
  it raises.

  At the moment, you can only transfer data from `Nx.BitStringDevice`
  to other devices and vice-versa but not between ad-hoc devices.

  ## Examples

  Move a tensor to a device:

      device_tensor = Nx.device_transfer(tensor, Exla.NxDevice, client: :cuda)

  Read the device tensor back to an Elixir bitstring:

      tensor = Nx.device_transfer(tensor)

  """
  def device_transfer(tensor, device \\ Nx.BitStringDevice, opts \\ [])

  def device_transfer(%T{data: {Nx.BitStringDevice, _data}} = t, device, opts) do
    %{type: type, shape: shape} = t
    %{t | data: device.allocate(data!(t), type, shape, opts)}
  end

  def device_transfer(%T{} = t, Nx.BitStringDevice, _opts) do
    new_t = device_read(t)
    _ = device_deallocate(t)
    new_t
  end

  def device_transfer(%T{data: {data_device, _}}, device, _opts) do
    raise ArgumentError, "cannot transfer from #{inspect(data_device)} to #{inspect(device)}"
  end

  @doc """
  Reads data allocated in a device.

  It returns a tensor where the device is `Nx.BitStringDevice`.
  The data is not deallocated from the current device. If the
  device has already been deallocated, it raises.
  """
  def device_read(%T{data: {device, state}} = t) do
    %{t | data: {Nx.BitStringDevice, device.read(state)}}
  end

  @doc """
  Deallocates data in a device.

  It returns either `:ok` or `:already_deallocated`.
  """
  def device_deallocate(%T{data: {device, state}} = _tensor), do: device.deallocate(state)

  ## Arith binary ops

  def_binary_op = fn name, op, cast ->
    cast = cast.(Macro.var(:output_type, nil))

    def unquote(name)(left, right)

    def unquote(name)(left, right) when is_number(left) and is_number(right) do
      tensor(unquote(op)(left, right))
    end

    def unquote(name)(scalar, %T{type: input_type} = t) when is_number(scalar) do
      data = data!(t)
      output_type = Nx.Type.merge_scalar(input_type, scalar)
      output_type = unquote(cast)

      data =
        match_types [input_type, output_type] do
          for <<match!(seg, 0) <- data>>, into: <<>> do
            <<write!(unquote(op)(scalar, read!(seg, 0)), 1)>>
          end
        end

      %{t | data: {Nx.BitStringDevice, data}, type: output_type}
    end

    def unquote(name)(%T{type: input_type} = t, scalar) when is_number(scalar) do
      data = data!(t)
      output_type = Nx.Type.merge_scalar(input_type, scalar)
      output_type = unquote(cast)

      data =
        match_types [input_type, output_type] do
          for <<match!(seg, 0) <- data>>, into: <<>> do
            <<write!(unquote(op)(read!(seg, 0), scalar), 1)>>
          end
        end

      %{t | data: {Nx.BitStringDevice, data}, type: output_type}
    end

    def unquote(name)(%T{type: left_type} = left, %T{type: right_type} = right) do
      output_type = Nx.Type.merge(left_type, right_type)
      output_type = unquote(cast)

      {data, shape} =
        match_types [left_type, right_type, output_type] do
          broadcast(left, right, fn left_dimension, right_dimension ->
            for <<match!(left_seg, 0) <- left_dimension>>,
                <<match!(right_seg, 1) <- right_dimension>>,
                into: <<>> do
              <<write!(unquote(op)(read!(left_seg, 0), read!(right_seg, 1)), 2)>>
            end
          end)
        end

      %T{data: {Nx.BitStringDevice, data}, type: output_type, shape: shape}
    end
  end

  @doc """
  Element-wise addition of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

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
  def_binary_op.(:add, :+, & &1)

  @doc """
  Element-wise subtraction of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Subtracting scalars

      iex> t = Nx.subtract(1, 2)
      iex> Nx.to_bitstring(t)
      <<-1::64-native>>

  ### Subtracting tensors and scalars

      iex> t = Nx.subtract(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<0::64-native, 1::64-native, 2::64-native>>

      iex> t = Nx.subtract(1, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<0.0::float-64-native, -1.0::float-64-native, -2.0::float-64-native>>

  ### Subtracting tensors

      iex> t = Nx.subtract(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      iex> Nx.to_bitstring(t)
      <<-9::64-native, -19::64-native, -8::64-native, -18::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.subtract(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<-9::8-native, -19::8-native, -8::8-native, -18::8-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.subtract(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<-9.0::float-32-native, -19.0::float-32-native, -8.0::float-32-native, -18.0::float-32-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def_binary_op.(:subtract, :-, & &1)

  @doc """
  Element-wise multiplication of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Multiplying scalars

      iex> t = Nx.multiply(1, 2)
      iex> Nx.to_bitstring(t)
      <<2::64-native>>

  ### Multiplying tensors and scalars

      iex> t = Nx.multiply(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<1::64-native, 2::64-native, 3::64-native>>

      iex> t = Nx.multiply(1, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<1.0::float-64-native, 2.0::float-64-native, 3.0::float-64-native>>

  ### Multiplying tensors

      iex> t = Nx.multiply(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      iex> Nx.to_bitstring(t)
      <<10::64-native, 20::64-native, 20::64-native, 40::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.multiply(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<10::8-native, 20::8-native, 20::8-native, 40::8-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.multiply(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<10.0::float-32-native, 20.0::float-32-native, 20.0::float-32-native, 40.0::float-32-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def_binary_op.(:multiply, :*, & &1)

  @doc """
  Element-wise remainder of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Remainder of scalars

      iex> t = Nx.remainder(1, 2)
      iex> Nx.to_bitstring(t)
      <<1::64-native>>

  ### Remainder of tensors and scalars

      iex> t = Nx.remainder(Nx.tensor([1, 2, 3]), 2)
      iex> Nx.to_bitstring(t)
      <<1::64-native, 0::64-native, 1::64-native>>

      iex> t = Nx.remainder(2, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<0.0::float-64-native, 0.0::float-64-native, 2.0::float-64-native>>

  ### Remainder of tensors

      iex> t = Nx.remainder(Nx.tensor([[10], [20]]), Nx.tensor([[3, 4]]))
      iex> Nx.to_bitstring(t)
      <<1::64-native, 2::64-native, 2::64-native, 0::64-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def_binary_op.(:remainder, :erlang_remainder, & &1)
  @compile {:inline, erlang_remainder: 2}
  defp erlang_remainder(a, b) when is_integer(a) and is_integer(b), do: rem(a, b)
  defp erlang_remainder(a, b), do: :math.fmod(a, b)

  @doc """
  Element-wise division of two tensors.

  If a number is given, it is converted to a tensor.

  It always returns a float tensor. If any of the input
  tensors are not float, they are converted to f64.
  Division by zero raises, but it may trigger undefined
  behaviour on some compilers.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Dividing scalars

      iex> t = Nx.divide(1, 2)
      iex> Nx.to_bitstring(t)
      <<0.5::float-64-native>>

  ### Dividing tensors and scalars

      iex> t = Nx.divide(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<1.0::float-64-native, 2.0::float-64-native, 3.0::64-native>>

      iex> t = Nx.divide(1, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<1.0::float-64-native, 0.5::float-64-native, (1/3)::float-64-native>>

  ### Dividing tensors

      iex> t = Nx.divide(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      iex> Nx.to_bitstring(t)
      <<0.1::float-64-native, 0.05::float-64-native, 0.2::float-64-native, 0.1::float-64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.divide(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<0.1::float-32-native, 0.05::float-32-native, 0.2::float-32-native, 0.1::float-32-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.divide(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<0.1::float-32-native, 0.05::float-32-native, 0.2::float-32-native, 0.1::float-32-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def_binary_op.(:divide, :/, &quote(do: Nx.Type.to_floating(unquote(&1))))

  @doc """
  Element-wise maximum of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Max between scalars

      iex> t = Nx.max(1, 2)
      iex> Nx.to_bitstring(t)
      <<2::64-native>>

  ### Max between tensors and scalars

      iex> t = Nx.max(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<1::64-native, 2::64-native, 3::64-native>>

      iex> t = Nx.max(1, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<1.0::float-64-native, 2.0::float-64-native, 3.0::float-64-native>>

  ### Max between tensors

      iex> t = Nx.max(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      iex> Nx.to_bitstring(t)
      <<10::64-native, 20::64-native, 10::64-native, 20::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.max(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<10::8-native, 20::8-native, 10::8-native, 20::8-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.max(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<10.0::float-32-native, 20.0::float-32-native, 10.0::float-32-native, 20.0::float-32-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def_binary_op.(:max, :erlang_max, & &1)
  @compile {:inline, erlang_max: 2}
  defp erlang_max(a, b), do: :erlang.max(a, b)

  @doc """
  Element-wise minimum of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Min between scalars

      iex> t = Nx.min(1, 2)
      iex> Nx.to_bitstring(t)
      <<1::64-native>>

  ### Min between tensors and scalars

      iex> t = Nx.min(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<1::64-native, 1::64-native, 1::64-native>>

      iex> t = Nx.min(1, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<1.0::float-64-native, 1.0::float-64-native, 1.0::float-64-native>>

  ### Min between tensors

      iex> t = Nx.min(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      iex> Nx.to_bitstring(t)
      <<1::64-native, 1::64-native, 2::64-native, 2::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.min(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<1::8-native, 1::8-native, 2::8-native, 2::8-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.min(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<1.0::float-32-native, 1.0::float-32-native, 2.0::float-32-native, 2.0::float-32-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def_binary_op.(:min, :erlang_min, & &1)
  @compile {:inline, erlang_min: 2}
  defp erlang_min(a, b), do: :erlang.min(a, b)

  ## Bitwise ops

  def_binary_bitwise_op = fn name, op ->
    def_binary_op.(name, op, &quote(do: assert_integer_type!(unquote(&1), unquote(name))))
  end

  defp assert_integer_type!({:s, _} = type, _op), do: type
  defp assert_integer_type!({:u, _} = type, _op), do: type

  defp assert_integer_type!(type, op) do
    raise ArgumentError,
          "#{op} expects integer tensors as inputs and outputs an integer tensor, " <>
            "got: #{inspect(type)}"
  end

  @doc """
  Element-wise bitwise AND of two tensors.

  Only integer tensors are supported. If a float or
  complex tensor is given, an error is raised.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### bitwise and between scalars

      iex> t = Nx.bitwise_and(1, 0)
      iex> Nx.to_bitstring(t)
      <<0::64-native>>

  ### bitwise and between tensors and scalars

      iex> t = Nx.bitwise_and(Nx.tensor([0, 1, 2]), 1)
      iex> Nx.to_bitstring(t)
      <<0::64-native, 1::64-native, 0::64-native>>

      iex> t = Nx.bitwise_and(Nx.tensor([0, -1, -2]), -1)
      iex> Nx.to_bitstring(t)
      <<0::64-native, -1::64-native, -2::64-native>>

  ### bitwise and between tensors

      iex> t = Nx.bitwise_and(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0, 1, 0, 1]))
      iex> Nx.to_bitstring(t)
      <<0::64-native, 0::64-native, 0::64-native, 1::64-native>>

  ### Error cases

      iex> Nx.bitwise_and(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise_and expects integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def_binary_bitwise_op.(:bitwise_and, :erlang_band)
  @compile {:inline, erlang_band: 2}
  defp erlang_band(a, b), do: :erlang.band(a, b)

  @doc """
  Element-wise bitwise OR of two tensors.

  Only integer tensors are supported. If a float or
  complex tensor is given, an error is raised.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### bitwise or between scalars

      iex> t = Nx.bitwise_or(1, 0)
      iex> Nx.to_bitstring(t)
      <<1::64-native>>

  ### bitwise or between tensors and scalars

      iex> t = Nx.bitwise_or(Nx.tensor([0, 1, 2]), 1)
      iex> Nx.to_bitstring(t)
      <<1::64-native, 1::64-native, 3::64-native>>

      iex> t = Nx.bitwise_or(Nx.tensor([0, -1, -2]), -1)
      iex> Nx.to_bitstring(t)
      <<-1::64-native, -1::64-native, -1::64-native>>

  ### bitwise or between tensors

      iex> t = Nx.bitwise_or(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0, 1, 0, 1]))
      iex> Nx.to_bitstring(t)
      <<0::64-native, 1::64-native, 1::64-native, 1::64-native>>

  ### Error cases

      iex> Nx.bitwise_or(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise_or expects integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def_binary_bitwise_op.(:bitwise_or, :erlang_bor)
  @compile {:inline, erlang_bor: 2}
  defp erlang_bor(a, b), do: :erlang.bor(a, b)

  @doc """
  Element-wise bitwise XOR of two tensors.

  Only integer tensors are supported. If a float or complex
  tensor is given, an error is raised.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Bitwise xor between scalars

      iex> t = Nx.bitwise_xor(1, 0)
      iex> Nx.to_bitstring(t)
      <<1::64-native>>

  ### Bitwise xor and between tensors and scalars

      iex> t = Nx.bitwise_xor(Nx.tensor([1, 2, 3]), 2)
      iex> Nx.to_bitstring(t)
      <<3::64-native, 0::64-native, 1::64-native>>

      iex> t = Nx.bitwise_xor(Nx.tensor([-1, -2, -3]), 2)
      iex> Nx.to_bitstring(t)
      <<-3::64-native, -4::64-native, -1::64-native>>

  ### Bitwise xor between tensors

      iex> t = Nx.bitwise_xor(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0, 1, 0, 1]))
      iex> Nx.to_bitstring(t)
      <<0::64-native, 1::64-native, 1::64-native, 0::64-native>>

  ### Error cases

      iex> Nx.bitwise_xor(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise_xor expects integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def_binary_bitwise_op.(:bitwise_xor, :erlang_bxor)
  @compile {:inline, erlang_bxor: 2}
  defp erlang_bxor(a, b), do: :erlang.bxor(a, b)

  @doc """
  Element-wise left shift of two tensors.

  Only integer tensors are supported. If a float or complex
  tensor is given, an error is raised. If the right side
  is negative, it will raise, but it may trigger undefined
  behaviour on some compilers.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Left shift between scalars

      iex> t = Nx.left_shift(1, 0)
      iex> Nx.to_bitstring(t)
      <<1::64-native>>

  ### Left shift between tensors and scalars

      iex> t = Nx.left_shift(Nx.tensor([1, 2, 3]), 2)
      iex> Nx.to_bitstring(t)
      <<4::64-native, 8::64-native, 12::64-native>>

  ### Left shift between tensors

      iex> t = Nx.left_shift(Nx.tensor([1, 1, -1, -1]), Nx.tensor([1, 2, 3, 4]))
      iex> Nx.to_bitstring(t)
      <<2::64-native, 4::64-native, -8::64-native, -16::64-native>>

  ### Error cases

      iex> Nx.left_shift(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) left_shift expects integer tensors as inputs and outputs an integer tensor, got: {:f, 64}

      iex> Nx.left_shift(Nx.tensor(1), -1)
      ** (ArgumentError) cannot left shift by -1
  """
  def_binary_bitwise_op.(:left_shift, :erlang_bsl)
  @compile {:inline, erlang_bsl: 2}
  defp erlang_bsl(a, b) when b >= 0, do: :erlang.bsl(a, b)
  defp erlang_bsl(_, b), do: raise(ArgumentError, "cannot left shift by #{b}")

  @doc """
  Element-wise right shift of two tensors.

  Only integer tensors are supported. If a float or complex
  tensor is given, an error is raised. If the right side
  is negative, it will raise, but it may trigger undefined
  behaviour on some compilers.

  It performs an arithmetic shift if the tensor is made of
  signed integers, it performs a logical shift otherwise.
  In other words, it preserves the sign for signed integers.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Right shift between scalars

      iex> t = Nx.right_shift(1, 0)
      iex> Nx.to_bitstring(t)
      <<1::64-native>>

  ### Right shift between tensors and scalars

      iex> t = Nx.right_shift(Nx.tensor([2, 4, 8]), 2)
      iex> Nx.to_bitstring(t)
      <<0::64-native, 1::64-native, 2::64-native>>

  ### Right shift between tensors

      iex> t = Nx.right_shift(Nx.tensor([16, 32, -64, -128]), Nx.tensor([1, 2, 3, 4]))
      iex> Nx.to_bitstring(t)
      <<8::64-native, 8::64-native, -8::64-native, -8::64-native>>

  ### Error cases

      iex> Nx.right_shift(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) right_shift expects integer tensors as inputs and outputs an integer tensor, got: {:f, 64}

      iex> Nx.right_shift(Nx.tensor(1), -1)
      ** (ArgumentError) cannot right shift by -1
  """
  def_binary_bitwise_op.(:right_shift, :erlang_bsr)
  @compile {:inline, erlang_bsr: 2}
  defp erlang_bsr(a, b) when b >= 0, do: :erlang.bsr(a, b)
  defp erlang_bsr(_, b), do: raise(ArgumentError, "cannot right shift by #{b}")

  ## Unary ops

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

      iex> t = Nx.exp(Nx.tensor([1, 2, 3], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<2.718281828459045::float-32-native, 7.38905609893065::float-32-native, 20.085536923187668::float-32-native>>

      iex> t = Nx.exp(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<2.718281828459045::float-native-32, 7.38905609893065::float-native-32, 20.085536923187668::float-native-32>>

      iex> t = Nx.exp(Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16}))
      iex> Nx.to_bitstring(t)
      <<16429::16-native, 16620::16-native, 16800::16-native>>

  """
  def exp(number)

  def exp(number) when is_number(number), do: tensor(:math.exp(number))

  def exp(%T{type: input_type} = t) do
    data = data!(t)
    output_type = Nx.Type.to_floating(input_type)

    data =
      match_types [input_type, output_type] do
        for <<match!(seg, 0) <- data>>, into: <<>>, do: <<write!(:math.exp(read!(seg, 0)), 1)>>
      end

    %{t | data: {Nx.BitStringDevice, data}, type: output_type}
  end

  ## Aggregate ops

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
  def sum(%T{type: type} = t) do
    data =
      match_types [type] do
        value =
          bin_reduce_all(data!(t), 0, fn <<match!(var, 0), rest::bitstring>>, acc ->
            {read!(var, 0) + acc, rest}
          end)

        <<write!(value, 0)>>
      end

    %{t | data: {Nx.BitStringDevice, data}, shape: {}}
  end

  ## Random Ops

  @doc """
  Returns a uniformly-distributed random tensor with the given shape.

  By default, distribution is bounded between `0.0` and
  `1.0`. Return type is one of `{:f, size}`, `{:s, size}`
  or `{:u, size}`.

  ## Examples

      iex> tensors = for i <- 1..100, do: Nx.random_uniform({i})
      iex> for t <- tensors, do: for <<x::float-64-native <- Nx.to_bitstring(t)>>, do: true = x >= 0.0 and x <= 1.0
      iex> for t <- tensors, do: Nx.shape(t)
      for i <- 1..100, do: {i}
      iex> for t <- tensors, do: Nx.type(t)
      for _ <- 1..100, do: {:f, 64}
      iex> tensors = for i <- 1..100, do: Nx.random_uniform({i, i}, 10, 20, type: {:s, 32})
      iex> for t <- tensors, do: for <<x::32-native <- Nx.to_bitstring(t)>>, do: true = x >= 10 and x <= 20
      iex> for t <- tensors, do: Nx.shape(t)
      for i <- 1..100, do: {i, i}
      iex> for t <- tensors, do: Nx.type(t)
      for _ <- 1..100, do: {:s, 32}
      iex> tensors = for _ <- 1..100, do: Nx.random_uniform({}, 0, 5, type: {:u, 64})
      iex> for t <- tensors, do: for <<x::64-unsigned-native <- Nx.to_bitstring(t)>>, do: true = x >= 0 and x <= 5
      iex> for t <- tensors, do: Nx.shape(t)
      for _ <- 1..100, do: {}
      iex> for t <- tensors, do: Nx.type(t)
      for _ <- 1..100, do: {:u, 64}
  """
  def random_uniform(shape, opts \\ []) when is_tuple(shape), do: random_uniform(shape, 0.0, 1.0, opts)

  def random_uniform(shape, min, max, opts \\ []) when is_tuple(shape) and is_number(min) and is_number(max) do
    type = opts[:type] || Nx.Type.infer(max - min)
    gen =
      case type do
        {:f, _} -> fn -> (max - min) * :rand.uniform() + min end
        {:s, _} -> fn -> max - (:rand.uniform(max - min)) end
        {:u, _} -> fn -> max - (:rand.uniform(max - min)) end
      end
    data = for _ <- 1..tuple_product(shape), into: "", do: scalar_to_binary(gen.(), type)
    %T{data: {Nx.BitStringDevice, data}, shape: shape, type: type}
  end

  @doc """
  Returns a normally-distributed random tensor with the given shape.

  By default, distribution has mean of `0.0` and
  standard deviation of `1.0`. Return type is one of
  `{:f, 32}` or `{:f, 64}`.

  ## Examples

      iex> tensors = for i <- 1..100, do: Nx.random_normal({i})
      iex> for t <- tensors, do: Nx.shape(t)
      for i <- 1..100, do: {i}
      iex> for t <- tensors, do: Nx.type(t)
      for _ <- 1..100, do: {:f, 64}
      iex> tensors = for i <- 1..100, do: Nx.random_normal({i, i}, type: {:f, 32})
      iex> for t <- tensors, do: Nx.shape(t)
      for i <- 1..100, do: {i, i}
      iex> for t <- tensors, do: Nx.type(t)
      for _ <- 1..100, do: {:f, 32}
  """
  def random_normal(shape, opts \\ []), do: random_normal(shape, 0.0, 1.0, opts)

  def random_normal(shape, mu, sigma, opts \\ []) when is_tuple(shape) when is_number(mu) and is_number(sigma) do
    type = {:f, _} = opts[:type] || {:f, 64}
    data = for _ <- 1..tuple_product(shape), into: "", do: scalar_to_binary(:rand.normal(mu, sigma), type)
    %T{data: {Nx.BitStringDevice, data}, shape: shape, type: type}
  end

  ## Device helpers

  defp data!(%T{data: {Nx.BitStringDevice, data}}), do: data

  defp data!(%T{data: {device, _data}}) do
    raise ArgumentError,
          "cannot read Nx.Tensor data because the data is allocated on device #{inspect(device)}. " <>
            "Please use Nx.device_transfer/1 to transfer data back to Elixir"
  end

  ## Broadcast helpers

  defp broadcast(
         %T{type: {_, left_size}, shape: shape} = left,
         %T{type: {_, right_size}, shape: shape} = right,
         fun
       ) do
    data = bin_zip_map_all(data!(left), left_size, data!(right), right_size, fun)
    {IO.iodata_to_binary(data), shape}
  end

  defp broadcast(
         %T{type: {_, left_size}, shape: left_shape} = left,
         %T{type: {_, right_size}, shape: right_shape} = right,
         fun
       ) do
    left_rank = tuple_size(left_shape)
    right_rank = tuple_size(right_shape)
    rank = :erlang.max(left_rank, right_rank)
    left_ordered = shape_to_ranked_ordered_list(left_shape, left_rank, rank)
    right_ordered = shape_to_ranked_ordered_list(right_shape, right_rank, rank)

    case broadcast_chunks(left_ordered, right_ordered, left_size, right_size, [fun], []) do
      {chunks, shape} ->
        {broadcast_recur(data!(left), data!(right), chunks), shape}

      :error ->
        raise ArgumentError,
              "cannot broadcast tensor of dimensions #{inspect(left_shape)} " <>
                "to #{inspect(right_shape)}"
    end
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

    # This is an optimization, we skip cross traversals on the left-side
    # if we are just before a previous cross traversal. If broadcasting is
    # failing, remove the if branch and see if it succeeds. :)
    chunks =
      if dir == :left and match?([{:cross, _, _} | _], chunks) do
        chunks
      else
        [{:cross, left_size, right_size} | chunks]
      end

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
    do: broadcast_split_chunks(lt, rt, ls * lh, rh * rs, n - 1, [:erlang.max(lh, rh) | shape])

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

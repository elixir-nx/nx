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
  import Bitwise, only: [>>>: 2, &&&: 2]

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

  @doc """
  Shortcut for `random_uniform(shape, 0.0, 1.0, opts)`.
  """
  def random_uniform(shape, opts \\ []) when is_tuple(shape), do: random_uniform(shape, 0.0, 1.0, opts)

  @doc """
  Returns a uniformly-distributed random tensor with the given shape.

  The distribution is bounded on the semi-open interval `[min, max)`.
  Return type is one of `{:f, size}`, `{:bf, size}`, `{:u, size}`,
  `{:s, size}`.

  ## Examples

  ### Generating Floats

      iex> t = Nx.random_uniform({10})
      iex> for <<x::float-64-native <- Nx.to_bitstring(t)>> do
      ...>   true = x >= 0.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:f, 64}

      iex> t = Nx.random_uniform({5, 5}, type: {:bf, 16})
      iex> byte_size(Nx.to_bitstring(t))
      50
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:bf, 16}

      iex> t = Nx.random_uniform({5, 5}, -1.0, 1.0, type: {:f, 32})
      iex> for <<x::float-32-native <- Nx.to_bitstring(t)>> do
      ...>   true = x >= -1.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:f, 32}

  ### Generating Integers

      iex> t = Nx.random_uniform({10}, 5, 10, type: {:u, 32})
      iex> for <<x::32-unsigned-native <- Nx.to_bitstring(t)>> do
      ...>   true = x >= 5 and x < 10
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:u, 32}

      iex> t = Nx.random_uniform({5, 5}, -5, 5, type: {:s, 64})
      iex> for <<x::64-signed-native <- Nx.to_bitstring(t)>> do
      ...>   true = x >= -5 and x < 5
      ...> end
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:s, 64}
  """
  def random_uniform(shape, min, max, opts \\ []) when is_tuple(shape) and is_number(min) and is_number(max) do
    type = opts[:type] || Nx.Type.infer(max - min)
    gen =
      case type do
        {:s, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {:u, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {_, _} -> fn -> (max - min) * :rand.uniform() + min end
      end
    data = for _ <- 1..tuple_product(shape), into: "", do: scalar_to_binary(gen.(), type)
    %T{data: {Nx.BitStringDevice, data}, shape: shape, type: type}
  end

  @doc """
  Shortcut for `random_normal(shape, 0.0, 1.0, opts)`.
  """
  def random_normal(shape, opts \\ []), do: random_normal(shape, 0.0, 1.0, opts)

  @doc """
  Returns a normally-distributed random tensor with the given shape.

  The distribution has mean of `mu` and standard deviation of
  `sigma`. Return type is one of `{:bf, 16}`, `{:f, 32}` or `{:f, 64}`.

  ## Examples

      iex> t = Nx.random_normal({10})
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:f, 64}

      iex> t = Nx.random_normal({5, 5}, 2.0, 1.0, type: {:bf, 16})
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:bf, 16}

      iex> t = Nx.random_normal({3, 3, 3}, -1.0, 1.0, type: {:f, 32})
      iex> Nx.shape(t)
      {3, 3, 3}
      iex> Nx.type(t)
      {:f, 32}
  """
  def random_normal(shape, mu, sigma, opts \\ []) when is_tuple(shape) when is_number(mu) and is_number(sigma) do
    type = opts[:type] || {:f, 64}
    data = for _ <- 1..tuple_product(shape), into: "", do: scalar_to_binary(:rand.normal(mu, sigma), type)
    %T{data: {Nx.BitStringDevice, data}, shape: shape, type: type}
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

  ## Element-wise binary ops

  defp_element_wise_bin_op = fn name, cast ->
    cast = cast.(Macro.var(:output_type, nil))

    defp unquote(name)(left, right, fun) when is_number(left) and is_number(right) do
      output_type = if is_float(left) or is_float(right), do: {:f, 64}, else: {:s, 64}
      output_type = unquote(cast)
      data = scalar_to_binary(fun.(output_type, left, right), output_type)
      %T{data: {Nx.BitStringDevice, data}, type: output_type, shape: {}}
    end

    defp unquote(name)(scalar, %T{type: input_type} = t, fun) when is_number(scalar) do
      data = data!(t)
      output_type = Nx.Type.merge_scalar(input_type, scalar)
      output_type = unquote(cast)

      data =
        match_types [input_type, output_type] do
          for <<match!(seg, 0) <- data>>, into: <<>> do
            <<write!(fun.(output_type, scalar, read!(seg, 0)), 1)>>
          end
        end

      %{t | data: {Nx.BitStringDevice, data}, type: output_type}
    end

    defp unquote(name)(%T{type: input_type} = t, scalar, fun) when is_number(scalar) do
      data = data!(t)
      output_type = Nx.Type.merge_scalar(input_type, scalar)
      output_type = unquote(cast)

      data =
        match_types [input_type, output_type] do
          for <<match!(seg, 0) <- data>>, into: <<>> do
            <<write!(fun.(output_type, read!(seg, 0), scalar), 1)>>
          end
        end

      %{t | data: {Nx.BitStringDevice, data}, type: output_type}
    end

    defp unquote(name)(%T{type: left_type} = left, %T{type: right_type} = right, fun) do
      output_type = Nx.Type.merge(left_type, right_type)
      output_type = unquote(cast)

      {data, shape} =
        match_types [left_type, right_type, output_type] do
          broadcast(left, right, fn left_dimension, right_dimension ->
            for <<match!(left_seg, 0) <- left_dimension>>,
                <<match!(right_seg, 1) <- right_dimension>>,
                into: <<>> do
              <<write!(fun.(output_type, read!(left_seg, 0), read!(right_seg, 1)), 2)>>
            end
          end)
        end

      %T{data: {Nx.BitStringDevice, data}, type: output_type, shape: shape}
    end
  end

  defp_element_wise_bin_op.(
    :element_wise_bin_arith,
    & &1
  )

  defp_element_wise_bin_op.(
    :element_wise_bin_float_arith,
    &quote(do: Nx.Type.to_floating(unquote(&1)))
  )

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
  def add(left, right), do: element_wise_bin_arith(left, right, &erlang_add/3)
  @compile {:inline, erlang_add: 3}
  defp erlang_add(_, a, b), do: a + b

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
  def subtract(left, right), do: element_wise_bin_arith(left, right, &erlang_subtract/3)
  @compile {:inline, erlang_subtract: 3}
  defp erlang_subtract(_, a, b), do: a - b

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
  def multiply(left, right), do: element_wise_bin_arith(left, right, &erlang_multiply/3)
  @compile {:inline, erlang_multiply: 3}
  defp erlang_multiply(_, a, b), do: a * b

  @doc """
  Element-wise power of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Power of scalars

      iex> t = Nx.power(2, 4)
      iex> Nx.to_bitstring(t)
      <<16::64-native>>

  ### Power of tensors and scalars

      iex> t = Nx.power(Nx.tensor([1, 2, 3]), 2)
      iex> Nx.to_bitstring(t)
      <<1::64-native, 4::64-native, 9::64-native>>

      iex> t = Nx.power(2, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<2.0::float-64-native, 4.0::float-64-native, 8.0::float-64-native>>

  ### Power of tensors

      iex> t = Nx.power(Nx.tensor([[2], [3]]), Nx.tensor([[4, 5]]))
      iex> Nx.to_bitstring(t)
      <<16::64-native, 32::64-native, 81::64-native, 243::64-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def power(left, right), do: element_wise_bin_arith(left, right, &erlang_power/3)
  @compile {:inline, erlang_remainder: 3}
  defp erlang_power({type, _}, a, b) when type in [:s, :u], do: integer_pow(a, b)
  defp erlang_power(_, a, b), do: :math.pow(a, b)

  # TODO: Use Integer.pow on Elixir v1.12
  defp integer_pow(base, exponent) when is_integer(base) and is_integer(exponent) do
    if exponent < 0, do: :erlang.error(:badarith, [base, exponent])
    guarded_pow(base, exponent)
  end

  defp guarded_pow(_, 0), do: 1
  defp guarded_pow(b, 1), do: b
  defp guarded_pow(b, e) when (e &&& 1) == 0, do: guarded_pow(b * b, e >>> 1)
  defp guarded_pow(b, e), do: b * guarded_pow(b * b, e >>> 1)

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
  def remainder(left, right), do: element_wise_bin_arith(left, right, &erlang_remainder/3)
  @compile {:inline, erlang_remainder: 3}
  defp erlang_remainder(_, a, b) when is_integer(a) and is_integer(b), do: rem(a, b)
  defp erlang_remainder(_, a, b), do: :math.fmod(a, b)

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
  def divide(left, right), do: element_wise_bin_float_arith(left, right, &erlang_divide/3)
  @compile {:inline, erlang_divide: 3}
  defp erlang_divide(_, a, b), do: a / b

  @doc """
  Element-wise arc tangent of two tensors.

  If a number is given, it is converted to a tensor.

  It always returns a float tensor. If any of the input
  tensors are not float, they are converted to f64.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Arc tangent between scalars

      iex> t = Nx.arctan2(1, 2)
      iex> Nx.to_bitstring(t)
      <<0.4636476090008061::float-64-native>>

  ### Arc tangent between tensors and scalars

      iex> t = Nx.arctan2(Nx.tensor([1, 2, 3]), 1)
      iex> Nx.to_bitstring(t)
      <<0.7853981633974483::float-64-native, 1.1071487177940904::float-64-native, 1.2490457723982544::float-64-native>>

      iex> t = Nx.arctan2(1, Nx.tensor([1.0, 2.0, 3.0]))
      iex> Nx.to_bitstring(t)
      <<0.7853981633974483::float-64-native, 0.4636476090008061::float-64-native, 0.3217505543966422::float-64-native>>

  ### Arc tangent between tensors

      # Note there is a bug in Erlang/OTP 23.0 and earlier where the compiler
      # optimizes -0.0 away as 0.0. So we do: -1.0*(Integer.parse("0")|>elem(0))
      iex> pos_and_neg_zero_x = Nx.multiply(Nx.tensor([[-1.0], [1.0]]), 0.0)
      iex> pos_and_neg_zero_y = Nx.multiply(Nx.tensor([-1.0, 1.0]), 0.0)
      iex> t = Nx.arctan2(pos_and_neg_zero_x, pos_and_neg_zero_y)
      iex> Nx.to_bitstring(t)
      <<-3.141592653589793::float-64-native, (-1.0*(Integer.parse("0")|>elem(0)))::float-64-native,
        3.141592653589793::float-64-native, 0.0::float-64-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def arctan2(left, right), do: element_wise_bin_float_arith(left, right, &erlang_arctan2/3)
  @compile {:inline, erlang_arctan2: 3}
  defp erlang_arctan2(_, a, b), do: :math.atan2(a, b)

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
  def max(left, right), do: element_wise_bin_arith(left, right, &erlang_max/3)
  @compile {:inline, erlang_max: 3}
  defp erlang_max(_, a, b), do: :erlang.max(a, b)

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
  def min(left, right), do: element_wise_bin_arith(left, right, &erlang_min/3)
  @compile {:inline, erlang_min: 3}
  defp erlang_min(_, a, b), do: :erlang.min(a, b)

  ## Bitwise ops

  defp_element_wise_bin_op.(
    :element_wise_bin_bitwise_arith,
    &quote(do: assert_bitwise_type!(unquote(&1)))
  )

  defp to_unsigned(integer, size) do
    <<integer::unsigned-size(size)>> = <<integer::signed-size(size)>>
    integer
  end

  defp assert_bitwise_type!({:s, _} = type), do: type
  defp assert_bitwise_type!({:u, _} = type), do: type

  defp assert_bitwise_type!(type) do
    raise ArgumentError,
          "bitwise operators expect integer tensors as inputs and outputs an integer tensor, " <>
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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_and(left, right),
    do: element_wise_bin_bitwise_arith(left, right, &erlang_bitwise_and/3)

  @compile {:inline, erlang_bitwise_and: 3}
  defp erlang_bitwise_and(_, a, b), do: :erlang.band(a, b)

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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_or(left, right),
    do: element_wise_bin_bitwise_arith(left, right, &erlang_bitwise_or/3)

  @compile {:inline, erlang_bitwise_or: 3}
  defp erlang_bitwise_or(_, a, b), do: :erlang.bor(a, b)

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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_xor(left, right),
    do: element_wise_bin_bitwise_arith(left, right, &erlang_bitwise_xor/3)

  @compile {:inline, erlang_bitwise_xor: 3}
  defp erlang_bitwise_xor(_, a, b), do: :erlang.bxor(a, b)

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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}

      iex> Nx.left_shift(Nx.tensor(1), -1)
      ** (ArgumentError) cannot left shift by -1
  """
  def left_shift(left, right),
    do: element_wise_bin_bitwise_arith(left, right, &erlang_left_shift/3)

  @compile {:inline, erlang_left_shift: 3}
  defp erlang_left_shift(_, a, b) when b >= 0, do: :erlang.bsl(a, b)
  defp erlang_left_shift(_, _, b), do: raise(ArgumentError, "cannot left shift by #{b}")

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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}

      iex> Nx.right_shift(Nx.tensor(1), -1)
      ** (ArgumentError) cannot right shift by -1
  """
  def right_shift(left, right),
    do: element_wise_bin_bitwise_arith(left, right, &erlang_right_shift/3)

  @compile {:inline, erlang_right_shift: 3}
  defp erlang_right_shift(_, a, b) when b >= 0, do: :erlang.bsr(a, b)
  defp erlang_right_shift(_, _, b), do: raise(ArgumentError, "cannot right shift by #{b}")

  ## Unary ops

  funs = [
    exp: {"exponential", &quote(do: :math.exp(unquote(&1)))},
    expm1: {"exponential minus one", &quote(do: :math.exp(unquote(&1)) - 1)},
    log: {"natural log (base 2)", &quote(do: :math.log(unquote(&1)))},
    log1p: {"natural log plus one", &quote(do: :math.log(unquote(&1) + 1))},
    logistic: {"standard logistic (a sigmoid)", &quote(do: 1 / (1 + :math.exp(-unquote(&1))))},
    cos: {"cosine", &quote(do: :math.cos(unquote(&1)))},
    sin: {"sine", &quote(do: :math.sin(unquote(&1)))},
    tanh: {"hyperbolic tangent", &quote(do: :math.tanh(unquote(&1)))},
    sqrt: {"square root", &quote(do: :math.sqrt(unquote(&1)))},
    rsqrt: {"reverse square root", &quote(do: 1 / :math.sqrt(unquote(&1)))},
    cbrt: {"cube root", &quote(do: :math.pow(unquote(&1), 1/3))}
  ]

  for {name, {desc, code}} <- funs do
    applied = code.(Macro.var(:x, nil))
    formula = Macro.to_string(applied)

    {one, _} = Code.eval_quoted(applied, x: 1)
    {two, _} = Code.eval_quoted(applied, x: 2)
    {three, _} = Code.eval_quoted(applied, x: 3)

    @doc """
    Calculates the #{desc} of each element in the tensor.

    It is equivalent to:

        #{formula}

    ## Examples

        iex> t = Nx.#{name}(1)
        iex> Nx.to_bitstring(t)
        <<#{one}::float-64-native>>

        iex> t = Nx.#{name}(Nx.tensor([1, 2, 3]))
        iex> Nx.to_bitstring(t)
        <<#{one}::float-64-native, #{two}::float-64-native, #{three}::float-64-native>>

        iex> t = Nx.#{name}(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
        iex> Nx.to_bitstring(t)
        <<#{one}::float-32-native, #{two}::float-32-native, #{three}::float-32-native>>

    """
    def unquote(name)(tensor), do: unary_float(tensor, fn x -> unquote(applied) end)
  end

  defp unary_float(number, fun) when is_number(number), do: tensor(fun.(number))

  defp unary_float(%T{type: input_type} = t, fun) do
    data = data!(t)
    output_type = Nx.Type.to_floating(input_type)

    data =
      match_types [input_type, output_type] do
        for <<match!(seg, 0) <- data>>, into: <<>> do
          <<write!(fun.(read!(seg, 0)), 1)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}, type: output_type}
  end

  @doc """
  Negates each element in the tensor.

  ## Examples

      iex> t = Nx.negate(1)
      iex> Nx.to_bitstring(t)
      <<-1::64-native>>

      iex> t = Nx.negate(Nx.tensor([-1, 0, 1]))
      iex> Nx.to_bitstring(t)
      <<1::64-native, 0::64-native, -1::64-native>>

      iex> t = Nx.negate(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
      iex> Nx.to_bitstring(t)
      <<-1.0::float-32-native, -2.0::float-32-native, -3.0::float-32-native>>

  If an unsigned tensor is given, it works as `bitwise_not`:

      iex> t = Nx.negate(Nx.tensor([0, 1, 2], type: {:u, 8}))
      iex> Nx.to_bitstring(t)
      <<0::8-unsigned, 255::8-unsigned, 254::8-unsigned>>

  """
  def negate(tensor)

  def negate(number) when is_number(number), do: tensor(-number)

  def negate(%T{type: input_type} = t) do
    data = data!(t)

    data =
      match_types [input_type] do
        for <<match!(seg, 0) <- data>>, into: <<>> do
          <<write!(-(read!(seg, 0)), 0)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}}
  end

  @doc """
  Computes the sign of each element in the tensor.

  If a number is less than zero, it returns -1.
  If a number is more than zero, it returns 1.
  Otherwise it returns zero (which may either be
  positive or negative for floats).

  ## Examples

      iex> t = Nx.sign(Nx.tensor([-2, -1, 0, 1, 2]))
      iex> Nx.to_bitstring(t)
      <<-1::64-native, -1::64-native, 0::64-native, 1::64-native, 1::64-native>>

  """
  def sign(tensor)

  def sign(number) when is_number(number), do: tensor(erlang_sign(number))

  def sign(%T{type: input_type} = t) do
    data = data!(t)

    data =
      match_types [input_type] do
        for <<match!(seg, 0) <- data>>, into: <<>> do
          <<write!(erlang_sign(read!(seg, 0)), 0)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}}
  end

  @compile {:inline, erlang_sign: 1}
  defp erlang_sign(n) when n < 0, do: -1
  defp erlang_sign(n) when n > 0, do: 1
  defp erlang_sign(n), do: n

  @doc """
  Computes the absolute value of each element in the tensor.

  ## Examples

      iex> t = Nx.abs(Nx.tensor([-2, -1, 0, 1, 2]))
      iex> Nx.to_bitstring(t)
      <<2::64-native, 1::64-native, 0::64-native, 1::64-native, 2::64-native>>

  """
  def abs(tensor)

  def abs(number) when is_number(number), do: tensor(:erlang.abs(number))

  def abs(%T{type: input_type} = t) do
    data = data!(t)

    data =
      match_types [input_type] do
        for <<match!(seg, 0) <- data>>, into: <<>> do
          <<write!(:erlang.abs(read!(seg, 0)), 0)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}}
  end

  @doc """
  Applies bitwise not to each element in the tensor.

  ## Examples

      iex> t = Nx.bitwise_not(1)
      iex> Nx.to_bitstring(t)
      <<-2::64-native>>

      iex> t = Nx.bitwise_not(Nx.tensor([-1, 0, 1], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<0::8, -1::8, -2::8>>

      iex> t = Nx.bitwise_not(Nx.tensor([0, 1, 254, 255], type: {:u, 8}))
      iex> Nx.to_bitstring(t)
      <<255::8-unsigned, 254::8-unsigned, 1::8-unsigned, 0::8-unsigned>>

  ### Error cases

      iex> Nx.bitwise_not(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_not(tensor)

  def bitwise_not(number) when is_integer(number), do: tensor(:erlang.bnot(number))
  def bitwise_not(number) when is_float(number), do: assert_bitwise_type!({:f, 64})

  def bitwise_not(%T{type: input_type} = t) do
    assert_bitwise_type!(input_type)
    data = data!(t)

    data =
      match_types [input_type] do
        for <<match!(seg, 0) <- data>>, into: <<>> do
          <<write!(:erlang.bnot(read!(seg, 0)), 0)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}}
  end

  @doc """
  Computes the bitwise population count of each element in the tensor.

  ## Examples

      iex> t = Nx.population_count(1)
      iex> Nx.to_bitstring(t)
      <<1::64-native>>

      iex> t = Nx.population_count(-128)
      iex> Nx.to_bitstring(t)
      <<57::64-native>>

      iex> t = Nx.population_count(Nx.tensor([0, 1, 254, 255]))
      iex> Nx.to_bitstring(t)
      <<0::64-native, 1::64-native, 7::64-native, 8::64-native>>

      iex> t = Nx.population_count(Nx.tensor([0, 1, 126, 127, -1, -127, -128], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<0, 1, 6, 7, 8, 2, 1>>

  ### Error cases

      iex> Nx.population_count(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def population_count(tensor)

  def population_count(number) when is_integer(number),
    do: tensor(erlang_popcount(to_unsigned(number, 64), 0))

  def population_count(number) when is_float(number),
    do: assert_bitwise_type!({:f, 64})

  def population_count(%T{type: {_, size} = input_type} = t) do
    assert_bitwise_type!(input_type)

    data =
      for <<seg::unsigned-size(size)-native <- data!(t)>>, into: <<>> do
        match_types [input_type] do
          <<write!(erlang_popcount(seg, 0), 0)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}}
  end

  # https://en.wikipedia.org/wiki/Hamming_weight
  # There are algorithms with faster worst case but they are size specific.
  # The implementation below is also the most efficient for low counts. Given
  # our integers are always 64 bits internally, we will have a lot of zeros
  # internally, so this should be the fastest.
  defp erlang_popcount(0, count), do: count
  defp erlang_popcount(n, count), do: erlang_popcount(n &&& (n - 1), count + 1)

  @doc """
  Counts the number of leading zeros of each element in the tensor.

  ## Examples

      iex> t = Nx.count_leading_zeros(1)
      iex> Nx.to_bitstring(t)
      <<63::64-native>>

      iex> t = Nx.count_leading_zeros(-1)
      iex> Nx.to_bitstring(t)
      <<0::64-native>>

      iex> t = Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF]))
      iex> Nx.to_bitstring(t)
      <<64::64-native, 60::64-native, 56::64-native, 48::64-native>>

      iex> t = Nx.count_leading_zeros(Nx.tensor([0xF000000000000000, 0x0F00000000000000]))
      iex> Nx.to_bitstring(t)
      <<0::64-native, 4::64-native>>

      iex> t = Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF], type: {:s, 32}))
      iex> Nx.to_bitstring(t)
      <<32::32-native, 28::32-native, 24::32-native, 16::32-native>>

      iex> t = Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF], type: {:s, 16}))
      iex> Nx.to_bitstring(t)
      <<16::16-native, 12::16-native, 8::16-native, 0::16-native>>

      iex> t = Nx.count_leading_zeros(Nx.tensor([0, 1, 2, 4, 8, 16, 32, 64, -1, -128], type: {:s, 8}))
      iex> Nx.to_bitstring(t)
      <<8, 7, 6, 5, 4, 3, 2, 1, 0, 0>>

      iex> t = Nx.count_leading_zeros(Nx.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128], type: {:u, 8}))
      iex> Nx.to_bitstring(t)
      <<8, 7, 6, 5, 4, 3, 2, 1, 0>>

  ### Error cases

      iex> Nx.population_count(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def count_leading_zeros(tensor)

  def count_leading_zeros(number) when is_integer(number),
    do: tensor(erlang_clz(to_unsigned(number, 64), 64))

  def count_leading_zeros(number) when is_float(number),
    do: assert_bitwise_type!({:f, 64})

  def count_leading_zeros(%T{type: {_, size} = input_type} = t) do
    assert_bitwise_type!(input_type)

    data =
      for <<seg::unsigned-size(size)-native <- data!(t)>>, into: <<>> do
        match_types [input_type] do
          <<write!(erlang_clz(seg, size), 0)>>
        end
      end

    %{t | data: {Nx.BitStringDevice, data}}
  end

  defp erlang_clz(0, size), do: size
  defp erlang_clz(n, 64), do: erlang_clz64(n)
  defp erlang_clz(n, 32), do: erlang_clz32(n)
  defp erlang_clz(n, 16), do: erlang_clz16(n)
  defp erlang_clz(n, 8), do: erlang_clz8(n)

  defp erlang_clz64(num) do
    case num &&& 0xFFFFFFFF00000000 do
      0 -> 32 + erlang_clz32(num)
      _ -> erlang_clz32(num >>> 32)
    end
  end

  defp erlang_clz32(num) do
    case num &&& 0xFFFF0000 do
      0 -> 16 + erlang_clz16(num)
      _ -> erlang_clz16(num >>> 16)
    end
  end

  defp erlang_clz16(num) do
    case num &&& 0xFF00 do
      0 -> 8 + erlang_clz8(num)
      _ -> erlang_clz8(num >>> 8)
    end
  end

  defp erlang_clz8(num) do
    case num &&& 0xF0 do
      0 -> 4 + erlang_clz4(num)
      _ -> erlang_clz4(num >>> 4)
    end
  end

  defp erlang_clz4(num) do
    case num &&& 0xC do
      0 -> 2 + erlang_clz2(num)
      _ -> erlang_clz2(num >>> 2)
    end
  end

  defp erlang_clz2(0), do: 2
  defp erlang_clz2(1), do: 1
  defp erlang_clz2(_), do: 0

  for {name, desc} <- [floor: "floor", ceil: "ceil", round: "round (away from zero)"] do
    [res1, res2, res3, res4] = Enum.map([-1.5, -0.5, 0.5, 1.5], &apply(:erlang, name, [&1]))

    @doc """
    Calculates the #{desc} of each element in the tensor.

    If a non-floating tensor is given, it is returned as is.
    If a floating tensor is given, then we apply the operation,
    but keep its type.

    ## Examples

        iex> t = Nx.#{name}(Nx.tensor([-1, 0, 1]))
        iex> Nx.to_bitstring(t)
        <<-1::64-native, 0::64-native, 1::64-native>>

        iex> t = Nx.#{name}(Nx.tensor([-1.5, -0.5, 0.5, 1.5]))
        iex> Nx.to_bitstring(t)
        <<#{res1}::float-64-native, #{res2}::float-64-native, #{res3}::float-64-native, #{res4}::float-64-native>>

    """
    def unquote(name)(tensor)

    def unquote(name)(number) when is_number(number), do: tensor(:erlang.unquote(name)(number))

    def unquote(name)(%T{type: {type, _}} = t) when type in [:s, :u], do: t

    def unquote(name)(%T{type: input_type} = t) do
      data = data!(t)

      data =
        match_types [input_type] do
          for <<match!(seg, 0) <- data>>, into: <<>> do
            <<write!(:erlang.unquote(name)(read!(seg, 0)), 0)>>
          end
        end

      %{t | data: {Nx.BitStringDevice, data}}
    end
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

  ## Matrix ops

  @doc """
  Returns the dot product of two tensors.

  Given `a` and `b`, computes the dot product according to
  the following rules:

    * If both `a` and `b` are scalars, it is equivalent to `a * b`.
    * If `a` is a scalar and `b` is a tensor, it is equivalent to `Nx.multiply(a, b)`.
    * If `a` is a tensor and `b` is a scalar, it is equivalent to `Nx.multiply(a, b)`.
    * If both `a` and `b` are 1-D tensors (vectors), it is the sum of the element-wise
    product between `a` and `b`. The lengths of `a` and `b` must be equal.
    * If both `a` and `b` are 2-D tensors (matrices), it is equivalent to matrix-multiplication.
    * If either `a` or `b` is a 1-D tensor, and the other is an n-D tensor, it is the
    sum of the element-wise product along the last axis of `a` or `b`. The length of the
    1-D tensor must match the last dimension of the n-D tensor.
    * If `a` is an n-D tensor and `b` is an m-D tensor, it is the sum of the element-wise
    product along the last axis of `a` and the second-to-last axis of `b`. The last dimension
    of `a` must match the second-to-last dimension of `b`.

  ## Examples

  ### Dot Product of Scalars

      iex> Nx.dot(5, 5)
      25
      iex> Nx.dot(-2.0, 5.0)
      -10.0

  ### Dot Product of Vectors

      iex> t = Nx.dot(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
      iex> Nx.to_bitstring(t)
      <<32::64-native>>

      iex> t = Nx.dot(Nx.tensor([2.0, 4.0, 3.0, 5.0]), Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      iex> Nx.to_bitstring(t)
      <<39.0::float-64-native>>

  ### Dot Product of Matrices

      TODO

  ### Dot Product of Vector and n-d tensor

      iex> t = Nx.dot(Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), Nx.tensor([5, 10]))
      iex> Nx.to_bitstring(t)
      <<25::64-native, 55::64-native, 85::64-native, 115::64-native>>
      iex> Nx.shape(t)
      {2, 2}

      iex> t = Nx.dot(Nx.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]]), Nx.tensor([2.0, 2.0]))
      iex> Nx.to_bitstring(t)
      <<6.0::float-64-native, 14.0::float-64-native, 22.0::float-64-native, 30.0::float-64-native>>
      iex> Nx.shape(t)
      {1, 1, 2, 2}

  ### Dot Product of n-D and m-D tensor

      TODO
  """
  def dot(a, b)

  def dot(a, b) when is_number(a) and is_number(b), do: a * b

  def dot(a = %T{}, b) when is_number(b), do: Nx.multiply(a, b)

  def dot(a, b = %T{}) when is_number(a), do: Nx.multiply(a, b)

  def dot(a = %T{type: left_type, shape: s1}, b = %T{type: right_type, shape: {n}}) do
    output_type = Nx.Type.merge(left_type, right_type)
    {_, left_size} = left_type
    {_, right_size} = right_type

    last_dim = elem(s1, tuple_size(s1) - 1)
    total_elems = div(tuple_product(s1), last_dim)

    output_shape =
      s1
      |> Tuple.to_list()
      |> Enum.take(tuple_size(s1) - 2)
      |> Kernel.++([last_dim])
      |> List.to_tuple()

    data =
      match_types [left_type, right_type, output_type] do
        for i <- 0..total_elems-1, into: <<>> do
          row = :binary.part(data!(a), div(i*n*left_size, 8), div(n*left_size, 8))
          value =
            bin_reduce_all(
              bin_zip_map_all(row, left_size, data!(b), right_size,
                fn <<match!(x, 0), _::bitstring>>, <<match!(y, 1), _::bitstring>> ->
                  <<write!(read!(x, 0) * read!(y, 1), 2)>>
                end
              ) |> IO.iodata_to_binary(), 0,
              fn <<match!(var, 0), rest::bitstring>>, acc ->
                {read!(var, 0) + acc, rest}
              end
            )

          <<write!(value, 0)>>
        end
      end

    %T{data: {Nx.BitStringDevice, data}, type: output_type, shape: output_shape}
  end

  def dot(a = %T{shape: {_}}, b = %T{}), do: Nx.dot(b, a)

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

  defp scalar_to_binary(value, type) do
    match_types([type], do: <<write!(value, 0)>>)
  end

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

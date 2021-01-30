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
      iex> Nx.divide(Nx.exp(t), Nx.sum(Nx.exp(t)))
      #Nx.Tensor<
        f64[2][2]
        [
          [0.03205860328008499, 0.08714431874203257],
          [0.23688281808991013, 0.6439142598879722]
        ]
      >

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
  For example, using the `EXLA` compiler:

      @defn_compiler {EXLA, platform: :host}
      defn softmax(t) do
        Nx.exp(t) / Nx.sum(Nx.exp(t))
      end

  This complements Erlang's JIT compiler as it compiles direct to
  native code with numerical compilation and performance in mind.

  ## Creating tensors

  The main APIs for creating tensors are `tensor/2`, `from_binary/2`,
  `iota/2`, `random_uniform/2`, `random_normal/2`, and `broadcast/3`.

  ## Broadcasting

  Broadcasting allows operations on two tensors of different shapes
  to match. For example, most often operations between tensors have
  the same shape:

      iex> a = Nx.tensor([1, 2, 3])
      iex> b = Nx.tensor([10, 20, 30])
      iex> Nx.add(a, b)
      #Nx.Tensor<
        s64[3]
        [11, 22, 33]
      >

  Now let's imagine you want to multiply a large tensor, dimensions
  1000x1000x1000, by 2. If you had to create a similarly large tensor
  only to perform this operation, it would be inneficient. Therefore,
  you can simply multiply this large tensor by the scalar 2, and Nx
  will propagate its dimensions at the time the operation happens,
  without allocating a large intermediate tensor:

      iex> Nx.multiply(Nx.tensor([1, 2, 3]), 2)
      #Nx.Tensor<
        s64[3]
        [2, 4, 6]
      >

  In practice, brodcasting is not restricted only to scalars, it
  is a general algorithm that applies to all dimensions of a tensor.
  When broadcasting, `Nx` compares the shapes of the two tensors,
  starting with the trailing ones, such that:

    * If the dimensions have equal size, then they are compatible

    * If one of the dimensions have size of 1, it is "broadcast"
      to match the dimension of the other

  In case on tensor has more dimensions than the other, the missing
  dimensions are considered to be of size one. Here are some examples
  of how broadcast would work when multiplying two tensors with the
  following shapes:

      s64[3] * s64
      #=> s64[3]

      s64[255][255][3] * s64[3]
      #=> s64[255][255][3]

      s64[2][1] * s[1][2]
      #=> s64[2][2]

      s64[5][1][4][1] * s64[3][4][5]
      #=> s64[5][3][4][5]

  If any of the dimensions do not match or are not 1, an error is
  raised.

  ## Devices

  The `Nx` library has built-in support for devices. A tensor is
  always allocated in a device, the default device being the
  `Nx.BinaryDevice`, which means the tensor is allocated as a
  binary within the Erlang VM.

  Most operations in the `Nx` module require the tensor to be
  allocated within the VM but, most often, when running `defn`
  functions that on the GPU, you want to keep the data on the
  GPU as much as possible. For example:

      @defn_compiler {EXLA, platform: :host, keep_on_device: true}
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

  import Nx.Shared
  alias Nx.Tensor, as: T

  @type t :: number | Nx.Tensor.t()
  @type shape :: Nx.Tensor.shape()
  @type axis :: Nx.Tensor.axis()
  @type axes :: NX.Tensor.axes()

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

      iex> Nx.tensor(0)
      #Nx.Tensor<
        s64
        0
      >

      iex> Nx.tensor(1.0)
      #Nx.Tensor<
        f64
        1.0
      >

  Giving a list returns a vector (an one-dimensional tensor):

      iex> Nx.tensor([1, 2, 3])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.tensor([1.2, 2.3, 3.4, 4.5])
      #Nx.Tensor<
        f64[4]
        [1.2, 2.3, 3.4, 4.5]
      >

  The type can be explicitly given. Integers and floats
  bigger than the given size overflow:

      iex> Nx.tensor([300, 301, 302], type: {:s, 8})
      #Nx.Tensor<
        s8[3]
        [44, 45, 46]
      >

  Mixed types get the highest precision type:

      iex> Nx.tensor([1, 2, 3.0])
      #Nx.Tensor<
        f64[3]
        [1.0, 2.0, 3.0]
      >

  Multi-dimensional tensors are also possible:

      iex> Nx.tensor([[1, 2, 3], [4, 5, 6]])
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
      >

      iex> Nx.tensor([[1, 2], [3, 4], [5, 6]])
      #Nx.Tensor<
        s64[3][2]
        [
          [1, 2],
          [3, 4],
          [5, 6]
        ]
      >

      iex> Nx.tensor([[[1, 2], [3, 4], [5, 6]], [[-1, -2], [-3, -4], [-5, -6]]])
      #Nx.Tensor<
        s64[2][3][2]
        [
          [
            [1, 2],
            [3, 4],
            [5, 6]
          ],
          [
            [-1, -2],
            [-3, -4],
            [-5, -6]
          ]
        ]
      >

  Brain-floating points are also supported, although they are
  emulated in Elixir and therefore perform slower without a
  compilation backend:

      iex> Nx.tensor([1, 2, 3], type: {:bf, 16})
      #Nx.Tensor<
        bf16[3]
        [1.0, 2.0, 3.0]
      >

  Given a tensor to `tensor/2` returns the tensor itself:

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.tensor(t) == t
      true

  However, if a :type is given and they don't match, an error is
  raised:

      iex> Nx.tensor(Nx.tensor([1, 2, 3]), type: {:f, 64})
      ** (ArgumentError) got a tensor with type {:f, 64} but tensor has type {:s, 64}

  You can also provide names for tensor dimensions. Names are either atoms or `nil`:

      iex> Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y])
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
      >

  Names make your code more expressive:

      iex> Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:batch, :height, :width])
      #Nx.Tensor<
        s64[batch: 1][height: 3][width: 3]
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        ]
      >

  You can also leave dimension names as `nil`:

      iex> Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:batch, nil, nil])
      #Nx.Tensor<
        s64[batch: 1][3][3]
        [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]
        ]
      >

  However, you must provide a name for every dimension in the tensor:

      iex> Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:batch])
      ** (ArgumentError) invalid names for tensor of rank 3, when specifying names every dimension must have a name or be nil

  ## Options

    * `:type` - sets the type of the tensor. If one is not given,
      one is automatically inferred based on the input. See `Nx.Type`
      and `Nx.Type.infer/1` for more information on types. If a
      tensor is given alongside this option, then it verifies the
      tensor matches the given `:type`

    * `:names` - dimension names. If you wish to specify dimension
      names you must specify a name for every dimension in the tensor.
      Only `nil` and atoms are supported as dimension names.

  """
  def tensor(arg, opts \\ [])

  def tensor(%T{} = t, opts) do
    assert_keys!(opts, [:type, :names, :backend])
    type = opts[:type]

    if type && type != t.type do
      raise ArgumentError,
            "got a tensor with type #{inspect(type)} but tensor has type #{inspect(t.type)}"
    end

    backend = opts[:backend] || Nx.BinaryTensor
    backend.tensor(t)
  end

  def tensor(arg, opts) do
    assert_keys!(opts, [:type, :names, :backend])
    type = Nx.Type.normalize!(opts[:type] || Nx.Type.infer(arg))
    {shape, data} = flatten(arg, type)

    if data == "" do
      raise "cannot build empty tensor"
    end

    names = Nx.Shape.named_axes!(opts[:names], shape)
    backend = opts[:backend] || Nx.BinaryTensor
    backend.from_binary(%T{shape: shape, type: type, names: names}, data)
  end

  defp flatten(list, type) when is_list(list) do
    {dimensions, acc} = flatten_list(list, type, [], [])

    {dimensions |> Enum.reverse() |> List.to_tuple(),
     acc |> Enum.reverse() |> :erlang.list_to_binary()}
  end

  defp flatten(other, type), do: {{}, number_to_binary(other, type)}

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
    {[length(list) | dimensions], Enum.reduce(list, acc, &[number_to_binary(&1, type) | &2])}
  end

  @doc """
  Shortcut for `random_uniform(shape, 0.0, 1.0, opts)`.
  """
  def random_uniform(tensor_or_shape, opts \\ []) do
    random_uniform(tensor_or_shape, 0.0, 1.0, opts)
  end

  @doc """
  Returns a uniformly-distributed random tensor with the given shape.

  The distribution is bounded on the semi-open interval `[min, max)`.
  If `min` and `max` are integers, then the tensor has type `{:s, 64}`.
  Otherwise, a `{:f, 64}` tensor is returned. You can also pass any
  valid type via the `:type` option.

  If a tensor or a number are given, the shape and default type are
  taken from them.

  ## Examples

  ### Generating Floats

      iex> t = Nx.random_uniform({10})
      iex> for <<x::float-64-native <- Nx.to_binary(t)>> do
      ...>   true = x >= 0.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:f, 64}

      iex> t = Nx.random_uniform({5, 5}, type: {:bf, 16})
      iex> byte_size(Nx.to_binary(t))
      50
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:bf, 16}

      iex> t = Nx.random_uniform({5, 5}, -1.0, 1.0, type: {:f, 32})
      iex> for <<x::float-32-native <- Nx.to_binary(t)>> do
      ...>   true = x >= -1.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:f, 32}

  ### Generating Integers

      iex> t = Nx.random_uniform({10}, 5, 10, type: {:u, 32})
      iex> for <<x::32-unsigned-native <- Nx.to_binary(t)>> do
      ...>   true = x >= 5 and x < 10
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:u, 32}

      iex> t = Nx.random_uniform({5, 5}, -5, 5, type: {:s, 64})
      iex> for <<x::64-signed-native <- Nx.to_binary(t)>> do
      ...>   true = x >= -5 and x < 5
      ...> end
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:s, 64}

  ### Tensors as shapes

  If given a tensor as a shape, it takes the shape and names from the tensor:

      iex> t = Nx.tensor([[1, 2], [3, 4]], names: [:batch, :data])
      iex> t = Nx.random_uniform(t)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.names(t)
      [:batch, :data]

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t = Nx.random_uniform(t, type: {:f, 32})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}
      iex> Nx.names(t)
      [nil, nil]

  The same applies to numbers:

      iex> t = Nx.random_uniform(10)
      iex> Nx.shape(t)
      {}
      iex> Nx.type(t)
      {:f, 64}

      iex> t = Nx.random_uniform(10.0)
      iex> Nx.shape(t)
      {}
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.names(t)
      []

  If you pass `:names` as an option, the resulting tensor will take on those names:

      iex> t = Nx.tensor([[1, 2], [3, 4]], names: [:batch, :data])
      iex> t = Nx.random_uniform(t, names: [:batch, nil])
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.names(t)
      [:batch, nil]
  """
  def random_uniform(tensor_or_shape, min, max, opts \\ [])
      when is_number(min) and is_number(max) do
    assert_keys!(opts, [:type, :names, :backend])
    shape = Nx.shape(tensor_or_shape)
    names = opts[:names] || Nx.Shape.named_axes!(names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type] || Nx.Type.infer(max - min))
    backend = opts[:backend] || Nx.BinaryTensor
    backend.random_uniform(%T{shape: shape, type: type, names: names}, min, max)
  end

  @doc """
  Shortcut for `random_normal(shape, 0.0, 1.0, opts)`.
  """
  def random_normal(tensor_or_shape, opts \\ []) do
    random_normal(tensor_or_shape, 0.0, 1.0, opts)
  end

  @doc """
  Returns a normally-distributed random tensor with the given shape.

  The distribution has mean of `mu` and standard deviation of
  `sigma`. Return type is one of `{:bf, 16}`, `{:f, 32}` or `{:f, 64}`.

  If a tensor or a number are given, the shape is taken from the tensor.

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

  If given a tensor as a shape, it takes the shape, names, and default type
  from the tensor:

      iex> t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], names: [:batch, :data])
      iex> t = Nx.random_normal(t)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.names(t)
      [:batch, :data]

      iex> t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      iex> t = Nx.random_normal(t, type: {:f, 32})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}
      iex> Nx.names(t)
      [nil, nil]

  The same applies to numbers:

      iex> t = Nx.random_normal(10.0)
      iex> Nx.shape(t)
      {}
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.names(t)
      []

  If you pass the `:names` option, the resulting tensor will take on those names:

      iex> t = Nx.tensor([[1, 2], [3, 4]], names: [:batch, :data])
      iex> t = Nx.random_normal(t, names: [:batch, nil])
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.names(t)
      [:batch, nil]
  """
  def random_normal(tensor_or_shape, mu, sigma, opts \\ [])
      when is_float(mu) and is_float(sigma) do
    assert_keys!(opts, [:type, :names, :backend])
    shape = Nx.shape(tensor_or_shape)
    names = opts[:names] || Nx.Shape.named_axes!(names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type] || {:f, 64})
    backend = opts[:backend] || Nx.BinaryTensor
    backend.random_normal(%T{shape: shape, type: type, names: names}, mu, sigma)
  end

  @doc """
  Creates a tensor with the given shape which increments
  along the provided axis. You may optionally provide dimension
  names.

  If no axis is provided, index counts up at each element.

  If a tensor or a number are given, the shape and names are taken from the tensor.

  ## Examples

      iex> Nx.iota({})
      #Nx.Tensor<
        s64
        0
      >

      iex> Nx.iota({5})
      #Nx.Tensor<
        s64[5]
        [0, 1, 2, 3, 4]
      >

      iex> Nx.iota({3, 2, 3}, names: [:batch, :height, :width])
      #Nx.Tensor<
        s64[batch: 3][height: 2][width: 3]
        [
          [
            [0, 1, 2],
            [3, 4, 5]
          ],
          [
            [6, 7, 8],
            [9, 10, 11]
          ],
          [
            [12, 13, 14],
            [15, 16, 17]
          ]
        ]
      >

      iex> Nx.iota({3, 3}, axis: 1, names: [:batch, nil])
      #Nx.Tensor<
        s64[batch: 3][3]
        [
          [0, 1, 2],
          [0, 1, 2],
          [0, 1, 2]
        ]
      >

      iex> Nx.iota({3, 3}, axis: -1)
      #Nx.Tensor<
        s64[3][3]
        [
          [0, 1, 2],
          [0, 1, 2],
          [0, 1, 2]
        ]
      >

      iex> Nx.iota({3, 4, 3}, axis: 0, type: {:f, 64})
      #Nx.Tensor<
        f64[3][4][3]
        [
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ],
          [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
          ],
          [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0]
          ]
        ]
      >

      iex> Nx.iota({1, 3, 2}, axis: 2)
      #Nx.Tensor<
        s64[1][3][2]
        [
          [
            [0, 1],
            [0, 1],
            [0, 1]
          ]
        ]
      >
  """
  def iota(tensor_or_shape, opts \\ []) do
    assert_keys!(opts, [:type, :axis, :names, :backend])
    shape = Nx.shape(tensor_or_shape)
    names = opts[:names] || Nx.Shape.named_axes!(names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type] || {:s, 64})
    backend = opts[:backend] || Nx.BinaryTensor

    if axis = opts[:axis] do
      axis = Nx.Shape.normalize_axis(shape, axis, names)
      backend.iota(%T{type: type, shape: shape, names: names}, axis)
    else
      backend.iota(%T{type: type, shape: shape, names: names}, nil)
    end
  end

  @doc """
  Returns the underlying tensor as a binary.

  The binary is returned as is (which is row-major).

  ## Examples

      iex> Nx.to_binary(1)
      <<1::64-native>>

      iex> Nx.to_binary(Nx.tensor([1.0, 2.0, 3.0]))
      <<1.0::float-native, 2.0::float-native, 3.0::float-native>>
  """
  def to_binary(tensor) do
    tensor = tensor!(tensor)
    impl!(tensor).to_binary(tensor)
  end

  ## Conversions

  @doc """
  Returns the underlying tensor as a flat list.

  ## Options

    * `:limit` - convert at most limit elements to a list

    * `:non_numbers` - control the behaviour when "Inf"
      and "NaN" are present in the tensor. Defaults to
      `:raise` but can be set to `:as_strings`

  ## Examples

      iex> Nx.to_flat_list(1)
      [1]

      iex> Nx.to_flat_list(Nx.tensor([1.0, 2.0, 3.0]))
      [1.0, 2.0, 3.0]

      iex> Nx.to_flat_list(Nx.tensor([1.0, 2.0, 3.0]), limit: 2)
      [1.0, 2.0]

      iex> bin = <<0xFFF0000000000000::64-native, 0x7FF0000000000000::64-native,
      ...>         0x7FF0000000000001::64-native, 0x7FF8000000000001::64-native>>
      iex> Nx.to_flat_list(Nx.from_binary(bin, {:f, 64}), non_numbers: :as_strings)
      ["-Inf", "Inf", "NaN", "NaN"]

  """
  def to_flat_list(tensor, opts \\ []) do
    assert_keys!(opts, [:limit, :non_numbers])
    tensor = Nx.tensor(tensor)
    binary = Nx.to_binary(tensor)
    {kind, size} = type = tensor.type

    binary =
      if limit = opts[:limit] do
        binary_part(binary, 0, Kernel.min(byte_size(binary), limit * div(size, 8)))
      else
        binary
      end

    case Keyword.get(opts, :non_numbers, :raise) do
      :raise ->
        for <<part::size(size)-bitstring <- binary>> do
          match_types [type] do
            <<match!(var, 0)>> = part
            read!(var, 0)
          end
        end

      :as_strings ->
        case kind do
          :s -> for <<x::size(size)-signed-native <- binary>>, do: x
          :u -> for <<x::size(size)-unsigned-native <- binary>>, do: x
          :f -> for <<x::size(size)-bitstring <- binary>>, do: read_float(x, size)
          :bf -> for <<x::16-bitstring <- binary>>, do: read_bf16(x)
        end
    end
  end

  defp read_bf16(<<0xFF80::16-native>>), do: "-Inf"
  defp read_bf16(<<0x7F80::16-native>>), do: "Inf"
  defp read_bf16(<<0xFFC1::16-native>>), do: "NaN"
  defp read_bf16(<<0xFF81::16-native>>), do: "NaN"

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

  defp read_float(data, 32) do
    case data do
      <<0xFF800000::32-native>> -> "-Inf"
      <<0x7F800000::32-native>> -> "Inf"
      <<0xFF800001::32-native>> -> "NaN"
      <<0xFFC00001::32-native>> -> "NaN"
      <<x::float-32-native>> -> x
    end
  end

  defp read_float(data, 64) do
    case data do
      <<0xFFF0000000000000::64-native>> -> "-Inf"
      <<0x7FF0000000000000::64-native>> -> "Inf"
      <<0x7FF0000000000001::64-native>> -> "NaN"
      <<0x7FF8000000000001::64-native>> -> "NaN"
      <<x::float-64-native>> -> x
    end
  end

  @doc """
  Returns the underlying tensor as a scalar.

  If the tensor has a dimension, it raises.

    ## Examples

      iex> Nx.to_scalar(1)
      1

      iex> Nx.to_scalar(Nx.tensor([1.0, 2.0, 3.0]))
      ** (ArgumentError) cannot convert tensor of shape {3} to scalar

  """
  def to_scalar(tensor)

  def to_scalar(number) when is_number(number), do: number

  def to_scalar(tensor) do
    tensor = Nx.tensor(tensor)

    if tensor.shape != {} do
      raise ArgumentError, "cannot convert tensor of shape #{inspect(tensor.shape)} to scalar"
    end

    match_types [tensor.type] do
      <<match!(x, 0)>> = Nx.to_binary(tensor)
      read!(x, 0)
    end
  end

  @doc """
  Creates a one-dimensional tensor from a `binary` with the given `type`.

  If the binary size does not match its type, an error is raised.

  ## Examples

      iex> Nx.from_binary(<<1, 2, 3, 4>>, {:s, 8})
      #Nx.Tensor<
        s8[4]
        [1, 2, 3, 4]
      >

      iex> Nx.from_binary(<<12.3::float-64-native>>, {:f, 64})
      #Nx.Tensor<
        f64[1]
        [12.3]
      >

      iex> Nx.from_binary(<<1, 2, 3, 4>>, {:f, 64})
      ** (ArgumentError) binary does not match the given size

  """
  def from_binary(binary, type, names \\ nil) when is_binary(binary) do
    {_, size} = Nx.Type.normalize!(type)
    dim = div(bit_size(binary), size)

    names = Nx.Shape.named_axes!(names, {dim})

    if binary == "" do
      raise ArgumentError, "cannot build an empty tensor"
    end

    if rem(bit_size(binary), size) != 0 do
      raise ArgumentError, "binary does not match the given size"
    end

    Nx.BinaryTensor.from_binary(%T{type: type, shape: {dim}, names: names}, binary)
  end

  ## Meta operations (do not invoke the backend)

  @doc """
  Changes the type of a tensor.

  Note it is not possible to cast from floats to integers.
  Use `round/1`, `floor/1`, and `ceil/1` instead.

  Casting from a higher precision may lead to overflows
  and underflows, which is platform and compiler dependent
  behaviour.

  ## Examples

      iex> Nx.as_type(Nx.tensor([0, 1, 2], names: [:data]), {:f, 32})
      #Nx.Tensor<
        f32[data: 3]
        [0.0, 1.0, 2.0]
      >

      iex> Nx.as_type(Nx.tensor([0.0, 1.0, 2.0], names: [:data]), {:bf, 16})
      #Nx.Tensor<
        bf16[data: 3]
        [0.0, 1.0, 2.0]
      >

  """
  def as_type(tensor, type) do
    tensor = tensor!(tensor)
    new_type = Nx.Type.normalize!(type)

    if tensor.type == new_type do
      tensor
    else
      impl!(tensor).as_type(%{tensor | type: new_type}, tensor)
    end
  end

  @doc """
  Changes the shape of a tensor.

  The new shape is either a tuple or a tensor which we will
  retrieve the current shape from. The shapes must be compatible:
  the product of each dimension in the shape must be equal.

  Reshaping only changes the tensor metadata, it doesn't copy
  the underlying structure.

  Reshape is a destructive operation with respect to names. You
  can optionally provide `:names` for each of the dimensions
  in the reshaped tensor. If you do not provide `:names`, they
  will be taken from the tensor the shape is taken from or
  all of the dimension names will be set to `nil`.

  ## Examples

      iex> t = Nx.tensor([1, 2, 3, 4], names: [:x])
      iex> Nx.reshape(t, {2, 2}, names: [:x, :y])
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [1, 2],
          [3, 4]
        ]
      >

  The shape can also be an existing tensor:

      iex> shape = Nx.tensor([[0], [0], [0], [0]], names: [:x, :y])
      iex> Nx.reshape(Nx.tensor([1, 2, 3, 4]), shape)
      #Nx.Tensor<
        s64[x: 4][y: 1]
        [
          [1],
          [2],
          [3],
          [4]
        ]
      >

  Even a scalar can be transformed into a 3-dimensional tensor:

      iex> t = Nx.tensor(1)
      iex> Nx.reshape(t, {1, 1, 1}, names: [:x, :y, :z])
      #Nx.Tensor<
        s64[x: 1][y: 1][z: 1]
        [
          [
            [1]
          ]
        ]
      >

  """
  def reshape(tensor, new_shape, opts \\ []) do
    %T{shape: old_shape} = tensor = tensor!(tensor)
    new_names = opts[:names] || names!(new_shape)
    new_shape = shape(new_shape)

    names = Nx.Shape.named_axes!(new_names, new_shape)

    if size(old_shape) != size(new_shape) do
      raise ArgumentError,
            "cannot reshape, current shape #{inspect(old_shape)} is not compatible with " <>
              "new shape #{inspect(new_shape)}"
    end

    if old_shape == new_shape do
      tensor
    else
      impl!(tensor).reshape(%{tensor | shape: new_shape, names: names}, tensor, new_shape)
    end
  end

  @doc """
  Squeezes the given size `1` dimensions out of the tensor.

  If no axes are given, squeezes all size `1` dimensions
  from the tensor.

  While this is equivalent to a reshape which eliminates
  the size `1` axes, squeeze preserves important information
  about which axes were squeezed out which can then be used
  later on in transformations.

  ## Examples

      iex> Nx.squeeze(Nx.tensor([[[[[1]]]]]))
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.squeeze(Nx.tensor([[[[1]]], [[[2]]]], names: [:x, :y, :z, :i]))
      #Nx.Tensor<
        s64[x: 2]
        [1, 2]
      >

      iex> Nx.squeeze(Nx.tensor([[1, 2, 3]], names: [:x, :y]), axes: [:x])
      #Nx.Tensor<
        s64[y: 3]
        [1, 2, 3]
      >

      iex> Nx.squeeze(Nx.tensor([[1], [2]], names: [:x, :y]), axes: [:y])
      #Nx.Tensor<
        s64[x: 2]
        [1, 2]
      >

  ### Error cases

      iex> Nx.squeeze(Nx.tensor([[1, 2, 3], [4, 5, 6]]), axes: [1])
      ** (ArgumentError) cannot squeeze dimensions whose sizes are not 1, got 3 for dimension 1

      iex> Nx.squeeze(Nx.tensor([[[[[1]]]]]), axes: [0, 0])
      ** (ArgumentError) axes [0, 0] must be unique integers between 0 and 4
  """
  def squeeze(tensor, opts \\ []) do
    assert_keys!(opts, [:axes])
    %T{shape: old_shape, names: names} = tensor = tensor!(tensor)
    axes = opts[:axes] || Nx.Shape.squeeze_axes(old_shape)
    axes = Nx.Shape.normalize_axes(old_shape, axes, names)
    {new_shape, new_names} = Nx.Shape.squeeze(old_shape, axes, names)

    if old_shape == new_shape do
      tensor
    else
      impl!(tensor).squeeze(%{tensor | shape: new_shape, names: new_names}, tensor, axes: axes)
    end
  end

  @doc """
  Broadcasts `tensor` to the given `broadcast_shape`.

  The new shape is either a tuple or a tensor which we will
  retrieve the current shape from. The broadcast shape must
  be of equal or higher rank than the current shape.

  An optional `:axes` can be given to customize how broadcasting
  happens. `axes` must be a list with the same length as the
  tensor shape. Each `axis` in the list maps to the dimension
  in the broadcast shape that must match. For example, an axis
  of `[1, 2]` says the 0 dimension of the tensor matches to
  the 1 dimension of the broadcast shape and the 1 dimension
  of the tensor matches the 2 dimension of the broadcast shape.
  Each matching dimension must either be 1, for implicit
  broadcasting, or match the dimension in the broadcast shape.

  Broadcasting is destructive with respect to names. You can
  optionally provide new `:names` for the new tensor. If you
  pass a tensor with named dimensions, the new tensor will
  inherit names from that tensor.

  ## Examples

  ### Without axes

  ## Examples

      iex> Nx.broadcast(1, {1, 2, 3})
      #Nx.Tensor<
        s64[1][2][3]
        [
          [
            [1, 1, 1],
            [1, 1, 1]
          ]
        ]
      >

      iex> Nx.broadcast(Nx.tensor([[1], [2]], names: [:x, :y]), Nx.tensor([[10, 20], [30, 40]], names: [:i, :j]))
      #Nx.Tensor<
        s64[i: 2][j: 2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> Nx.broadcast(Nx.tensor([[1, 2]], names: [:x, :y]), Nx.tensor([[10, 20], [30, 40]], names: [:i, :j]))
      #Nx.Tensor<
        s64[i: 2][j: 2]
        [
          [1, 2],
          [1, 2]
        ]
      >

  Note that, even if there is no broadcasting because the
  shape is the name, names are discarded if none are given:

      iex> Nx.broadcast(Nx.iota({2, 2}, names: [:x, :y]), {2, 2})
      #Nx.Tensor<
        s64[2][2]
        [
          [0, 1],
          [2, 3]
        ]
      >

      iex> Nx.broadcast(Nx.iota({2, 2}, names: [:x, :y]), {2, 2}, names: [:i, :j])
      #Nx.Tensor<
        s64[i: 2][j: 2]
        [
          [0, 1],
          [2, 3]
        ]
      >

  ### With axes

  Using the default broadcast rules, we cannot broadcast a
  tensor of shape (3) to the shape (3, 2), because the lower
  dimensions must match. But with `Nx.broadcast/3` we can
  configure how the dimensions match:

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.broadcast(t, {3, 2}, axes: [0], names: [:x, :y])
      #Nx.Tensor<
        s64[x: 3][y: 2]
        [
          [1, 1],
          [2, 2],
          [3, 3]
        ]
      >

  Or a more complex example:

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.broadcast(t, {2, 3, 2}, axes: [1], names: [:x, :y, :z])
      #Nx.Tensor<
        s64[x: 2][y: 3][z: 2]
        [
          [
            [1, 1],
            [2, 2],
            [3, 3]
          ],
          [
            [1, 1],
            [2, 2],
            [3, 3]
          ]
        ]
      >

  """
  def broadcast(tensor, shape, opts \\ []) do
    assert_keys!(opts, [:axes, :names])

    tensor = tensor!(tensor)
    broadcast_names = opts[:names] || names!(shape)
    broadcast_shape = shape(shape)
    opts_axes = opts[:axes]

    axes =
      if opts_axes do
        Nx.Shape.normalize_axes(broadcast_shape, opts_axes, tensor.names)
      else
        Nx.Shape.broadcast_axes(tensor.shape, broadcast_shape)
      end

    broadcast_names = Nx.Shape.named_axes!(broadcast_names, broadcast_shape)
    out = %{tensor | names: broadcast_names, shape: broadcast_shape}

    if tensor.shape == broadcast_shape and is_nil(opts_axes) do
      out
    else
      _ = Nx.Shape.broadcast!(tensor.shape, broadcast_shape, axes)
      impl!(tensor).broadcast(out, tensor, broadcast_shape, axes)
    end
  end

  @doc """
  Pads a tensor with a given value.

  You must specify a padding configuration. A padding
  configuration is a list of tuples consisting of
  `{pad_width_low, pad_width_high}` for each dimension
  in the input tensor. The padding configuration must
  be of the same length as the tensor shape.

  Padding widths can be negative. If they are negative,
  the tensor is clipped on either end according to the
  padding width.

  ## Examples

      iex> Nx.pad(Nx.tensor(1), 0, [])
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.pad(Nx.tensor([1, 2, 3], names: [:data]), 0, [{1, 1, 0}])
      #Nx.Tensor<
        s64[data: 5]
        [0, 1, 2, 3, 0]
      >

      iex> Nx.pad(Nx.tensor([[1, 2, 3], [4, 5, 6]]), 0, [{0, 0, 1}, {0, 0, 1}])
      #Nx.Tensor<
        s64[3][5]
        [
          [1, 0, 2, 0, 3],
          [0, 0, 0, 0, 0],
          [4, 0, 5, 0, 6]
        ]
      >

      iex> Nx.pad(Nx.tensor([[1, 2, 3], [4, 5, 6]]), 0, [{1, 1, 0}, {1, 1, 0}])
      #Nx.Tensor<
        s64[4][5]
        [
          [0, 0, 0, 0, 0],
          [0, 1, 2, 3, 0],
          [0, 4, 5, 6, 0],
          [0, 0, 0, 0, 0]
        ]
      >

      iex> tensor = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      iex> Nx.pad(tensor, 0, [{0, 2, 0}, {1, 1, 0}, {1, 0, 0}])
      #Nx.Tensor<
        s64[4][4][3]
        [
          [
            [0, 0, 0],
            [0, 1, 2],
            [0, 3, 4],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [0, 5, 6],
            [0, 7, 8],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
          ]
        ]
      >

      iex> tensor = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      iex> Nx.pad(tensor, 0, [{1, 0, 0}, {1, 1, 0}, {0, 1, 0}])
      #Nx.Tensor<
        s64[3][4][3]
        [
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [5, 6, 0],
            [7, 8, 0],
            [0, 0, 0]
          ]
        ]
      >

      iex> tensor = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
      iex> Nx.pad(tensor, 0.0, [{1, 2, 0}, {1, 0, 0}, {0, 1, 0}])
      #Nx.Tensor<
        f64[5][3][3]
        [
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [5.0, 6.0, 0.0],
            [7.0, 8.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ],
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ]
        ]
      >

      iex> Nx.pad(Nx.tensor([0, 1, 2, 3, 0]), 0, [{-1, -1, 0}])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> tensor = Nx.tensor([
      ...>   [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      ...>   [[0, 0, 0], [1, 2, 0], [3, 4, 0], [0, 0, 0]],
      ...>   [[0, 0, 0], [5, 6, 0], [7, 8, 0], [0, 0, 0]]
      ...> ])
      iex> Nx.pad(tensor, 0, [{-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}])
      #Nx.Tensor<
        s64[2][2][2]
        [
          [
            [1, 2],
            [3, 4]
          ],
          [
            [5, 6],
            [7, 8]
          ]
        ]
      >

      iex> Nx.pad(Nx.tensor([[0, 1, 2, 3], [0, 4, 5, 6]]), 0, [{0, 0, 0}, {-1, 1, 0}])
      #Nx.Tensor<
        s64[2][4]
        [
          [1, 2, 3, 0],
          [4, 5, 6, 0]
        ]
      >

      iex> Nx.pad(Nx.tensor([[0, 1, 2], [3, 4, 5]], type: {:f, 32}), 0, [{-1, 2, 0}, {1, -1, 0}])
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 3.0, 4.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ]
      >

  """
  def pad(tensor, pad_value, padding_config) when is_list(padding_config) do
    tensor = tensor!(tensor)
    pad_value = tensor!(pad_value)

    output_type = binary_type(tensor, pad_value)

    if pad_value.shape != {} do
      raise ArgumentError, "padding value must be a scalar"
    end

    shape = Nx.Shape.pad(tensor.shape, padding_config)

    out = %{tensor | type: output_type, shape: shape}
    impl!(tensor).pad(out, tensor, pad_value, padding_config)
  end

  ## Reflection

  @doc """
  Returns the type of the tensor.

  See `Nx.Type` for more information.

  ## Examples

      iex> Nx.type(Nx.tensor([1, 2, 3]))
      {:s, 64}

      iex> Nx.type(Nx.tensor([1, 2, 3], type: {:f, 32}))
      {:f, 32}

      iex> Nx.type(1)
      {:s, 64}

      iex> Nx.type(1.0)
      {:f, 64}
  """
  def type(tensor) do
    %T{type: type} = tensor!(tensor)
    type
  end

  @doc """
  Returns the shape of the tensor as a tuple.

  The size of this tuple gives the rank of the tensor.

  If a shape as a tuple is given, it returns the shape itself.

  ### Examples

      iex> Nx.shape(Nx.tensor(1))
      {}

      iex> Nx.shape(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      {2, 3}

      iex> Nx.shape(1)
      {}

      iex> Nx.shape({1, 2, 3})
      {1, 2, 3}

  """
  def shape(shape) when is_tuple(shape), do: validate_shape(shape, tuple_size(shape))
  def shape(%T{shape: shape}), do: shape
  def shape(number) when is_number(number), do: {}

  def shape(other) do
    raise ArgumentError,
          "expected a shape. A shape is a n-element tuple with the size of each dimension. " <>
            "Alternatively you can pass a tensor (or a number) and the shape will be retrieved from the tensor. " <>
            "Got: #{inspect(other)}"
  end

  defp validate_shape(shape, 0), do: shape

  defp validate_shape(shape, pos) do
    dim = :erlang.element(pos, shape)

    if is_integer(dim) and dim > 0 do
      validate_shape(shape, pos - 1)
    else
      raise ArgumentError,
            "invalid dimension #{inspect(dim)} in shape #{inspect(shape)}. Each dimension must be a positive integer"
    end
  end

  @doc """
  Returns the rank of a tensor.

  If a tuple is given as a shape, it computes the rank
  of the given tuple.

  ### Examples

      iex> Nx.rank(Nx.tensor(1))
      0

      iex> Nx.rank(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      2

      iex> Nx.rank(1)
      0

      iex> Nx.rank({1, 2, 3})
      3

  """
  def rank(shape) when is_tuple(shape), do: tuple_size(shape)
  def rank(tensor), do: tuple_size(shape(tensor))

  @doc """
  Returns how many elements they are in the tensor.

  If a tuple is given as a shape, it computes the size
  of the given tuple.

  ### Examples

      iex> Nx.size(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      6

      iex> Nx.size(1)
      1

      iex> Nx.size({1, 2, 3})
      6

  """
  def size(shape) when is_tuple(shape), do: tuple_product(shape, tuple_size(shape))
  def size(tensor), do: size(shape(tensor))

  @doc """
  Returns all of the axes in a tensor.

  If a shape is given, it returns the axes for the given shape.

  ### Examples

      iex> Nx.axes(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      [0, 1]

      iex> Nx.axes(1)
      []

      iex> Nx.axes({1, 2, 3})
      [0, 1, 2]

  """
  def axes(shape), do: count_up(rank(shape), 0)

  @doc """
  Returns all of the names in a tensor.

  ### Examples

      iex> Nx.names(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:batch, :data]))
      [:batch, :data]

      iex> Nx.names(Nx.tensor([1, 2, 3]))
      [nil]

      iex> Nx.names(5)
      []
  """
  def names(%T{names: names}), do: names
  def names(a) when is_number(a), do: []

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  # TODO: Use Tuple.product on Elixir v1.12
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  ## Device API

  @doc """
  Transfers data to the given device.

  If a device is not given, `Nx.BinaryDevice` is used, which means
  the data is read into an Elixir binary. If the device is already
  `Nx.BinaryDevice`, it returns the tensor as is.

  If a separate device is given, the data will be moved to the new
  device. Once transfer is done, the data is deallocated from the
  current tensor device. If the device has already been deallocated,
  it raises.

  At the moment, you can only transfer data from `Nx.BinaryDevice`
  to other devices and vice-versa but not between ad-hoc devices.

  ## Examples

  Move a tensor to a device:

      device_tensor = Nx.device_transfer(tensor, EXLA.NxDevice, client: :cuda)

  Read the device tensor back to an Elixir binary:

      tensor = Nx.device_transfer(tensor)

  """
  def device_transfer(tensor, device \\ Nx.BinaryDevice, opts \\ []) do
    tensor = tensor!(tensor)
    impl!(tensor).device_transfer(tensor, device, opts)
  end

  @doc """
  Reads data allocated in a device.

  It returns a tensor where the device is `Nx.BinaryDevice`.
  The data is not deallocated from the current device. If the
  device has already been deallocated, it raises.
  """
  def device_read(tensor) do
    tensor = tensor!(tensor)
    impl!(tensor).device_read(tensor)
  end

  @doc """
  Deallocates data in a device.

  It returns either `:ok` or `:already_deallocated`.
  """
  def device_deallocate(tensor) do
    tensor = tensor!(tensor)
    impl!(tensor).device_deallocate(tensor)
  end

  ## Element-wise binary ops

  defp element_wise_bin_op(left, right, op, fun) do
    type = binary_type(left, right) |> fun.()
    %T{shape: left_shape, names: left_names} = left = tensor!(left)
    %T{shape: right_shape, names: right_names} = right = tensor!(right)

    {shape, names} = Nx.Shape.binary_broadcast(left_shape, left_names, right_shape, right_names)

    apply(impl!(left, right), op, [%{left | type: type, shape: shape, names: names}, left, right])
  end

  defp element_wise_pred_op(left, right, op) do
    %T{shape: left_shape, names: left_names} = left = tensor!(left)
    %T{shape: right_shape, names: right_names} = right = tensor!(right)

    {shape, names} = Nx.Shape.binary_broadcast(left_shape, left_names, right_shape, right_names)

    out = %{left | type: {:u, 8}, shape: shape, names: names}
    apply(impl!(left, right), op, [out, left, right])
  end

  @doc """
  Element-wise addition of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Adding scalars

      iex> Nx.add(1, 2)
      #Nx.Tensor<
        s64
        3
      >

      iex> Nx.add(1, 2.2)
      #Nx.Tensor<
        f64
        3.2
      >

  ### Adding a scalar to a tensor

      iex> Nx.add(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        s64[data: 3]
        [2, 3, 4]
      >

      iex> Nx.add(1, Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        s64[data: 3]
        [2, 3, 4]
      >

  Given a float scalar converts the tensor to a float:

      iex> Nx.add(Nx.tensor([1, 2, 3], names: [:data]), 1.0)
      #Nx.Tensor<
        f64[data: 3]
        [2.0, 3.0, 4.0]
      >

      iex> Nx.add(Nx.tensor([1.0, 2.0, 3.0], names: [:data]), 1)
      #Nx.Tensor<
        f64[data: 3]
        [2.0, 3.0, 4.0]
      >

      iex> Nx.add(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}, names: [:data]), 1)
      #Nx.Tensor<
        f32[data: 3]
        [2.0, 3.0, 4.0]
      >

  Unsigned tensors become signed and double their size if a
  negative number is given:

      iex> Nx.add(Nx.tensor([0, 1, 2], type: {:u, 8}, names: [:data]), -1)
      #Nx.Tensor<
        s16[data: 3]
        [-1, 0, 1]
      >

  ### Adding tensors of the same shape

      iex> Nx.add(Nx.tensor([[1, 2], [3, 4]], names: [:x, :y]), Nx.tensor([[10, 20], [30, 40]], names: [nil, :y]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [11, 22],
          [33, 44]
        ]
      >

  ### Adding tensors with broadcasting

      iex> Nx.add(Nx.tensor([[1], [2]], names: [nil, :y]), Nx.tensor([[10, 20]], names: [:x, nil]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [11, 21],
          [12, 22]
        ]
      >

      iex> Nx.add(Nx.tensor([[10, 20]], names: [:x, nil]), Nx.tensor([[1], [2]], names: [nil, :y]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [11, 21],
          [12, 22]
        ]
      >

      iex> Nx.add(Nx.tensor([[1], [2]], names: [:x, nil]), Nx.tensor([[10, 20], [30, 40]]))
      #Nx.Tensor<
        s64[x: 2][2]
        [
          [11, 21],
          [32, 42]
        ]
      >

      iex> Nx.add(Nx.tensor([[1, 2]]), Nx.tensor([[10, 20], [30, 40]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [11, 22],
          [31, 42]
        ]
      >

  """
  def add(left, right), do: element_wise_bin_op(left, right, :add, & &1)

  @doc """
  Element-wise subtraction of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Subtracting scalars

      iex> Nx.subtract(1, 2)
      #Nx.Tensor<
        s64
        -1
      >

  ### Subtracting tensors and scalars

      iex> Nx.subtract(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        s64[data: 3]
        [0, 1, 2]
      >

      iex> Nx.subtract(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [0.0, -1.0, -2.0]
      >

  ### Subtracting tensors

      iex> Nx.subtract(Nx.tensor([[1], [2]], names: [:x, :y]), Nx.tensor([[10, 20]], names: [:x, :y]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [-9, -19],
          [-8, -18]
        ]
      >

      iex> Nx.subtract(Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, nil]), Nx.tensor([[10, 20]], type: {:s, 8}, names: [nil, :y]))
      #Nx.Tensor<
        s8[x: 2][y: 2]
        [
          [-9, -19],
          [-8, -18]
        ]
      >

      iex> Nx.subtract(Nx.tensor([[1], [2]], type: {:f, 32}, names: [nil, :y]), Nx.tensor([[10, 20]], type: {:f, 32}, names: [:x, nil]))
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [-9.0, -19.0],
          [-8.0, -18.0]
        ]
      >

  """
  def subtract(left, right), do: element_wise_bin_op(left, right, :subtract, & &1)

  @doc """
  Element-wise multiplication of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Multiplying scalars

      iex> Nx.multiply(1, 2)
      #Nx.Tensor<
        s64
        2
      >

  ### Multiplying tensors and scalars

      iex> Nx.multiply(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        s64[data: 3]
        [1, 2, 3]
      >

      iex> Nx.multiply(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [1.0, 2.0, 3.0]
      >

  ### Multiplying tensors

      iex> Nx.multiply(Nx.tensor([[1], [2]], names: [:x, :y]), Nx.tensor([[10, 20]], names: [:x, :y]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [10, 20],
          [20, 40]
        ]
      >

      iex> Nx.multiply(Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, nil]), Nx.tensor([[10, 20]], type: {:s, 8}, names: [nil, :y]))
      #Nx.Tensor<
        s8[x: 2][y: 2]
        [
          [10, 20],
          [20, 40]
        ]
      >

      iex> Nx.multiply(Nx.tensor([[1], [2]], type: {:f, 32}, names: [nil, :y]), Nx.tensor([[10, 20]], type: {:f, 32}, names: [:x, nil]))
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [10.0, 20.0],
          [20.0, 40.0]
        ]
      >

  """
  def multiply(left, right), do: element_wise_bin_op(left, right, :multiply, & &1)

  @doc """
  Element-wise power of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Power of scalars

      iex> Nx.power(2, 4)
      #Nx.Tensor<
        s64
        16
      >

  ### Power of tensors and scalars

      iex> Nx.power(Nx.tensor([1, 2, 3], names: [:data]), 2)
      #Nx.Tensor<
        s64[data: 3]
        [1, 4, 9]
      >

      iex> Nx.power(2, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [2.0, 4.0, 8.0]
      >

  ### Power of tensors

      iex> Nx.power(Nx.tensor([[2], [3]], names: [:x, nil]), Nx.tensor([[4, 5]], names: [nil, :y]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [16, 32],
          [81, 243]
        ]
      >

  """
  def power(left, right), do: element_wise_bin_op(left, right, :power, & &1)

  @doc """
  Element-wise remainder of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Remainder of scalars

      iex> Nx.remainder(1, 2)
      #Nx.Tensor<
        s64
        1
      >

  ### Remainder of tensors and scalars

      iex> Nx.remainder(Nx.tensor([1, 2, 3], names: [:data]), 2)
      #Nx.Tensor<
        s64[data: 3]
        [1, 0, 1]
      >

      iex> Nx.remainder(2, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [0.0, 0.0, 2.0]
      >

  ### Remainder of tensors

      iex> Nx.remainder(Nx.tensor([[10], [20]], names: [:x, :y]), Nx.tensor([[3, 4]], names: [nil, :y]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [1, 2],
          [2, 0]
        ]
      >

  """
  def remainder(left, right), do: element_wise_bin_op(left, right, :remainder, & &1)

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

      iex> Nx.divide(1, 2)
      #Nx.Tensor<
        f64
        0.5
      >

  ### Dividing tensors and scalars

      iex> Nx.divide(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        f64[data: 3]
        [1.0, 2.0, 3.0]
      >

      iex> Nx.divide(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [1.0, 0.5, 0.3333333333333333]
      >

  ### Dividing tensors

      iex> Nx.divide(Nx.tensor([[1], [2]], names: [:x, nil]), Nx.tensor([[10, 20]], names: [nil, :y]))
      #Nx.Tensor<
        f64[x: 2][y: 2]
        [
          [0.1, 0.05],
          [0.2, 0.1]
        ]
      >

      iex> Nx.divide(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}, names: [:x, :y]))
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      >

      iex> Nx.divide(Nx.tensor([[1], [2]], type: {:f, 32}, names: [:x, nil]), Nx.tensor([[10, 20]], type: {:f, 32}, names: [nil, :y]))
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      >

  """
  def divide(left, right), do: element_wise_bin_op(left, right, :divide, &Nx.Type.to_floating/1)

  @doc """
  Element-wise arc tangent of two tensors.

  If a number is given, it is converted to a tensor.

  It always returns a float tensor. If any of the input
  tensors are not float, they are converted to f64.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Arc tangent between scalars

      iex> Nx.arctan2(1, 2)
      #Nx.Tensor<
        f64
        0.4636476090008061
      >

  ### Arc tangent between tensors and scalars

      iex> Nx.arctan2(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        f64[data: 3]
        [0.7853981633974483, 1.1071487177940904, 1.2490457723982544]
      >

      iex> Nx.arctan2(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [0.7853981633974483, 0.4636476090008061, 0.3217505543966422]
      >

  ### Arc tangent between tensors

      # Note there is a bug in Erlang/OTP 23.0 and earlier where the compiler
      # optimizes -0.0 away as 0.0. So we do: -1.0*(Integer.parse("0")|>elem(0))
      iex> pos_and_neg_zero_x = Nx.multiply(Nx.tensor([[-1.0], [1.0]]), 0.0)
      iex> pos_and_neg_zero_y = Nx.multiply(Nx.tensor([-1.0, 1.0]), 0.0)
      iex> t = Nx.arctan2(pos_and_neg_zero_x, pos_and_neg_zero_y)
      iex> Nx.to_binary(t)
      <<-3.141592653589793::float-64-native, (-1.0*(Integer.parse("0")|>elem(0)))::float-64-native,
        3.141592653589793::float-64-native, 0.0::float-64-native>>
      iex> Nx.shape(t)
      {2, 2}

  """
  def arctan2(left, right), do: element_wise_bin_op(left, right, :arctan2, &Nx.Type.to_floating/1)

  @doc """
  Element-wise maximum of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Max between scalars

      iex> Nx.max(1, 2)
      #Nx.Tensor<
        s64
        2
      >

  ### Max between tensors and scalars

      iex> Nx.max(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        s64[data: 3]
        [1, 2, 3]
      >

      iex> Nx.max(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [1.0, 2.0, 3.0]
      >

  ### Max between tensors

      iex> Nx.max(Nx.tensor([[1], [2]], names: [:x, :y]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [10, 20],
          [10, 20]
        ]
      >

      iex> Nx.max(Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, nil]), Nx.tensor([[10, 20]], type: {:s, 8}))
      #Nx.Tensor<
        s8[x: 2][2]
        [
          [10, 20],
          [10, 20]
        ]
      >

      iex> Nx.max(Nx.tensor([[1], [2]], type: {:f, 32}, names: [:x, nil]), Nx.tensor([[10, 20]], type: {:f, 32}, names: [nil, :y]))
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [10.0, 20.0],
          [10.0, 20.0]
        ]
      >

  """
  def max(left, right), do: element_wise_bin_op(left, right, :max, & &1)

  @doc """
  Element-wise minimum of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Min between scalars

      iex> Nx.min(1, 2)
      #Nx.Tensor<
        s64
        1
      >

  ### Min between tensors and scalars

      iex> Nx.min(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        s64[data: 3]
        [1, 1, 1]
      >

      iex> Nx.min(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f64[data: 3]
        [1.0, 1.0, 1.0]
      >

  ### Min between tensors

      iex> Nx.min(Nx.tensor([[1], [2]], names: [:x, nil]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[x: 2][2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> Nx.min(Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, :y]), Nx.tensor([[10, 20]], type: {:s, 8}))
      #Nx.Tensor<
        s8[x: 2][y: 2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> Nx.min(Nx.tensor([[1], [2]], type: {:f, 32}, names: [:x, nil]), Nx.tensor([[10, 20]], type: {:f, 32}, names: [nil, :y]))
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [1.0, 1.0],
          [2.0, 2.0]
        ]
      >

  """
  def min(left, right), do: element_wise_bin_op(left, right, :min, & &1)

  ## Bitwise ops

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

      iex> Nx.bitwise_and(1, 0)
      #Nx.Tensor<
        s64
        0
      >

  ### bitwise and between tensors and scalars

      iex> Nx.bitwise_and(Nx.tensor([0, 1, 2], names: [:data]), 1)
      #Nx.Tensor<
        s64[data: 3]
        [0, 1, 0]
      >

      iex> Nx.bitwise_and(Nx.tensor([0, -1, -2], names: [:data]), -1)
      #Nx.Tensor<
        s64[data: 3]
        [0, -1, -2]
      >

  ### bitwise and between tensors

      iex> Nx.bitwise_and(Nx.tensor([0, 0, 1, 1], names: [:data]), Nx.tensor([0, 1, 0, 1]))
      #Nx.Tensor<
        s64[data: 4]
        [0, 0, 0, 1]
      >

  ### Error cases

      iex> Nx.bitwise_and(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_and(left, right),
    do: element_wise_bin_op(left, right, :bitwise_and, &assert_bitwise_type!/1)

  @doc """
  Element-wise bitwise OR of two tensors.

  Only integer tensors are supported. If a float or
  complex tensor is given, an error is raised.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### bitwise or between scalars

      iex> Nx.bitwise_or(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### bitwise or between tensors and scalars

      iex> Nx.bitwise_or(Nx.tensor([0, 1, 2], names: [:data]), 1)
      #Nx.Tensor<
        s64[data: 3]
        [1, 1, 3]
      >

      iex> Nx.bitwise_or(Nx.tensor([0, -1, -2], names: [:data]), -1)
      #Nx.Tensor<
        s64[data: 3]
        [-1, -1, -1]
      >

  ### bitwise or between tensors

      iex> Nx.bitwise_or(Nx.tensor([0, 0, 1, 1], names: [:data]), Nx.tensor([0, 1, 0, 1], names: [:data]))
      #Nx.Tensor<
        s64[data: 4]
        [0, 1, 1, 1]
      >

  ### Error cases

      iex> Nx.bitwise_or(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_or(left, right),
    do: element_wise_bin_op(left, right, :bitwise_or, &assert_bitwise_type!/1)

  @doc """
  Element-wise bitwise XOR of two tensors.

  Only integer tensors are supported. If a float or complex
  tensor is given, an error is raised.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Bitwise xor between scalars

      iex> Nx.bitwise_xor(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### Bitwise xor and between tensors and scalars

      iex> Nx.bitwise_xor(Nx.tensor([1, 2, 3], names: [:data]), 2)
      #Nx.Tensor<
        s64[data: 3]
        [3, 0, 1]
      >

      iex> Nx.bitwise_xor(Nx.tensor([-1, -2, -3], names: [:data]), 2)
      #Nx.Tensor<
        s64[data: 3]
        [-3, -4, -1]
      >

  ### Bitwise xor between tensors

      iex> Nx.bitwise_xor(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0, 1, 0, 1], names: [:data]))
      #Nx.Tensor<
        s64[data: 4]
        [0, 1, 1, 0]
      >

  ### Error cases

      iex> Nx.bitwise_xor(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_xor(left, right),
    do: element_wise_bin_op(left, right, :bitwise_xor, &assert_bitwise_type!/1)

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

      iex> Nx.left_shift(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### Left shift between tensors and scalars

      iex> Nx.left_shift(Nx.tensor([1, 2, 3], names: [:data]), 2)
      #Nx.Tensor<
        s64[data: 3]
        [4, 8, 12]
      >

  ### Left shift between tensors

      iex> Nx.left_shift(Nx.tensor([1, 1, -1, -1], names: [:data]), Nx.tensor([1, 2, 3, 4], names: [:data]))
      #Nx.Tensor<
        s64[data: 4]
        [2, 4, -8, -16]
      >

  ### Error cases

      iex> Nx.left_shift(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}

      iex> Nx.left_shift(Nx.tensor(1), -1)
      ** (ArgumentError) cannot left shift by -1
  """
  def left_shift(left, right),
    do: element_wise_bin_op(left, right, :left_shift, &assert_bitwise_type!/1)

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

      iex> Nx.right_shift(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### Right shift between tensors and scalars

      iex> Nx.right_shift(Nx.tensor([2, 4, 8], names: [:data]), 2)
      #Nx.Tensor<
        s64[data: 3]
        [0, 1, 2]
      >

  ### Right shift between tensors

      iex> Nx.right_shift(Nx.tensor([16, 32, -64, -128], names: [:data]), Nx.tensor([1, 2, 3, 4]))
      #Nx.Tensor<
        s64[data: 4]
        [8, 8, -8, -8]
      >

  ### Error cases

      iex> Nx.right_shift(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}

      iex> Nx.right_shift(Nx.tensor(1), -1)
      ** (ArgumentError) cannot right shift by -1
  """
  def right_shift(left, right),
    do: element_wise_bin_op(left, right, :right_shift, &assert_bitwise_type!/1)

  @doc """
  Element-wise equality comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Comparison of scalars

      iex> Nx.equal(1, 2)
      #Nx.Tensor<
        u8
        0
      >

  ### Comparison of tensors and scalars

      iex> Nx.equal(1, Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 0, 0]
      >

  ### Comparison of tensors

      iex> Nx.equal(Nx.tensor([1, 2, 3], names: [:data]), Nx.tensor([1, 2, 5]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 1, 0]
      >

      iex> Nx.equal(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, nil]), Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        u8[x: 2][3]
        [
          [1, 1, 1],
          [0, 0, 0]
        ]
      >
  """
  def equal(left, right), do: element_wise_pred_op(left, right, :equal)

  @doc """
  Element-wise logical and of two tensors.

  Zero is considered false, any other number is considered
  true.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

      iex> Nx.logical_and(1, Nx.tensor([-1, 0, 1], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 0, 1]
      >

      iex> Nx.logical_and(Nx.tensor([-1, 0, 1], names: [:data]), Nx.tensor([[-1], [0], [1]]))
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 0, 1],
          [0, 0, 0],
          [1, 0, 1]
        ]
      >

      iex> Nx.logical_and(Nx.tensor([-1.0, 0.0, 1.0], names: [:data]), Nx.tensor([[-1], [0], [1]]))
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 0, 1],
          [0, 0, 0],
          [1, 0, 1]
        ]
      >
  """
  def logical_and(left, right), do: element_wise_pred_op(left, right, :logical_and)

  @doc """
  Element-wise logical or of two tensors.

  Zero is considered false, any other number is considered
  true.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

      iex> Nx.logical_or(0, Nx.tensor([-1, 0, 1], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 0, 1]
      >

      iex> Nx.logical_or(Nx.tensor([-1, 0, 1], names: [:data]), Nx.tensor([[-1], [0], [1]]))
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ]
      >

      iex> Nx.logical_or(Nx.tensor([-1.0, 0.0, 1.0], names: [:data]), Nx.tensor([[-1], [0], [1]]))
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ]
      >
  """
  def logical_or(left, right), do: element_wise_pred_op(left, right, :logical_or)

  @doc """
  Element-wise logical xor of two tensors.

  Zero is considered false, any other number is considered
  true.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

      iex> Nx.logical_xor(0, Nx.tensor([-1, 0, 1], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 0, 1]
      >

      iex> Nx.logical_xor(Nx.tensor([-1, 0, 1], names: [:data]), Nx.tensor([[-1], [0], [1]]))
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]
        ]
      >

      iex> Nx.logical_xor(Nx.tensor([-1.0, 0.0, 1.0], names: [:data]), Nx.tensor([[-1], [0], [1]]))
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]
        ]
      >

  """
  def logical_xor(left, right), do: element_wise_pred_op(left, right, :logical_xor)

  @doc """
  Element-wise logical not a tensor.

  Zero is considered false, any other number is considered
  true.

  ## Examples

      iex> Nx.logical_not(Nx.tensor([-1, 0, 1], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [0, 1, 0]
      >

      iex> Nx.logical_not(Nx.tensor([-1.0, 0.0, 1.0], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [0, 1, 0]
      >

  """
  def logical_not(tensor) do
    tensor = tensor!(tensor)
    type = tensor.type
    out = %T{shape: {}, type: type, names: []}
    zero = Nx.BinaryTensor.from_binary(out, number_to_binary(0, type))
    element_wise_pred_op(tensor, zero, :equal)
  end

  @doc """
  Element-wise not-equal comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Comparison of scalars

      iex> Nx.not_equal(1, 2)
      #Nx.Tensor<
        u8
        1
      >

  ### Comparison of tensor and scalar

      iex> Nx.not_equal(Nx.tensor([1, 2, 3], names: [:data]), Nx.tensor(1))
      #Nx.Tensor<
        u8[data: 3]
        [0, 1, 1]
      >

  ### Comparison of tensors

      iex> Nx.not_equal(Nx.tensor([1, 1, 2]), Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [0, 1, 1]
      >

      iex> Nx.not_equal(Nx.tensor([[1, 4, 2], [4, 5, 6]], names: [:x, :y]), Nx.tensor([[1, 3, 2], [4, 2, 1]], names: [:x, :y]))
      #Nx.Tensor<
        u8[x: 2][y: 3]
        [
          [0, 1, 0],
          [0, 1, 1]
        ]
      >
  """
  def not_equal(left, right), do: element_wise_pred_op(left, right, :not_equal)

  @doc """
  Element-wise greater than comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Comparison of scalars

      iex> Nx.greater(1, 2)
      #Nx.Tensor<
        u8
        0
      >

  ### Comparison of tensors and scalars

      iex> Nx.greater(1, Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [0, 0, 0]
      >

  ### Comparison of tensors

      iex> Nx.greater(Nx.tensor([1, 2, 3], names: [:data]), Nx.tensor([1, 2, 2]))
      #Nx.Tensor<
        u8[data: 3]
        [0, 0, 1]
      >

      iex> Nx.greater(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, :y]), Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        u8[x: 2][y: 3]
        [
          [0, 0, 0],
          [1, 1, 1]
        ]
      >
  """
  def greater(left, right), do: element_wise_pred_op(left, right, :greater)

  @doc """
  Element-wise less than comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Comparison of scalars

      iex> Nx.less(1, 2)
      #Nx.Tensor<
        u8
        1
      >

  ### Comparison of tensors and scalars

      iex> Nx.less(1, Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [0, 1, 1]
      >

  ### Comparison of tensors

      iex> Nx.less(Nx.tensor([1, 2, 1]), Nx.tensor([1, 2, 2], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [0, 0, 1]
      >

      iex> Nx.less(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]], names: [:x, :y]), Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        u8[x: 2][y: 3]
        [
          [0, 0, 0],
          [0, 0, 1]
        ]
      >
  """
  def less(left, right), do: element_wise_pred_op(left, right, :less)

  @doc """
  Element-wise greater than or equal comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Comparison of scalars

      iex> Nx.greater_equal(1, 2)
      #Nx.Tensor<
        u8
        0
      >

  ### Comparison of tensors and scalars

      iex> Nx.greater_equal(1, Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 0, 0]
      >

  ### Comparison of tensors

      iex> Nx.greater_equal(Nx.tensor([1, 2, 3], names: [:data]), Nx.tensor([1, 2, 2]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 1, 1]
      >

      iex> Nx.greater_equal(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, :y]), Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        u8[x: 2][y: 3]
        [
          [1, 1, 1],
          [1, 1, 1]
        ]
      >
  """
  def greater_equal(left, right), do: element_wise_pred_op(left, right, :greater_equal)

  @doc """
  Element-wise less than or equal comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Comparison of scalars

      iex> Nx.less_equal(1, 2)
      #Nx.Tensor<
        u8
        1
      >

  ### Comparison of tensors and scalars

      iex> Nx.less_equal(1, Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 1, 1]
      >

  ### Comparison of tensors

      iex> Nx.less_equal(Nx.tensor([1, 2, 3], names: [:data]), Nx.tensor([1, 2, 2]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 1, 0]
      >

      iex> Nx.less_equal(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Nx.tensor([1, 2, 3], names: [:y]))
      #Nx.Tensor<
        u8[2][y: 3]
        [
          [1, 1, 1],
          [0, 0, 0]
        ]
      >
  """
  def less_equal(left, right), do: element_wise_pred_op(left, right, :less_equal)

  @doc """
  Constructs a tensor from two tensors, based on a predicate.

  The resulting tensor is built by evaluating each element of
  `pred` and returning either the corresponding element from
  `on_true` or `on_false`.

  `pred` must either be `1` or `0` or a tensor of predicates
  with a shape that matches the largest shape between `s1` or `s2`.

  If the shape of `on_true` or `on_false` do not match the shape of
  `pred`, attemps to broadcast both so they match the shape of `pred`.

  ## Examples

      iex> Nx.select(1, Nx.tensor([1, 2, 3], names: [:x]), Nx.tensor([4, 5, 6], names: [:x]))
      #Nx.Tensor<
        s64[x: 3]
        [1, 2, 3]
      >

      iex> Nx.select(0, Nx.tensor([1, 2, 3], names: [:y]), Nx.tensor([4, 5, 6], names: [:y]))
      #Nx.Tensor<
        s64[y: 3]
        [4, 5, 6]
      >

      iex> Nx.select(0, Nx.tensor([[1, 2]], names: [:x, :y]), Nx.tensor([[3], [4]], names: [:x, :y]))
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [3, 3],
          [4, 4]
        ]
      >

      iex> Nx.select(Nx.tensor([0, 1, 0], names: [:x]), Nx.tensor([1, 2, 3], names: [:y]), Nx.tensor([4, 5, 6], names: [:z]))
      #Nx.Tensor<
        s64[x: 3]
        [4, 2, 6]
      >

      iex> x = Nx.tensor([2, 4, 6], names: [:x])
      iex> y = Nx.tensor([3, 2, 1])
      iex> Nx.select(Nx.greater(x, y), Nx.tensor([2, 4, 6], names: [:i]), Nx.tensor([1, 3, 5], names: [:j]))
      #Nx.Tensor<
        s64[x: 3]
        [1, 4, 6]
      >

      iex> x = Nx.tensor([2, 4, 6, 8, 10], names: [:x])
      iex> y = Nx.tensor([1, 6, 2, 11, 2], names: [:x])
      iex> Nx.select(Nx.greater(x, y), Nx.tensor(2), Nx.tensor([1, 3, 5, 7, 9], names: [:x]))
      #Nx.Tensor<
        s64[x: 5]
        [2, 3, 2, 7, 2]
      >
  """
  def select(pred, on_true, on_false) do
    output_type = binary_type(on_true, on_false)

    %T{shape: pred_shape, names: pred_names} = pred = tensor!(pred)
    %T{shape: true_shape, names: true_names} = on_true = tensor!(on_true)
    %T{shape: false_shape, names: false_names} = on_false = tensor!(on_false)

    {output_shape, output_names} =
      case pred_shape do
        {} ->
          Nx.Shape.binary_broadcast(true_shape, true_names, false_shape, false_names)

        _ ->
          {pred_shape, pred_names}
      end

    _ =
      Nx.Shape.broadcast!(
        true_shape,
        output_shape,
        Nx.Shape.broadcast_axes(true_shape, output_shape)
      )

    _ =
      Nx.Shape.broadcast!(
        false_shape,
        output_shape,
        Nx.Shape.broadcast_axes(false_shape, output_shape)
      )

    out = %{pred | shape: output_shape, type: output_type, names: output_names}
    impl!(pred, on_true, on_false).select(out, pred, on_true, on_false)
  end

  ## Unary ops

  for {name, {desc, code}} <- Nx.Shared.unary_math_funs() do
    formula = code |> Macro.to_string() |> String.replace("var!(x)", "x")

    {one, _} = Code.eval_quoted(code, x: 1)
    {two, _} = Code.eval_quoted(code, x: 2)
    {three, _} = Code.eval_quoted(code, x: 3)

    @doc """
    Calculates the #{desc} of each element in the tensor.

    It is equivalent to:

        #{formula}

    ## Examples

        iex> Nx.#{name}(1)
        #Nx.Tensor<
          f64
          #{one}
        >

        iex> Nx.#{name}(Nx.tensor([1.0, 2.0, 3.0], names: [:x]))
        #Nx.Tensor<
          f64[x: 3]
          [#{one}, #{two}, #{three}]
        >

    """
    def unquote(name)(tensor) do
      tensor = tensor!(tensor)
      type = Nx.Type.to_floating(tensor.type)
      impl!(tensor).unquote(name)(%{tensor | type: type}, tensor)
    end
  end

  @doc """
  Negates each element in the tensor.

  ## Examples

      iex> Nx.negate(1)
      #Nx.Tensor<
        s64
        -1
      >

      iex> Nx.negate(Nx.tensor([-1, 0, 1]))
      #Nx.Tensor<
        s64[3]
        [1, 0, -1]
      >

      iex> Nx.negate(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[3]
        [-1.0, -2.0, -3.0]
      >

  If an unsigned tensor is given, it works as `bitwise_not`:

      iex> Nx.negate(Nx.tensor([0, 1, 2], type: {:u, 8}, names: [:x]))
      #Nx.Tensor<
        u8[x: 3]
        [0, 255, 254]
      >

  """
  def negate(tensor) do
    tensor = tensor!(tensor)
    impl!(tensor).negate(tensor, tensor)
  end

  @doc """
  Computes the sign of each element in the tensor.

  If a number is less than zero, it returns -1.
  If a number is more than zero, it returns 1.
  Otherwise it returns zero (which may either be
  positive or negative for floats).

  ## Examples

      iex> Nx.sign(Nx.tensor([-2, -1, 0, 1, 2], names: [:x]))
      #Nx.Tensor<
        s64[x: 5]
        [-1, -1, 0, 1, 1]
      >

  """
  def sign(tensor) do
    tensor = tensor!(tensor)
    impl!(tensor).sign(tensor, tensor)
  end

  @doc """
  Computes the absolute value of each element in the tensor.

  ## Examples

      iex> Nx.abs(Nx.tensor([-2, -1, 0, 1, 2], names: [:x]))
      #Nx.Tensor<
        s64[x: 5]
        [2, 1, 0, 1, 2]
      >

  """
  def abs(tensor) do
    tensor = tensor!(tensor)

    case tensor.type do
      {:u, _} -> tensor
      _ -> impl!(tensor).abs(tensor, tensor)
    end
  end

  @doc """
  Applies bitwise not to each element in the tensor.

  ## Examples

      iex> Nx.bitwise_not(1)
      #Nx.Tensor<
        s64
        -2
      >

      iex> Nx.bitwise_not(Nx.tensor([-1, 0, 1], type: {:s, 8}, names: [:x]))
      #Nx.Tensor<
        s8[x: 3]
        [0, -1, -2]
      >

      iex> Nx.bitwise_not(Nx.tensor([0, 1, 254, 255], type: {:u, 8}, names: [:x]))
      #Nx.Tensor<
        u8[x: 4]
        [255, 254, 1, 0]
      >

  ### Error cases

      iex> Nx.bitwise_not(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_not(tensor) do
    tensor = tensor!(tensor)
    assert_bitwise_type!(tensor.type)
    impl!(tensor).bitwise_not(tensor, tensor)
  end

  @doc """
  Computes the bitwise population count of each element in the tensor.

  ## Examples

      iex> Nx.population_count(1)
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.population_count(-128)
      #Nx.Tensor<
        s64
        57
      >

      iex> Nx.population_count(Nx.tensor([0, 1, 254, 255], names: [:x]))
      #Nx.Tensor<
        s64[x: 4]
        [0, 1, 7, 8]
      >

      iex> Nx.population_count(Nx.tensor([0, 1, 126, 127, -1, -127, -128], type: {:s, 8}, names: [:x]))
      #Nx.Tensor<
        s8[x: 7]
        [0, 1, 6, 7, 8, 2, 1]
      >

  ### Error cases

      iex> Nx.population_count(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def population_count(tensor) do
    tensor = tensor!(tensor)
    assert_bitwise_type!(tensor.type)
    impl!(tensor).population_count(tensor, tensor)
  end

  @doc """
  Counts the number of leading zeros of each element in the tensor.

  ## Examples

      iex> Nx.count_leading_zeros(1)
      #Nx.Tensor<
        s64
        63
      >

      iex> Nx.count_leading_zeros(-1)
      #Nx.Tensor<
        s64
        0
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF], names: [:x]))
      #Nx.Tensor<
        s64[x: 4]
        [64, 60, 56, 48]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0xF000000000000000, 0x0F00000000000000], names: [:x]))
      #Nx.Tensor<
        s64[x: 2]
        [0, 4]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF], type: {:s, 32}, names: [:x]))
      #Nx.Tensor<
        s32[x: 4]
        [32, 28, 24, 16]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF], type: {:s, 16}, names: [:x]))
      #Nx.Tensor<
        s16[x: 4]
        [16, 12, 8, 0]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 1, 2, 4, 8, 16, 32, 64, -1, -128], type: {:s, 8}, names: [:x]))
      #Nx.Tensor<
        s8[x: 10]
        [8, 7, 6, 5, 4, 3, 2, 1, 0, 0]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128], type: {:u, 8}, names: [:x]))
      #Nx.Tensor<
        u8[x: 9]
        [8, 7, 6, 5, 4, 3, 2, 1, 0]
      >

  ### Error cases

      iex> Nx.count_leading_zeros(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def count_leading_zeros(tensor) do
    tensor = tensor!(tensor)
    assert_bitwise_type!(tensor.type)
    impl!(tensor).count_leading_zeros(tensor, tensor)
  end

  for {name, desc} <- [floor: "floor", ceil: "ceil", round: "round (away from zero)"] do
    [res1, res2, res3, res4] = Enum.map([-1.5, -0.5, 0.5, 1.5], &apply(:erlang, name, [&1]))

    @doc """
    Calculates the #{desc} of each element in the tensor.

    If a non-floating tensor is given, it is returned as is.
    If a floating tensor is given, then we apply the operation,
    but keep its type.

    ## Examples

        iex> Nx.#{name}(Nx.tensor([-1, 0, 1], names: [:x]))
        #Nx.Tensor<
          s64[x: 3]
          [-1, 0, 1]
        >

        iex> Nx.#{name}(Nx.tensor([-1.5, -0.5, 0.5, 1.5], names: [:x]))
        #Nx.Tensor<
          f64[x: 4]
          [#{res1}.0, #{res2}.0, #{res3}.0, #{res4}.0]
        >

    """
    def unquote(name)(tensor) do
      case tensor!(tensor) do
        %T{type: {type, _}} = tensor when type in [:s, :u] -> tensor
        %T{} = tensor -> impl!(tensor).unquote(name)(tensor, tensor)
      end
    end
  end

  ## Aggregate ops

  @doc """
  Returns a scalar tensor of value 1 if all of the
  tensor values are not zero. Otherwise the value is 0.

  If the `:axes` option is given, it aggregates over
  the given dimensions, effectively removing them.
  `axes: [0]` implies aggregating over the highest order
  dimension and so forth. If the axis is negative, then
  counts the axis from the back. For example, `axes: [-1]`
  will always aggregate all rows.

  ## Examples

      iex> Nx.all?(Nx.tensor([0, 1, 2]))
      #Nx.Tensor<
        u8
        0
      >

      iex> Nx.all?(Nx.tensor([[-1, 0, 1], [2, 3, 4]], names: [:x, :y]), axes: [:x])
      #Nx.Tensor<
        u8[y: 3]
        [1, 0, 1]
      >

      iex> Nx.all?(Nx.tensor([[-1, 0, 1], [2, 3, 4]], names: [:x, :y]), axes: [:y])
      #Nx.Tensor<
        u8[x: 2]
        [0, 1]
      >
  """
  def all?(tensor, opts \\ []) do
    assert_keys!(opts, [:axes])
    %{shape: shape, names: names} = tensor = tensor!(tensor)

    {shape, names, axes} =
      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes, names)
        {new_shape, new_names} = Nx.Shape.contract(shape, axes, names, false)
        {new_shape, new_names, axes}
      else
        {{}, [], nil}
      end

    out = %{tensor | type: {:u, 8}, shape: shape, names: names}
    impl!(tensor).all?(out, tensor, axes: axes)
  end

  @doc """
  Returns a scalar tensor of value 1 if any of the
  tensor values are not zero. Otherwise the value is 0.

  If the `:axes` option is given, it aggregates over
  the given dimensions, effectively removing them.
  `axes: [0]` implies aggregating over the highest order
  dimension and so forth. If the axis is negative, then
  counts the axis from the back. For example, `axes: [-1]`
  will always aggregate all rows.

  ## Examples

      iex> Nx.any?(Nx.tensor([0, 1, 2]))
      #Nx.Tensor<
        u8
        1
      >

      iex> Nx.any?(Nx.tensor([[0, 1, 0], [0, 1, 2]], names: [:x, :y]), axes: [:x])
      #Nx.Tensor<
        u8[y: 3]
        [0, 1, 1]
      >

      iex> Nx.any?(Nx.tensor([[0, 1, 0], [0, 1, 2]], names: [:x, :y]), axes: [:y])
      #Nx.Tensor<
        u8[x: 2]
        [1, 1]
      >
  """
  def any?(tensor, opts \\ []) do
    assert_keys!(opts, [:axes])
    %{shape: shape, names: names} = tensor = tensor!(tensor)

    {shape, names, axes} =
      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes, names)
        {new_shape, new_names} = Nx.Shape.contract(shape, axes, names, false)
        {new_shape, new_names, axes}
      else
        {{}, [], nil}
      end

    out = %{tensor | type: {:u, 8}, shape: shape, names: names}
    impl!(tensor).any?(out, tensor, axes: axes)
  end

  @doc """
  Returns the sum for the tensor.

  If the `:axes` option is given, it aggregates over
  the given dimensions, effectively removing them.
  `axes: [0]` implies aggregating over the highest order
  dimension and so forth. If the axis is negative, then
  counts the axis from the back. For example, `axes: [-1]`
  will always aggregate all rows.

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the summed
  axes to size 1.

  ## Examples

      iex> Nx.sum(Nx.tensor(42))
      #Nx.Tensor<
        s64
        42
      >

      iex> Nx.sum(Nx.tensor([1, 2, 3], names: [:x]))
      #Nx.Tensor<
        s64
        6
      >

      iex> Nx.sum(Nx.tensor([[1.0, 2.0], [3.0, 4.0]], names: [:x, :y]))
      #Nx.Tensor<
        f64
        10.0
      >

  Giving a tensor with low precision casts it to a higher
  precision to make sure the sum does not overflow:

      iex> Nx.sum(Nx.tensor([[101, 102], [103, 104]], type: {:u, 8}, names: [:x, :y]))
      #Nx.Tensor<
        u64
        410
      >

      iex> Nx.sum(Nx.tensor([[101, 102], [103, 104]], type: {:s, 8}, names: [:x, :y]))
      #Nx.Tensor<
        s64
        410
      >

  ### Aggregating over an axis

      iex> Nx.sum(Nx.tensor([1, 2, 3], names: [:x]), axes: [0])
      #Nx.Tensor<
        s64
        6
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:x])
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:y])
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [5, 7, 9],
          [17, 19, 21]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:z])
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [6, 15],
          [24, 33]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:x, :z])
      #Nx.Tensor<
        s64[y: 2]
        [30, 48]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:z])
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [6, 15],
          [24, 33]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [-3])
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

  ### Keeping dimensions

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:z], keep_axes: true)
      #Nx.Tensor<
        s64[x: 2][y: 2][z: 1]
        [
          [
            [6],
            [15]
          ],
          [
            [24],
            [33]
          ]
        ]
      >

  ### Errors

      iex> Nx.sum(Nx.tensor([[1, 2]]), axes: [2])
      ** (ArgumentError) given axis (2) invalid for shape with rank 2

  """
  def sum(tensor, opts \\ []) do
    assert_keys!(opts, [:axes, :keep_axes])
    keep_axes = opts[:keep_axes] || false

    %{shape: shape, type: type, names: names} = tensor = tensor!(tensor)

    {shape, names, axes} =
      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes, names)
        {new_shape, new_names} = Nx.Shape.contract(shape, axes, names, keep_axes)
        {new_shape, new_names, axes}
      else
        if keep_axes do
          shape = List.to_tuple(List.duplicate(1, Nx.rank(shape)))
          {shape, names, nil}
        else
          {{}, [], nil}
        end
      end

    type =
      case type do
        {:u, _} -> {:u, 64}
        {:s, _} -> {:s, 64}
        type -> type
      end

    impl!(tensor).sum(%{tensor | type: type, shape: shape, names: names}, tensor,
      axes: axes,
      keep_axes: keep_axes
    )
  end

  @doc """
  Returns the mean for the tensor.

  If the `:axis` option is given, it aggregates over
  that dimension, effectively removing it. `axes: [0]`
  implies aggregating over the highest order dimension
  and so forth. If the axis is negative, then counts
  the axis from the back. For example, `axes: [-1]` will
  always aggregate all rows.

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the averaged
  axes to size 1.

  ## Examples

      iex> Nx.mean(Nx.tensor(42))
      #Nx.Tensor<
        f64
        42.0
      >

      iex> Nx.mean(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f64
        2.0
      >

  ### Aggregating over an axis

      iex> Nx.mean(Nx.tensor([1, 2, 3], names: [:x]), axes: [0])
      #Nx.Tensor<
        f64
        2.0
      >

      iex> Nx.mean(Nx.tensor([1, 2, 3], type: {:u, 8}, names: [:x]), axes: [:x])
      #Nx.Tensor<
        f64
        2.0
      >

      iex> Nx.mean(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:x])
      #Nx.Tensor<
        f64[y: 2][z: 3]
        [
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]
        ]
      >

      iex> Nx.mean(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [:x, :z])
      #Nx.Tensor<
        f64[y: 2]
        [5.0, 8.0]
      >

      iex> Nx.mean(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [-1])
      #Nx.Tensor<
        f64[x: 2][y: 2]
        [
          [2.0, 5.0],
          [8.0, 11.0]
        ]
      >

  ### Keeping dimensions

      iex> Nx.mean(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z]), axes: [-1], keep_axes: true)
      #Nx.Tensor<
        f64[x: 2][y: 2][z: 1]
        [
          [
            [2.0],
            [5.0]
          ],
          [
            [8.0],
            [11.0]
          ]
        ]
      >

  """
  def mean(tensor, opts \\ []) do
    %T{shape: shape, names: names} = tensor = tensor!(tensor)

    mean_den =
      if axes = opts[:axes] do
        mean_den(shape, Nx.Shape.normalize_axes(shape, axes, names))
      else
        mean_den(shape, nil)
      end

    divide(sum(tensor, opts), mean_den)
  end

  defp mean_den(shape, nil), do: size(shape)
  defp mean_den(_shape, []), do: 1

  defp mean_den(shape, [axis | axes]) when axis >= 0,
    do: elem(shape, axis) * mean_den(shape, axes)

  defp mean_den(shape, [axis | axes]),
    do: elem(shape, tuple_size(shape) + axis) * mean_den(shape, axes)

  @doc """
  Returns the indices of the maximum values.

  ## Options

    * `:axis` - the axis to aggregate on. If no axis is given,
      returns the index of the absolute maximum value in the tensor.

    * `:tie_break` - how to break ties. one of `:high`, or `:low``.
      default behavior is to always return the lower index.

  ## Examples

      iex> Nx.argmax(4)
      #Nx.Tensor<
        s64
        0
      >

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]))
      #Nx.Tensor<
        s64
        10
      >

  If a tensor of floats is given, it still returns integers:

      iex> Nx.argmax(Nx.tensor([2.0, 4.0]))
      #Nx.Tensor<
        s64
        1
      >

  ### Aggregating over an axis

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), axis: :x)
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [1, 0, 0],
          [1, 1, 0]
        ]
      >

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [0, 0, 0],
          [0, 1, 0]
        ]
      >

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [0, 2],
          [0, 1]
        ]
      >

  ### Tie breaks

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), tie_break: :low, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [0, 0, 0],
          [0, 1, 0]
        ]
      >

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), tie_break: :high, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [0, 0, 1],
          [0, 1, 1]
        ]
      >
  """
  def argmax(tensor, opts \\ []) do
    argmin_or_max(tensor, :argmax, opts)
  end

  @doc """
  Returns the indices of the minimum values.

  ## Options

    * `:axis` - the axis to aggregate on. If no axis is given,
      returns the index of the absolute minimum value in the tensor.

    * `:tie_break` - how to break ties. one of `:high`, or `:low`.
      default behavior is to always return the lower index.

  ## Examples

      iex> Nx.argmin(4)
      #Nx.Tensor<
        s64
        0
      >

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]))
      #Nx.Tensor<
        s64
        4
      >

  If a tensor of floats is given, it still returns integers:

      iex> Nx.argmin(Nx.tensor([2.0, 4.0]))
      #Nx.Tensor<
        s64
        0
      >

  ### Aggregating over an axis

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), axis: :x)
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [0, 0, 0],
          [0, 0, 0]
        ]
      >

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), axis: 1)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [1, 1, 0],
          [1, 0, 0]
        ]
      >

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [1, 1],
          [1, 2]
        ]
      >

  ### Tie breaks

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), tie_break: :low, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [1, 1, 0],
          [1, 0, 0]
        ]
      >

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z]), tie_break: :high, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [1, 1, 1],
          [1, 0, 1]
        ]
      >
  """
  def argmin(tensor, opts \\ []) do
    argmin_or_max(tensor, :argmin, opts)
  end

  defp argmin_or_max(tensor, op, opts) do
    assert_keys!(opts, [:axis, :tie_break])

    tie_break =
      case opts[:tie_break] || :low do
        :high ->
          :high

        :low ->
          :low

        other ->
          raise ArgumentError,
                "unknown value for :tie_break, expected :high or :low, got: #{inspect(other)}"
      end

    %{shape: shape, names: names} = tensor = tensor!(tensor)

    {shape, names, axis} =
      if axis = opts[:axis] do
        axis = Nx.Shape.normalize_axis(shape, axis, names)
        {new_shape, new_names} = Nx.Shape.contract(shape, [axis], names, false)
        {new_shape, new_names, axis}
      else
        {{}, [], nil}
      end

    opts = [tie_break: tie_break, axis: axis]

    apply(impl!(tensor), op, [
      %{tensor | type: {:s, 64}, shape: shape, names: names},
      tensor,
      opts
    ])
  end

  @doc """
  Reduces over a tensor with the given accumulator.

  The given `fun` will receive two tensors and it must
  return the reduced value.

  The tensor may be reduced in parallel and the reducer
  function can be called with arguments in any order, the
  initial accumulator may be given multiples, and it may
  be non-deterministic. Therefore, the reduction function
  should be associative (or as close as possible to
  associativity considered floats themselves are not
  strictly associative).

  By default, it reduces all dimensions of the tensor and
  return a scalar. If the `:axes` option is given, it
  aggregates over multiple dimensions, effectively removing
  them. `axes: [0]` implies aggregating over the highest
  order dimension and so forth. If the axis is negative,
  then counts the axis from the back. For example,
  `axes: [-1]` will always aggregate all rows.

  The type of the returned tensor will be computed based on
  the given tensor and the initial value. For example,
  a tensor of integers with a float accumulator will be
  cast integer, as done by most binary operator. You can
  also pass a `:type` option to change this behaviour.

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the reduced
  axes to size 1.

  ## Examples

      iex> Nx.reduce(Nx.tensor(42), 0, fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64
        42
      >

      iex> Nx.reduce(Nx.tensor([1, 2, 3]), 0, fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64
        6
      >

      iex> Nx.reduce(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        f64
        10.0
      >

  ### Aggregating over axes

      iex> t = Nx.tensor([1, 2, 3], names: [:x])
      iex> Nx.reduce(t, 0, [axes: [:x]], fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64
        6
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.reduce(t, 0, [axes: [:x]], fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.reduce(t, 0, [axes: [:y]], fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [5, 7, 9],
          [17, 19, 21]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.reduce(t, 0, [axes: [:x, 2]], fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64[y: 2]
        [30, 48]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.reduce(t, 0, [axes: [-1]], fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [6, 15],
          [24, 33]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.reduce(t, 0, [axes: [:x]], fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.reduce(t, 0, [axes: [:x], keep_axes: true], fn x, y -> Nx.add(x, y) end)
      #Nx.Tensor<
        s64[x: 1][y: 2][z: 3]
        [
          [
            [8, 10, 12],
            [14, 16, 18]
          ]
        ]
      >

  """
  def reduce(tensor, acc, opts \\ [], fun) when is_function(fun, 2) do
    assert_keys!(opts, [:axes, :type, :keep_axes])
    keep_axes = opts[:keep_axes] || false
    type = Nx.Type.normalize!(opts[:type] || binary_type(tensor, acc))
    %{shape: shape, names: names} = tensor = tensor!(tensor)
    acc = tensor!(acc)

    {shape, names, axes} =
      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes, names)
        {new_shape, new_names} = Nx.Shape.contract(shape, axes, names, keep_axes)
        {new_shape, new_names, axes}
      else
        if keep_axes do
          shape = List.to_tuple(List.duplicate(1, Nx.rank(shape)))
          {shape, names, nil}
        else
          {{}, [], nil}
        end
      end

    out = %{tensor | type: type, shape: shape, names: names}
    impl!(tensor).reduce(out, tensor, acc, [axes: axes, keep_axes: keep_axes], fun)
  end

  @doc """
  Reduces elements in a window.

  The rank of the input tensor and window dimensions must match.

  Padding can either be `:valid`, `:same`, or a general padding
  configuration of edge-high and edge-low paddings.

  If specifying stride, the rank of the stride must match the
  rank of the input tensor.

  ### Examples

      iex> Nx.reduce_window(Nx.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [11, 12, 13, 14]]),
      ...>  :first, {2, 2},
      ...>  fn x, acc -> if acc == :first, do: x, else: max(x, acc) end
      ...> )
      #Nx.Tensor<
        s64[3][3]
        [
          [5, 6, 7],
          [8, 9, 10],
          [12, 13, 14]
        ]
      >

      iex> Nx.reduce_window(Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
      ...>  :first, {2, 2},
      ...>  [padding: :same, strides: {1, 1}],
      ...>  fn x, acc -> if acc == :first, do: x, else: max(x, acc) end
      ...> )
      #Nx.Tensor<
        s64[3][3]
        [
          [5, 6, 6],
          [8, 9, 9],
          [8, 9, 9]
        ]
      >

      iex> Nx.reduce_window(Nx.tensor([[1, 2, 3], [4, 5, 6]]),
      ...>  0, {1, 2},
      ...>  [padding: :same, strides: {1, 1}],
      ...>  fn x, acc -> x + acc end
      ...> )
      #Nx.Tensor<
        s64[2][3]
        [
          [3, 5, 3],
          [9, 11, 6]
        ]
      >
  """
  def reduce_window(tensor, acc, window_dimensions, opts \\ [], fun) do
    assert_keys!(opts, [:padding, :strides])
    %T{shape: shape} = tensor = tensor!(tensor)

    window_strides = opts[:strides] || List.to_tuple(List.duplicate(1, rank(tensor.shape)))
    padding = opts[:padding] || :valid

    padding_config =
      case padding do
        :valid ->
          for _ <- 0..(tuple_size(shape) - 1), do: {0, 0}

        :same ->
          padding_config = Nx.Shape.calculate_padding(shape, window_dimensions, window_strides)
          padding_config

        config when is_list(config) ->
          config

        _ ->
          raise ArgumentError,
                "invalid padding configuration, padding must be" <>
                  " :valid or :same, or a padding configuration for" <>
                  " the spatial dimensions of the input tensor"
      end

    padded_shape = Nx.Shape.pad(shape, Enum.map(padding_config, &Tuple.append(&1, 0)))
    output_shape = Nx.Shape.window(padded_shape, window_dimensions, window_strides)

    out = %{tensor | shape: output_shape}

    impl!(tensor).reduce_window(
      out,
      tensor,
      acc,
      window_dimensions,
      [padding: padding_config, strides: window_strides],
      fun
    )
  end

  @doc """
  Maps the given scalar function over the entire
  tensor.

  The type of the returned tensor will be the same type
  as the given tensor, unless the `:type` option is given.
  Therefore, keep in mind explicit casting may be necessary.
  For example, if you have an integer tensor and you convert
  it to a float, it will fail:

      iex> Nx.map(Nx.tensor([[1, 2, 3], [4, 5, 6]]), fn x -> x * 1.0 end)
      ** (ArgumentError) argument error

  You need to explicitly pass the output type in such cases:

      iex> Nx.map(Nx.tensor([[1, 2, 3], [4, 5, 6]]), [type: {:f, 32}], fn x -> x * 1.0 end)
      #Nx.Tensor<
        f32[2][3]
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]
        ]
      >

  Generally, you should prefer other using more idiomatic
  tensor operators to this function.

  ### Examples

      iex> Nx.map(Nx.tensor([[1, 2, 3], [4, 5, 6]]), fn x -> x + 1 end)
      #Nx.Tensor<
        s64[2][3]
        [
          [2, 3, 4],
          [5, 6, 7]
        ]
      >

      iex> Nx.map(Nx.tensor(1), fn x -> x + 1 end)
      #Nx.Tensor<
        s64
        2
      >

      iex> Nx.map(Nx.tensor([[1, 2, 3], [4, 5, 6]]), [type: {:f, 64}], fn x -> x + 1 end)
      #Nx.Tensor<
        f64[2][3]
        [
          [2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0]
        ]
      >

      iex> Nx.map(Nx.tensor([[1, 2, 3], [4, 5, 6]]), [type: {:f, 64}], fn x -> Nx.add(x, 1) end)
      #Nx.Tensor<
        f64[2][3]
        [
          [2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0]
        ]
      >
  """
  def map(tensor, opts \\ [], fun) do
    assert_keys!(opts, [:type])
    %T{type: type} = tensor = tensor!(tensor)
    output_type = opts[:type] || type
    out = %{tensor | type: output_type}
    impl!(tensor).map(out, tensor, fun)
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

  For a more general `dot` function where you control which axes contract,
  see `dot/4`.

  ## Examples

  ### Dot product of scalars

      iex> Nx.dot(5, 5)
      #Nx.Tensor<
        s64
        25
      >

      iex> Nx.dot(-2.0, 5.0)
      #Nx.Tensor<
        f64
        -10.0
      >

      iex> Nx.dot(2, 2.0)
      #Nx.Tensor<
        f64
        4.0
      >

  ### Dot product of vectors

      iex> Nx.dot(Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
      #Nx.Tensor<
        s64
        32
      >

      iex> Nx.dot(Nx.tensor([2.0, 4.0, 3.0, 5.0]), Nx.tensor([1.0, 2.0, 3.0, 4.0]))
      #Nx.Tensor<
        f64
        39.0
      >

      iex> Nx.dot(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f64
        14.0
      >

  ### Dot product of matrices

      iex> Nx.dot(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:i, :j]), Nx.tensor([[7, 8], [9, 10], [11, 12]], names: [:x, :y]))
      #Nx.Tensor<
        s64[i: 2][y: 2]
        [
          [58, 64],
          [139, 154]
        ]
      >

      iex> Nx.dot(Nx.tensor([[10.0, 13.0, 14.0, 15.0], [59.0, 20.0, 10.0, 30.0]], names: [:i, :j]), Nx.tensor([[2.0, 4.0], [5.0, 1.0], [6.0, 8.0], [9.0, 10.0]], names: [:x, :y]))
      #Nx.Tensor<
        f64[i: 2][y: 2]
        [
          [304.0, 315.0],
          [548.0, 636.0]
        ]
      >

      iex> Nx.dot(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:i, :j]), Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], names: [:x, :y]))
      #Nx.Tensor<
        f64[i: 2][y: 2]
        [
          [58.0, 64.0],
          [139.0, 154.0]
        ]
      >

  ### Dot product of vector and n-d tensor

      iex> Nx.dot(Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], names: [:i, :j, :k]), Nx.tensor([5, 10], names: [:x]))
      #Nx.Tensor<
        s64[i: 2][j: 2]
        [
          [25, 55],
          [85, 115]
        ]
      >

      iex> Nx.dot(Nx.tensor([5, 10], names: [:x]), Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:i, :j]))
      #Nx.Tensor<
        s64[j: 3]
        [45, 60, 75]
      >

      iex> Nx.dot(Nx.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], names: [:shard, :batch, :x, :y, :z]), Nx.tensor([2.0, 2.0], names: [:data]))
      #Nx.Tensor<
        f64[shard: 1][batch: 1][x: 2][y: 2]
        [
          [
            [
              [6.0, 14.0],
              [22.0, 30.0]
            ]
          ]
        ]
      >

  ### Dot product of n-D and m-D tensor

      iex> a = Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:x, :y, :z])
      iex> b = Nx.tensor([[[1, 2, 3], [3, 4, 5], [5, 6, 7]]], names: [:i, :j, :k])
      iex> Nx.dot(a, b)
      #Nx.Tensor<
        s64[x: 2][y: 3][i: 1][k: 3]
        [
          [
            [
              [22, 28, 34]
            ],
            [
              [49, 64, 79]
            ],
            [
              [76, 100, 124]
            ]
          ],
          [
            [
              [22, 28, 34]
            ],
            [
              [49, 64, 79]
            ],
            [
              [76, 100, 124]
            ]
          ]
        ]
      >

  ### Error Cases

      iex> Nx.dot(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2]))
      ** (ArgumentError) dot/zip expects shapes to be compatible, dimension 0 of left-side (3) does not equal dimension 0 of right-side (2)
  """
  def dot(t1, t2) do
    %T{shape: s1} = t1 = tensor!(t1)
    %T{shape: s2} = t2 = tensor!(t2)

    case {tuple_size(s1), tuple_size(s2)} do
      {0, _} -> multiply(t1, t2)
      {_, 0} -> multiply(t1, t2)
      {n, 1} -> dot(t1, [n - 1], t2, [0])
      {1, m} -> dot(t1, [0], t2, [m - 2])
      {n, m} when n >= 2 and m >= 2 -> dot(t1, [n - 1], t2, [m - 2])
    end
  end

  @doc """
  Computes the dot product of two tensors over the given axes.

  The dot product is computed by multiplying the values from `t1`
  given by `axes1` against the values from `t2` given by `axes2`.
  For instance, the first axis in `axes1` will be matched against
  the first axis in `axes2` and so on. The axes given by `axes1`
  and `axes2` are effectively removed from the final tensor, which
  is why they are often called the contraction axes.

  ## Examples

      iex> t1 = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      iex> t2 = Nx.tensor([[10, 20], [30, 40]], names: [:height, :width])
      iex> Nx.dot(t1, [0], t2, [0])
      #Nx.Tensor<
        s64[y: 2][width: 2]
        [
          [100, 140],
          [140, 200]
        ]
      >
      iex> Nx.dot(t1, [0], t2, [1])
      #Nx.Tensor<
        s64[y: 2][height: 2]
        [
          [70, 150],
          [100, 220]
        ]
      >
      iex> Nx.dot(t1, [1], t2, [0])
      #Nx.Tensor<
        s64[x: 2][width: 2]
        [
          [70, 100],
          [150, 220]
        ]
      >
      iex> Nx.dot(t1, [1], t2, [1])
      #Nx.Tensor<
        s64[x: 2][height: 2]
        [
          [50, 110],
          [110, 250]
        ]
      >
      iex> Nx.dot(t1, [0, 1], t2, [0, 1])
      #Nx.Tensor<
        s64
        300
      >

  If no axes are given, it works like `outer/2`:

      iex> t1 = Nx.tensor([[1, 2], [3, 4]])
      iex> t2 = Nx.tensor([[10, 20], [30, 40]])
      iex> Nx.dot(t1, [], t2, [])
      #Nx.Tensor<
        s64[2][2][2][2]
        [
          [
            [
              [10, 20],
              [30, 40]
            ],
            [
              [20, 40],
              [60, 80]
            ]
          ],
          [
            [
              [30, 60],
              [90, 120]
            ],
            [
              [40, 80],
              [120, 160]
            ]
          ]
        ]
      >

  """
  def dot(t1, axes1, t2, axes2) do
    output_type = binary_type(t1, t2)
    %T{shape: s1, names: names1} = t1 = tensor!(t1)
    %T{shape: s2, names: names2} = t2 = tensor!(t2)
    axes1 = Nx.Shape.normalize_axes(s1, axes1, names1)
    axes2 = Nx.Shape.normalize_axes(s2, axes2, names2)
    {output_shape, output_names} = Nx.Shape.zip_reduce(s1, axes1, names1, s2, axes2, names2)
    out = %{t1 | type: output_type, names: output_names, shape: output_shape}
    impl!(t1, t2).dot(out, t1, axes1, t2, axes2)
  end

  @doc """
  Computes the outer product of two tensors.

  The output is always a two-dimensional tensor.

  ## Examples

      iex> Nx.outer(Nx.tensor([1, 2, 3], names: [:x]), 100)
      #Nx.Tensor<
        s64[x: 3][1]
        [
          [100],
          [200],
          [300]
        ]
      >

      iex> Nx.outer(Nx.tensor([1, 2, 3], names: [:x]), Nx.tensor([10, 20], names: [:y]))
      #Nx.Tensor<
        s64[x: 3][y: 2]
        [
          [10, 20],
          [20, 40],
          [30, 60]
        ]
      >

      iex> Nx.outer(Nx.tensor([[1, 2], [3, 4]], names: [:x, :y]), Nx.tensor([10, 20, 30], names: [:z]))
      #Nx.Tensor<
        s64[x: 4][z: 3]
        [
          [10, 20, 30],
          [20, 40, 60],
          [30, 60, 90],
          [40, 80, 120]
        ]
      >

  """
  def outer(t1, t2) do
    type = binary_type(t1, t2)
    %T{shape: s1, names: n1} = t1 = tensor!(t1)
    %T{shape: s2, names: n2} = t2 = tensor!(t2)
    new_shape = {size(s1), size(s2)}

    names =
      case {n1, n2} do
        {[], rhs} -> [nil, List.last(rhs)]
        {lhs, rhs} -> [hd(lhs), List.last(rhs)]
      end

    impl!(t1, t2).outer(%{t1 | type: type, shape: new_shape, names: names}, t1, t2)
  end

  @doc """
  Transposes a tensor to the given `axes`.

  If no axes are given, the default behavior is to
  reverse the order of the original tensor's axes.

  The axes is a list of integers or dimension names
  containing how the new dimensions must be ordered.
  The highest dimension is zero.

  ## Examples

      iex> Nx.transpose(Nx.tensor(1))
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.transpose(Nx.iota({2, 3, 4}, names: [:x, :y, :z]))
      #Nx.Tensor<
        s64[z: 4][y: 3][x: 2]
        [
          [
            [0, 12],
            [4, 16],
            [8, 20]
          ],
          [
            [1, 13],
            [5, 17],
            [9, 21]
          ],
          [
            [2, 14],
            [6, 18],
            [10, 22]
          ],
          [
            [3, 15],
            [7, 19],
            [11, 23]
          ]
        ]
      >

      iex> Nx.transpose(Nx.tensor(1), axes: [])
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.transpose(Nx.iota({2, 3, 4}, names: [:batch, :x, :y]), axes: [2, 1, :batch])
      #Nx.Tensor<
        s64[y: 4][x: 3][batch: 2]
        [
          [
            [0, 12],
            [4, 16],
            [8, 20]
          ],
          [
            [1, 13],
            [5, 17],
            [9, 21]
          ],
          [
            [2, 14],
            [6, 18],
            [10, 22]
          ],
          [
            [3, 15],
            [7, 19],
            [11, 23]
          ]
        ]
      >

      iex> Nx.transpose(Nx.iota({2, 3, 4}, names: [:batch, :x, :y]), axes: [:y, :batch, :x])
      #Nx.Tensor<
        s64[y: 4][batch: 2][x: 3]
        [
          [
            [0, 4, 8],
            [12, 16, 20]
          ],
          [
            [1, 5, 9],
            [13, 17, 21]
          ],
          [
            [2, 6, 10],
            [14, 18, 22]
          ],
          [
            [3, 7, 11],
            [15, 19, 23]
          ]
        ]
      >

      iex> Nx.transpose(Nx.iota({2, 3, 4}, names: [:batch, :x, :y]), axes: [:batch, :y, :x])
      #Nx.Tensor<
        s64[batch: 2][y: 4][x: 3]
        [
          [
            [0, 4, 8],
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11]
          ],
          [
            [12, 16, 20],
            [13, 17, 21],
            [14, 18, 22],
            [15, 19, 23]
          ]
        ]
      >

  ### Errors

      iex> Nx.transpose(Nx.iota({2, 2}, names: [:batch, :x]), axes: [:batch])
      ** (ArgumentError) expected length of permutation (1) to match rank of shape (2)

      iex> Nx.transpose(Nx.iota({2, 2}), axes: [1, 2])
      ** (ArgumentError) given axis (2) invalid for shape with rank 2

  """
  def transpose(tensor, opts \\ []) do
    %{shape: shape, names: names} = tensor = tensor!(tensor)
    axes = opts[:axes] || Nx.Shape.transpose_axes(shape)
    axes = Nx.Shape.normalize_axes(shape, axes, names)

    if axes == Nx.axes(shape) do
      tensor
    else
      {shape, names} = Nx.Shape.transpose(shape, axes, names)
      impl!(tensor).transpose(%{tensor | shape: shape, names: names}, tensor, axes: axes)
    end
  end

  @doc """
  Reverses the tensor in the given dimensions.

  If no axes are provided, reverses every axis.

  You can pass either names or numbers for the reverse
  dimensions. Dimensions must be unique, but they do not
  have to be successive.

  ### Examples

      iex> Nx.reverse(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        s64[3]
        [3, 2, 1]
      >

      iex> Nx.reverse(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      #Nx.Tensor<
        s64[2][3]
        [
          [6, 5, 4],
          [3, 2, 1]
        ]
      >

      iex> Nx.reverse(Nx.tensor([1, 2, 3], names: [:x]), axes: [:x])
      #Nx.Tensor<
        s64[x: 3]
        [3, 2, 1]
      >

      iex> Nx.reverse(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y]), axes: [:x])
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [4, 5, 6],
          [1, 2, 3]
        ]
      >

      iex> Nx.reverse(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y]), axes: [:y])
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [3, 2, 1],
          [6, 5, 4]
        ]
      >

      iex> Nx.reverse(Nx.iota({2, 2, 2}, type: {:f, 32}, names: [:x, :y, :z]), axes: [:x, :z])
      #Nx.Tensor<
        f32[x: 2][y: 2][z: 2]
        [
          [
            [5.0, 4.0],
            [7.0, 6.0]
          ],
          [
            [1.0, 0.0],
            [3.0, 2.0]
          ]
        ]
      >
  """
  def reverse(tensor, opts \\ []) do
    assert_keys!(opts, [:axes])
    %{shape: shape, names: names} = tensor = tensor!(tensor)
    axes = opts[:axes] || axes(shape)

    case Nx.Shape.normalize_axes(shape, axes, names) do
      [] -> tensor
      axes -> impl!(tensor).reverse(tensor, tensor, axes: Enum.sort(axes))
    end
  end

  ## Conv

  @doc """
  Convolution operation.

  ### Examples

      iex> lhs = Nx.iota({9})
      iex> lhs = Nx.reshape(lhs, {1, 1, 3, 3})
      iex> rhs = Nx.iota({4})
      iex> rhs = Nx.reshape(rhs, {4, 1, 1, 1})
      iex> Nx.conv(lhs, rhs, strides: {1, 1})
      #Nx.Tensor<
        f64[1][4][3][3]
        [
          [
            [
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]
            ],
            [
              [0.0, 1.0, 2.0],
              [3.0, 4.0, 5.0],
              [6.0, 7.0, 8.0]
            ],
            [
              [0.0, 2.0, 4.0],
              [6.0, 8.0, 10.0],
              [12.0, 14.0, 16.0]
            ],
            [
              [0.0, 3.0, 6.0],
              [9.0, 12.0, 15.0],
              [18.0, 21.0, 24.0]
            ]
          ]
        ]
      >
  """
  def conv(tensor, kernel, opts \\ []) when is_list(opts) do
    assert_keys!(opts, [:padding, :strides, :input_dilation, :kernel_dilation])

    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1

    %{shape: input_shape, names: input_names} = tensor = tensor!(tensor)
    %{shape: kernel_shape, names: kernel_names} = kernel = tensor!(kernel)

    if rank(input_shape) < 3 do
      raise ArgumentError, "input shape in conv requires at least rank 3"
    end

    if rank(kernel_shape) < 3 do
      raise ArgumentError, "kernel shape in conv requires at least rank 3"
    end

    tensor_input_channels = elem(input_shape, 1)
    kernel_input_channels = elem(kernel_shape, 1)

    if tensor_input_channels != kernel_input_channels do
      raise ArgumentError,
            "size of input dimension 1 must match size of kernel dimension 1," <>
              " got #{tensor_input_channels} != #{kernel_input_channels}"
    end

    filter_shape =
      kernel_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    spatial_dims =
      input_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    strides = opts[:strides] || List.to_tuple(List.duplicate(1, rank(spatial_dims)))

    if rank(strides) != rank(spatial_dims) do
      raise ArgumentError, "must specify stride for each spatial dimension"
    end

    cond do
      is_integer(input_dilation) and input_dilation < 1 ->
        raise ArgumentError,
              "input dilation must be greater than or equal to 1, got #{input_dilation}"

      is_tuple(input_dilation) and rank(input_dilation) != rank(spatial_dims) ->
        raise ArgumentError,
              "must specify dilation for each spatial dimension of the input" <>
                " or specify an integer dilation factor"

      is_tuple(input_dilation) and Enum.any?(Tuple.to_list(input_dilation), &(&1 < 1)) ->
        raise ArgumentError, "input dilation of each dimension must be greater than or equal to 1"

      is_integer(kernel_dilation) and kernel_dilation < 1 ->
        raise ArgumentError,
              "kernel dilation must be greater than or equal to 1, got #{kernel_dilation}"

      is_tuple(kernel_dilation) and rank(kernel_dilation) != rank(filter_shape) ->
        raise ArgumentError,
              "must specify dilation for each spatial dimension of the kernel" <>
                " or specify an integer dilation factor"

      is_tuple(kernel_dilation) and Enum.any?(Tuple.to_list(kernel_dilation), &(&1 < 1)) ->
        raise ArgumentError,
              "kernel dilation of each dimension must be greater than or equal to 1"

      true ->
        :ok
    end

    kernel_dilation =
      if is_tuple(kernel_dilation),
        do: kernel_dilation,
        else: List.to_tuple(for _ <- 1..tuple_size(filter_shape), do: kernel_dilation)

    kernel_dilation_padding_config = [
      {0, 0, 0},
      {0, 0, 0} | Enum.map(Tuple.to_list(kernel_dilation), &{0, 0, &1 - 1})
    ]

    dilated_kernel_shape = Nx.Shape.pad(kernel_shape, kernel_dilation_padding_config)

    input_dilation =
      if is_tuple(input_dilation),
        do: input_dilation,
        else: List.to_tuple(for _ <- 1..tuple_size(spatial_dims), do: input_dilation)

    input_dilation_padding_config = [
      {0, 0, 0},
      {0, 0, 0} | Enum.map(Tuple.to_list(input_dilation), &{0, 0, &1 - 1})
    ]

    dilated_input_shape = Nx.Shape.pad(input_shape, input_dilation_padding_config)

    # Always send the padding as an actual padding configuration
    # so backends don't deal with atoms themselves
    #
    # We assume padding is specified only for spatial dims and only
    # as {edge_high, edge_low} tuples, this conceptually simplfies
    # things a bit
    padding_config =
      case padding do
        :valid ->
          for _ <- 0..(tuple_size(input_shape) - 3), do: {0, 0}

        :same ->
          padding_config = Nx.Shape.calculate_padding(spatial_dims, filter_shape)
          padding_config

        config when is_list(config) ->
          config

        _ ->
          raise ArgumentError,
                "invalid padding configuration, padding must be" <>
                  " :valid or :same, or a padding configuration for" <>
                  " the spatial dimensions of the input tensor"
      end

    {shape, names} =
      Nx.Shape.conv(
        dilated_input_shape,
        input_names,
        dilated_kernel_shape,
        kernel_names,
        strides,
        padding_config
      )

    type = binary_type(tensor, kernel) |> Nx.Type.to_floating()

    out = %{tensor | type: type, shape: shape, names: names}

    impl!(tensor).conv(
      out,
      tensor,
      kernel,
      strides: strides,
      padding: padding_config,
      input_dilation: input_dilation,
      kernel_dilation: kernel_dilation
    )
  end

  @doc """
  Clips the values of the tensor on the closed
  interval `[min, max]`.

  You can pass a tensor to `min` or `max` as long
  as the tensor has a scalar shape.

  ### Examples

      iex> Nx.clip(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y]), 2, 4)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [2, 2, 3],
          [4, 4, 4]
        ]
      >

      iex> Nx.clip(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y]), 2.0, 3)
      #Nx.Tensor<
        f64[x: 2][y: 3]
        [
          [2.0, 2.0, 3.0],
          [3.0, 3.0, 3.0]
        ]
      >

      iex> Nx.clip(Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y]), Nx.tensor(2.0), Nx.max(1.0, 3.0))
      #Nx.Tensor<
        f64[x: 2][y: 3]
        [
          [2.0, 2.0, 3.0],
          [3.0, 3.0, 3.0]
        ]
      >

      iex> Nx.clip(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, :y]), 2, 6.0)
      #Nx.Tensor<
        f64[x: 2][y: 3]
        [
          [2.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]
        ]
      >

      iex> Nx.clip(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: {:f, 32}, names: [:x, :y]), 1, 4)
      #Nx.Tensor<
        f64[x: 2][y: 3]
        [
          [1.0, 2.0, 3.0],
          [4.0, 4.0, 4.0]
        ]
      >
  """
  def clip(tensor, min, max) do
    %T{type: type} = tensor = tensor!(tensor)
    %T{type: min_type} = min = tensor!(min)
    %T{type: max_type} = max = tensor!(max)

    if min.shape != {} do
      raise ArgumentError, "min value must be a scalar shape, got: #{min.shape}"
    end

    if max.shape != {} do
      raise ArgumentError, "max value must be a scalar shape, got: #{max.shape}"
    end

    output_type = Nx.Type.merge(type, Nx.Type.merge(min_type, max_type))
    impl!(tensor).clip(%{tensor | type: output_type}, tensor, min, max)
  end

  @doc """
  Slices a tensor from `start_indices` to `limit_indices`.
  You can optionally provide a `stride` to specify the amount
  of stride in each dimension.

  Both start indices and limit indices must match the rank
  of the input tensor shape. You cannot slice in reverse.
  All of the limit indices must be strictly greater than
  their corresponding start index. The resulting shape
  cannot have any zero-sized dimensions or negative
  sized dimensions.

  ### Examples

      iex> t = Nx.iota({900})
      iex> t = Nx.reshape(t, {2, 15, 30})
      iex> Nx.slice(t, [1, 4, 10], [2, 5, 20], [1, 2, 3])
      #Nx.Tensor<
        s64[1][1][4]
        [
          [
            [580, 583, 586, 589]
          ]
        ]
      >

      iex> t = Nx.iota({900})
      iex> t = Nx.reshape(t, {2, 15, 30})
      iex> Nx.slice(t, [0, 6, 2], [2, 7, 5])
      #Nx.Tensor<
        s64[2][1][3]
        [
          [
            [182, 183, 184]
          ],
          [
            [632, 633, 634]
          ]
        ]
      >

      iex> t = Nx.iota({900})
      iex> t = Nx.reshape(t, {2, 15, 30})
      iex> Nx.slice(t, [0, 4, 11], [2, 7, 20], [2, 1, 3])
      #Nx.Tensor<
        s64[1][3][3]
        [
          [
            [131, 134, 137],
            [161, 164, 167],
            [191, 194, 197]
          ]
        ]
      >

      iex> t = Nx.tensor([
      ...>   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ...>   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ...>   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ...>   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ...>   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ...>   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      ...> ])
      iex> Nx.slice(t, [0, 0], [6, 7], [5, 3])
      #Nx.Tensor<
        f64[2][3]
        [
          [0.0, 0.0, 0.0],
          [1.0, 1.0, 1.0]
        ]
      >
  """
  def slice(tensor, start_indices, limit_indices, strides \\ nil) do
    %T{shape: shape} = tensor = tensor!(tensor)

    strides = if strides, do: strides, else: List.duplicate(1, rank(shape))

    if length(strides) != rank(shape) do
      raise ArgumentError, "invalid strides for shape of rank #{rank(shape)}"
    end

    if length(start_indices) != rank(shape) do
      raise ArgumentError, "invalid start indices for shape of rank #{rank(shape)}"
    end

    if length(limit_indices) != rank(shape) do
      raise ArgumentError, "invalid limit indices for shape of rank #{rank(shape)}"
    end

    output_shape = Nx.Shape.slice(shape, start_indices, limit_indices, strides)

    out = %{tensor | shape: output_shape}
    impl!(tensor).slice(out, tensor, start_indices, limit_indices, strides)
  end

  @doc """
  Concatenates tensors along the given axis.

  If no axis is provided, defaults to 0.

  All tensors must have the same rank and all of their
  dimension sizes but the concatenated dimension must match.

  If tensors are named, the names must be able to be merged.

  If tensors with mixed types are given, the types will
  be merged to a higher type and all of the tensors will
  be cast to the higher type before concatenating.

  ### Examples

      iex> Nx.concatenate([Nx.tensor([1, 2, 3])])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.concatenate([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
      #Nx.Tensor<
        s64[6]
        [1, 2, 3, 4, 5, 6]
      >

      iex> t1 = Nx.iota({2, 2, 2}, names: [:x, :y, :z], type: {:f, 32})
      iex> t2 = Nx.iota({1, 2, 2}, names: [:x, :y, :z], type: {:u, 8})
      iex> t3 = Nx.iota({1, 2, 2}, names: [:x, :y, :z], type: {:s, 64})
      iex> Nx.concatenate([t1, t2, t3], axis: :x)
      #Nx.Tensor<
        f64[x: 4][y: 2][z: 2]
        [
          [
            [0.0, 1.0],
            [2.0, 3.0]
          ],
          [
            [4.0, 5.0],
            [6.0, 7.0]
          ],
          [
            [0.0, 1.0],
            [2.0, 3.0]
          ],
          [
            [0.0, 1.0],
            [2.0, 3.0]
          ]
        ]
      >

      iex> t1 = Nx.iota({1, 3, 2}, names: [:x, :y, :z])
      iex> t2 = Nx.iota({1, 1, 2}, names: [:x, :y, :z])
      iex> t3 = Nx.iota({1, 2, 2}, names: [:x, :y, :z])
      iex> Nx.concatenate([t1, t2, t3], axis: :y)
      #Nx.Tensor<
        s64[x: 1][y: 6][z: 2]
        [
          [
            [0, 1],
            [2, 3],
            [4, 5],
            [0, 1],
            [0, 1],
            [2, 3]
          ]
        ]
      >

      iex> t1 = Nx.iota({2, 1, 4}, names: [:x, :y, :z])
      iex> t2 = Nx.iota({2, 1, 1}, names: [:x, :y, :z])
      iex> t3 = Nx.iota({2, 1, 3}, names: [:x, :y, :z])
      iex> Nx.concatenate([t1, t2, t3], axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 1][z: 8]
        [
          [
            [0, 1, 2, 3, 0, 0, 1, 2]
          ],
          [
            [4, 5, 6, 7, 1, 3, 4, 5]
          ]
        ]
      >

      iex> t1 = Nx.iota({2, 1, 4}, names: [:x, :y, :z])
      iex> Nx.concatenate([t1], axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 1][z: 4]
        [
          [
            [0, 1, 2, 3]
          ],
          [
            [4, 5, 6, 7]
          ]
        ]
      >
  """
  def concatenate(tensors, opts \\ []) when is_list(tensors) do
    assert_keys!(opts, [:axis])
    axis = opts[:axis] || 0

    case tensors do
      [] ->
        raise ArgumentError, "empty list passed to concatenate"

      [t | []] ->
        t

      [t1 | _] = tensors ->
        {tensors, [type1 | rest], [s1 | _] = shapes, [n1 | _] = names} =
          tensors
          |> Enum.map(fn t ->
            %T{type: type, shape: shape, names: names} = t = tensor!(t)
            {t, type, shape, names}
          end)
          |> unzip4()

        axis = Nx.Shape.normalize_axis(s1, axis, n1)
        {output_shape, output_names} = Nx.Shape.concatenate(shapes, names, axis)

        output_type =
          rest
          |> Enum.reduce(type1, fn t1, t2 -> Nx.Type.merge(t1, t2) end)

        out = %{t1 | type: output_type, shape: output_shape, names: output_names}
        impl!(t1).concatenate(out, tensors, axis: axis)
    end
  end

  defp unzip4(enumerable) do
    {list1, list2, list3, list4} =
      Enum.reduce(enumerable, {[], [], [], []}, fn {el1, el2, el3, el4},
                                                   {list1, list2, list3, list4} ->
        {[el1 | list1], [el2 | list2], [el3 | list3], [el4 | list4]}
      end)

    {Enum.reverse(list1), Enum.reverse(list2), Enum.reverse(list3), Enum.reverse(list4)}
  end

  @doc """
  Performs a cholesky decomposition of a square matrix.

  The matrix must be positive-definite and either Hermitian
  if complex or symmetric if real. An error is raised by the
  default backend if those conditions are not met. Other
  backends may emit undefined behaviour.

  ### Examples

    iex> Nx.cholesky(Nx.tensor([[6.0, 3.0, 4.0, 8.0], [3.0, 6.0, 5.0, 1.0], [4.0, 5.0, 10.0, 7.0], [8.0, 1.0, 7.0, 25.0]]))
    #Nx.Tensor<
      f64[4][4]
      [
        [2.449489742783178, 0.0, 0.0, 0.0],
        [1.2247448713915892, 2.1213203435596424, 0.0, 0.0],
        [1.6329931618554523, 1.414213562373095, 2.309401076758503, 0.0],
        [3.2659863237109046, -1.4142135623730956, 1.5877132402714704, 3.1324910215354165]
      ]
    >

    iex> Nx.cholesky(Nx.tensor([[20.0, 17.6], [17.6, 16.0]]))
    #Nx.Tensor<
      f64[2][2]
      [
        [4.47213595499958, 0.0],
        [3.93547964039963, 0.7155417527999305]
      ]
    >

  ### Error cases

      iex> Nx.cholesky(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      ** (ArgumentError) matrix must be symmetric, a matrix is symmetric iff X = X.T
  """
  def cholesky(tensor) do
    %T{type: type, shape: shape, names: names} = tensor = tensor!(tensor)

    output_type = Nx.Type.to_floating(type)

    {output_shape, output_names} = Nx.Shape.cholesky(shape, names)

    out = %{tensor | type: output_type, shape: output_shape, names: output_names}
    impl!(tensor).cholesky(out, tensor)
  end

  @doc """
  Sorts the tensor along the given axis with the
  given comparator.

  If no axis is given, defaults to `0`.

  ### Examples

      iex> Nx.sort(Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y]), axis: :x)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [2, 1, 4],
          [3, 5, 7]
        ]
      >

      iex> Nx.sort(Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y]), axis: :y)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [1, 3, 7],
          [2, 4, 5]
        ]
      >

      iex> Nx.sort(Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y]), axis: :y, comparator: :asc)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [7, 3, 1],
          [5, 4, 2]
        ]
      >

      iex> Nx.sort(Nx.tensor([[[4, 5, 2], [2, 5, 3], [5, 0, 2]], [[1, 9, 8], [2, 1, 3], [2, 1, 4]]], names: [:x, :y, :z]), axis: :x)
      #Nx.Tensor<
        s64[x: 2][y: 3][z: 3]
        [
          [
            [1, 5, 2],
            [2, 1, 3],
            [2, 0, 2]
          ],
          [
            [4, 9, 8],
            [2, 5, 3],
            [5, 1, 4]
          ]
        ]
      >

      iex> Nx.sort(Nx.tensor([[[4, 5, 2], [2, 5, 3], [5, 0, 2]], [[1, 9, 8], [2, 1, 3], [2, 1, 4]]], names: [:x, :y, :z]), axis: :y)
      #Nx.Tensor<
        s64[x: 2][y: 3][z: 3]
        [
          [
            [2, 0, 2],
            [4, 5, 2],
            [5, 5, 3]
          ],
          [
            [1, 1, 3],
            [2, 1, 4],
            [2, 9, 8]
          ]
        ]
      >

      iex> Nx.sort(Nx.tensor([[[4, 5, 2], [2, 5, 3], [5, 0, 2]], [[1, 9, 8], [2, 1, 3], [2, 1, 4]]], names: [:x, :y, :z]), axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 3][z: 3]
        [
          [
            [2, 4, 5],
            [2, 3, 5],
            [0, 2, 5]
          ],
          [
            [1, 8, 9],
            [1, 2, 3],
            [1, 2, 4]
          ]
        ]
      >

      iex> Nx.sort(Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y]), axis: :x, comparator: &Nx.less/2)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [2, 1, 4],
          [3, 5, 7]
        ]
      >
  """
  def sort(tensor, opts \\ []) do
    assert_keys!(opts, [:axis, :comparator])
    comparator = opts[:comparator] || :desc

    %T{shape: shape, names: names} = tensor = tensor!(tensor)

    axis = opts[:axis] || 0
    axis = Nx.Shape.normalize_axis(shape, axis, names)

    impl!(tensor).sort(tensor, tensor, axis: axis, comparator: comparator)
  end

  ## Helpers

  defp tensor!(%T{} = t),
    do: t

  defp tensor!(number) when is_number(number) do
    type = Nx.Type.infer(number)
    out = %T{shape: {}, type: type, names: []}
    Nx.BinaryTensor.from_binary(out, number_to_binary(number, type))
  end

  defp number_to_binary(number, type),
    do: match_types([type], do: <<write!(number, 0)>>)

  defp names!(%T{names: names}), do: names
  defp names!(_), do: nil

  defp assert_keys!(keyword, valid) do
    for {k, _} <- keyword, k not in valid do
      raise "unknown key #{inspect(k)} in #{inspect(keyword)}, expected one of #{inspect(valid)}"
    end
  end
end

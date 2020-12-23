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
  For example, using the `Exla` compiler:

      @defn_compiler {Exla, platform: :host}
      defn softmax(t) do
        Nx.exp(t) / Nx.sum(Nx.exp(t))
      end

  This complements Erlang's JIT compiler as it compiles direct to
  native code with numerical compilation and performance in mind.

  ## Creating tensors

  TODO: Summarize functions for creating tensors: tensor, iota,
  random_*, broadcast.

  ## Broadcasting

  TODO: Write docs.

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
  import Nx.Shared

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

  ## Options

    * `:type` - sets the type of the tensor. If one is not given,
      one is automatically inferred based on the input. See `Nx.Type`
      and `Nx.Type.infer/1` for more information on types. If a
      tensor is given alongside this option, then it verifies the
      tensor matches the given `:type`

  """
  def tensor(arg, opts \\ [])

  def tensor(%T{} = t, []), do: t

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

    if data == "" do
      raise "cannot build empty tensor"
    end

    %T{shape: dimensions, type: type, data: {Nx.BitStringDevice, data}}
  end

  defp flatten(list, type) when is_list(list) do
    {dimensions, acc} = flatten_list(list, type, [], [])

    {dimensions |> Enum.reverse() |> List.to_tuple(),
     acc |> Enum.reverse() |> :erlang.list_to_bitstring()}
  end

  defp flatten(other, type), do: {{}, scalar_to_bin(other, type)}

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
    {[length(list) | dimensions], Enum.reduce(list, acc, &[scalar_to_bin(&1, type) | &2])}
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
      iex> for <<x::float-64-native <- Nx.Util.to_bitstring(t)>> do
      ...>   true = x >= 0.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:f, 64}

      iex> t = Nx.random_uniform({5, 5}, type: {:bf, 16})
      iex> byte_size(Nx.Util.to_bitstring(t))
      50
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:bf, 16}

      iex> t = Nx.random_uniform({5, 5}, -1.0, 1.0, type: {:f, 32})
      iex> for <<x::float-32-native <- Nx.Util.to_bitstring(t)>> do
      ...>   true = x >= -1.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:f, 32}

  ### Generating Integers

      iex> t = Nx.random_uniform({10}, 5, 10, type: {:u, 32})
      iex> for <<x::32-unsigned-native <- Nx.Util.to_bitstring(t)>> do
      ...>   true = x >= 5 and x < 10
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:u, 32}

      iex> t = Nx.random_uniform({5, 5}, -5, 5, type: {:s, 64})
      iex> for <<x::64-signed-native <- Nx.Util.to_bitstring(t)>> do
      ...>   true = x >= -5 and x < 5
      ...> end
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:s, 64}

  ### Tensors as shapes

  If given a tensor as a shape, it takes the shape from the tensor:

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t = Nx.random_uniform(t)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 64}

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t = Nx.random_uniform(t, type: {:f, 32})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

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

  """
  def random_uniform(tensor_or_shape, min, max, opts \\ [])
      when is_number(min) and is_number(max) do
    shape = shape!(tensor_or_shape)
    type = opts[:type] || Nx.Type.infer(max - min)

    gen =
      case type do
        {:s, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {:u, _} -> fn -> min + :rand.uniform(max - min) - 1 end
        {_, _} -> fn -> (max - min) * :rand.uniform() + min end
      end

    data = for _ <- 1..Nx.Shape.size(shape), into: "", do: scalar_to_bin(gen.(), type)
    %T{data: {Nx.BitStringDevice, data}, shape: shape, type: type}
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

  If given a tensor as a shape, it takes the shape and default type
  from the tensor:

      iex> t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      iex> t = Nx.random_normal(t)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 64}

      iex> t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      iex> t = Nx.random_normal(t, type: {:f, 32})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

  The same applies to numbers:

      iex> t = Nx.random_normal(10.0)
      iex> Nx.shape(t)
      {}
      iex> Nx.type(t)
      {:f, 64}

  """
  def random_normal(tensor_or_shape, mu, sigma, opts \\ [])
      when is_float(mu) and is_float(sigma) do
    shape = shape!(tensor_or_shape)
    type = opts[:type] || {:f, 64}

    data =
      for _ <- 1..Nx.Shape.size(shape),
          into: "",
          do: scalar_to_bin(:rand.normal(mu, sigma), type)

    %T{data: {Nx.BitStringDevice, data}, shape: shape, type: type}
  end

  @doc """
  Creates a tensor with the given shape which increments
  along the provided axis.

  If no axis is provided, index counts up at each element.

  If a tensor or a number are given, the shape is taken from the tensor.

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

      iex> Nx.iota({3, 2, 3})
      #Nx.Tensor<
        s64[3][2][3]
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

      iex> Nx.iota({3, 3}, axis: 1)
      #Nx.Tensor<
        s64[3][3]
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
  def iota(tensor_or_shape, opts \\ [])

  def iota({}, opts), do: tensor(0, opts)

  def iota({n}, opts) do
    output_type = opts[:type] || {:s, 64}
    # Validate axis
    axis = opts[:axis] || 0
    Nx.Shape.normalize_axis({n}, axis)
    data = for i <- 0..(n - 1), do: scalar_to_bin(i, output_type)
    %T{data: {Nx.BitStringDevice, IO.iodata_to_binary(data)}, shape: {n}, type: output_type}
  end

  def iota(tensor_or_shape, opts) do
    shape = shape!(tensor_or_shape)
    output_type = opts[:type] || {:s, 64}

    if axis = opts[:axis] do
      axis = Nx.Shape.normalize_axis(shape, axis)
      {dims_before, [dim | dims_after]} =
        shape
        |> Tuple.to_list()
        |> Enum.split(axis)

      # Number of repetitions of an index in memory
      repeat_blocks =
        dims_after
        |> Enum.reduce(1, &*/2)

      # Number of cycles of the counting pattern
      cycles =
        dims_before
        |> Enum.reduce(1, &*/2)

      data =
        for _ <- 1..cycles,
            i <- 0..(dim - 1),
            _ <- 1..repeat_blocks,
            into: "",
            do: scalar_to_bin(i, output_type)

      %T{data: {Nx.BitStringDevice, data}, shape: shape, type: output_type}
    else
      t = iota({Nx.Shape.size(shape)}, opts)
      reshape(t, shape)
    end
  end

  ## Shape

  @doc """
  Changes the shape of a tensor.

  The new shape is either a tuple or a tensor which we will
  retrieve the current shape from. The shapes must be compatible:
  the product of each dimension in the shape must be equal.

  Reshaping only changes the tensor metadata, it doesn't copy
  the underlying structure.

  ## Examples

      iex> t = Nx.tensor([1, 2, 3, 4])
      iex> Nx.reshape(t, {2, 2})
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >

  The shape can also be an existing tensor:

      iex> shape = Nx.tensor([[0], [0], [0], [0]])
      iex> Nx.reshape(Nx.tensor([1, 2, 3, 4]), shape)
      #Nx.Tensor<
        s64[4][1]
        [
          [1],
          [2],
          [3],
          [4]
        ]
      >

  Even a scalar can be transformed into a 3-dimensional tensor:

      iex> t = Nx.tensor(1)
      iex> Nx.reshape(t, {1, 1, 1})
      #Nx.Tensor<
        s64[1][1][1]
        [
          [
            [1]
          ]
        ]
      >

  """
  def reshape(tensor, new_shape) do
    %T{shape: old_shape} = t = tensor(tensor)
    new_shape = shape!(new_shape)
    # This does the shape validation
    output_shape = Nx.Shape.reshape(old_shape, new_shape)
    %{t | shape: output_shape}
  end

  @doc """
  Broadcasts the tensor to the given shape.

  The new shape is either a tuple or a tensor which we will
  retrieve the current shape from. The broadcast shape must
  be of equal or higher rank than the current shape. The
  matching dimensions of the broadcast shape must be either
  one or of the same size as of the current tensor.

  The default broadcasting implementation copies the data in
  memory to match the new dimensions.

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

      iex> Nx.broadcast(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> Nx.broadcast(Nx.tensor([[1, 2]]), Nx.tensor([[10], [20]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [1, 2]
        ]
      >

      iex> Nx.broadcast(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20], [30, 40]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> Nx.broadcast(Nx.tensor([[1, 2]]), Nx.tensor([[10, 20], [30, 40]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [1, 2]
        ]
      >

  """
  def broadcast(tensor, new_shape) do

    case {tensor(tensor), new_shape} do

      {%T{shape: {}} = t, new_shape} ->
        new_shape = shape!(new_shape)
        data = :binary.copy(Nx.Util.to_bitstring(t), Nx.Shape.size(new_shape))
        %{t | data: {Nx.BitStringDevice, data}, shape: new_shape}

      {%T{shape: new_shape} = t, new_shape} -> t

      {%T{shape: old_shape, type: {_, size}} = t, new_shape} ->
        new_shape = shape!(new_shape)

        old_rank = tuple_size(old_shape)
        new_rank = tuple_size(new_shape)
        rank = :erlang.max(old_rank, new_rank)

        old_lower = shape_to_lower_ranked_list(old_shape, old_rank, rank)
        new_lower = shape_to_lower_ranked_list(new_shape, new_rank, rank)

        case unary_broadcast_shape(old_lower, new_lower, []) do
          {:ok, new_higher} ->
            chunk_size = size * Nx.Shape.size(old_shape)
            old_higher = Enum.reverse(old_lower)
            data = unary_broadcast(old_higher, new_higher, Nx.Util.to_bitstring(t), chunk_size)
            data = IO.iodata_to_binary(data)
            %{t | data: {Nx.BitStringDevice, data}, shape: List.to_tuple(new_higher)}

          :error ->
            raise ArgumentError,
                  "cannot broadcast tensor of dimensions #{inspect(old_shape)} " <>
                    "to #{inspect(new_shape)}"
        end
    end

  end

  defp unary_broadcast_shape([odim | odims], [ndim | ndims], acc)
       when ndim == 1 or ndim == odim,
       do: unary_broadcast_shape(odims, ndims, [odim | acc])

  defp unary_broadcast_shape([1 | odims], [ndim | ndims], acc),
    do: unary_broadcast_shape(odims, ndims, [ndim | acc])

  defp unary_broadcast_shape([], [], acc),
    do: {:ok, acc}

  defp unary_broadcast_shape(_, _, _),
    do: :error

  defp unary_broadcast([dim | odims], [dim | ndims], data, chunk_size) do
    chunk_size = div(chunk_size, dim)

    for <<chunk::size(chunk_size)-bitstring <- data>> do
      unary_broadcast(odims, ndims, chunk, chunk_size)
    end
  end

  defp unary_broadcast([1 | odims], [ndim | ndims], data, chunk_size) do
    for _ <- 1..ndim do
      unary_broadcast(odims, ndims, data, chunk_size)
    end
  end

  defp unary_broadcast([], [], data, _chunk_size), do: data

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
    %T{type: type} = tensor(tensor)
    type
  end

  @doc """
  Returns the shape of the tensor as a tuple.

  The size of this tuple gives the rank of the tensor.

  ### Examples

      iex> Nx.shape(Nx.tensor(1))
      {}

      iex> Nx.shape(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      {2, 3}

      iex> Nx.shape(1)
      {}

  """
  def shape(tensor) do
    %T{shape: shape} = tensor(tensor)
    shape
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

  """
  def rank(tensor) do
    %T{shape: shape} = tensor(tensor)
    tuple_size(shape)
  end

  @doc """
  Returns how many elements they are in the tensor.

  If a tuple is given as a shape, it computes the size
  of the given tuple.

  ### Examples

      iex> Nx.size(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      6

      iex> Nx.size(1)
      1

  """
  def size(tensor) do
    %T{shape: shape} = tensor(tensor)
    Nx.Shape.size(shape)
  end

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
    %{t | data: device.allocate(Nx.Util.to_bitstring(t), type, shape, opts)}
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

    defp unquote(name)(left, right, fun) do
      output_type = Nx.Type.merge_tensors(left, right)

      %T{type: left_type} = left = tensor(left)
      %T{type: right_type} = right = tensor(right)

      # Merging scalars has less aggressive type promotion than a straight merge

      output_type = unquote(cast)

      {data, shape} =
        match_types [left_type, right_type, output_type] do
          binary_broadcast(left, right, fn left_dimension, right_dimension ->
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

  defp_element_wise_bin_op.(
    :element_wise_bin_bool,
    &quote(do: Nx.Type.to_predicate(unquote(&1)))
  )

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

      iex> Nx.add(Nx.tensor([1, 2, 3]), 1)
      #Nx.Tensor<
        s64[3]
        [2, 3, 4]
      >

      iex> Nx.add(1, Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        s64[3]
        [2, 3, 4]
      >

  Given a float scalar converts the tensor to a float:

      iex> Nx.add(Nx.tensor([1, 2, 3]), 1.0)
      #Nx.Tensor<
        f64[3]
        [2.0, 3.0, 4.0]
      >

      iex> Nx.add(Nx.tensor([1.0, 2.0, 3.0]), 1)
      #Nx.Tensor<
        f64[3]
        [2.0, 3.0, 4.0]
      >

      iex> Nx.add(Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}), 1)
      #Nx.Tensor<
        f32[3]
        [2.0, 3.0, 4.0]
      >

  Unsigned tensors become signed and double their size if a
  negative number is given:

      iex> Nx.add(Nx.tensor([0, 1, 2], type: {:u, 8}), -1)
      #Nx.Tensor<
        s16[3]
        [-1, 0, 1]
      >

  ### Adding tensors of the same shape

      iex> Nx.add(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[10, 20], [30, 40]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [11, 22],
          [33, 44]
        ]
      >

  ### Adding tensors with broadcasting

      iex> Nx.add(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [11, 21],
          [12, 22]
        ]
      >

      iex> Nx.add(Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [11, 21],
          [12, 22]
        ]
      >

      iex> Nx.add(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20], [30, 40]]))
      #Nx.Tensor<
        s64[2][2]
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

      iex> Nx.subtract(1, 2)
      #Nx.Tensor<
        s64
        -1
      >

  ### Subtracting tensors and scalars

      iex> Nx.subtract(Nx.tensor([1, 2, 3]), 1)
      #Nx.Tensor<
        s64[3]
        [0, 1, 2]
      >

      iex> Nx.subtract(1, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [0.0, -1.0, -2.0]
      >

  ### Subtracting tensors

      iex> Nx.subtract(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [-9, -19],
          [-8, -18]
        ]
      >

      iex> Nx.subtract(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      #Nx.Tensor<
        s8[2][2]
        [
          [-9, -19],
          [-8, -18]
        ]
      >

      iex> Nx.subtract(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      #Nx.Tensor<
        f32[2][2]
        [
          [-9.0, -19.0],
          [-8.0, -18.0]
        ]
      >

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

      iex> Nx.multiply(1, 2)
      #Nx.Tensor<
        s64
        2
      >

  ### Multiplying tensors and scalars

      iex> Nx.multiply(Nx.tensor([1, 2, 3]), 1)
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.multiply(1, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [1.0, 2.0, 3.0]
      >

  ### Multiplying tensors

      iex> Nx.multiply(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [10, 20],
          [20, 40]
        ]
      >

      iex> Nx.multiply(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      #Nx.Tensor<
        s8[2][2]
        [
          [10, 20],
          [20, 40]
        ]
      >

      iex> Nx.multiply(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      #Nx.Tensor<
        f32[2][2]
        [
          [10.0, 20.0],
          [20.0, 40.0]
        ]
      >

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

      iex> Nx.power(2, 4)
      #Nx.Tensor<
        s64
        16
      >

  ### Power of tensors and scalars

      iex> Nx.power(Nx.tensor([1, 2, 3]), 2)
      #Nx.Tensor<
        s64[3]
        [1, 4, 9]
      >

      iex> Nx.power(2, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [2.0, 4.0, 8.0]
      >

  ### Power of tensors

      iex> Nx.power(Nx.tensor([[2], [3]]), Nx.tensor([[4, 5]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [16, 32],
          [81, 243]
        ]
      >

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

      iex> Nx.remainder(1, 2)
      #Nx.Tensor<
        s64
        1
      >

  ### Remainder of tensors and scalars

      iex> Nx.remainder(Nx.tensor([1, 2, 3]), 2)
      #Nx.Tensor<
        s64[3]
        [1, 0, 1]
      >

      iex> Nx.remainder(2, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [0.0, 0.0, 2.0]
      >

  ### Remainder of tensors

      iex> Nx.remainder(Nx.tensor([[10], [20]]), Nx.tensor([[3, 4]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [2, 0]
        ]
      >

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

      iex> Nx.divide(1, 2)
      #Nx.Tensor<
        f64
        0.5
      >

  ### Dividing tensors and scalars

      iex> Nx.divide(Nx.tensor([1, 2, 3]), 1)
      #Nx.Tensor<
        f64[3]
        [1.0, 2.0, 3.0]
      >

      iex> Nx.divide(1, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [1.0, 0.5, 0.3333333333333333]
      >

  ### Dividing tensors

      iex> Nx.divide(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        f64[2][2]
        [
          [0.1, 0.05],
          [0.2, 0.1]
        ]
      >

      iex> Nx.divide(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      #Nx.Tensor<
        f32[2][2]
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      >

      iex> Nx.divide(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      #Nx.Tensor<
        f32[2][2]
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      >

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

      iex> Nx.arctan2(1, 2)
      #Nx.Tensor<
        f64
        0.4636476090008061
      >

  ### Arc tangent between tensors and scalars

      iex> Nx.arctan2(Nx.tensor([1, 2, 3]), 1)
      #Nx.Tensor<
        f64[3]
        [0.7853981633974483, 1.1071487177940904, 1.2490457723982544]
      >

      iex> Nx.arctan2(1, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [0.7853981633974483, 0.4636476090008061, 0.3217505543966422]
      >

  ### Arc tangent between tensors

      # Note there is a bug in Erlang/OTP 23.0 and earlier where the compiler
      # optimizes -0.0 away as 0.0. So we do: -1.0*(Integer.parse("0")|>elem(0))
      iex> pos_and_neg_zero_x = Nx.multiply(Nx.tensor([[-1.0], [1.0]]), 0.0)
      iex> pos_and_neg_zero_y = Nx.multiply(Nx.tensor([-1.0, 1.0]), 0.0)
      iex> t = Nx.arctan2(pos_and_neg_zero_x, pos_and_neg_zero_y)
      iex> Nx.Util.to_bitstring(t)
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

      iex> Nx.max(1, 2)
      #Nx.Tensor<
        s64
        2
      >

  ### Max between tensors and scalars

      iex> Nx.max(Nx.tensor([1, 2, 3]), 1)
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.max(1, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [1.0, 2.0, 3.0]
      >

  ### Max between tensors

      iex> Nx.max(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [10, 20],
          [10, 20]
        ]
      >

      iex> Nx.max(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      #Nx.Tensor<
        s8[2][2]
        [
          [10, 20],
          [10, 20]
        ]
      >

      iex> Nx.max(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      #Nx.Tensor<
        f32[2][2]
        [
          [10.0, 20.0],
          [10.0, 20.0]
        ]
      >

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

      iex> Nx.min(1, 2)
      #Nx.Tensor<
        s64
        1
      >

  ### Min between tensors and scalars

      iex> Nx.min(Nx.tensor([1, 2, 3]), 1)
      #Nx.Tensor<
        s64[3]
        [1, 1, 1]
      >

      iex> Nx.min(1, Nx.tensor([1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[3]
        [1.0, 1.0, 1.0]
      >

  ### Min between tensors

      iex> Nx.min(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> Nx.min(Nx.tensor([[1], [2]], type: {:s, 8}), Nx.tensor([[10, 20]], type: {:s, 8}))
      #Nx.Tensor<
        s8[2][2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> Nx.min(Nx.tensor([[1], [2]], type: {:f, 32}), Nx.tensor([[10, 20]], type: {:f, 32}))
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 1.0],
          [2.0, 2.0]
        ]
      >

  """
  def min(left, right), do: element_wise_bin_arith(left, right, &erlang_min/3)
  @compile {:inline, erlang_min: 3}
  defp erlang_min(_, a, b), do: :erlang.min(a, b)

  ## Bitwise ops

  defp_element_wise_bin_op.(
    :element_wise_bin_bitwise_arith,
    &quote(do: assert_bitwise_type!(unquote(&1)))
  )

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

      iex> Nx.bitwise_and(Nx.tensor([0, 1, 2]), 1)
      #Nx.Tensor<
        s64[3]
        [0, 1, 0]
      >

      iex> Nx.bitwise_and(Nx.tensor([0, -1, -2]), -1)
      #Nx.Tensor<
        s64[3]
        [0, -1, -2]
      >

  ### bitwise and between tensors

      iex> Nx.bitwise_and(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0, 1, 0, 1]))
      #Nx.Tensor<
        s64[4]
        [0, 0, 0, 1]
      >

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

      iex> Nx.bitwise_or(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### bitwise or between tensors and scalars

      iex> Nx.bitwise_or(Nx.tensor([0, 1, 2]), 1)
      #Nx.Tensor<
        s64[3]
        [1, 1, 3]
      >

      iex> Nx.bitwise_or(Nx.tensor([0, -1, -2]), -1)
      #Nx.Tensor<
        s64[3]
        [-1, -1, -1]
      >

  ### bitwise or between tensors

      iex> Nx.bitwise_or(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0, 1, 0, 1]))
      #Nx.Tensor<
        s64[4]
        [0, 1, 1, 1]
      >

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

      iex> Nx.bitwise_xor(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### Bitwise xor and between tensors and scalars

      iex> Nx.bitwise_xor(Nx.tensor([1, 2, 3]), 2)
      #Nx.Tensor<
        s64[3]
        [3, 0, 1]
      >

      iex> Nx.bitwise_xor(Nx.tensor([-1, -2, -3]), 2)
      #Nx.Tensor<
        s64[3]
        [-3, -4, -1]
      >

  ### Bitwise xor between tensors

      iex> Nx.bitwise_xor(Nx.tensor([0, 0, 1, 1]), Nx.tensor([0, 1, 0, 1]))
      #Nx.Tensor<
        s64[4]
        [0, 1, 1, 0]
      >

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

      iex> Nx.left_shift(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### Left shift between tensors and scalars

      iex> Nx.left_shift(Nx.tensor([1, 2, 3]), 2)
      #Nx.Tensor<
        s64[3]
        [4, 8, 12]
      >

  ### Left shift between tensors

      iex> Nx.left_shift(Nx.tensor([1, 1, -1, -1]), Nx.tensor([1, 2, 3, 4]))
      #Nx.Tensor<
        s64[4]
        [2, 4, -8, -16]
      >

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

      iex> Nx.right_shift(1, 0)
      #Nx.Tensor<
        s64
        1
      >

  ### Right shift between tensors and scalars

      iex> Nx.right_shift(Nx.tensor([2, 4, 8]), 2)
      #Nx.Tensor<
        s64[3]
        [0, 1, 2]
      >

  ### Right shift between tensors

      iex> Nx.right_shift(Nx.tensor([16, 32, -64, -128]), Nx.tensor([1, 2, 3, 4]))
      #Nx.Tensor<
        s64[4]
        [8, 8, -8, -8]
      >

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

    iex> Nx.equal(1, Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[3]
      [1, 0, 0]
    >

  ### Comparison of tensors

    iex> Nx.equal(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 5]))
    #Nx.Tensor<
      u8[3]
      [1, 1, 0]
    >

    iex> Nx.equal(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[2][3]
      [
        [1, 1, 1],
        [0, 0, 0]
      ]
    >
  """
  def equal(left, right),
    do: element_wise_bin_bool(left, right, &erlang_equal/3)
  @compile {:inline, erlang_equal: 3}
  defp erlang_equal(_, a, b), do: if a == b, do: 1, else: 0

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

      iex> Nx.not_equal(Nx.tensor([1, 2, 3]), Nx.tensor(1))
      #Nx.Tensor<
        u8[3]
        [0, 1, 1]
      >

  ### Comparison of tensors

      iex> Nx.not_equal(Nx.tensor([1, 1, 2]), Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        u8[3]
        [0, 1, 1]
      >

      iex> Nx.not_equal(Nx.tensor([[1, 4, 2], [4, 5, 6]]), Nx.tensor([[1, 3, 2], [4, 2, 1]]))
      #Nx.Tensor<
        u8[2][3]
        [
          [0, 1, 0],
          [0, 1, 1]
        ]
      >
  """
  def not_equal(left, right),
    do: element_wise_bin_bool(left, right, &erlang_not_equal/3)
  @compile {:inline, erlang_not_equal: 3}
  defp erlang_not_equal(_, a, b), do: if a != b, do: 1, else: 0

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

    iex> Nx.greater(1, Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[3]
      [0, 0, 0]
    >

  ### Comparison of tensors

    iex> Nx.greater(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 2]))
    #Nx.Tensor<
      u8[3]
      [0, 0, 1]
    >

    iex> Nx.greater(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[2][3]
      [
        [0, 0, 0],
        [1, 1, 1]
      ]
    >
  """
  def greater(left, right),
    do: element_wise_bin_bool(left, right, &erlang_greater/3)
  @compile {:inline, erlang_greater: 3}
  defp erlang_greater(_, a, b), do: if a > b, do: 1, else: 0

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

    iex> Nx.less(1, Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[3]
      [0, 1, 1]
    >

  ### Comparison of tensors

    iex> Nx.less(Nx.tensor([1, 2, 1]), Nx.tensor([1, 2, 2]))
    #Nx.Tensor<
      u8[3]
      [0, 0, 1]
    >

    iex> Nx.less(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]]), Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[2][3]
      [
        [0, 0, 0],
        [0, 0, 1]
      ]
    >
  """
  def less(left, right),
    do: element_wise_bin_bool(left, right, &erlang_less/3)
  @compile {:inline, erlang_less: 3}
  defp erlang_less(_, a, b), do: if a < b, do: 1, else: 0

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

    iex> Nx.greater_equal(1, Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[3]
      [1, 0, 0]
    >

  ### Comparison of tensors

    iex> Nx.greater_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 2]))
    #Nx.Tensor<
      u8[3]
      [1, 1, 1]
    >

    iex> Nx.greater_equal(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[2][3]
      [
        [1, 1, 1],
        [1, 1, 1]
      ]
    >
  """
  def greater_equal(left, right),
    do: element_wise_bin_bool(left, right, &erlang_greater_equal/3)
  @compile {:inline, erlang_greater_equal: 3}
  defp erlang_greater_equal(_, a, b), do: if a >= b, do: 1, else: 0

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

    iex> Nx.less_equal(1, Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[3]
      [1, 1, 1]
    >

  ### Comparison of tensors

    iex> Nx.less_equal(Nx.tensor([1, 2, 3]), Nx.tensor([1, 2, 2]))
    #Nx.Tensor<
      u8[3]
      [1, 1, 0]
    >

    iex> Nx.less_equal(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Nx.tensor([1, 2, 3]))
    #Nx.Tensor<
      u8[2][3]
      [
        [1, 1, 1],
        [0, 0, 0]
      ]
    >
  """
  def less_equal(left, right),
    do: element_wise_bin_bool(left, right, &erlang_less_equal/3)
  @compile {:inline, erlang_less_equal: 3}
  defp erlang_less_equal(_, a, b), do: if a <= b, do: 1, else: 0

  @doc """
  Constructs a tensor from two tensors, based on a predicate.

  The resulting tensor is built by evaluating each element of
  `pred` and returning either the corresponding element from
  `on_true` or `on_false`.

  `pred` must either be a boolean, `1` or `0`, or a tensor
  of predicates with a shape that matches the largest shape between
  `s1` or `s2`.

  If `pred` is a boolean, takes either the entire `on_true` tensor
  or the entire `on_false` tensor.

  If the shape of `on_true` or `on_false` do not match the shape of
  `pred`, attemps to broadcast both so they match the shape of `pred`.

  ## Examples


      iex> Nx.select(1, Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.select(0, Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
      #Nx.Tensor<
        s64[3]
        [4, 5, 6]
      >

      iex> x = Nx.tensor([2, 4, 6])
      iex> y = Nx.tensor([3, 2, 1])
      iex> Nx.select(Nx.greater(x, y), Nx.tensor([2, 4, 6]), Nx.tensor([1, 3, 5]))
      #Nx.Tensor<
        s64[3]
        [1, 4, 6]
      >

      iex> x = Nx.tensor([2, 4, 6, 8, 10])
      iex> y = Nx.tensor([1, 6, 2, 11, 2])
      iex> Nx.select(Nx.greater(x, y), Nx.tensor(2), Nx.tensor([1, 3, 5, 7, 9]))
      #Nx.Tensor<
        s64[5]
        [2, 3, 2, 7, 2]
      >
  """
  def select(pred, on_true, on_false) do
    output_type = Nx.Type.merge_tensors(on_true, on_false)

    case {tensor(pred), tensor(on_true), tensor(on_false)} do
      {%T{shape: {}} = pred, on_true, on_false} ->
        if one?(pred), do: on_true, else: on_false

      {%T{shape: shape} = pred, %T{type: {_, left_size} = left_type} = on_true, %T{type: {_, right_size} = right_type} = on_false} ->

        on_true_bcast = broadcast(on_true, shape)
        on_false_bcast = broadcast(on_false, shape)

        pred_data = Nx.Util.to_bitstring(pred)
        on_true_data = Nx.Util.to_bitstring(on_true_bcast)
        on_false_data = Nx.Util.to_bitstring(on_false_bcast)

        pred_size = Nx.Shape.size(shape)

        data =
          for i <- 0..pred_size - 1, into: <<>> do
            pred_consumed = i * 8
            <<_::size(pred_consumed), pred::size(8)-unsigned-native, _::bitstring>> = pred_data
            match_types [left_type, right_type, output_type] do
              if pred == 1 do
                consumed = i*left_size
                <<_::size(consumed), match!(x, 0), _::bitstring>> = on_true_data
                <<write!(read!(x, 0), 2)>>
              else
                consumed = i*right_size
                <<_::size(consumed), match!(y, 1), _::bitstring>> = on_false_data
                <<write!(read!(y, 1), 2)>>
              end
            end
          end

        %T{data: {Nx.BitStringDevice, data}, shape: shape, type: output_type}
    end
  end

  defp one?(%T{type: type} = t) do
    data = Nx.Util.to_bitstring(t)

    match_types [type] do
      <<match!(x, 0)>> = data
      read!(x, 0) == 1
    end
  end

  ## Unary ops

  funs = [
    exp: {"exponential", &quote(do: :math.exp(unquote(&1)))},
    expm1: {"exponential minus one", &quote(do: :math.exp(unquote(&1)) - 1)},
    log: {"natural log", &quote(do: :math.log(unquote(&1)))},
    log1p: {"natural log plus one", &quote(do: :math.log(unquote(&1) + 1))},
    logistic: {"standard logistic (a sigmoid)", &quote(do: 1 / (1 + :math.exp(-unquote(&1))))},
    cos: {"cosine", &quote(do: :math.cos(unquote(&1)))},
    sin: {"sine", &quote(do: :math.sin(unquote(&1)))},
    tanh: {"hyperbolic tangent", &quote(do: :math.tanh(unquote(&1)))},
    sqrt: {"square root", &quote(do: :math.sqrt(unquote(&1)))},
    rsqrt: {"reverse square root", &quote(do: 1 / :math.sqrt(unquote(&1)))},
    cbrt: {"cube root", &quote(do: :math.pow(unquote(&1), 1 / 3))}
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

        iex> Nx.#{name}(1)
        #Nx.Tensor<
          f64
          #{one}
        >

        iex> Nx.#{name}(Nx.tensor([1.0, 2.0, 3.0]))
        #Nx.Tensor<
          f64[3]
          [#{one}, #{two}, #{three}]
        >

    """
    def unquote(name)(tensor), do: unary_float(tensor, fn x -> unquote(applied) end)
  end

  defp unary_float(tensor, fun) do
    %T{type: input_type} = t = tensor(tensor)

    data = Nx.Util.to_bitstring(t)
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

      iex> Nx.negate(Nx.tensor([0, 1, 2], type: {:u, 8}))
      #Nx.Tensor<
        u8[3]
        [0, 255, 254]
      >

  """
  def negate(tensor) do
    %T{type: input_type} = t = tensor(tensor)

    data = Nx.Util.to_bitstring(t)

    data =
      match_types [input_type] do
        for <<match!(seg, 0) <- data>>, into: <<>> do
          <<write!(-read!(seg, 0), 0)>>
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

      iex> Nx.sign(Nx.tensor([-2, -1, 0, 1, 2]))
      #Nx.Tensor<
        s64[5]
        [-1, -1, 0, 1, 1]
      >

  """
  def sign(tensor) do
    %T{type: input_type} = t = tensor(tensor)

    data = Nx.Util.to_bitstring(t)

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

      iex> Nx.abs(Nx.tensor([-2, -1, 0, 1, 2]))
      #Nx.Tensor<
        s64[5]
        [2, 1, 0, 1, 2]
      >

  """
  def abs(tensor) do
    %T{type: input_type} = t = tensor(tensor)

    data = Nx.Util.to_bitstring(t)

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

      iex> Nx.bitwise_not(1)
      #Nx.Tensor<
        s64
        -2
      >

      iex> Nx.bitwise_not(Nx.tensor([-1, 0, 1], type: {:s, 8}))
      #Nx.Tensor<
        s8[3]
        [0, -1, -2]
      >

      iex> Nx.bitwise_not(Nx.tensor([0, 1, 254, 255], type: {:u, 8}))
      #Nx.Tensor<
        u8[4]
        [255, 254, 1, 0]
      >

  ### Error cases

      iex> Nx.bitwise_not(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def bitwise_not(tensor) do
    %T{type: input_type} = t = tensor(tensor)

    assert_bitwise_type!(input_type)
    data = Nx.Util.to_bitstring(t)

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

      iex> Nx.population_count(Nx.tensor([0, 1, 254, 255]))
      #Nx.Tensor<
        s64[4]
        [0, 1, 7, 8]
      >

      iex> Nx.population_count(Nx.tensor([0, 1, 126, 127, -1, -127, -128], type: {:s, 8}))
      #Nx.Tensor<
        s8[7]
        [0, 1, 6, 7, 8, 2, 1]
      >

  ### Error cases

      iex> Nx.population_count(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def population_count(tensor) do
    %T{type: {_, size} = input_type} = t = tensor(tensor)

    assert_bitwise_type!(input_type)

    data =
      for <<seg::unsigned-size(size)-native <- Nx.Util.to_bitstring(t)>>, into: <<>> do
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
  defp erlang_popcount(n, count), do: erlang_popcount(n &&& n - 1, count + 1)

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

      iex> Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF]))
      #Nx.Tensor<
        s64[4]
        [64, 60, 56, 48]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0xF000000000000000, 0x0F00000000000000]))
      #Nx.Tensor<
        s64[2]
        [0, 4]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF], type: {:s, 32}))
      #Nx.Tensor<
        s32[4]
        [32, 28, 24, 16]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 0xF, 0xFF, 0xFFFF], type: {:s, 16}))
      #Nx.Tensor<
        s16[4]
        [16, 12, 8, 0]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 1, 2, 4, 8, 16, 32, 64, -1, -128], type: {:s, 8}))
      #Nx.Tensor<
        s8[10]
        [8, 7, 6, 5, 4, 3, 2, 1, 0, 0]
      >

      iex> Nx.count_leading_zeros(Nx.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128], type: {:u, 8}))
      #Nx.Tensor<
        u8[9]
        [8, 7, 6, 5, 4, 3, 2, 1, 0]
      >

  ### Error cases

      iex> Nx.population_count(Nx.tensor([0.0, 1.0]))
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 64}
  """
  def count_leading_zeros(tensor) do
    %T{type: {_, size} = input_type} = t = tensor(tensor)

    assert_bitwise_type!(input_type)

    data =
      for <<seg::unsigned-size(size)-native <- Nx.Util.to_bitstring(t)>>, into: <<>> do
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

        iex> Nx.#{name}(Nx.tensor([-1, 0, 1]))
        #Nx.Tensor<
          s64[3]
          [-1, 0, 1]
        >

        iex> Nx.#{name}(Nx.tensor([-1.5, -0.5, 0.5, 1.5]))
        #Nx.Tensor<
          f64[4]
          [#{res1}.0, #{res2}.0, #{res3}.0, #{res4}.0]
        >

    """
    def unquote(name)(tensor) do
      case tensor(tensor) do
        %T{type: {type, _}} = t when type in [:s, :u] -> t
        %T{type: input_type} = t ->
          data = Nx.Util.to_bitstring(t)

          data =
            match_types [input_type] do
              for <<match!(seg, 0) <- data>>, into: <<>> do
                <<write!(:erlang.unquote(name)(read!(seg, 0)), 0)>>
              end
            end

          %{t | data: {Nx.BitStringDevice, data}}
      end
    end
  end

  ## Aggregate ops

  @doc """
  Returns the sum for the tensor.

  If the `:axis` option is given, it aggregates over
  that dimension, effectively removing it. `axes: [0]`
  implies aggregating over the highest order dimension
  and so forth. If the axis is negative, then counts
  the axis from the back. For example, `axes: [-1]` will
  always aggregate all rows.

  ## Examples

      iex> Nx.sum(Nx.tensor(42))
      #Nx.Tensor<
        s64
        42
      >

      iex> Nx.sum(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        s64
        6
      >

      iex> Nx.sum(Nx.tensor([[1.0, 2.0], [3.0, 4.0]]))
      #Nx.Tensor<
        f64
        10.0
      >

  ### Aggregating over an axis

      iex> Nx.sum(Nx.tensor([1, 2, 3]), axes: [0])
      #Nx.Tensor<
        s64
        6
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [0])
      #Nx.Tensor<
        s64[2][3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [1])
      #Nx.Tensor<
        s64[2][3]
        [
          [5, 7, 9],
          [17, 19, 21]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [2])
      #Nx.Tensor<
        s64[2][2]
        [
          [6, 15],
          [24, 33]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [0, 2])
      #Nx.Tensor<
        s64[2]
        [30, 48]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [-1])
      #Nx.Tensor<
        s64[2][2]
        [
          [6, 15],
          [24, 33]
        ]
      >

      iex> Nx.sum(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [-3])
      #Nx.Tensor<
        s64[2][3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

  ### Errors

      iex> Nx.sum(Nx.tensor([[1, 2]]), axes: [2])
      ** (ArgumentError) axes [2] must be unique integers between 0 and 1

  """
  def sum(tensor, opts \\ []) do
    {tensor, _} = Nx.Util.reduce(tensor, 0, opts, fn x, acc -> {x + acc, x + acc} end)
    tensor
  end

  @doc """
  Returns the mean for the tensor.

  If the `:axis` option is given, it aggregates over
  that dimension, effectively removing it. `axes: [0]`
  implies aggregating over the highest order dimension
  and so forth. If the axis is negative, then counts
  the axis from the back. For example, `axes: [-1]` will
  always aggregate all rows.

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

      iex> Nx.mean(Nx.tensor([1, 2, 3]), axes: [0])
      #Nx.Tensor<
        f64
        2.0
      >

      iex> Nx.mean(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [0])
      #Nx.Tensor<
        f64[2][3]
        [
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]
        ]
      >

      iex> Nx.mean(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [0, 2])
      #Nx.Tensor<
        f64[2]
        [5.0, 8.0]
      >

      iex> Nx.mean(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), axes: [-1])
      #Nx.Tensor<
        f64[2][2]
        [
          [2.0, 5.0],
          [8.0, 11.0]
        ]
      >

  ### Errors

      iex> Nx.mean(Nx.tensor([[1, 2]]), axes: [2])
      ** (ArgumentError) axes [2] must be unique integers between 0 and 1

  """
  def mean(tensor, opts \\ []) do
    tensor = tensor(tensor)
    divide(sum(tensor, opts), mean_den(tensor.shape, opts[:axes]))
  end

  defp mean_den(shape, nil), do: Nx.Shape.size(shape)
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

  ### Aggregating over an axis

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), axis: 0)
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 0, 0],
          [1, 1, 0]
        ]
      >

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), axis: 1)
      #Nx.Tensor<
        s64[2][3]
        [
          [0, 0, 0],
          [0, 1, 0]
        ]
      >

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), axis: 2)
      #Nx.Tensor<
        s64[2][2]
        [
          [0, 2],
          [0, 1]
        ]
      >

  ### Tie breaks

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), tie_break: :low, axis: 1)
      #Nx.Tensor<
        s64[2][3]
        [
          [0, 0, 0],
          [0, 1, 0]
        ]
      >

      iex> Nx.argmax(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), tie_break: :high, axis: 1)
      #Nx.Tensor<
        s64[2][3]
        [
          [0, 0, 1],
          [0, 1, 1]
        ]
      >
  """
  def argmax(tensor, opts \\ []) do
    comparator =
      case opts[:tie_break] || :low do
        :high -> &>=/2
        :low -> &>/2
      end

    argmin_or_max(tensor, comparator, opts)
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

  ### Aggregating over an axis

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), axis: 0)
      #Nx.Tensor<
        s64[2][3]
        [
          [0, 0, 0],
          [0, 0, 0]
        ]
      >

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), axis: 1)
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 1, 0],
          [1, 0, 0]
        ]
      >

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), axis: 2)
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 1],
          [1, 2]
        ]
      >

  ### Tie breaks

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), tie_break: :low, axis: 1)
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 1, 0],
          [1, 0, 0]
        ]
      >

      iex> Nx.argmin(Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]]), tie_break: :high, axis: 1)
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 1, 1],
          [1, 0, 1]
        ]
      >
  """
  def argmin(tensor, opts \\ []) do
    comparator =
      case opts[:tie_break] || :low do
        :high -> &<=/2
        :low -> &</2
      end

    argmin_or_max(tensor, comparator, opts)
  end

  defp argmin_or_max(number, _comparator, opts) when is_number(number), do: tensor(0, opts)

  defp argmin_or_max(t = %T{}, comparator, opts) do
    case tensor(t) do
      %T{shape: {}} -> tensor(0, opts)
      %T{} = t ->
        axes = if axis = opts[:axis], do: [axis], else: nil

        {tensor, _accs} =
          Nx.Util.reduce(t, {0, :first, -1}, [axes: axes], fn x, {i, cur_extreme_x, cur_extreme_i} ->
            if comparator.(x, cur_extreme_x) or cur_extreme_x == :first do
              {i, {i + 1, x, i}}
            else
              {cur_extreme_i, {i + 1, cur_extreme_x, cur_extreme_i}}
            end
          end)

        tensor
    end
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

      iex> Nx.dot(Nx.tensor([[1, 2, 3], [4, 5, 6]]), Nx.tensor([[7, 8], [9, 10], [11, 12]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [58, 64],
          [139, 154]
        ]
      >

      iex> Nx.dot(Nx.tensor([[10.0, 13.0, 14.0, 15.0], [59.0, 20.0, 10.0, 30.0]]), Nx.tensor([[2.0, 4.0], [5.0, 1.0], [6.0, 8.0], [9.0, 10.0]]))
      #Nx.Tensor<
        f64[2][2]
        [
          [304.0, 315.0],
          [548.0, 636.0]
        ]
      >

      iex> Nx.dot(Nx.tensor([[1, 2, 3], [4, 5, 6]]), Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]))
      #Nx.Tensor<
        f64[2][2]
        [
          [58.0, 64.0],
          [139.0, 154.0]
        ]
      >

  ### Dot product of vector and n-d tensor

      iex> Nx.dot(Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), Nx.tensor([5, 10]))
      #Nx.Tensor<
        s64[2][2]
        [
          [25, 55],
          [85, 115]
        ]
      >

      iex> Nx.dot(Nx.tensor([5, 10]), Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      #Nx.Tensor<
        s64[3]
        [45, 60, 75]
      >

      iex> Nx.dot(Nx.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]]), Nx.tensor([2.0, 2.0]))
      #Nx.Tensor<
        f64[1][1][2][2]
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

      iex> a = Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
      iex> b = Nx.tensor([[[1, 2, 3], [3, 4, 5], [5, 6, 7]]])
      iex> Nx.dot(a, b)
      #Nx.Tensor<
        s64[2][3][1][3]
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
      ** (ArgumentError) dot product expects shapes to be compatible, dimension 0 of left-side (3) does not equal dimension 0 of right-side (2)
  """
  def dot(t1, t2) do
    {%T{shape: s1} = t1, %T{shape: s2} = t2} = {tensor(t1), tensor(t2)}
    case {tuple_size(s1), tuple_size(s2)} do
      {0, _} -> multiply(t1, t2)
      {_, 0} -> multiply(t1, t2)
      {n, 1} -> Nx.Util.dot(t1, [n - 1], t2, [0])
      {1, m} -> Nx.Util.dot(t1, [0], t2, [m - 2])
      {n, m} when n >= 2 and m >= 2 -> Nx.Util.dot(t1, [n - 1], t2, [m - 2])
    end
  end

  @doc """
  Computes the outer product of two one-dimensional tensors.

  ## Examples

      iex> Nx.outer(Nx.tensor([1, 2, 3]), 100)
      #Nx.Tensor<
        s64[3]
        [100, 200, 300]
      >

      iex> Nx.outer(Nx.tensor([1, 2, 3]), Nx.tensor([10, 20]))
      #Nx.Tensor<
        s64[3][2]
        [
          [10, 20],
          [20, 40],
          [30, 60]
        ]
      >

      iex> Nx.outer(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([10, 20, 30]))
      #Nx.Tensor<
        s64[2][2][3]
        [
          [
            [10, 20, 30],
            [20, 40, 60]
          ],
          [
            [30, 60, 90],
            [40, 80, 120]
          ]
        ]
      >

  """
  def outer(t1, t2) do
    output_type = Nx.Type.merge_tensors(t1, t2)

    case {tensor(t1), tensor(t2)} do
      {%T{shape: {}} = t1, %T{shape: {}} = t2} -> multiply(t1, t2)

      {%T{shape: s1, type: left_type} = t1, %T{shape: s2, type: right_type} = t2} ->

        b1 = Nx.Util.to_bitstring(t1)
        b2 = Nx.Util.to_bitstring(t2)

        data =
          match_types [left_type, right_type, output_type] do
            for <<match!(left, 0) <- b1>>,
                <<match!(right, 1) <- b2>>,
                into: <<>>,
                do: <<write!(read!(left, 0) * read!(right, 1), 2)>>
          end

        shape = List.to_tuple(Tuple.to_list(s1) ++ Tuple.to_list(s2))
        %T{shape: shape, type: output_type, data: {Nx.BitStringDevice, data}}
    end
  end

  @doc """
  Transposes a tensor by reversing its axes.

  See `transpose/2` for more information.

  ## Examples

      iex> Nx.transpose(Nx.tensor(1))
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.transpose(Nx.iota({2, 3, 4}))
      #Nx.Tensor<
        s64[4][3][2]
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
  """
  def transpose(tensor) do
    case tensor(tensor) do
      %T{shape: {}} = t -> t
      %T{shape: s} = t -> transpose(t, Nx.Shape.transpose_axes(s))
    end
  end

  @doc """
  Transposes a tensor to the given `axes`.

  The axes is a tuple of integers containing how the new
  dimensions must be ordered. The highest dimension is zero.

  ## Examples

      iex> Nx.transpose(Nx.tensor(1), [])
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.transpose(Nx.iota({2, 3, 4}), [2, 1, 0])
      #Nx.Tensor<
        s64[4][3][2]
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

      iex> Nx.transpose(Nx.iota({2, 3, 4}), [2, 0, 1])
      #Nx.Tensor<
        s64[4][2][3]
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

      iex> Nx.transpose(Nx.iota({2, 3, 4}), [0, 2, 1])
      #Nx.Tensor<
        s64[2][4][3]
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

      iex> Nx.transpose(1, [0])
      ** (ArgumentError) axes [0] must be of the same rank as shape {}

      iex> Nx.transpose(Nx.iota({2, 2}), [1, 2])
      ** (ArgumentError) axes [1, 2] must be unique integers between 0 and 1

  """
  def transpose(tensor, axes) when is_list(axes) do
    case {tensor(tensor), axes} do
      {%T{shape: {}} = t, []} -> t

      {%T{shape: {_}} = t, [0]} -> t

      {%T{shape: shape, type: {_, size}} = t, axes} ->
        data = Nx.Util.to_bitstring(t)
        {list, min, max} = transpose_axes(shape, axes)
        weighted_shape = weighted_shape(shape, size)

        # The chunk size is computed based on all dimensions
        # before the minimum one being changed. For example,
        # for {0, 1, 2, 3} and the swap is between 1 and 2,
        # the chunk_size will be d1 * d2 * d3 * size.
        chunk_size = weighted_chunk(weighted_shape, min, size)

        # All of the major dimensions not being transposed can be
        # read at once. For example, for {0, 1, 2, 3} and the swap
        # is between 1 and 2, the read_size will be d3 * size.
        read_size = weighted_chunk(weighted_shape, max + 1, size)

        # And now how we will traverse
        traverse_list = Enum.map(list, &Enum.fetch!(weighted_shape, &1))

        data =
          for <<chunk::size(chunk_size)-bitstring <- data>> do
            weighted_traverse(traverse_list, chunk, read_size)
          end

        shape = Enum.map(axes, &elem(shape, &1)) |> List.to_tuple()
        %{t | data: {Nx.BitStringDevice, IO.iodata_to_binary(data)}, shape: shape}
    end
  end

  defp transpose_axes(shape, axes) when tuple_size(shape) != length(axes) do
    raise ArgumentError,
          "axes #{inspect(axes)} must be of the same rank as shape #{inspect(shape)}"
  end

  defp transpose_axes(shape, axes) do
    size = tuple_size(shape)

    unless Enum.all?(axes, &(&1 >= 0 and &1 < size)) and length(Enum.uniq(axes)) == size do
      raise ArgumentError,
            "axes #{inspect(axes)} must be unique integers between 0 and #{size - 1}"
    end

    {axes, min} = transpose_min(axes, 0)
    {axes, max} = transpose_max(Enum.reverse(axes), size - 1)
    {axes, min, max}
  end

  defp transpose_min([head | tail], head), do: transpose_min(tail, head + 1)
  defp transpose_min(tail, head), do: {tail, head}

  defp transpose_max([head | tail], head), do: transpose_max(tail, head - 1)
  defp transpose_max(tail, head), do: {Enum.reverse(tail), head}

  ## Shape

  defp shape!(shape) when is_tuple(shape), do: shape
  defp shape!(%T{shape: shape}), do: shape
  defp shape!(number) when is_number(number), do: {}

  defp shape!(other) do
    raise ArgumentError,
          "expected a shape as argument. A shape is a n-element tuple with the size of each dimension. " <>
            "Alternatively you can pass a tensor (or a number) and the shape will be retrieved from the tensor. " <>
            "Got: #{inspect(other)}"
  end

  ## Broadcast helpers

  defp shape_to_lower_ranked_list(_tuple, 0, 0),
    do: []

  defp shape_to_lower_ranked_list(tuple, 0, rank),
    do: [1 | shape_to_lower_ranked_list(tuple, 0, rank - 1)]

  defp shape_to_lower_ranked_list(tuple, size, rank),
    do: [:erlang.element(size, tuple) | shape_to_lower_ranked_list(tuple, size - 1, rank - 1)]

  defp binary_broadcast(
         %T{type: {_, left_size}, shape: shape} = left,
         %T{type: {_, right_size}, shape: shape} = right,
         fun
       ) do
    data =
      bin_zip_map(
        Nx.Util.to_bitstring(left),
        left_size,
        Nx.Util.to_bitstring(right),
        right_size,
        fun
      )

    {IO.iodata_to_binary(data), shape}
  end

  defp binary_broadcast(
         %T{type: {_, left_size}, shape: left_shape} = left,
         %T{type: {_, right_size}, shape: right_shape} = right,
         fun
       ) do
    left_rank = tuple_size(left_shape)
    right_rank = tuple_size(right_shape)
    rank = :erlang.max(left_rank, right_rank)
    left_ordered = shape_to_lower_ranked_list(left_shape, left_rank, rank)
    right_ordered = shape_to_lower_ranked_list(right_shape, right_rank, rank)

    case broadcast_chunks(left_ordered, right_ordered, left_size, right_size, [fun], []) do
      {chunks, shape} ->
        {broadcast_recur(Nx.Util.to_bitstring(left), Nx.Util.to_bitstring(right), chunks), shape}

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
    |> bin_zip_map(left_chunk, right_data, right_chunk, &broadcast_recur(&1, &2, chunks))
    |> IO.iodata_to_binary()
  end

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

  defp bin_zip_map(<<>>, _left_size, <<>>, _right_size, _fun), do: []

  defp bin_zip_map(left_data, left_size, right_data, right_size, fun) do
    <<left_head::bitstring-size(left_size), left_rest::bitstring>> = left_data
    <<right_head::bitstring-size(right_size), right_rest::bitstring>> = right_data

    [
      fun.(left_head, right_head)
      | bin_zip_map(left_rest, left_size, right_rest, right_size, fun)
    ]
  end
end

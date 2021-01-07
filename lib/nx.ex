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
  `Nx.BinaryDevice`, which means the tensor is allocated as a
  binary within the Erlang VM.

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

  import Nx.Shared
  alias Nx.Tensor, as: T

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
    assert_keys!(opts, [:type])
    type = opts[:type]

    if type && type != t.type do
      raise ArgumentError,
            "got a tensor with type #{inspect(type)} but tensor has type #{inspect(t.type)}"
    end

    t
  end

  def tensor(arg, opts) do
    assert_keys!(opts, [:type])
    type = Nx.Type.normalize!(opts[:type] || Nx.Type.infer(arg))
    Nx.BinaryTensor.tensor(arg, type)
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
    assert_keys!(opts, [:type])
    shape = shape(tensor_or_shape)
    type = Nx.Type.normalize!(opts[:type] || Nx.Type.infer(max - min))
    Nx.BinaryTensor.random_uniform(%T{shape: shape, type: type}, min, max)
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
    assert_keys!(opts, [:type])
    shape = shape(tensor_or_shape)
    type = Nx.Type.normalize!(opts[:type] || {:f, 64})
    Nx.BinaryTensor.random_normal(%T{shape: shape, type: type}, mu, sigma)
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

  def iota(tensor_or_shape, opts) do
    assert_keys!(opts, [:type, :axis])
    shape = shape(tensor_or_shape)
    type = Nx.Type.normalize!(opts[:type] || {:s, 64})

    if axis = opts[:axis] do
      axis = Nx.Shape.normalize_axis(shape, axis)
      Nx.BinaryTensor.iota(%T{type: type, shape: shape}, axis)
    else
      %T{type: type, shape: {Nx.size(shape)}}
      |> Nx.BinaryTensor.iota(0)
      |> Map.replace!(:shape, shape)
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
    tensor = tensor(tensor)
    impl!(tensor).to_binary(tensor)
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
  def from_binary(binary, type) when is_binary(binary) do
    {_, size} = Nx.Type.normalize!(type)
    dim = div(bit_size(binary), size)

    if binary == "" do
      raise ArgumentError, "cannot build an empty tensor"
    end

    if rem(bit_size(binary), size) != 0 do
      raise ArgumentError, "binary does not match the given size"
    end

    Nx.BinaryTensor.from_binary(%T{type: type, shape: {dim}}, binary)
  end

  ## Meta operations (do not invoke the backend)

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
    %T{shape: old_shape} = tensor = tensor(tensor)
    new_shape = shape(new_shape)

    if Nx.size(old_shape) != Nx.size(new_shape) do
      raise ArgumentError,
            "cannot reshape, current shape #{inspect(old_shape)} is not compatible with " <>
              "new shape #{inspect(new_shape)}"
    end

    if old_shape == new_shape do
      tensor
    else
      impl!(tensor).reshape(%{tensor | shape: new_shape}, tensor, new_shape)
    end
  end

  @doc """
  Squeezes all of the size `1` dimensions out of the tensor.

  While this is equivalent to a reshape which eliminates
  the size `1` axes, squeeze preserves important information
  about which axes were squeezed out which can then be used
  later on in transformations.

  ## Examples

      iex> Nx.squeeze(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
      >

      iex> Nx.squeeze(Nx.tensor([[[[[1]]]]]))
      #Nx.Tensor<
        s64
        1
      >

      iex> Nx.squeeze(Nx.tensor([[[[1]]], [[[2]]]]))
      #Nx.Tensor<
        s64[2]
        [1, 2]
      >

  """
  def squeeze(tensor) do
    %T{shape: shape} = tensor = tensor(tensor)
    squeeze(tensor, Nx.Shape.squeeze_axes(shape))
  end

  @doc """
  Squeezes the given size `1` dimensions out of the tensor.

  While this is equivalent to a reshape which eliminates
  the size `1` axes, squeeze preserves important information
  about which axes were squeezed out which can then be used
  later on in transformations.

  ## Examples

      iex> Nx.squeeze(Nx.tensor([[1, 2, 3]]), [0])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.squeeze(Nx.tensor([[1], [2]]), [1])
      #Nx.Tensor<
        s64[2]
        [1, 2]
      >

  ### Error cases

      iex> Nx.squeeze(Nx.tensor([[1, 2, 3], [4, 5, 6]]), [1])
      ** (ArgumentError) cannot squeeze dimensions whose sizes are not 1, got 3 for dimension 1

      iex> Nx.squeeze(Nx.tensor([[[[[1]]]]]), [0, 0])
      ** (ArgumentError) axes [0, 0] must be unique integers between 0 and 4
  """
  def squeeze(tensor, axes) do
    %T{shape: old_shape} = tensor = tensor(tensor)
    axes = Nx.Shape.normalize_axes(old_shape, axes)
    new_shape = Nx.Shape.squeeze(old_shape, axes)

    if old_shape == new_shape do
      tensor
    else
      impl!(tensor).squeeze(%{tensor | shape: new_shape}, tensor, axes)
    end
  end

  @doc """
  Broadcasts `tensor` to the given `broadcast_shape`.

  The new shape is either a tuple or a tensor which we will
  retrieve the current shape from. The broadcast shape must
  be of equal or higher rank than the current shape.

  The lower dimensions of the tensor shape must match the
  equivalent lower dimension of the broadcast shape or be 1.
  To customize this behaviour, see `broadcast/3`.

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
  def broadcast(tensor, broadcast_shape) do
    tensor = tensor(tensor)
    broadcast_shape = shape(broadcast_shape)

    if tensor.shape == broadcast_shape do
      tensor
    else
      broadcast(tensor, broadcast_shape, Nx.Shape.broadcast_axes(tensor.shape, broadcast_shape))
    end
  end

  @doc """
  Broadcasts `tensor` to the given `broadcast_shape` with `axes`.

  The new shape is either a tuple or a tensor which we will
  retrieve the current shape from. The broadcast shape must
  be of equal or higher rank than the current shape.

  `axes` must be a list with the same length as the tensor
  shape. Each `axis` in the list maps to the dimension in the
  broadcast shape that must match. For example, an axis of
  `[1, 2]` says the 0 dimension of the tensor matches to the
  1 dimension of the broadcast shape and the 1 dimension of
  the tensor matches the 2 dimension of the broadcast shape.
  Each matching dimension must either be 1, for implicit
  broadcasting, or match the dimension in the broadcast shape.

  The default broadcasting implementation copies the data in
  memory to match the new dimensions.

  ## Examples

  Using the default broadcast rules, we cannot broadcast a
  tensor of shape (3) to the shape (3, 2), because the lower
  dimensions must match. But with `Nx.broadcast/3` we can
  configure how the dimensions match:

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.broadcast(t, {3, 2}, [0])
      #Nx.Tensor<
        s64[3][2]
        [
          [1, 1],
          [2, 2],
          [3, 3]
        ]
      >

  Or a more complex example:

      iex> t = Nx.tensor([1, 2, 3])
      iex> Nx.broadcast(t, {2, 3, 2}, [1])
      #Nx.Tensor<
        s64[2][3][2]
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
  def broadcast(tensor, shape, axes) do
    tensor = tensor(tensor)

    shape = shape(shape)
    axes = Nx.Shape.normalize_axes(shape, axes)
    _ = Nx.Shape.broadcast!(tensor.shape, shape, axes)

    impl!(tensor).broadcast(%{tensor | shape: shape}, tensor, shape, axes)
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

      iex> Nx.pad(Nx.tensor([1, 2, 3]), 0, [{1, 1}])
      #Nx.Tensor<
        s64[5]
        [0, 1, 2, 3, 0]
      >

      iex> Nx.pad(Nx.tensor([[1, 2, 3], [4, 5, 6]]), 0, [{1, 1}, {1, 1}])
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
      iex> Nx.pad(tensor, 0, [{0, 2}, {1, 1}, {1, 0}])
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
      iex> Nx.pad(tensor, 0, [{1, 0}, {1, 1}, {0, 1}])
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
    iex> Nx.pad(tensor, 0.0, [{1, 2}, {1, 0}, {0, 1}])
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

    iex> Nx.pad(Nx.tensor([0, 1, 2, 3, 0]), 0, [{-1, -1}])
    #Nx.Tensor<
      s64[3]
      [1, 2, 3]
    >

    iex> tensor = Nx.tensor([
    ...>   [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ...>   [[0, 0, 0], [1, 2, 0], [3, 4, 0], [0, 0, 0]],
    ...>   [[0, 0, 0], [5, 6, 0], [7, 8, 0], [0, 0, 0]]
    ...> ])
    iex> Nx.pad(tensor, 0, [{-1, 0}, {-1, -1}, {0, -1}])
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

    iex> Nx.pad(Nx.tensor([[0, 1, 2, 3], [0, 4, 5, 6]]), 0, [{0, 0}, {-1, 1}])
    #Nx.Tensor<
      s64[2][4]
      [
        [1, 2, 3, 0],
        [4, 5, 6, 0]
      ]
    >
  """
  def pad(tensor, pad_value, padding_config) when is_list(padding_config) do
    tensor = tensor(tensor)
    pad_value = tensor(pad_value)

    if pad_value.shape != {} do
      raise ArgumentError, "padding value must be a scalar"
    end

    shape = Nx.Shape.pad(tensor.shape, padding_config)
    impl!(tensor).pad(%{tensor | shape: shape}, tensor, pad_value, padding_config)
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
    %T{type: type} = tensor(tensor)
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

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

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

      device_tensor = Nx.device_transfer(tensor, Exla.NxDevice, client: :cuda)

  Read the device tensor back to an Elixir binary:

      tensor = Nx.device_transfer(tensor)

  """
  def device_transfer(tensor, device \\ Nx.BinaryDevice, opts \\ []) do
    tensor = tensor(tensor)
    impl!(tensor).device_transfer(tensor, device, opts)
  end

  @doc """
  Reads data allocated in a device.

  It returns a tensor where the device is `Nx.BinaryDevice`.
  The data is not deallocated from the current device. If the
  device has already been deallocated, it raises.
  """
  def device_read(tensor) do
    tensor = tensor(tensor)
    impl!(tensor).device_read(tensor)
  end

  @doc """
  Deallocates data in a device.

  It returns either `:ok` or `:already_deallocated`.
  """
  def device_deallocate(tensor) do
    tensor = tensor(tensor)
    impl!(tensor).device_deallocate(tensor)
  end

  ## Element-wise binary ops

  defp element_wise_bin_op(left, right, op, fun) do
    type = binary_type(left, right) |> fun.()
    %T{shape: left_shape} = left = tensor(left)
    %T{shape: right_shape} = right = tensor(right)

    shape = Nx.Shape.binary_broadcast(left_shape, right_shape)
    apply(impl!(left, right), op, [%{left | type: type, shape: shape}, left, right])
  end

  defp element_wise_pred_op(left, right, op) do
    %T{shape: left_shape} = left = tensor(left)
    %T{shape: right_shape} = right = tensor(right)

    shape = Nx.Shape.binary_broadcast(left_shape, right_shape)
    apply(impl!(left, right), op, [%{left | type: {:u, 8}, shape: shape}, left, right])
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
  def equal(left, right), do: element_wise_pred_op(left, right, :equal)

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

      iex> Nx.select(0, Nx.tensor([[1, 2]]), Nx.tensor([[3], [4]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [3, 3],
          [4, 4]
        ]
      >

      iex> Nx.select(Nx.tensor([0, 1, 0]), Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6]))
      #Nx.Tensor<
        s64[3]
        [4, 2, 6]
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
    output_type = binary_type(on_true, on_false)

    %T{shape: pred_shape} = pred = tensor(pred)
    %T{shape: true_shape} = on_true = tensor(on_true)
    %T{shape: false_shape} = on_false = tensor(on_false)

    output_shape =
      case pred_shape do
        {} ->
          Nx.Shape.binary_broadcast(true_shape, false_shape)

        _ ->
          pred_shape
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

    out = %{pred | shape: output_shape, type: output_type}
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

        iex> Nx.#{name}(Nx.tensor([1.0, 2.0, 3.0]))
        #Nx.Tensor<
          f64[3]
          [#{one}, #{two}, #{three}]
        >

    """
    def unquote(name)(tensor) do
      tensor = tensor(tensor)
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

      iex> Nx.negate(Nx.tensor([0, 1, 2], type: {:u, 8}))
      #Nx.Tensor<
        u8[3]
        [0, 255, 254]
      >

  """
  def negate(tensor) do
    tensor = tensor(tensor)
    impl!(tensor).negate(tensor, tensor)
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
    tensor = tensor(tensor)
    impl!(tensor).sign(tensor, tensor)
  end

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
    tensor = tensor(tensor)

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
    tensor = tensor(tensor)
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
    tensor = tensor(tensor)
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
    tensor = tensor(tensor)
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
        %T{type: {type, _}} = tensor when type in [:s, :u] -> tensor
        %T{} = tensor -> impl!(tensor).unquote(name)(tensor, tensor)
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

  Giving a tensor with low precision casts it to a higher
  precision to make sure the sum does not overflow:

      iex> Nx.sum(Nx.tensor([[101, 102], [103, 104]], type: {:u, 8}))
      #Nx.Tensor<
        u64
        410
      >

      iex> Nx.sum(Nx.tensor([[101, 102], [103, 104]], type: {:s, 8}))
      #Nx.Tensor<
        s64
        410
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
      ** (ArgumentError) given axis (2) invalid for shape with rank 2

  """
  def sum(tensor, opts \\ []) do
    assert_keys!(opts, [:axes])
    %{shape: shape, type: type} = tensor = tensor(tensor)

    {shape, axes} =
      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes)
        {Nx.Shape.contract(shape, axes), axes}
      else
        {{}, nil}
      end

    type =
      case type do
        {:u, _} -> {:u, 64}
        {:s, _} -> {:s, 64}
        type -> type
      end

    impl!(tensor).sum(%{tensor | type: type, shape: shape}, tensor, axes: axes)
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

      iex> Nx.mean(Nx.tensor([1, 2, 3], type: {:u, 8}), axes: [0])
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

  """
  def mean(tensor, opts \\ []) do
    tensor = tensor(tensor)
    divide(sum(tensor, opts), mean_den(tensor.shape, opts[:axes]))
  end

  defp mean_den(shape, nil), do: Nx.size(shape)
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

    %{shape: shape} = tensor = tensor(tensor)

    {shape, axis} =
      if axis = opts[:axis] do
        axis = Nx.Shape.normalize_axis(shape, axis)
        {Nx.Shape.contract(shape, [axis]), axis}
      else
        {{}, nil}
      end

    opts = [tie_break: tie_break, axis: axis]
    apply(impl!(tensor), op, [%{tensor | type: {:s, 64}, shape: shape}, tensor, opts])
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
      ** (ArgumentError) dot/zip expects shapes to be compatible, dimension 0 of left-side (3) does not equal dimension 0 of right-side (2)
  """
  def dot(t1, t2) do
    %T{shape: s1} = t1 = tensor(t1)
    %T{shape: s2} = t2 = tensor(t2)

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

      iex> t1 = Nx.tensor([[1, 2], [3, 4]])
      iex> t2 = Nx.tensor([[10, 20], [30, 40]])
      iex> Nx.dot(t1, [0], t2, [0])
      #Nx.Tensor<
        s64[2][2]
        [
          [100, 140],
          [140, 200]
        ]
      >
      iex> Nx.dot(t1, [0], t2, [1])
      #Nx.Tensor<
        s64[2][2]
        [
          [70, 150],
          [100, 220]
        ]
      >
      iex> Nx.dot(t1, [1], t2, [0])
      #Nx.Tensor<
        s64[2][2]
        [
          [70, 100],
          [150, 220]
        ]
      >
      iex> Nx.dot(t1, [1], t2, [1])
      #Nx.Tensor<
        s64[2][2]
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
    %T{shape: s1} = t1 = tensor(t1)
    %T{shape: s2} = t2 = tensor(t2)
    axes1 = Nx.Shape.normalize_axes(s1, axes1)
    axes2 = Nx.Shape.normalize_axes(s2, axes2)
    output_shape = Nx.Shape.zip_reduce(s1, axes1, s2, axes2)
    impl!(t1, t2).dot(%{t1 | type: output_type, shape: output_shape}, t1, axes1, t2, axes2)
  end

  @doc """
  Computes the outer product of two tensors.

  The output is always a two-dimensional tensor.

  ## Examples

      iex> Nx.outer(Nx.tensor([1, 2, 3]), 100)
      #Nx.Tensor<
        s64[3][1]
        [
          [100],
          [200],
          [300]
        ]
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
        s64[4][3]
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
    %T{shape: s1} = t1 = tensor(t1)
    %T{shape: s2} = t2 = tensor(t2)
    impl!(t1, t2).outer(%{t1 | type: type, shape: Nx.Shape.outer(s1, s2)}, t1, t2)
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
    tensor = tensor(tensor)
    transpose(tensor, Nx.Shape.transpose_axes(tensor.shape))
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

      iex> Nx.transpose(Nx.iota({2, 2}), [0])
      ** (ArgumentError) expected length of permutation (1) to match rank of shape (2)

      iex> Nx.transpose(Nx.iota({2, 2}), [1, 2])
      ** (ArgumentError) given axis (2) invalid for shape with rank 2

  """
  def transpose(tensor, axes) when is_list(axes) do
    %{shape: shape} = tensor = tensor(tensor)
    axes = Nx.Shape.normalize_axes(shape, axes)

    if axes == Nx.axes(shape) do
      tensor
    else
      shape = Nx.Shape.transpose(shape, axes)
      impl!(tensor).transpose(%{tensor | shape: shape}, tensor, axes)
    end
  end

  ## Type

  defp binary_type(a, b) when is_number(a) and is_number(b), do: Nx.Type.infer(a + b)
  defp binary_type(a, b) when is_number(a), do: Nx.Type.merge_scalar(b.type, a)
  defp binary_type(a, b) when is_number(b), do: Nx.Type.merge_scalar(a.type, b)
  defp binary_type(a, b), do: Nx.Type.merge(a.type, b.type)
end

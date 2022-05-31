defmodule Nx do
  @moduledoc """
  Numerical Elixir.

  The `Nx` library is a collection of functions and data
  types to work with Numerical Elixir. This module defines
  the main entry point for building and working with said
  data-structures. For example, to create an n-dimensional
  tensor, do:

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> Nx.shape(t)
      {2, 2}

  `Nx` also provides the so-called numerical definitions under
  the `Nx.Defn` module. They are a subset of Elixir tailored for
  numerical computations. For example, it overrides Elixir's
  default operators so they are tensor-aware:

      defn softmax(t) do
        Nx.exp(t) / Nx.sum(Nx.exp(t))
      end

  Code inside `defn` functions can also be given to custom compilers,
  which can compile said functions just-in-time (JIT) to run on the
  CPU or on the GPU.

  ## References

  Here is a general outline of the main references in this library:

    * For an introduction, see our [Intro to Nx](intro-to-nx.livemd) guide

    * This module provides the main API for working with tensors

    * `Nx.Defn` provides numerical definitions, CPU/GPU compilation, gradients, and more

    * `Nx.LinAlg` provides functions related to linear algebra

    * `Nx.Constants` declares many constants commonly used in numerical code

  Continue reading this documentation for an overview of creating,
  broadcasting, and accessing/slicing Nx tensors.

  ## Creating tensors

  The main APIs for creating tensors are `tensor/2`, `from_binary/2`,
  `iota/2`, `eye/2`, `random_uniform/2`, `random_normal/2`, and
  `broadcast/3`.

  The tensor types can be one of:

    * unsigned integers (`u8`, `u16`, `u32`, `u64`)
    * signed integers (`s8`, `s16`, `s32`, `s64`)
    * floats (`f16`, `f32`, `f64`)
    * brain floats (`bf16`)
    * and complex numbers (`c64`, `c128`)

  The types are tracked as tuples:

      iex> Nx.tensor([1, 2, 3], type: {:f, 32})
      #Nx.Tensor<
        f32[3]
        [1.0, 2.0, 3.0]
      >

  But a shortcut atom notation is also available:

      iex> Nx.tensor([1, 2, 3], type: :f32)
      #Nx.Tensor<
        f32[3]
        [1.0, 2.0, 3.0]
      >

  The tensor dimensions can also be named, via the `:names` option
  available to all creation functions:

      iex> Nx.iota({2, 3}, names: [:x, :y])
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [0, 1, 2],
          [3, 4, 5]
        ]
      >

  Finally, for creating vectors and matrices, a sigil notation
  is available:

      iex> import Nx, only: :sigils
      iex> ~V[1 2 3]f32
      #Nx.Tensor<
        f32[3]
        [1.0, 2.0, 3.0]
      >

      iex> import Nx, only: :sigils
      iex> ~M'''
      ...> 1 2 3
      ...> 4 5 6
      ...> '''s32
      #Nx.Tensor<
        s32[2][3]
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
      >

  All other APIs accept exclusively numbers or tensors, unless
  explicitly noted otherwise.

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

  Now let's imagine you want to multiply a large tensor of dimensions
  1000x1000x1000 by 2. If you had to create a similarly large tensor
  only to perform this operation, it would be inefficient. Therefore,
  you can simply multiply this large tensor by the scalar 2, and Nx
  will propagate its dimensions at the time the operation happens,
  without allocating a large intermediate tensor:

      iex> Nx.multiply(Nx.tensor([1, 2, 3]), 2)
      #Nx.Tensor<
        s64[3]
        [2, 4, 6]
      >

  In practice, broadcasting is not restricted only to scalars; it
  is a general algorithm that applies to all dimensions of a tensor.
  When broadcasting, `Nx` compares the shapes of the two tensors,
  starting with the trailing ones, such that:

    * If the dimensions have equal size, then they are compatible

    * If one of the dimensions have size of 1, it is "broadcast"
      to match the dimension of the other

  In case one tensor has more dimensions than the other, the missing
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

  ## Access syntax (slicing)

  Nx tensors implement Elixir's access syntax. This allows developers
  to slice tensors up and easily access sub-dimensions and values.

  Access accepts integers:

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t[0]
      #Nx.Tensor<
        s64[2]
        [1, 2]
      >
      iex> t[1]
      #Nx.Tensor<
        s64[2]
        [3, 4]
      >
      iex> t[1][1]
      #Nx.Tensor<
        s64
        4
      >

  If a negative index is given, it accesses the element from the back:

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t[-1][-1]
      #Nx.Tensor<
        s64
        4
      >

  Out of bound access will raise:

      iex> Nx.tensor([1, 2])[2]
      ** (ArgumentError) index 2 is out of bounds for axis 0 in shape {2}

      iex> Nx.tensor([1, 2])[-3]
      ** (ArgumentError) index -3 is out of bounds for axis 0 in shape {2}

  The index can also be another tensor but in such cases it must be
  a scalar between 0 and the dimension size. Out of bound dynamic indexes
  are always clamped to the tensor dimensions:

      iex> two = Nx.tensor(2)
      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t[two][two]
      #Nx.Tensor<
        s64
        4
      >

  For example, a `minus_one` dynamic index will be clamped to zero:

      iex> minus_one = Nx.tensor(-1)
      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> t[minus_one][minus_one]
      #Nx.Tensor<
        s64
        1
      >

  Access also accepts ranges. Ranges in Elixir are inclusive:

      iex> t = Nx.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
      iex> t[0..1]
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [3, 4]
        ]
      >

  Ranges can receive negative positions and they will read from
  the back. In such cases, the range step must be explicitly given
  and the right-side of the range must be equal or greater than
  the left-side:

      iex> t = Nx.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
      iex> t[1..-2//1]
      #Nx.Tensor<
        s64[2][2]
        [
          [3, 4],
          [5, 6]
        ]
      >

  As you can see, accessing with a range does not eliminate the
  accessed axis, therefore, when wanting to slice across multiple
  axes with ranges, it is often desired to use a list:

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      iex> t[[1..2, 1..2]]
      #Nx.Tensor<
        s64[2][2]
        [
          [5, 6],
          [8, 9]
        ]
      >

  You can mix both ranges and integers in the list too:

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      iex> t[[1..2, 2]]
      #Nx.Tensor<
        s64[2]
        [6, 9]
      >

  If the list has less elements than axes, the remaining dimensions
  are returned in full:

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      iex> t[[1..2]]
      #Nx.Tensor<
        s64[2][3]
        [
          [4, 5, 6],
          [7, 8, 9]
        ]
      >

  The access syntax also pairs nicely with named tensors. By
  using named tensors, you can pass only the axis you want to
  slice, leaving the other axis intact:

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], names: [:x, :y])
      iex> t[x: 1..2]
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [4, 5, 6],
          [7, 8, 9]
        ]
      >
      iex> t[x: 1..2, y: 0..1]
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [4, 5],
          [7, 8]
        ]
      >
      iex> t[x: 1, y: 0..1]
      #Nx.Tensor<
        s64[y: 2]
        [4, 5]
      >

  For a more complex slicing rules, including strides, you
  can always fallback to `Nx.slice/4`.

  ## Backends

  The `Nx` library has built-in support for multiple backends.
  A tensor is always handled by a backend, the default backend
  being `Nx.BinaryBackend`, which means the tensor is allocated
  as a binary within the Erlang VM.

  Most often backends are used to provide a completely different
  implementation of tensor operations, often accelerated to the GPU.
  In such cases, you want to guarantee all tensors are allocated in
  the new backend. This can be done by configuring your runtime:

      # config/runtime.exs
      import Config
      config :nx, default_backend: Lib.CustomBackend

  Or by calling `Nx.default_backend/1`:

      Nx.default_backend({Lib.CustomBackend, device: :cuda})

  To implement your own backend, check the `Nx.Tensor` behaviour.
  """

  import Nx.Shared
  import Nx.Defn.Kernel, only: [keyword!: 2]

  alias Nx.Tensor, as: T

  @typedoc """
  Represents a numerical value.

  Can be a plain number, a `Complex` number or an `Nx.Tensor`.

  See also: `is_tensor/1`
  """
  @type t :: number | Complex.t() | Nx.Tensor.t()
  @type shape :: number() | Nx.Tensor.t() | Nx.Tensor.shape()
  @type axis :: Nx.Tensor.axis()
  @type axes :: Nx.Tensor.axes()

  @file_version 1

  @non_finite [:neg_infinity, :infinity, :nan]

  @doc """
  Checks whether the value is a valid numerical value.

  Returns true if the value is a `number`, a `Complex` number or an `Nx.Tensor`.

  See also: `t:t/0`
  """
  @doc type: :guards
  defguard is_tensor(t) when is_number(t) or is_struct(t, T) or is_struct(t, Complex)

  ## Creation API

  @doc """
  Builds a tensor.

  The argument is either a number, which means the tensor is a scalar
  (zero-dimensions), a list of those (the tensor is a vector) or
  a list of n-lists of those, leading to n-dimensional tensors.
  The tensor will be allocated in `Nx.default_backend/1`, unless the
  `:backend` option is given, which overrides the default one.

  ## Examples

  A number returns a tensor of zero dimensions:

      iex> Nx.tensor(0)
      #Nx.Tensor<
        s64
        0
      >

      iex> Nx.tensor(1.0)
      #Nx.Tensor<
        f32
        1.0
      >

  Giving a list returns a vector (a one-dimensional tensor):

      iex> Nx.tensor([1, 2, 3])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.tensor([1.2, 2.3, 3.4, 4.5])
      #Nx.Tensor<
        f32[4]
        [1.2000000476837158, 2.299999952316284, 3.4000000953674316, 4.5]
      >

  The type can be explicitly given. Integers and floats
  bigger than the given size overflow:

      iex> Nx.tensor([300, 301, 302], type: {:s, 8})
      #Nx.Tensor<
        s8[3]
        [44, 45, 46]
      >

  Mixed types give higher priority to floats:

      iex> Nx.tensor([1, 2, 3.0])
      #Nx.Tensor<
        f32[3]
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

  Besides single-precision (32 bits), floats can also have
  half-precision (16) or double-precision (64):

      iex> Nx.tensor([1, 2, 3], type: {:f, 16})
      #Nx.Tensor<
        f16[3]
        [1.0, 2.0, 3.0]
      >

      iex> Nx.tensor([1, 2, 3], type: {:f, 64})
      #Nx.Tensor<
        f64[3]
        [1.0, 2.0, 3.0]
      >

  Brain-floating points are also supported:

      iex> Nx.tensor([1, 2, 3], type: {:bf, 16})
      #Nx.Tensor<
        bf16[3]
        [1.0, 2.0, 3.0]
      >

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
      one is automatically inferred based on the input.

    * `:names` - dimension names. If you wish to specify dimension
      names you must specify a name for every dimension in the tensor.
      Only `nil` and atoms are supported as dimension names.

    * `:backend` - the backend to allocate the tensor on. It is either
      an atom or a tuple in the shape `{backend, options}`. This option
      is ignored inside `defn`

  """
  @doc type: :creation
  def tensor(arg, opts \\ []) do
    opts = keyword!(opts, [:type, :names, :backend])
    type = Nx.Type.normalize!(opts[:type] || infer_type(arg))
    tensor(arg, type, opts)
  end

  defp infer_type([head | tail]) when is_list(tail) do
    Enum.reduce(tail, infer_type(head), &Nx.Type.merge(infer_type(&1), &2))
  end

  defp infer_type(number)
       when is_number(number) or is_struct(number, Complex) or number in @non_finite do
    Nx.Type.infer(number)
  end

  defp infer_type(value) do
    raise ArgumentError, "invalid value given to Nx.tensor/1, got: #{inspect(value)}"
  end

  defp tensor(arg, type, opts) when is_number(arg) do
    names = Nx.Shape.named_axes!(opts[:names], {})
    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    backend.constant(%T{shape: {}, type: type, names: names}, arg, backend_options)
  end

  defp tensor(%Complex{} = arg, {:c, size}, opts) do
    names = Nx.Shape.named_axes!(opts[:names], {})
    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    backend.constant(%T{shape: {}, type: {:c, size}, names: names}, arg, backend_options)
  end

  defp tensor(%Complex{}, type, _) do
    raise ArgumentError,
          "invalid type for complex number. Expected {:c, 64} or {:c, 128}, got: #{inspect(type)}"
  end

  defp tensor(arg, type, opts) when arg in @non_finite do
    names = Nx.Shape.named_axes!(opts[:names], {})
    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    data = number_to_binary(arg, type)
    backend.from_binary(%T{shape: {}, type: type, names: names}, data, backend_options)
  end

  defp tensor(arg, type, opts) when is_list(arg) do
    {shape, data} = flatten_list(arg, type)

    if data == "" do
      raise "cannot build empty tensor"
    end

    names = Nx.Shape.named_axes!(opts[:names], shape)
    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    backend.from_binary(%T{shape: shape, type: type, names: names}, data, backend_options)
  end

  defp flatten_list(list, type) do
    {dimensions, acc} = flatten_list(list, type, [], [])

    {dimensions |> Enum.reverse() |> List.to_tuple(),
     acc |> Enum.reverse() |> :erlang.list_to_binary()}
  end

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
    {[length(list) | dimensions],
     Enum.reduce(list, acc, &[tensor_or_number_to_binary(&1, type) | &2])}
  end

  defp tensor_or_number_to_binary(%Complex{re: re, im: im}, {:c, size}) do
    number_to_binary(re, {:f, div(size, 2)}) <> number_to_binary(im, {:f, div(size, 2)})
  end

  defp tensor_or_number_to_binary(number, type) when is_number(number) do
    number_to_binary(number, type)
  end

  defp tensor_or_number_to_binary(number, type) when number in @non_finite do
    number_to_binary(number, type)
  end

  defp tensor_or_number_to_binary(value, _type) do
    raise ArgumentError, "invalid value given to Nx.tensor/1, got: #{inspect(value)}"
  end

  @doc """
  Creates a tensor template.

  You can't perform any operation on this tensor.
  It exists exclusively to define APIs that say
  a tensor with a certain type, shape, and names
  is expected in the future.

  ## Examples

      iex> Nx.template({2, 3}, {:f, 32})
      #Nx.Tensor<
        f32[2][3]
        Nx.TemplateBackend
      >

      iex> Nx.template({2, 3}, {:f, 32}, names: [:rows, :columns])
      #Nx.Tensor<
        f32[rows: 2][columns: 3]
        Nx.TemplateBackend
      >

  Although note it is impossible to perform any operation on a tensor template:

      iex> t = Nx.template({2, 3}, {:f, 32}, names: [:rows, :columns])
      iex> Nx.abs(t)
      ** (RuntimeError) cannot perform operations on a Nx.TemplateBackend tensor

  To convert existing tensors to templates, use `to_template/1`.
  """
  @doc type: :creation
  def template(shape, type, opts \\ []) when is_tuple(shape) do
    opts = keyword!(opts, [:names])
    type = Nx.Type.normalize!(type)
    names = Nx.Shape.named_axes!(opts[:names], shape)
    %T{shape: shape, type: type, names: names, data: %Nx.TemplateBackend{}}
  end

  @doc """
  Converts a tensor (or tuples and maps of tensors) to tensor templates.

  Templates are useful when you need to pass types and shapes to
  operations and the data is not yet available.

  For convenience, this function accepts tensors and any container
  (such as maps and tuples as defined by the `Nx.Container` protocol)
  and recursively converts all tensors to templates.

  ## Examples

      iex> Nx.iota({2, 3}) |> Nx.to_template()
      #Nx.Tensor<
        s64[2][3]
        Nx.TemplateBackend
      >

      iex> {int, float} = Nx.to_template({1, 2.0})
      iex> int
      #Nx.Tensor<
        s64
        Nx.TemplateBackend
      >
      iex> float
      #Nx.Tensor<
        f32
        Nx.TemplateBackend
      >

  Although note it is impossible to perform any operation on a tensor template:

      iex> t = Nx.iota({2, 3}) |> Nx.to_template()
      iex> Nx.abs(t)
      ** (RuntimeError) cannot perform operations on a Nx.TemplateBackend tensor

  To build a template from scratch, use `template/3`.
  """
  @doc type: :conversion
  def to_template(tensor_or_container) do
    Nx.Defn.Composite.traverse(tensor_or_container, fn tensor ->
      %{to_tensor(tensor) | data: %Nx.TemplateBackend{}}
    end)
  end

  @doc """
  Shortcut for `random_uniform(shape, 0.0, 1.0, opts)`.
  """
  @doc type: :random
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
      iex> for <<x::float-32-native <- Nx.to_binary(t)>> do
      ...>   true = x >= 0.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Nx.random_uniform({5, 5}, type: {:bf, 16})
      iex> byte_size(Nx.to_binary(t))
      50
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:bf, 16}

      iex> t = Nx.random_uniform({5, 5}, -1.0, 1.0, type: {:f, 64})
      iex> for <<x::float-64-native <- Nx.to_binary(t)>> do
      ...>   true = x >= -1.0 and x < 1.0
      ...> end
      iex> Nx.shape(t)
      {5, 5}
      iex> Nx.type(t)
      {:f, 64}

  ### Generating Integers

      iex> t = Nx.random_uniform({10}, 5, 10, type: {:u, 8})
      iex> for <<x::8-unsigned-native <- Nx.to_binary(t)>> do
      ...>   true = x >= 5 and x < 10
      ...> end
      iex> Nx.shape(t)
      {10}
      iex> Nx.type(t)
      {:u, 8}

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
      {:f, 32}
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
      {:f, 32}

      iex> t = Nx.random_uniform(10.0)
      iex> Nx.shape(t)
      {}
      iex> Nx.type(t)
      {:f, 32}
      iex> Nx.names(t)
      []

  If you pass `:names` as an option, the resulting tensor will take on those names:

      iex> t = Nx.tensor([[1, 2], [3, 4]], names: [:batch, :data])
      iex> t = Nx.random_uniform(t, names: [:batch, nil])
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}
      iex> Nx.names(t)
      [:batch, nil]

  ## Options

    * `:type` - the type of the tensor

    * `:names` - the names of the tensor dimensions

    * `:backend` - the backend to allocate the tensor on. It is either
      an atom or a tuple in the shape `{backend, options}`. This option
      is ignored inside `defn`

  """
  @doc type: :random
  def random_uniform(tensor_or_shape, min, max, opts \\ []) do
    opts = keyword!(opts, [:type, :names, :backend])
    %T{type: min_type, shape: min_shape} = min = to_tensor(min)
    %T{type: max_type, shape: max_shape} = max = to_tensor(max)

    shape = shape(tensor_or_shape)
    names = Nx.Shape.named_axes!(opts[:names] || names!(tensor_or_shape), shape)
    range_type = Nx.Type.merge(min_type, max_type)
    type = Nx.Type.normalize!(opts[:type] || range_type)

    unless min_shape == {} and max_shape == {} do
      raise ArgumentError,
            "random_uniform/3 expects min and max to be scalars, got:" <>
              " min shape: #{inspect(min_shape)} and max shape: #{inspect(max_shape)}"
    end

    unless Nx.Type.float?(type) or (Nx.Type.integer?(type) and Nx.Type.integer?(range_type)) do
      raise ArgumentError,
            "random_uniform/3 expects compatible types, got: #{inspect(type)}" <>
              " with range #{inspect(range_type)}"
    end

    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    backend.random_uniform(%T{shape: shape, type: type, names: names}, min, max, backend_options)
  end

  @doc """
  Shortcut for `random_normal(shape, 0.0, 1.0, opts)`.
  """
  @doc type: :random
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
      {:f, 32}

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
      {:f, 32}
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
      {:f, 32}
      iex> Nx.names(t)
      []

  If you pass the `:names` option, the resulting tensor will take on those names:

      iex> t = Nx.tensor([[1, 2], [3, 4]], names: [:batch, :data])
      iex> t = Nx.random_normal(t, names: [:batch, nil])
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}
      iex> Nx.names(t)
      [:batch, nil]

  ## Options

    * `:type` - the type of the tensor

    * `:names` - the names of the tensor dimensions

    * `:backend` - the backend to allocate the tensor on. It is either
      an atom or a tuple in the shape `{backend, options}`. This option
      is ignored inside `defn`

  """
  @doc type: :random
  def random_normal(tensor_or_shape, mu, sigma, opts \\ []) do
    opts = keyword!(opts, [:type, :names, :backend])
    %T{type: mu_type, shape: mu_shape} = mu = to_tensor(mu)
    %T{type: sigma_type, shape: sigma_shape} = sigma = to_tensor(sigma)

    shape = shape(tensor_or_shape)
    names = Nx.Shape.named_axes!(opts[:names] || names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type] || {:f, 32})

    unless mu_shape == {} and sigma_shape == {} do
      raise ArgumentError,
            "random_normal/3 expects mu and sigma to be scalars" <>
              " got: mu shape: #{inspect(mu_shape)} and sigma shape: #{inspect(sigma_shape)}"
    end

    unless Nx.Type.float?(mu_type) and Nx.Type.float?(sigma_type) do
      raise ArgumentError,
            "random_normal/3 expects mu and sigma to be float types," <>
              " got: mu type: #{inspect(mu_type)} and sigma type: #{inspect(sigma_type)}"
    end

    unless Nx.Type.float?(type) do
      raise ArgumentError, "random_normal/3 expects float type, got: #{inspect(type)}"
    end

    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    backend.random_normal(%T{shape: shape, type: type, names: names}, mu, sigma, backend_options)
  end

  @doc """
  Shuffles tensor elements.

  By default, shuffles elements within the whole tensor. When `:axis`
  is given, shuffles the tensor along the specific axis instead.

  ## Options

    * `:axis` - the axis to shuffle along

  ## Examples

  Shuffling all elements:

      t = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      Nx.shuffle(t)
      #=>
      #Nx.Tensor<
        s64[3][2]
        [
          [5, 1],
          [2, 3],
          [6, 4]
        ]
      >

  Shuffling rows in a two-dimensional tensor:

      t = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      Nx.shuffle(t, axis: 0)
      #=>
      #Nx.Tensor<
        s64[3][2]
        [
          [5, 6],
          [1, 2],
          [3, 4]
        ]
      >
  """
  @doc type: :random
  def shuffle(tensor, opts \\ []) do
    opts = keyword!(opts, [:axis])
    %T{shape: shape, names: names} = tensor = to_tensor(tensor)

    if axis = opts[:axis] do
      axis = Nx.Shape.normalize_axis(shape, axis, names)
      size = Nx.axis_size(tensor, axis)
      permutation = Nx.random_uniform({size}) |> Nx.argsort()
      Nx.take(tensor, permutation, axis: axis)
    else
      flattened = Nx.flatten(tensor)
      permutation = flattened |> Nx.random_uniform() |> Nx.argsort()
      flattened |> Nx.take(permutation) |> Nx.reshape(tensor)
    end
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

  ## Options

    * `:type` - the type of the tensor

    * `:axis` - an axis to repeat the iota over

    * `:names` - the names of the tensor dimensions

    * `:backend` - the backend to allocate the tensor on. It is either
      an atom or a tuple in the shape `{backend, options}`. This option
      is ignored inside `defn`

  """
  @doc type: :creation
  def iota(tensor_or_shape, opts \\ []) do
    opts = keyword!(opts, [:axis, :names, :backend, type: {:s, 64}])
    shape = shape(tensor_or_shape)
    names = Nx.Shape.named_axes!(opts[:names] || names!(tensor_or_shape), shape)
    type = Nx.Type.normalize!(opts[:type])
    {backend, backend_options} = backend_from_options!(opts) || default_backend()

    if axis = opts[:axis] do
      axis = Nx.Shape.normalize_axis(shape, axis, names)
      backend.iota(%T{type: type, shape: shape, names: names}, axis, backend_options)
    else
      backend.iota(%T{type: type, shape: shape, names: names}, nil, backend_options)
    end
  end

  @doc """
  Creates the identity matrix of size `n`.

  ## Examples

      iex> Nx.eye(2)
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 0],
          [0, 1]
        ]
      >

      iex> Nx.eye(3, type: {:f, 32}, names: [:height, :width])
      #Nx.Tensor<
        f32[height: 3][width: 3]
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ]
      >

  The first argument can also be a tensor or a shape of a square
  matrix:

      iex> Nx.eye(Nx.iota({2, 2}))
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 0],
          [0, 1]
        ]
      >

      iex> Nx.eye({1, 1})
      #Nx.Tensor<
        s64[1][1]
        [
          [1]
        ]
      >

  ## Options

    * `:type` - the type of the tensor

    * `:names` - the names of the tensor dimensions

    * `:backend` - the backend to allocate the tensor on. It is either
      an atom or a tuple in the shape `{backend, options}`. This option
      is ignored inside `defn`

  """
  @doc type: :creation
  def eye(n_or_shape_or_tensor, opts \\ [])

  def eye(n, opts) when is_integer(n) and n > 0 do
    eye({n, n}, opts)
  end

  def eye(shape_or_tensor, opts) do
    opts = keyword!(opts, [:names, :backend, type: {:s, 64}])

    shape =
      case shape(shape_or_tensor) do
        {n, n} -> {n, n}
        other -> raise ArgumentError, "eye/2 expects a square matrix, got: #{inspect(other)}"
      end

    names = Nx.Shape.named_axes!(opts[:names], shape)
    type = Nx.Type.normalize!(opts[:type] || {:s, 64})

    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    backend.eye(%T{type: type, shape: shape, names: names}, backend_options)
  end

  @doc """
  Extracts the diagonal of a 2D tensor.

  Converse of `make_diagonal/2`.

  ## Examples

  Given a 2D tensor without offset:

      iex> Nx.take_diagonal(Nx.tensor([
      ...> [0, 1, 2],
      ...> [3, 4, 5],
      ...> [6, 7, 8]
      ...> ]))
      #Nx.Tensor<
        s64[3]
        [0, 4, 8]
      >

  And if given a 2D tensor along with an offset:

      iex> Nx.take_diagonal(Nx.iota({3, 3}), offset: 1)
      #Nx.Tensor<
        s64[2]
        [1, 5]
      >

      iex> Nx.take_diagonal(Nx.iota({3, 3}), offset: -1)
      #Nx.Tensor<
        s64[2]
        [3, 7]
      >

  ## Options

    * `:offset` - offset used for extracting the diagonal.
      Use offset > 0 for diagonals above the main diagonal,
      and offset < 0 for diagonals below the main diagonal.
      Defaults to 0.

  ## Error cases

      iex> Nx.take_diagonal(Nx.tensor([0, 1, 2]))
      ** (ArgumentError) take_diagonal/2 expects tensor of rank 2, got tensor of rank: 1

      iex> Nx.take_diagonal(Nx.iota({3, 3}), offset: 3)
      ** (ArgumentError) offset must be less than length of axis 1 when positive, got: 3

      iex> Nx.take_diagonal(Nx.iota({3, 3}), offset: -4)
      ** (ArgumentError) absolute value of offset must be less than length of axis 0 when negative, got: -4
  """
  @doc type: :creation
  def take_diagonal(tensor, opts \\ []) do
    tensor = to_tensor(tensor)

    opts = keyword!(opts, offset: 0)

    shape = Nx.Shape.take_diagonal(tensor.shape)
    offset = opts[:offset]

    Nx.Shape.validate_diag_offset!(shape, offset)

    Nx.gather(tensor, diag_indices(shape, offset))
  end

  @doc """
  Creates a diagonal tensor from a 1D tensor.

  Converse of `take_diagonal/2`.

  The returned tensor will be a square matrix of dimensions equal
  to the size of the tensor. If an offset is given, the absolute value
  of the offset is added to the matrix dimensions sizes.

  ## Examples

    Given a 1D tensor:

      iex> Nx.make_diagonal(Nx.tensor([1, 2, 3, 4]))
      #Nx.Tensor<
        s64[4][4]
        [
          [1, 0, 0, 0],
          [0, 2, 0, 0],
          [0, 0, 3, 0],
          [0, 0, 0, 4]
        ]
      >

    Given a 1D tensor with an offset:

      iex> Nx.make_diagonal(Nx.tensor([1, 2, 3]), offset: 1)
      #Nx.Tensor<
        s64[4][4]
        [
          [0, 1, 0, 0],
          [0, 0, 2, 0],
          [0, 0, 0, 3],
          [0, 0, 0, 0]
        ]
      >

      iex> Nx.make_diagonal(Nx.tensor([1, 2, 3]), offset: -1)
      #Nx.Tensor<
        s64[4][4]
        [
          [0, 0, 0, 0],
          [1, 0, 0, 0],
          [0, 2, 0, 0],
          [0, 0, 3, 0]
        ]
      >

    You can also have offsets with an abs greater than the tensor length:

      iex> Nx.make_diagonal(Nx.tensor([1, 2, 3]), offset: -4)
      #Nx.Tensor<
        s64[7][7]
        [
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0],
          [0, 2, 0, 0, 0, 0, 0],
          [0, 0, 3, 0, 0, 0, 0]
        ]
      >

      iex> Nx.make_diagonal(Nx.tensor([1, 2, 3]), offset: 4)
      #Nx.Tensor<
        s64[7][7]
        [
          [0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 2, 0],
          [0, 0, 0, 0, 0, 0, 3],
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0]
        ]
      >

  ## Options

    * `:offset` - offset used for making the diagonal.
      Use offset > 0 for diagonals above the main diagonal,
      and offset < 0 for diagonals below the main diagonal.
      Defaults to 0.

  ## Error cases

      iex> Nx.make_diagonal(Nx.tensor([[0, 0], [0, 1]]))
      ** (ArgumentError) make_diagonal/2 expects tensor of rank 1, got tensor of rank: 2
  """
  @doc type: :creation
  def make_diagonal(tensor, opts \\ []) do
    tensor = to_tensor(tensor)

    opts = keyword!(opts, offset: 0)

    {len} = Nx.Shape.make_diagonal(tensor.shape)
    offset = opts[:offset]

    diag_len = len + Kernel.abs(offset)
    diag_shape = {diag_len, diag_len}

    0
    |> Nx.broadcast(diag_shape)
    |> Nx.indexed_add(diag_indices(diag_shape, offset), tensor)
  end

  # Returns the indices of the diagonal of a tensor of the given shape
  defp diag_indices(shape, offset) do
    {len, breadth} = shape

    indices =
      case offset do
        i when i >= 0 ->
          Enum.zip_with(0..(len - 1), i..(breadth - 1), fn x, y -> [x, y] end)

        i when i < 0 ->
          Enum.zip_with(-i..(len - 1), 0..(breadth - 1), fn x, y -> [x, y] end)
      end

    Nx.tensor(indices)
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

  The atom notation for types is also supported:

      iex> Nx.from_binary(<<12.3::float-64-native>>, :f64)
      #Nx.Tensor<
        f64[1]
        [12.3]
      >

  An error is raised for incompatible sizes:

      iex> Nx.from_binary(<<1, 2, 3, 4>>, {:f, 64})
      ** (ArgumentError) binary does not match the given size

  ## Options

    * `:backend` - the backend to allocate the tensor on. It is either
      an atom or a tuple in the shape `{backend, options}`. This option
      is ignored inside `defn`
  """
  @doc type: :creation
  def from_binary(binary, type, opts \\ []) when is_binary(binary) do
    opts = keyword!(opts, [:backend])
    {_, size} = type = Nx.Type.normalize!(type)
    dim = div(bit_size(binary), size)

    if binary == "" do
      raise ArgumentError, "cannot build an empty tensor"
    end

    if rem(bit_size(binary), size) != 0 do
      raise ArgumentError, "binary does not match the given size"
    end

    {backend, backend_options} = backend_from_options!(opts) || default_backend()
    backend.from_binary(%T{type: type, shape: {dim}, names: [nil]}, binary, backend_options)
  end

  ## Conversions

  @doc """
  Returns the underlying tensor as a binary.

  **Warning**: converting a tensor to a binary can
  potentially be a very expensive operation, as it
  may copy a GPU tensor fully to the machine memory.

  It returns the in-memory binary representation of
  the tensor in a row-major fashion. The binary is
  in the system endianness, which has to be taken into
  account if the binary is meant to be serialized to
  other systems.

  ## Options

    * `:limit` - limit the number of entries represented in the binary

  ## Examples

      iex> Nx.to_binary(1)
      <<1::64-native>>

      iex> Nx.to_binary(Nx.tensor([1.0, 2.0, 3.0]))
      <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>

      iex> Nx.to_binary(Nx.tensor([1.0, 2.0, 3.0]), limit: 2)
      <<1.0::float-32-native, 2.0::float-32-native>>

  """
  @doc type: :conversion
  def to_binary(tensor, opts \\ []) do
    opts = keyword!(opts, [:limit])
    tensor = to_tensor(tensor)
    limit = if limit = opts[:limit], do: Kernel.min(size(tensor), limit), else: size(tensor)
    impl!(tensor).to_binary(tensor, limit)
  end

  @doc """
  Converts the given number (or tensor) to a tensor.

  This function exists for data normalization. If your
  goal is to create tensors from lists, see `tensor/2`.
  If you want to create a tensor from binary, see
  `from_binary/3`.
  """
  @doc type: :conversion
  def to_tensor(%T{} = t),
    do: t

  def to_tensor(number) when is_number(number) do
    {backend, options} = default_backend()
    type = Nx.Type.infer(number)
    out = %T{shape: {}, type: type, names: []}
    backend.constant(out, number, options)
  end

  def to_tensor(%Complex{re: re, im: im} = number) do
    {backend, options} = default_backend()
    {_, size} = re |> Nx.Type.infer() |> Nx.Type.merge(Nx.Type.infer(im))
    out = %T{shape: {}, type: {:c, size * 2}, names: []}
    backend.constant(out, number, options)
  end

  def to_tensor(t) do
    raise ArgumentError, "expected a %Nx.Tensor{} or a number, got: #{inspect(t)}"
  end

  @doc """
  Returns the underlying tensor as a flat list.

  Negative infinity, infinity, and NaN will be respectively returned
  as the atoms `:neg_infinity`, `:infinity`, and `:nan`.

  ## Examples

      iex> Nx.to_flat_list(1)
      [1]

      iex> Nx.to_flat_list(Nx.tensor([1.0, 2.0, 3.0]))
      [1.0, 2.0, 3.0]

      iex> Nx.to_flat_list(Nx.tensor([1.0, 2.0, 3.0]), limit: 2)
      [1.0, 2.0]

  Non-finite numbers are returned as atoms:

      iex> t = Nx.tensor([:neg_infinity, :nan, :infinity])
      iex> Nx.to_flat_list(t)
      [:neg_infinity, :nan, :infinity]

  """
  @doc type: :conversion
  def to_flat_list(tensor, opts \\ []) do
    opts = keyword!(opts, [:limit])
    %{type: {_, size} = type} = tensor = to_tensor(tensor)

    for <<part::size(size)-bitstring <- to_binary(tensor, Keyword.take(opts, [:limit]))>> do
      match_types [type] do
        <<match!(var, 0)>> = part
        read!(var, 0)
      end
    end
  end

  @doc """
  Converts the underlying tensor to a list of tensors.

  The first dimension (axis 0) is divided by `batch_size`.
  In case the dimension cannot be evenly divided by
  `batch_size`, you may specify what to do with leftover
  data using `:leftover`. `:leftover` must be one of `:repeat`
  or `:discard`. `:repeat` repeats the first `n` values to
  make the last batch match the desired batch size. `:discard`
  discards excess elements.

  ## Examples

      iex> [first, second] = Nx.to_batched_list(Nx.iota({2, 2, 2}), 1)
      iex> first
      #Nx.Tensor<
        s64[1][2][2]
        [
          [
            [0, 1],
            [2, 3]
          ]
        ]
      >
      iex> second
      #Nx.Tensor<
        s64[1][2][2]
        [
          [
            [4, 5],
            [6, 7]
          ]
        ]
      >

      iex> [first, second, third] = Nx.to_batched_list(Nx.iota({6, 2}, names: [:x, :y]), 2)
      iex> first
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [0, 1],
          [2, 3]
        ]
      >
      iex> second
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [4, 5],
          [6, 7]
        ]
      >
      iex> third
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [8, 9],
          [10, 11]
        ]
      >

  If the batch size would result in uneven batches, you can repeat or discard excess data:

      iex> [first, second, third] = Nx.to_batched_list(Nx.iota({5, 2}, names: [:x, :y]), 2)
      iex> first
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [0, 1],
          [2, 3]
        ]
      >
      iex> second
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [4, 5],
          [6, 7]
        ]
      >
      iex> third
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [8, 9],
          [0, 1]
        ]
      >

      iex> [first, second] = Nx.to_batched_list(Nx.iota({5, 2}, names: [:x, :y]), 2, leftover: :discard)
      iex> first
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [0, 1],
          [2, 3]
        ]
      >
      iex> second
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [4, 5],
          [6, 7]
        ]
      >
  """
  @doc type: :conversion
  def to_batched_list(tensor, batch_size, opts \\ [])
      when is_integer(batch_size) and batch_size >= 1 do
    opts = keyword!(opts, leftover: :repeat)

    %{shape: shape} = to_tensor(tensor)

    if shape == {} do
      raise ArgumentError, "cannot batch scalar tensor #{inspect(tensor)}"
    end

    if elem(shape, 0) < batch_size do
      raise ArgumentError, "cannot batch beyond original tensor"
    end

    impl!(tensor).to_batched_list(%{tensor | shape: put_elem(shape, 0, batch_size)}, tensor, opts)
  end

  @doc """
  Returns the underlying tensor as a number.

  If the tensor has a dimension, it raises.

  ## Examples

      iex> Nx.to_number(1)
      1

      iex> Nx.to_number(Nx.tensor([1.0, 2.0, 3.0]))
      ** (ArgumentError) cannot convert tensor of shape {3} to number

  """
  @doc type: :conversion
  def to_number(tensor)

  def to_number(number) when is_number(number), do: number

  def to_number(tensor) do
    tensor = to_tensor(tensor)

    if tensor.shape != {} do
      raise ArgumentError, "cannot convert tensor of shape #{inspect(tensor.shape)} to number"
    end

    match_types [tensor.type] do
      <<match!(x, 0)>> = to_binary(tensor)
      read!(x, 0)
    end
  end

  @doc ~S"""
  Returns a heatmap struct with the tensor data.

  On terminals, coloring is done via ANSI colors. If ANSI
  is not enabled, the tensor is normalized to show numbers
  between 0 and 9.

  ## Terminal coloring

  Coloring is enabled by default on most Unix terminals.
  It is also available on Windows consoles from Windows
  10, although it must be explicitly enabled for the current
  user in the registry by running the following command:

      reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1

  After running the command above, you must restart your current
  console.

  ## Options

    * `:ansi_enabled` - forces ansi to be enabled or disabled.
      Defaults to `IO.ANSI.enabled?/0`

    * `:ansi_whitespace` - which whitespace character to use when
      printing. By default it uses `"\u3000"`, which is a full-width
      whitespace which often prints more precise shapes

  """
  @doc type: :conversion
  def to_heatmap(tensor, opts \\ []) when is_list(opts) do
    tensor = to_tensor(tensor)

    if tensor.shape == {} do
      raise ArgumentError, "cannot show heatmap for scalar tensors, got: #{inspect(tensor)}"
    end

    %Nx.Heatmap{tensor: to_tensor(tensor), opts: opts}
  end

  ## Reflection operations (do not invoke the backend)

  @doc """
  Changes the type of a tensor.

  Note conversion between float and integers truncates the
  result. Consider using `round/1`, `floor/1`, or `ceil/1`
  before casting from float to integer to guarantee consistent
  behavior.

  Casting from a higher precision may lead to an overflow
  or underflow, which is platform and compiler dependent
  behaviour.

  Casting of non-finite types to integer types are handled
  such as:

    * negative infinity becomes the minimum value for said type
    * positive infinity becomes the maximum value for said type
    * nan becomes zero

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

      iex> Nx.as_type(Nx.tensor([0.0, 1.0, 2.0], names: [:data]), {:s, 64})
      #Nx.Tensor<
        s64[data: 3]
        [0, 1, 2]
      >

  Casting numbers as complex will return the corresponding complex with 0 imaginary component:

      iex> Nx.as_type(Nx.tensor([1, -2]), {:c, 64})
      #Nx.Tensor<
        c64[2]
        [1.0+0.0i, -2.0+0.0i]
      >

  Casting complex numbers will return their real parts as the target type:

      iex> Nx.as_type(Nx.tensor([Complex.new(1, 2), Complex.new(0, 3), Complex.new(4, 5)]), {:f, 64})
      #Nx.Tensor<
        f64[3]
        [1.0, 0.0, 4.0]
      >

      iex> Nx.as_type(Nx.tensor([Complex.new(-1, 2), Complex.new(-2, 3), Complex.new(3, -4)]), {:s, 64})
      #Nx.Tensor<
        s64[3]
        [-1, -2, 3]
      >

  Casting of non-finite values to integer types convert to pre-determined
  integer values:

      iex> non_finite = Nx.tensor([:infinity, :nan, :neg_infinity])
      iex> Nx.as_type(non_finite, {:u, 8})
      #Nx.Tensor<
        u8[3]
        [255, 0, 0]
      >
      iex> Nx.as_type(non_finite, {:s, 32})
      #Nx.Tensor<
        s32[3]
        [2147483647, 0, -2147483648]
      >

  Non-finite values between float types are preserved:

      iex> non_finite = Nx.tensor([:infinity, :nan])
      iex> Nx.as_type(non_finite, {:f, 64})
      #Nx.Tensor<
        f64[2]
        [Inf, NaN]
      >
      iex> Nx.as_type(non_finite, {:f, 16})
      #Nx.Tensor<
        f16[2]
        [Inf, NaN]
      >

  """
  @doc type: :type
  def as_type(tensor, type) do
    tensor = to_tensor(tensor)
    new_type = Nx.Type.normalize!(type)

    cond do
      tensor.type == new_type ->
        tensor

      true ->
        impl!(tensor).as_type(%{tensor | type: new_type}, tensor)
    end
  end

  @doc """
  Changes the type of a tensor, using a bitcast.

  The width of input tensor's type must match the width
  of the output type. `bitcast/1` does not change the
  underlying tensor data, but instead changes how
  the tensor data is viewed.

  Machines with different floating-point representations
  will give different results.

  For complex numbers, the last axis will change in size
  depending on whether you are upcasting or downcasting.

  ## Examples

      iex> t = Nx.bitcast(Nx.tensor([0, 0, 0], names: [:data], type: {:s, 32}), {:f, 32})
      #Nx.Tensor<
        f32[data: 3]
        [0.0, 0.0, 0.0]
      >
      iex> Nx.bitcast(t, {:s, 32})
      #Nx.Tensor<
        s32[data: 3]
        [0, 0, 0]
      >

  ### Error cases

      iex> Nx.bitcast(Nx.tensor([0, 1, 2], names: [:data], type: {:s, 16}), {:f, 32})
      ** (ArgumentError) input type width must match new type width, got input type {:s, 16} and output type {:f, 32}

      iex> Nx.bitcast(Nx.tensor([0], type: {:c, 64}), {:s, 64})
      ** (ArgumentError) Nx.bitcast/2 does not support complex inputs

      iex> Nx.bitcast(Nx.tensor([0], type: {:s, 64}), {:c, 64})
      ** (ArgumentError) Nx.bitcast/2 does not support complex inputs
  """
  @doc type: :type
  def bitcast(tensor, type) do
    %T{type: {_, bits} = input_type} = tensor = to_tensor(tensor)
    {_, new_bits} = new_type = Nx.Type.normalize!(type)

    Nx.Shared.raise_complex_not_supported(input_type, :bitcast, 2)
    Nx.Shared.raise_complex_not_supported(new_type, :bitcast, 2)

    unless new_bits == bits do
      raise ArgumentError,
            "input type width must match new type width," <>
              " got input type #{inspect(input_type)} and" <>
              " output type #{inspect(type)}"
    end

    impl!(tensor).bitcast(%{tensor | type: new_type}, tensor)
  end

  @doc """
  Changes the shape of a tensor.

  The new shape is either a tuple or a tensor which we will
  retrieve the current shape from. The shapes must be compatible:
  the product of each dimension in the shape must be equal.

  You may specify one of the dimensions as `:auto`. Nx will compute
  the size of the dimension based on the original shape and new shape.

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

  You can use `:auto` to infer dimension sizes. This is useful when you
  don't know the size some dimension should be ahead of time:

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.reshape(t, {:auto, 2}, names: [:x, :y])
      #Nx.Tensor<
        s64[x: 3][y: 2]
        [
          [1, 2],
          [3, 4],
          [5, 6]
        ]
      >
  """
  @doc type: :shape
  def reshape(tensor, new_shape, opts \\ []) do
    %T{shape: old_shape} = tensor = to_tensor(tensor)
    new_names = opts[:names] || names!(new_shape)
    new_shape = if is_tuple(new_shape), do: new_shape, else: shape(new_shape)
    new_shape = Nx.Shape.reshape(old_shape, new_shape)

    names = Nx.Shape.named_axes!(new_names, new_shape)

    if old_shape == new_shape do
      %{tensor | names: names}
    else
      impl!(tensor).reshape(%{tensor | shape: new_shape, names: names}, tensor)
    end
  end

  @doc """
  Flattens a n-dimensional tensor to a 1-dimensional tensor.

  Flattening only changes the tensor metadata, it doesn't
  copy the underlying structure.

  Flatten is a destructive operation with respect to names.

  ## Examples

      iex> t = Nx.iota({2, 2, 2, 2})
      #Nx.Tensor<
        s64[2][2][2][2]
        [
          [
            [
              [0, 1],
              [2, 3]
            ],
            [
              [4, 5],
              [6, 7]
            ]
          ],
          [
            [
              [8, 9],
              [10, 11]
            ],
            [
              [12, 13],
              [14, 15]
            ]
          ]
        ]
      >
      iex> Nx.flatten(t)
      #Nx.Tensor<
        s64[16]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      >

  And if the tensor is already 1-dimensional:

      iex> t = Nx.iota({16})
      #Nx.Tensor<
        s64[16]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      >
      iex> Nx.flatten(t)
      #Nx.Tensor<
        s64[16]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      >
  """
  @doc type: :shape
  def flatten(tensor) do
    reshape(tensor, {size(tensor)})
  end

  @doc """
  Creates a new tensor by repeating the input tensor
  along the given axes.

  If the `tensor` has less dimensions than the repetitions given,
  the tensor will grow in dimensionality.

  If the `tensor` has more dimensions than the repetitions given,
  tiling is done from the rightmost dimensions (i.e. if the input
  shape is `{1,2,3}` and `repetitions = [2]`, the result is the same
  as if `repetitions = [1,1,2]`).

  ## Examples

      iex> a = Nx.tensor([0, 1, 2])
      iex> Nx.tile(a, [2])
      #Nx.Tensor<
        s64[6]
        [0, 1, 2, 0, 1, 2]
      >
      iex> Nx.tile(a, [1, 2])
      #Nx.Tensor<
        s64[1][6]
        [
          [0, 1, 2, 0, 1, 2]
        ]
      >
      iex> Nx.tile(a, [2, 2])
      #Nx.Tensor<
        s64[2][6]
        [
          [0, 1, 2, 0, 1, 2],
          [0, 1, 2, 0, 1, 2]
        ]
      >
      iex> Nx.tile(a, [2, 1])
      #Nx.Tensor<
        s64[2][3]
        [
          [0, 1, 2],
          [0, 1, 2]
        ]
      >
      iex> Nx.tile(a, [2, 1, 2])
      #Nx.Tensor<
        s64[2][1][6]
        [
          [
            [0, 1, 2, 0, 1, 2]
          ],
          [
            [0, 1, 2, 0, 1, 2]
          ]
        ]
      >

      iex> b = Nx.tensor([[1,2],[3,4]])
      iex> Nx.tile(b, [2])
      #Nx.Tensor<
        s64[2][4]
        [
          [1, 2, 1, 2],
          [3, 4, 3, 4]
        ]
      >
      iex> Nx.tile(b, [2, 1])
      #Nx.Tensor<
        s64[4][2]
        [
          [1, 2],
          [3, 4],
          [1, 2],
          [3, 4]
        ]
      >
      iex> Nx.tile(b, [1, 2])
      #Nx.Tensor<
        s64[2][4]
        [
          [1, 2, 1, 2],
          [3, 4, 3, 4]
        ]
      >

      iex> c = Nx.tensor([1,2,3,4])
      iex> Nx.tile(c, [4,1])
      #Nx.Tensor<
        s64[4][4]
        [
          [1, 2, 3, 4],
          [1, 2, 3, 4],
          [1, 2, 3, 4],
          [1, 2, 3, 4]
        ]
      >

  ### Error cases

      iex> Nx.tile(Nx.tensor([1,2]), 1.0)
      ** (ArgumentError) repetitions must be a list of integers, got: 1.0

      iex> Nx.tile(Nx.tensor([1,2]), [1, 1.0])
      ** (ArgumentError) repetitions must be a list of integers, got: [1, 1.0]

      iex> Nx.tile(Nx.tensor([1,2]), nil)
      ** (ArgumentError) repetitions must be a list of integers, got: nil
  """
  @doc type: :shape, from_backend: false
  def tile(tensor, repetitions) do
    tensor = to_tensor(tensor)

    unless tile_valid_repetitions?(repetitions) do
      raise ArgumentError,
            "repetitions must be a list of integers, got: #{inspect(repetitions)}"
    end

    {tensor_reshape, broadcast_shape, result_shape} = Nx.Shape.tile(tensor, repetitions)

    tensor
    |> reshape(tensor_reshape)
    |> broadcast(broadcast_shape)
    |> reshape(result_shape)
  end

  defp tile_valid_repetitions?(reps) when not is_list(reps), do: false

  defp tile_valid_repetitions?(reps) do
    Enum.all?(reps, &(is_integer(&1) and &1 >= 1))
  end

  @doc """
  Adds a new `axis` of size 1 with optional `name`.

  ## Examples

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.new_axis(t, 0, :new)
      #Nx.Tensor<
        s64[new: 1][2][3]
        [
          [
            [1, 2, 3],
            [4, 5, 6]
          ]
        ]
      >
      iex> Nx.new_axis(t, 1, :new)
      #Nx.Tensor<
        s64[2][new: 1][3]
        [
          [
            [1, 2, 3]
          ],
          [
            [4, 5, 6]
          ]
        ]
      >
      iex> Nx.new_axis(t, 2, :new)
      #Nx.Tensor<
        s64[2][3][new: 1]
        [
          [
            [1],
            [2],
            [3]
          ],
          [
            [4],
            [5],
            [6]
          ]
        ]
      >

  Axis can also be negative, which will start from the back:

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.new_axis(t, -1, :new)
      #Nx.Tensor<
        s64[2][3][new: 1]
        [
          [
            [1],
            [2],
            [3]
          ],
          [
            [4],
            [5],
            [6]
          ]
        ]
      >

  """
  @doc type: :shape, from_backend: false
  def new_axis(tensor, axis, name \\ nil) when is_integer(axis) do
    %{shape: shape, names: names} = tensor = to_tensor(tensor)
    rank = tuple_size(shape)
    norm = if axis < 0, do: axis + rank + 1, else: axis

    if norm not in 0..tuple_size(shape) do
      raise ArgumentError,
            "new axis position for shape #{inspect(shape)} must be " <>
              "a number between #{-rank - 1} and #{rank}, got: #{axis}"
    end

    new_shape = Tuple.insert_at(shape, norm, 1)
    new_names = List.insert_at(names, norm, name)
    impl!(tensor).reshape(%{tensor | shape: new_shape, names: new_names}, tensor)
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
  @doc type: :shape
  def squeeze(tensor, opts \\ []) do
    opts = keyword!(opts, [:axes])
    %T{shape: old_shape, names: names} = tensor = to_tensor(tensor)
    axes = opts[:axes] || Nx.Shape.squeeze_axes(old_shape)
    axes = Nx.Shape.normalize_axes(old_shape, axes, names)
    {new_shape, new_names} = Nx.Shape.squeeze(old_shape, axes, names)

    if old_shape == new_shape do
      tensor
    else
      impl!(tensor).squeeze(%{tensor | shape: new_shape, names: new_names}, tensor, axes)
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
  shape is the same, names are discarded if none are given:

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
  @doc type: :shape
  def broadcast(tensor, shape, opts \\ []) do
    opts = keyword!(opts, [:axes, :names])

    tensor = to_tensor(tensor)
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
  `{pad_width_low, pad_width_high, pad_width_interior}`
  for each dimension in the input tensor. The padding
  configuration must be of the same length as the tensor shape.

  Padding widths can be negative. If they are negative,
  the tensor is clipped on either end according to the
  padding width. Interior padding widths cannot be negative.

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
        f32[5][3][3]
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

      iex> tensor = Nx.tensor([[0, 1, 2, 3], [0, 4, 5, 6]])
      iex> Nx.pad(tensor, 0, [{0, 0, 0}, {-1, 1, 0}])
      #Nx.Tensor<
        s64[2][4]
        [
          [1, 2, 3, 0],
          [4, 5, 6, 0]
        ]
      >

      iex> tensor = Nx.tensor([[0, 1, 2], [3, 4, 5]], type: {:f, 32})
      iex> Nx.pad(tensor, 0, [{-1, 2, 0}, {1, -1, 0}])
      #Nx.Tensor<
        f32[3][3]
        [
          [0.0, 3.0, 4.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ]
      >

  """
  @doc type: :shape
  def pad(tensor, pad_value, padding_config) when is_list(padding_config) do
    output_type = binary_type(tensor, pad_value)
    tensor = to_tensor(tensor)
    pad_value = to_tensor(pad_value)

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
      {:f, 32}
  """
  @doc type: :type
  def type(tensor) do
    %T{type: type} = to_tensor(tensor)
    type
  end

  @doc """
  Checks if two tensors have the same shape, type, and compatible names.

  The data in the tensor is ignored.

  For convenience, this function accepts tensors and any container
  (such as maps and tuples as defined by the `Nx.Container` protocol)
  and recursively compares them, observing their container data
  structures are also the same.

  ## Examples

      iex> Nx.compatible?(Nx.iota({3, 2}), Nx.iota({3, 2}))
      true

      iex> Nx.compatible?(Nx.iota({3, 2}), Nx.iota({3, 2}, names: [:rows, :columns]))
      true

      iex> Nx.compatible?(
      ...>   Nx.iota({3, 2}, names: [:rows, nil]),
      ...>   Nx.iota({3, 2}, names: [nil, :columns])
      ...> )
      true

      iex> Nx.compatible?(
      ...>   Nx.iota({3, 2}, names: [:foo, :bar]),
      ...>   Nx.iota({3, 2}, names: [:rows, :columns])
      ...> )
      false

      iex> Nx.compatible?(Nx.iota({3, 2}), Nx.iota({2, 3}))
      false

      iex> Nx.compatible?(Nx.iota({2, 2}), Nx.iota({2, 2}, type: {:f, 32}))
      false

  Using collections:

      iex> Nx.compatible?({Nx.iota({3, 2}), {1, 2}}, {Nx.iota({3, 2}), {3, 4}})
      true

      iex> Nx.compatible?(%{foo: Nx.iota({3, 2})}, %{foo: Nx.iota({3, 2})})
      true

      iex> Nx.compatible?(%{foo: Nx.iota({3, 2})}, %{bar: Nx.iota({3, 2})})
      false

  """
  @doc type: :shape
  def compatible?(left, right)

  def compatible?(%T{} = left, %T{} = right) do
    %{type: type, shape: shape, names: left_names} = left

    case to_tensor(right) do
      %{type: ^type, shape: ^shape, names: right_names} ->
        compatible_names?(left_names, right_names)

      %{} ->
        false
    end
  end

  def compatible?(left, right) when is_number(left), do: compatible?(to_tensor(left), right)
  def compatible?(left, right) when is_number(right), do: compatible?(left, to_tensor(right))
  def compatible?(left, right), do: Nx.Defn.Composite.compatible?(left, right, &compatible?/2)

  defp compatible_names?([name | lnames], [name | rnames]), do: compatible_names?(lnames, rnames)
  defp compatible_names?([nil | lnames], [_ | rnames]), do: compatible_names?(lnames, rnames)
  defp compatible_names?([_ | lnames], [nil | rnames]), do: compatible_names?(lnames, rnames)
  defp compatible_names?([], []), do: true
  defp compatible_names?(_, _), do: false

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
  @doc type: :shape
  def shape(%T{shape: shape}), do: shape
  def shape(number) when is_number(number), do: {}
  def shape(shape) when is_tuple(shape), do: Nx.Shape.validate!(shape, :shape)

  def shape(other) do
    raise ArgumentError,
          "expected a shape. A shape is a n-element tuple with the size of each dimension. " <>
            "Alternatively, you can pass a tensor (or a number) and the shape will be retrieved from the tensor. " <>
            "Got: #{inspect(other)}"
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
  @doc type: :shape
  def rank(shape) when is_tuple(shape), do: tuple_size(shape)
  def rank(tensor), do: tuple_size(shape(tensor))

  @doc """
  Returns the size of a given axis of a tensor.

  It accepts either an atom as the name or an integer as the axis.
  It raises if the axis/name does not exist.

  ### Examples

      iex> Nx.axis_size(Nx.iota({100, 10, 20}), 0)
      100

      iex> Nx.axis_size(Nx.iota({100, 10, 20}, names: [:batch, :x, :y]), :y)
      20

  """
  @doc type: :shape
  def axis_size(tensor, axis) do
    shape = shape(tensor)
    index = Nx.Shape.normalize_axis(shape, axis, names(tensor))
    elem(shape, index)
  end

  @doc """
  Returns the index of the given axis in the tensor.

  ### Examples

      iex> Nx.axis_index(Nx.iota({100, 10, 20}), 0)
      0

      iex> Nx.axis_index(Nx.iota({100, 10, 20}), -1)
      2

      iex> Nx.axis_index(Nx.iota({100, 10, 20}, names: [:batch, :x, :y]), :x)
      1

  ### Error cases

      iex> Nx.axis_index(Nx.iota({100, 10, 20}), 3)
      ** (ArgumentError) given axis (3) invalid for shape with rank 3

      iex> Nx.axis_index(Nx.iota({100, 10, 20}, names: [:batch, :x, :y]), :z)
      ** (ArgumentError) key :z not found in tensor with names [:batch, :x, :y]

  """
  @doc type: :shape
  def axis_index(tensor, axis) do
    shape = shape(tensor)
    Nx.Shape.normalize_axis(shape, axis, names(tensor))
  end

  @doc """
  Returns the number of elements in the tensor.

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
  @doc type: :shape
  def size(shape) when is_tuple(shape), do: Tuple.product(shape)
  def size(tensor), do: size(shape(tensor))

  @doc """
  Returns the byte size of the data in the tensor
  computed from its shape and type.

  ### Examples

      iex> Nx.byte_size(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
      48
      iex> Nx.byte_size(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
      24
      iex> Nx.byte_size(Nx.tensor([[1, 2, 3], [4, 5, 6]], type: {:u, 8}))
      6
      iex> Nx.byte_size(1)
      8

  """
  @doc type: :shape
  def byte_size(tensor) do
    %{type: {_, bit_size}, shape: shape} = to_tensor(tensor)
    size(shape) * div(bit_size, 8)
  end

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
  @doc type: :shape
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
  @doc type: :shape
  def names(%T{names: names}), do: names
  def names(a) when is_number(a), do: []

  defp count_up(0, _n), do: []
  defp count_up(i, n), do: [n | count_up(i - 1, n + 1)]

  ## Backend API

  @backend_key {Nx, :default_backend}

  @doc """
  Sets the current process default backend to `backend` with the given `opts`.

  The default backend is stored only in the process dictionary.
  This means if you start a separate process, such as `Task`,
  the default backend must be set on the new process too.

  This function is mostly used for scripting and testing. In your
  applications, you typically set the backend in your config files:

      config :nx, :default_backend, {Lib.CustomBackend, device: :cuda}

  ## Examples

      iex> Nx.default_backend({Lib.CustomBackend, device: :cuda})
      {Nx.BinaryBackend, []}
      iex> Nx.default_backend()
      {Lib.CustomBackend, device: :cuda}

  """
  @doc type: :backend
  def default_backend(backend) do
    Process.put(@backend_key, backend!(backend)) ||
      backend!(Application.fetch_env!(:nx, :default_backend))
  end

  @doc """
  Sets the default backend globally.

  You must avoid calling this function at runtime. It is mostly
  useful during scripts or code notebooks to set a default.
  If you need to configure a global default backend in your
  applications, you can do so in your `config/*.exs` files:

      config :nx, :default_backend, {Lib.CustomBackend, []}

  """
  @doc type: :backend
  def global_default_backend(backend) do
    current = backend!(Application.fetch_env!(:nx, :default_backend))
    Application.put_env(:nx, :default_backend, backend!(backend))
    current
  end

  @doc """
  Gets the default backend for the current process.
  """
  @doc type: :backend
  def default_backend() do
    Process.get(@backend_key) || backend!(Application.fetch_env!(:nx, :default_backend))
  end

  @doc """
  Copies data to the given backend.

  If a backend is not given, `Nx.Tensor` is used, which means
  the given tensor backend will pick the most appropriate
  backend to copy the data to.

  Note this function keeps the data in the original backend.
  Therefore, use this function with care, as it may duplicate
  large amounts of data across backends. Generally speaking,
  you may want to use `backend_transfer/2`, unless you explicitly
  want to copy the data.

  For convenience, this function accepts tensors and any container
  (such as maps and tuples as defined by the `Nx.Container` protocol)
  and recursively copies all tensors in them. This behaviour exists
  as it is common to transfer data before and after `defn` functions.

  *Note: `Nx.default_backend/1` does not affect the behaviour of
  this function.

  ### Examples

    iex> Nx.backend_copy(Nx.tensor([[1, 2, 3], [4, 5, 6]]))
    #Nx.Tensor<
      s64[2][3]
      [
        [1, 2, 3],
        [4, 5, 6]
      ]
    >
  """
  @doc type: :backend
  def backend_copy(tensor_or_container, backend \\ Nx.Tensor) do
    {backend, opts} = backend!(backend)

    Nx.Defn.Composite.traverse(tensor_or_container, fn tensor ->
      tensor = to_tensor(tensor)
      impl!(tensor).backend_copy(tensor, backend, opts)
    end)
  end

  @doc """
  Transfers data to the given backend.

  This operation can be seen as an equivalent to `backend_copy/3`
  followed by a `backend_deallocate/1` on the initial tensor:

      new_tensor = Nx.backend_copy(old_tensor, new_backend)
      Nx.backend_deallocate(old_tensor)

  If a backend is not given, `Nx.Tensor` is used, which means
  the given tensor backend will pick the most appropriate
  backend to transfer to.

  For Elixir's builtin tensor, transferring to another backend
  will call `new_backend.from_binary(tensor, binary, opts)`.
  Transferring from a mutable backend, such as GPU memory,
  implies the data is copied from the GPU to the Erlang VM
  and then deallocated from the device.

  For convenience, this function accepts tensors and any container
  (such as maps and tuples as defined by the `Nx.Container` protocol)
  and transfers all tensors in them. This behaviour exists as it is
  common to transfer data from tuples and maps before and after `defn`
  functions.

  *Note: `Nx.default_backend/1` does not affect the behaviour of
  this function.

  ## Examples

  Transfer a tensor to an EXLA device backend, stored in the GPU:

      device_tensor = Nx.backend_transfer(tensor, {EXLA.Backend, client: :cuda})

  Transfer the device tensor back to an Elixir tensor:

      tensor = Nx.backend_transfer(device_tensor)

  """
  @doc type: :backend
  def backend_transfer(tensor_or_container, backend \\ Nx.Tensor) do
    {backend, opts} = backend!(backend)

    Nx.Defn.Composite.traverse(tensor_or_container, fn tensor ->
      tensor = to_tensor(tensor)
      impl!(tensor).backend_transfer(tensor, backend, opts)
    end)
  end

  @doc """
  Deallocates data in a device.

  It returns either `:ok` or `:already_deallocated`.

  For convenience, this function accepts tensors and any container
  (such as maps and tuples as defined by the `Nx.Container` protocol)
  and deallocates all devices in them. This behaviour exists as it is
  common to deallocate data after `defn` functions.
  """
  @doc type: :backend
  def backend_deallocate(tensor_or_container) do
    Nx.Defn.Composite.reduce(tensor_or_container, :ok, fn tensor, :ok ->
      if is_number(tensor) do
        :ok
      else
        impl!(tensor).backend_deallocate(tensor)
      end
    end)
  end

  ## Element-wise binary ops

  defp non_complex_element_wise_bin_op(left, right, op, fun) do
    type = binary_type(left, right) |> fun.()
    Nx.Shared.raise_complex_not_supported(type, op, 2)
    element_wise_bin_op(left, right, op, fun)
  end

  defp element_wise_bin_op(left, right, op, fun) do
    type = binary_type(left, right) |> fun.()

    %T{shape: left_shape, names: left_names} = left = to_tensor(left)
    %T{shape: right_shape, names: right_names} = right = to_tensor(right)

    {shape, names} = Nx.Shape.binary_broadcast(left_shape, left_names, right_shape, right_names)

    apply(impl!(left, right), op, [%{left | type: type, shape: shape, names: names}, left, right])
  end

  defp non_complex_element_wise_pred_op(left, right, op) do
    Nx.Shared.raise_complex_not_supported(type(left), op, 2)
    Nx.Shared.raise_complex_not_supported(type(right), op, 2)
    element_wise_pred_op(left, right, op)
  end

  defp element_wise_pred_op(left, right, op) do
    %T{shape: left_shape, names: left_names} = left = to_tensor(left)
    %T{shape: right_shape, names: right_names} = right = to_tensor(right)

    {shape, names} = Nx.Shape.binary_broadcast(left_shape, left_names, right_shape, right_names)

    out = %{left | type: {:u, 8}, shape: shape, names: names}
    apply(impl!(left, right), op, [out, left, right])
  end

  @doc """
  Element-wise addition of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `+` operator
  in place of this function: `left + right`.

  ## Examples

  ### Adding scalars

      iex> Nx.add(1, 2)
      #Nx.Tensor<
        s64
        3
      >

      iex> Nx.add(1, 2.2)
      #Nx.Tensor<
        f32
        3.200000047683716
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
        f32[data: 3]
        [2.0, 3.0, 4.0]
      >

      iex> Nx.add(Nx.tensor([1.0, 2.0, 3.0], names: [:data]), 1)
      #Nx.Tensor<
        f32[data: 3]
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

      iex> left = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      iex> right = Nx.tensor([[10, 20], [30, 40]], names: [nil, :y])
      iex> Nx.add(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [11, 22],
          [33, 44]
        ]
      >

  ### Adding tensors with broadcasting

      iex> left = Nx.tensor([[1], [2]], names: [nil, :y])
      iex> right = Nx.tensor([[10, 20]], names: [:x, nil])
      iex> Nx.add(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [11, 21],
          [12, 22]
        ]
      >

      iex> left = Nx.tensor([[10, 20]], names: [:x, nil])
      iex> right = Nx.tensor([[1], [2]], names: [nil, :y])
      iex> Nx.add(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [11, 21],
          [12, 22]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], names: [:x, nil])
      iex> right = Nx.tensor([[10, 20], [30, 40]])
      iex> Nx.add(left, right)
      #Nx.Tensor<
        s64[x: 2][2]
        [
          [11, 21],
          [32, 42]
        ]
      >

      iex> left = Nx.tensor([[1, 2]])
      iex> right = Nx.tensor([[10, 20], [30, 40]])
      iex> Nx.add(left, right)
      #Nx.Tensor<
        s64[2][2]
        [
          [11, 22],
          [31, 42]
        ]
      >

  """
  @doc type: :element
  def add(left, right), do: element_wise_bin_op(left, right, :add, & &1)

  @doc """
  Element-wise subtraction of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `-` operator
  in place of this function: `left - right`.

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
        f32[data: 3]
        [0.0, -1.0, -2.0]
      >

  ### Subtracting tensors

      iex> left = Nx.tensor([[1], [2]], names: [:x, :y])
      iex> right = Nx.tensor([[10, 20]], names: [:x, :y])
      iex> Nx.subtract(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [-9, -19],
          [-8, -18]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]], type: {:s, 8}, names: [nil, :y])
      iex> Nx.subtract(left, right)
      #Nx.Tensor<
        s8[x: 2][y: 2]
        [
          [-9, -19],
          [-8, -18]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:f, 32}, names: [nil, :y])
      iex> right = Nx.tensor([[10, 20]], type: {:f, 32}, names: [:x, nil])
      iex> Nx.subtract(left, right)
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [-9.0, -19.0],
          [-8.0, -18.0]
        ]
      >

  """
  @doc type: :element
  def subtract(left, right), do: element_wise_bin_op(left, right, :subtract, & &1)

  @doc """
  Element-wise multiplication of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `*` operator
  operator in place of this function as `left * right`.

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
        f32[data: 3]
        [1.0, 2.0, 3.0]
      >

  ### Multiplying tensors

      iex> left = Nx.tensor([[1], [2]], names: [:x, :y])
      iex> right = Nx.tensor([[10, 20]], names: [:x, :y])
      iex> Nx.multiply(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [10, 20],
          [20, 40]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]], type: {:s, 8}, names: [nil, :y])
      iex> Nx.multiply(left, right)
      #Nx.Tensor<
        s8[x: 2][y: 2]
        [
          [10, 20],
          [20, 40]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:f, 32}, names: [nil, :y])
      iex> right = Nx.tensor([[10, 20]], type: {:f, 32}, names: [:x, nil])
      iex> Nx.multiply(left, right)
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [10.0, 20.0],
          [20.0, 40.0]
        ]
      >

  """
  @doc type: :element
  def multiply(left, right), do: element_wise_bin_op(left, right, :multiply, & &1)

  @doc """
  Element-wise power of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If both tensors are integers and the exponent is
  negative, it will raise, but it may trigger undefined
  behaviour on some compilers.

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
        f32[data: 3]
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
  @doc type: :element
  def power(left, right), do: element_wise_bin_op(left, right, :power, & &1)

  @doc """
  Element-wise remainder of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `rem/2` function
  in place of this function: `rem(left, right)`.

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
        f32[data: 3]
        [0.0, 0.0, 2.0]
      >

  ### Remainder of tensors

      iex> left = Nx.tensor([[10], [20]], names: [:x, :y])
      iex> right = Nx.tensor([[3, 4]], names: [nil, :y])
      iex> Nx.remainder(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [1, 2],
          [2, 0]
        ]
      >

  """
  @doc type: :element
  def remainder(left, right), do: non_complex_element_wise_bin_op(left, right, :remainder, & &1)

  @doc """
  Element-wise division of two tensors.

  If a number is given, it is converted to a tensor.

  It always returns a float tensor. If any of the input
  tensors are not float, they are converted to f32.
  Division by zero raises, but it may trigger undefined
  behaviour on some compilers.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `/` operator
  in place of this function: `left / right`.

  ## Examples

  ### Dividing scalars

      iex> Nx.divide(1, 2)
      #Nx.Tensor<
        f32
        0.5
      >

  ### Dividing tensors and scalars

      iex> Nx.divide(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        f32[data: 3]
        [1.0, 2.0, 3.0]
      >

      iex> Nx.divide(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 3]
        [1.0, 0.5, 0.3333333432674408]
      >

  ### Dividing tensors

      iex> left = Nx.tensor([[1], [2]], names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]], names: [nil, :y])
      iex> Nx.divide(left, right)
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:s, 8})
      iex> right = Nx.tensor([[10, 20]], type: {:s, 8}, names: [:x, :y])
      iex> Nx.divide(left, right)
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:f, 32}, names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]], type: {:f, 32}, names: [nil, :y])
      iex> Nx.divide(left, right)
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      >

  """
  @doc type: :element
  def divide(left, right), do: element_wise_bin_op(left, right, :divide, &Nx.Type.to_floating/1)

  defp assert_quotient_type!(type) do
    if Nx.Type.integer?(type) do
      type
    else
      raise ArgumentError,
            "quotient expects integer tensors as inputs and outputs an integer tensor, " <>
              "got: #{inspect(type)}"
    end
  end

  @doc """
  Element-wise integer division of two tensors.

  If a number is given, it is converted to a tensor.

  It always returns an integer tensor. Input tensors and
  numbers must be integer types. Division by zero raises,
  but it may trigger undefined behaviour on some compilers.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Caveat for `grad`

  The `grad` operation is not supported for `quotient/2`.
  Since integer division is, by definition, a closed operation
  for the set of integers and grad involves floating points,
  `grad` is undefined.

  If you need to support gradients, you might consider using
  floor division, but beware of precision errors caused by
  floating points:

      a |> Nx.divide(b) |> Nx.floor()

  ## Examples

  ### Integer dividing scalars

      iex> Nx.quotient(11, 2)
      #Nx.Tensor<
        s64
        5
      >

  ### Integer dividing tensors and scalars

      iex> Nx.quotient(Nx.tensor([2, 4, 5], names: [:data]), 2)
      #Nx.Tensor<
        s64[data: 3]
        [1, 2, 2]
      >

      iex> Nx.quotient(10, Nx.tensor([1, 2, 3], names: [:data]))
      #Nx.Tensor<
        s64[data: 3]
        [10, 5, 3]
      >

  ### Dividing tensors

      iex> left = Nx.tensor([[10, 20]], names: [nil, :y])
      iex> right = Nx.tensor([[1], [2]], names: [:x, nil])
      iex> Nx.quotient(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [10, 20],
          [5, 10]
        ]
      >

      iex> left = Nx.tensor([[10, 20]], type: {:s, 8}, names: [:x, :y])
      iex> right = Nx.tensor([[1], [2]], type: {:s, 8})
      iex> Nx.quotient(left, right)
      #Nx.Tensor<
        s8[x: 2][y: 2]
        [
          [10, 20],
          [5, 10]
        ]
      >

      iex> left = Nx.tensor([[10, 20]], type: {:u, 8}, names: [:x, :y])
      iex> right = Nx.tensor([[1], [2]], type: {:u, 32})
      iex> Nx.quotient(left, right)
      #Nx.Tensor<
        u32[x: 2][y: 2]
        [
          [10, 20],
          [5, 10]
        ]
      >

  """
  @doc type: :element
  def quotient(left, right),
    do: element_wise_bin_op(left, right, :quotient, &assert_quotient_type!/1)

  @doc """
  Element-wise arc tangent of two tensors.

  If a number is given, it is converted to a tensor.

  It always returns a float tensor. If any of the input
  tensors are not float, they are converted to f32.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  ## Examples

  ### Arc tangent between scalars

      iex> Nx.atan2(1, 2)
      #Nx.Tensor<
        f32
        0.46364760398864746
      >

  ### Arc tangent between tensors and scalars

      iex> Nx.atan2(Nx.tensor([1, 2, 3], names: [:data]), 1)
      #Nx.Tensor<
        f32[data: 3]
        [0.7853981852531433, 1.1071487665176392, 1.249045729637146]
      >

      iex> Nx.atan2(1, Nx.tensor([1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 3]
        [0.7853981852531433, 0.46364760398864746, 0.32175055146217346]
      >

  ### Arc tangent between tensors

      iex> neg_and_pos_zero_columns = Nx.tensor([[-0.0], [0.0]], type: {:f, 64})
      iex> neg_and_pos_zero_rows = Nx.tensor([-0.0, 0.0], type: {:f, 64})
      iex> Nx.atan2(neg_and_pos_zero_columns, neg_and_pos_zero_rows)
      #Nx.Tensor<
        f64[2][2]
        [
          [-3.141592653589793, -0.0],
          [3.141592653589793, 0.0]
        ]
      >

  """
  @doc type: :element
  def atan2(left, right), do: element_wise_bin_op(left, right, :atan2, &Nx.Type.to_floating/1)

  @doc """
  Element-wise maximum of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `max/2` function
  in place of this function: `max(left, right)`.

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
        f32[data: 3]
        [1.0, 2.0, 3.0]
      >

  ### Max between tensors

      iex> left = Nx.tensor([[1], [2]], names: [:x, :y])
      iex> right = Nx.tensor([[10, 20]])
      iex> Nx.max(left, right)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [10, 20],
          [10, 20]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]], type: {:s, 8})
      iex> Nx.max(left, right)
      #Nx.Tensor<
        s8[x: 2][2]
        [
          [10, 20],
          [10, 20]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:f, 32}, names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]], type: {:f, 32}, names: [nil, :y])
      iex> Nx.max(left, right)
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [10.0, 20.0],
          [10.0, 20.0]
        ]
      >

  """
  @doc type: :element
  def max(left, right), do: non_complex_element_wise_bin_op(left, right, :max, & &1)

  @doc """
  Element-wise minimum of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `min/2` function
  in place of this function: `min(left, right)`.

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
        f32[data: 3]
        [1.0, 1.0, 1.0]
      >

  ### Min between tensors

      iex> left = Nx.tensor([[1], [2]], names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]])
      iex> Nx.min(left, right)
      #Nx.Tensor<
        s64[x: 2][2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:s, 8}, names: [:x, :y])
      iex> right = Nx.tensor([[10, 20]], type: {:s, 8})
      iex> Nx.min(left, right)
      #Nx.Tensor<
        s8[x: 2][y: 2]
        [
          [1, 1],
          [2, 2]
        ]
      >

      iex> left = Nx.tensor([[1], [2]], type: {:f, 32}, names: [:x, nil])
      iex> right = Nx.tensor([[10, 20]], type: {:f, 32}, names: [nil, :y])
      iex> Nx.min(left, right)
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [1.0, 1.0],
          [2.0, 2.0]
        ]
      >

  """
  @doc type: :element
  def min(left, right), do: non_complex_element_wise_bin_op(left, right, :min, & &1)

  ## Bitwise ops

  defp assert_bitwise_type!(type) do
    if Nx.Type.integer?(type) do
      type
    else
      raise ArgumentError,
            "bitwise operators expect integer tensors as inputs and outputs an integer tensor, " <>
              "got: #{inspect(type)}"
    end
  end

  @doc """
  Element-wise bitwise AND of two tensors.

  Only integer tensors are supported. If a float or
  complex tensor is given, an error is raised.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `&&&` operator
  in place of this function: `left &&& right`.

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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}
  """
  @doc type: :element
  def bitwise_and(left, right),
    do: element_wise_bin_op(left, right, :bitwise_and, &assert_bitwise_type!/1)

  @doc """
  Element-wise bitwise OR of two tensors.

  Only integer tensors are supported. If a float or
  complex tensor is given, an error is raised.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `|||` operator
  in place of this function: `left ||| right`.

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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}
  """
  @doc type: :element
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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}
  """
  @doc type: :element
  def bitwise_xor(left, right),
    do: element_wise_bin_op(left, right, :bitwise_xor, &assert_bitwise_type!/1)

  @doc """
  Element-wise left shift of two tensors.

  Only integer tensors are supported. If a float or complex
  tensor is given, an error is raised. If the right side
  is negative, it will raise, but it may trigger undefined
  behaviour on some compilers.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible. If the number of
  shifts are negative, Nx's default backend will raise,
  but it may trigger undefined behaviour in other backends.

  If you're using `Nx.Defn.defn/2`, you can use the `<<<` operator
  in place of this function: `left <<< right`.

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

      iex> left = Nx.tensor([1, 1, -1, -1], names: [:data])
      iex> right = Nx.tensor([1, 2, 3, 4], names: [:data])
      iex> Nx.left_shift(left, right)
      #Nx.Tensor<
        s64[data: 4]
        [2, 4, -8, -16]
      >

  ### Error cases

      iex> Nx.left_shift(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}

  """
  @doc type: :element
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
  not match and broadcasting is possible. If the number of
  shifts are negative, Nx's default backend will raise,
  but it may trigger undefined behaviour in other backends.

  If you're using `Nx.Defn.defn/2`, you can use the `>>>` operator
  in place of this function: `left >>> right`.

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

      iex> left = Nx.tensor([16, 32, -64, -128], names: [:data])
      iex> right = Nx.tensor([1, 2, 3, 4])
      iex> Nx.right_shift(left, right)
      #Nx.Tensor<
        s64[data: 4]
        [8, 8, -8, -8]
      >

  ### Error cases

      iex> Nx.right_shift(Nx.tensor([0, 0, 1, 1]), 1.0)
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}

  """
  @doc type: :element
  def right_shift(left, right),
    do: element_wise_bin_op(left, right, :right_shift, &assert_bitwise_type!/1)

  @doc """
  Element-wise equality comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `==` operator
  in place of this function: `left == right`.

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

      iex> left = Nx.tensor([1, 2, 3], names: [:data])
      iex> right = Nx.tensor([1, 2, 5])
      iex> Nx.equal(left, right)
      #Nx.Tensor<
        u8[data: 3]
        [1, 1, 0]
      >

      iex> left = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, nil])
      iex> right = Nx.tensor([1, 2, 3])
      iex> Nx.equal(left, right)
      #Nx.Tensor<
        u8[x: 2][3]
        [
          [1, 1, 1],
          [0, 0, 0]
        ]
      >
  """
  @doc type: :element
  def equal(left, right), do: element_wise_pred_op(left, right, :equal)

  @doc """
  Element-wise logical and of two tensors.

  Zero is considered false, any other number is considered
  true.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `and` operator
  in place of this function: `left and right`.

  ## Examples

      iex> Nx.logical_and(1, Nx.tensor([-1, 0, 1], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 0, 1]
      >

      iex> left = Nx.tensor([-1, 0, 1], names: [:data])
      iex> right = Nx.tensor([[-1], [0], [1]])
      iex> Nx.logical_and(left, right)
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 0, 1],
          [0, 0, 0],
          [1, 0, 1]
        ]
      >

      iex> left = Nx.tensor([-1.0, 0.0, 1.0], names: [:data])
      iex> right = Nx.tensor([[-1], [0], [1]])
      iex> Nx.logical_and(left, right)
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 0, 1],
          [0, 0, 0],
          [1, 0, 1]
        ]
      >
  """
  @doc type: :element
  def logical_and(left, right), do: element_wise_pred_op(left, right, :logical_and)

  @doc """
  Element-wise logical or of two tensors.

  Zero is considered false, any other number is considered
  true.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `or` operator
  in place of this function: `left or right`.

  ## Examples

      iex> Nx.logical_or(0, Nx.tensor([-1, 0, 1], names: [:data]))
      #Nx.Tensor<
        u8[data: 3]
        [1, 0, 1]
      >

      iex> left = Nx.tensor([-1, 0, 1], names: [:data])
      iex> right = Nx.tensor([[-1], [0], [1]])
      iex> Nx.logical_or(left, right)
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ]
      >

      iex> left = Nx.tensor([-1.0, 0.0, 1.0], names: [:data])
      iex> right = Nx.tensor([[-1], [0], [1]])
      iex> Nx.logical_or(left, right)
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ]
      >
  """
  @doc type: :element
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

      iex> left = Nx.tensor([-1, 0, 1], names: [:data])
      iex> right = Nx.tensor([[-1], [0], [1]])
      iex> Nx.logical_xor(left, right)
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]
        ]
      >

      iex> left = Nx.tensor([-1.0, 0.0, 1.0], names: [:data])
      iex> right = Nx.tensor([[-1], [0], [1]])
      iex> Nx.logical_xor(left, right)
      #Nx.Tensor<
        u8[3][data: 3]
        [
          [0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]
        ]
      >

  """
  @doc type: :element
  def logical_xor(left, right), do: element_wise_pred_op(left, right, :logical_xor)

  @doc """
  Element-wise logical not a tensor.

  Zero is considered false, any other number is considered
  true.

  If you're using `Nx.Defn.defn/2`, you can use the `not` operator
  in place of this function: `not tensor`.

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
  @doc type: :element
  def logical_not(tensor) do
    tensor = to_tensor(tensor)
    output = Nx.template(tensor.shape, {:u, 8}, names: tensor.names)

    Nx.Shared.optional(:logical_not, [tensor], output, fn tensor ->
      type = tensor.type

      zero =
        Nx.BinaryBackend.from_binary(
          %T{shape: {}, type: type, names: []},
          number_to_binary(0, type),
          []
        )

      element_wise_pred_op(tensor, zero, :equal)
    end)
  end

  @doc """
  Element-wise not-equal comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `!=` operator
  in place of this function: `left != right`.

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

      iex> left = Nx.tensor([1, 1, 2])
      iex> right = Nx.tensor([1, 2, 3], names: [:data])
      iex> Nx.not_equal(left, right)
      #Nx.Tensor<
        u8[data: 3]
        [0, 1, 1]
      >

      iex> left = Nx.tensor([[1, 4, 2], [4, 5, 6]], names: [:x, :y])
      iex> right = Nx.tensor([[1, 3, 2], [4, 2, 1]], names: [:x, :y])
      iex> Nx.not_equal(left, right)
      #Nx.Tensor<
        u8[x: 2][y: 3]
        [
          [0, 1, 0],
          [0, 1, 1]
        ]
      >
  """
  @doc type: :element
  def not_equal(left, right), do: element_wise_pred_op(left, right, :not_equal)

  @doc """
  Element-wise greater than comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `>` operator
  in place of this function: `left > right`.

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

      iex> left = Nx.tensor([1, 2, 3], names: [:data])
      iex> right = Nx.tensor([1, 2, 2])
      iex> Nx.greater(left, right)
      #Nx.Tensor<
        u8[data: 3]
        [0, 0, 1]
      >

      iex> left = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, :y])
      iex> right = Nx.tensor([1, 2, 3])
      iex> Nx.greater(left, right)
      #Nx.Tensor<
        u8[x: 2][y: 3]
        [
          [0, 0, 0],
          [1, 1, 1]
        ]
      >
  """
  @doc type: :element
  def greater(left, right), do: non_complex_element_wise_pred_op(left, right, :greater)

  @doc """
  Element-wise less than comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `<` operator
  in place of this function: `left < right`.

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
  @doc type: :element
  def less(left, right), do: non_complex_element_wise_pred_op(left, right, :less)

  @doc """
  Element-wise greater than or equal comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `>=` operator
  in place of this function: `left >= right`.

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

      iex> left = Nx.tensor([1, 2, 3], names: [:data])
      iex> right = Nx.tensor([1, 2, 2])
      iex> Nx.greater_equal(left, right)
      #Nx.Tensor<
        u8[data: 3]
        [1, 1, 1]
      >

      iex> left = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, :y])
      iex> right = Nx.tensor([1, 2, 3])
      iex> Nx.greater_equal(left, right)
      #Nx.Tensor<
        u8[x: 2][y: 3]
        [
          [1, 1, 1],
          [1, 1, 1]
        ]
      >

  """
  @doc type: :element
  def greater_equal(left, right),
    do: non_complex_element_wise_pred_op(left, right, :greater_equal)

  @doc """
  Element-wise less than or equal comparison of two tensors.

  If a number is given, it is converted to a tensor.

  It will broadcast tensors whenever the dimensions do
  not match and broadcasting is possible.

  If you're using `Nx.Defn.defn/2`, you can use the `<=` operator
  in place of this function: `left <= right`.

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

      iex> left = Nx.tensor([1, 2, 3], names: [:data])
      iex> right = Nx.tensor([1, 2, 2])
      iex> Nx.less_equal(left, right)
      #Nx.Tensor<
        u8[data: 3]
        [1, 1, 0]
      >

      iex> left = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      iex> right = Nx.tensor([1, 2, 3], names: [:y])
      iex> Nx.less_equal(left, right)
      #Nx.Tensor<
        u8[2][y: 3]
        [
          [1, 1, 1],
          [0, 0, 0]
        ]
      >

  """
  @doc type: :element
  def less_equal(left, right), do: non_complex_element_wise_pred_op(left, right, :less_equal)

  @doc """
  Constructs a tensor from two tensors, based on a predicate.

  The resulting tensor is built by evaluating each element of
  `pred` and returning either the corresponding element from
  `on_true` or `on_false`.

  `pred` must either be `1` or `0` or a tensor of predicates
  with a shape that matches the largest shape between `s1` or `s2`.

  If the shape of `on_true` or `on_false` do not match the shape of
  `pred`, attempts to broadcast both so they match the shape of `pred`.

  ## Examples

  When the first argument is a scalar:

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

  When the first argument is a tensor:

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

  If the tensor has other values, any non-zero value is considered true:

      iex> Nx.select(Nx.tensor([0, 1, 2], type: {:u, 8}), Nx.tensor([0, 0, 0]), Nx.tensor([1, 1, 1]))
      #Nx.Tensor<
        s64[3]
        [1, 0, 0]
      >

      iex> Nx.select(Nx.tensor([0, 1, 0]), Nx.tensor([1, 1, 1]), Nx.tensor([2.0, 2.0, 2.0]))
      #Nx.Tensor<
        f32[3]
        [2.0, 1.0, 2.0]
      >
  """
  @doc type: :element
  def select(pred, on_true, on_false) do
    output_type = binary_type(on_true, on_false)

    %T{shape: pred_shape, names: pred_names} = pred = to_tensor(pred)
    %T{shape: true_shape, names: true_names} = on_true = to_tensor(on_true)
    %T{shape: false_shape, names: false_names} = on_false = to_tensor(on_false)

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

  @doc """
  Performs a `window_reduce` to select the maximum index in each
  window of the input tensor according to and scatters source tensor
  to corresponding maximum indices in the output tensor.

  Output tensor is initialized as a full tensor with values
  `init_value`. If indices overlap, adds overlapping source values.
  The shape of the source tensor must match the valid windows in the
  input tensor. This means the shape of the source tensor must match
  the shape of the input tensor after a `window_reduce` op with padding
  `padding` and strides `strides`.

  This function is the gradient of `window_max`.

  ## Examples

      iex> t = Nx.tensor([
      ...>   [7, 2, 5, 3, 10, 2],
      ...>   [3, 8, 9, 3, 4, 2],
      ...>   [1, 5, 7, 5, 6, 1],
      ...>   [0, 6, 2, 7, 2, 8]
      ...> ])
      iex> opts = [strides: [2, 3], padding: :valid]
      iex> Nx.window_scatter_max(t, Nx.tensor([[2, 6], [3, 1]]), 0, {2, 3}, opts)
      #Nx.Tensor<
        s64[4][6]
        [
          [0, 0, 0, 0, 6, 0],
          [0, 0, 2, 0, 0, 0],
          [0, 0, 3, 0, 0, 0],
          [0, 0, 0, 0, 0, 1]
        ]
      >

      iex> t = Nx.tensor([
      ...>   [7, 2, 5, 3, 8],
      ...>   [3, 8, 9, 3, 4],
      ...>   [1, 5, 7, 5, 6],
      ...>   [0, 6, 2, 10, 2]
      ...> ])
      iex> opts = [strides: [2, 2], padding: :valid]
      iex> Nx.window_scatter_max(t, Nx.tensor([[2, 6], [3, 1]]), 0, {2, 3}, opts)
      #Nx.Tensor<
        s64[4][5]
        [
          [0, 0, 0, 0, 0],
          [0, 0, 8, 0, 0],
          [0, 0, 3, 0, 0],
          [0, 0, 0, 1, 0]
        ]
      >
  """
  @doc type: :window
  def window_scatter_max(tensor, source, init_value, window_dimensions, opts \\ []) do
    opts = keyword!(opts, padding: :valid, strides: 1)
    Nx.Shape.validate!(window_dimensions, :window_dimensions)

    %T{shape: input_shape} = tensor = to_tensor(tensor)
    %T{shape: source_shape, type: source_type} = source = to_tensor(source)
    %T{type: value_type} = init_value = to_tensor(init_value)

    padding = opts[:padding]
    strides = opts[:strides]

    strides =
      if is_integer(strides),
        do: List.duplicate(strides, rank(input_shape)),
        else: strides

    dilations = List.duplicate(1, rank(input_shape))

    {output_window_shape, padding_config} =
      Nx.Shape.pool(input_shape, window_dimensions, strides, padding, dilations)

    unless output_window_shape == source_shape do
      raise ArgumentError, "source shape must match valid windows in input tensor"
    end

    output_type = Nx.Type.merge(source_type, value_type)
    Nx.Shared.raise_complex_not_supported(output_type, :window_scatter_max, 5)

    impl!(tensor, source).window_scatter_max(
      %{tensor | type: output_type},
      tensor,
      source,
      init_value,
      window_dimensions,
      padding: padding_config,
      strides: strides
    )
  end

  @doc """
  Performs a `window_reduce` to select the minimum index in each
  window of the input tensor according to and scatters source tensor
  to corresponding minimum indices in the output tensor.

  Output tensor is initialized as a full tensor with values
  `init_value`. If indices overlap, adds overlapping source values.
  The shape of the source tensor must match the valid windows in the
  input tensor. This means the shape of the source tensor must match
  the shape of the input tensor after a `window_reduce` op with padding
  `padding` and strides `strides`.

  This function is the gradient of `window_min`.

  ## Examples

      iex> t = Nx.tensor([
      ...>   [7, 2, 5, 3, 10, 2],
      ...>   [3, 8, 9, 3, 4, 2],
      ...>   [1, 5, 7, 5, 6, 1],
      ...>   [0, 6, 2, 7, 2, 8]
      ...> ])
      iex> opts = [strides: [2, 3], padding: :valid]
      iex> Nx.window_scatter_min(t, Nx.tensor([[2, 6], [3, 1]]), 0, {2, 3}, opts)
      #Nx.Tensor<
        s64[4][6]
        [
          [0, 2, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 6],
          [0, 0, 0, 0, 0, 1],
          [3, 0, 0, 0, 0, 0]
        ]
      >

      iex> t = Nx.tensor([
      ...>   [7, 2, 5, 3, 8],
      ...>   [3, 8, 9, 3, 4],
      ...>   [1, 5, 7, 5, 6],
      ...>   [0, 6, 2, 10, 2]
      ...> ])
      iex> opts = [strides: [2, 2], padding: :valid]
      iex> Nx.window_scatter_min(t, Nx.tensor([[2, 6], [3, 1]]), 0, {2, 3}, opts)
      #Nx.Tensor<
        s64[4][5]
        [
          [0, 2, 0, 0, 0],
          [0, 0, 0, 6, 0],
          [0, 0, 0, 0, 0],
          [3, 0, 0, 0, 1]
        ]
      >
  """
  @doc type: :window
  def window_scatter_min(tensor, source, init_value, window_dimensions, opts \\ []) do
    opts = keyword!(opts, padding: :valid, strides: 1)

    %T{shape: input_shape} = tensor = to_tensor(tensor)
    %T{shape: source_shape, type: source_type} = source = to_tensor(source)
    %T{type: value_type} = init_value = to_tensor(init_value)

    padding = opts[:padding]
    strides = opts[:strides]

    strides =
      if is_integer(strides),
        do: List.duplicate(strides, rank(input_shape)),
        else: strides

    dilations = List.duplicate(1, rank(input_shape))

    {output_window_shape, padding_config} =
      Nx.Shape.pool(input_shape, window_dimensions, strides, padding, dilations)

    unless output_window_shape == source_shape do
      raise ArgumentError, "source shape must match valid windows in input tensor"
    end

    output_type = Nx.Type.merge(source_type, value_type)
    Nx.Shared.raise_complex_not_supported(output_type, :window_scatter_min, 5)

    impl!(tensor, source).window_scatter_min(
      %{tensor | type: output_type},
      tensor,
      source,
      init_value,
      window_dimensions,
      padding: padding_config,
      strides: strides
    )
  end

  @doc """
  Performs an indexed `add` operation on the `target` tensor,
  adding the `updates` into the corresponding `indices` positions.

  This operation is the grad for `gather/2` and gather-like operations such as
  `take/3` and `take_along_axis/3`.

  `indices` must be a fully qualified tensor of shape `{n, Nx.rank(target)}`, with `n`
  being an arbitrary number of indices, while `updates` must have a compatible `{n}` shape.

  ### Examples

      iex> t = Nx.iota({1, 2, 3})
      #Nx.Tensor<
        s64[1][2][3]
        [
          [
            [0, 1, 2],
            [3, 4, 5]
          ]
        ]
      >
      iex> indices = Nx.tensor([[0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 2], [0, 1, 2]])
      iex> updates = Nx.tensor([1, 3, 1, -2, 5])
      iex> Nx.indexed_add(t, indices, updates)
      #Nx.Tensor<
        s64[1][2][3]
        [
          [
            [2, 1, 0],
            [3, 7, 10]
          ]
        ]
      >

  Type promotions should happen automatically.

      iex> Nx.indexed_add(Nx.tensor([1.0]), Nx.tensor([[0], [0]]), Nx.tensor([1, 1]))
      #Nx.Tensor<
        f32[1]
        [3.0]
      >

      iex> Nx.indexed_add(Nx.tensor([1]), Nx.tensor([[0], [0]]), Nx.tensor([1.0, 1.0]))
      #Nx.Tensor<
        f32[1]
        [3.0]
      >

      iex> Nx.indexed_add(Nx.tensor([1], type: {:s, 32}), Nx.tensor([[0], [0]]), Nx.tensor([1, 1], type: {:s, 64}))
      #Nx.Tensor<
        s64[1]
        [3]
      >

  ### Error cases
      iex> Nx.indexed_add(Nx.tensor([[1], [2]]), Nx.tensor([[[1, 2, 3]]]), Nx.tensor([0]))
      ** (ArgumentError) indices must be a rank 2 tensor, got: 3

      iex> Nx.indexed_add(Nx.tensor([[1], [2]]), Nx.tensor([[1, 2]]), Nx.tensor([[0]]))
      ** (ArgumentError) updates must be a rank 1 tensor, got: 2

      iex> Nx.indexed_add(Nx.tensor([[1], [2]]), Nx.tensor([[1, 2, 3]]), Nx.tensor([0]))
      ** (ArgumentError) expected indices to have shape {*, 2}, got: {1, 3}

      iex> Nx.indexed_add(Nx.tensor([[1], [2]]), Nx.tensor([[1, 2]]), Nx.tensor([0, 1]))
      ** (ArgumentError) expected updates tensor to match the first axis of indices tensor with shape {1, 2}, got {2}
  """
  @doc type: :indexed
  def indexed_add(target, indices, updates) do
    target = to_tensor(target)
    indices = to_tensor(indices)
    updates = to_tensor(updates)

    type = binary_type(target, updates)

    Nx.Shape.indexed_add(target, indices, updates)

    impl!(target, indices, updates).indexed_add(%{target | type: type}, target, indices, updates)
  end

  ## Unary ops
  @disallow_complex_type_unary_ops [:erf, :erfc, :erf_inv]

  for {name, {desc, code, formula}} <- Nx.Shared.unary_math_funs() do
    inputs =
      if name in [:acos, :asin, :atan, :atanh, :erf_inv] do
        [to_float32(0.1), to_float32(0.5), to_float32(0.9)]
      else
        [1, 2, 3]
      end

    outputs =
      for input <- inputs do
        {res, _} = Code.eval_quoted(code, x: input)
        to_float32(res)
      end

    complex_check_block =
      if name in @disallow_complex_type_unary_ops do
        quote do
          Nx.Shared.raise_complex_not_supported(var!(type), unquote(name), 1)
        end
      end

    @doc """
    Calculates the #{desc} of each element in the tensor.

    It is equivalent to:

    #{formula}

    ## Examples

        iex> Nx.#{name}(#{hd(inputs)})
        #Nx.Tensor<
          f32
          #{hd(outputs)}
        >

        iex> Nx.#{name}(Nx.tensor(#{inspect(inputs)}, names: [:x]))
        #Nx.Tensor<
          f32[x: 3]
          #{inspect(outputs)}
        >

    """
    @doc type: :element
    def unquote(name)(tensor) do
      tensor = to_tensor(tensor)
      type = Nx.Type.to_floating(tensor.type)

      unquote(complex_check_block)

      impl!(tensor).unquote(name)(%{tensor | type: type}, tensor)
    end
  end

  @doc """
  Negates each element in the tensor.

  If you're using `Nx.Defn.defn/2`, you can use the `-` unary operator
  in place of this function: `-tensor`.

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
  @doc type: :element
  def negate(tensor) do
    tensor = to_tensor(tensor)
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
  @doc type: :element
  def sign(tensor) do
    tensor = to_tensor(tensor)
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
  @doc type: :element
  def abs(tensor) do
    tensor = to_tensor(tensor)

    case tensor.type do
      {:u, _} -> tensor
      {:c, size} -> impl!(tensor).abs(%{tensor | type: {:f, div(size, 2)}}, tensor)
      _ -> impl!(tensor).abs(tensor, tensor)
    end
  end

  @doc """
  Calculates the complex conjugate of each element in the tensor.

  If $z = a + bi = r e^\\theta$, $conjugate(z) = z^* = a - bi =  r e^{-\\theta}$

  ## Examples

       iex> Nx.conjugate(Complex.new(1, 2))
       #Nx.Tensor<
         c64
         1.0-2.0i
       >

       iex> Nx.conjugate(1)
       #Nx.Tensor<
         c64
         1.0+0.0i
       >

       iex> Nx.conjugate(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)]))
       #Nx.Tensor<
         c64[2]
         [1.0-2.0i, 2.0+4.0i]
       >
  """
  @doc type: :element
  def conjugate(tensor) do
    tensor = to_tensor(tensor)

    impl!(tensor).conjugate(%{tensor | type: Nx.Type.to_complex(tensor.type)}, tensor)
  end

  @doc """
  Calculates the complex phase angle of each element in the tensor.
  $phase(z) = atan2(b, a), z = a + bi \\in \\Complex$

  ## Examples

       iex> Nx.phase(Complex.new(1, 2))
       #Nx.Tensor<
         f32
         1.1071487665176392
       >

       iex> Nx.phase(1)
       #Nx.Tensor<
         f32
         0.0
       >

       iex> import Nx, only: [sigil_V: 2]
       iex> Nx.phase(~V[1+2i -2+1i])
       #Nx.Tensor<
         f32[2]
         [1.1071487665176392, 2.677945137023926]
       >
  """
  @doc type: :element
  def phase(tensor) do
    tensor = to_tensor(tensor)
    output = %{tensor | type: Nx.Type.to_real(tensor.type)}

    Nx.Shared.optional(:phase, [tensor], output, fn tensor ->
      tensor
      |> imag
      |> atan2(real(tensor))
    end)
  end

  @doc """
  Returns the real component of each entry in a complex tensor
  as a floating point tensor.

  ## Examples

      iex> Nx.real(Complex.new(1, 2))
      #Nx.Tensor<
        f32
        1.0
      >

      iex> Nx.real(Nx.tensor(1))
      #Nx.Tensor<
        f32
        1.0
      >

      iex> Nx.real(Nx.tensor(1, type: {:bf, 16}))
      #Nx.Tensor<
        bf16
        1.0
      >

      iex> Nx.real(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)]))
      #Nx.Tensor<
        f32[2]
        [1.0, 2.0]
      >
  """
  @doc type: :element
  def real(tensor) do
    %{type: type} = tensor = to_tensor(tensor)

    cond do
      match?({:c, _}, type) ->
        {:c, size} = type
        impl!(tensor).real(%{tensor | type: {:f, div(size, 2)}}, tensor)

      Nx.Type.float?(type) ->
        tensor

      tensor ->
        as_type(tensor, {:f, 32})
    end
  end

  @doc """
  Returns the imaginary component of each entry in a complex tensor
  as a floating point tensor.

  ## Examples

      iex> Nx.imag(Complex.new(1, 2))
      #Nx.Tensor<
        f32
        2.0
      >

      iex> Nx.imag(Nx.tensor(1))
      #Nx.Tensor<
        f32
        0.0
      >

      iex> Nx.imag(Nx.tensor(1, type: {:bf, 16}))
      #Nx.Tensor<
        bf16
        0.0
      >

      iex> Nx.imag(Nx.tensor([Complex.new(1, 2), Complex.new(2, -4)]))
      #Nx.Tensor<
        f32[2]
        [2.0, -4.0]
      >
  """
  @doc type: :element
  def imag(tensor) do
    case to_tensor(tensor) do
      %{type: {:c, size}} = tensor ->
        impl!(tensor).imag(%{tensor | type: {:f, div(size, 2)}}, tensor)

      tensor ->
        floating = Nx.Type.to_floating(tensor.type)
        zero = Nx.tensor(0.0, type: floating)
        broadcast(zero, tensor)
    end
  end

  @doc """
  Constructs a complex tensor from two equally-shaped tensors.

  Does not accept complex tensors as inputs.

  ### Examples

      iex> Nx.complex(Nx.tensor(1), Nx.tensor(2))
      #Nx.Tensor<
        c64
        1.0+2.0i
      >

      iex> Nx.complex(Nx.tensor([1, 2]), Nx.tensor([3, 4]))
      #Nx.Tensor<
        c64[2]
        [1.0+3.0i, 2.0+4.0i]
      >
  """
  @doc type: :element
  def complex(real, imag) do
    if elem(type(real), 0) == :c or elem(type(imag), 0) == :c do
      Nx.Shared.raise_complex_not_supported("complex", 2)
    end

    imag
    |> multiply(Nx.Constants.i())
    |> add(real)
  end

  @doc """
  Applies bitwise not to each element in the tensor.

  If you're using `Nx.Defn.defn/2`, you can use the `~~~` operator
  in place of this function: `~~~tensor`.

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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}
  """
  @doc type: :element
  def bitwise_not(tensor) do
    tensor = to_tensor(tensor)
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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}
  """
  @doc type: :element
  def population_count(tensor) do
    tensor = to_tensor(tensor)
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
      ** (ArgumentError) bitwise operators expect integer tensors as inputs and outputs an integer tensor, got: {:f, 32}
  """
  @doc type: :element
  def count_leading_zeros(tensor) do
    tensor = to_tensor(tensor)
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
          f32[x: 4]
          [#{res1}.0, #{res2}.0, #{res3}.0, #{res4}.0]
        >

    """
    @doc type: :element
    def unquote(name)(tensor) do
      case to_tensor(tensor) do
        %T{type: {type, _}} = tensor when type in [:s, :u] -> tensor
        %T{type: {:c, _}} -> Nx.Shared.raise_complex_not_supported(unquote(name), 1)
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

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the reduced
  axes to size 1.

  ## Examples

      iex> Nx.all(Nx.tensor([0, 1, 2]))
      #Nx.Tensor<
        u8
        0
      >

      iex> Nx.all(Nx.tensor([[-1, 0, 1], [2, 3, 4]], names: [:x, :y]), axes: [:x])
      #Nx.Tensor<
        u8[y: 3]
        [1, 0, 1]
      >

      iex> Nx.all(Nx.tensor([[-1, 0, 1], [2, 3, 4]], names: [:x, :y]), axes: [:y])
      #Nx.Tensor<
        u8[x: 2]
        [0, 1]
      >
  """
  @doc type: :aggregation
  def all(tensor, opts \\ []) do
    aggregate_axes_op(to_tensor(tensor), :all, {:u, 8}, opts)
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

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the reduced
  axes to size 1.

  ## Examples

      iex> Nx.any(Nx.tensor([0, 1, 2]))
      #Nx.Tensor<
        u8
        1
      >

      iex> Nx.any(Nx.tensor([[0, 1, 0], [0, 1, 2]], names: [:x, :y]), axes: [:x])
      #Nx.Tensor<
        u8[y: 3]
        [0, 1, 1]
      >

      iex> Nx.any(Nx.tensor([[0, 1, 0], [0, 1, 2]], names: [:x, :y]), axes: [:y])
      #Nx.Tensor<
        u8[x: 2]
        [1, 1]
      >
  """
  @doc type: :aggregation
  def any(tensor, opts \\ []) do
    aggregate_axes_op(to_tensor(tensor), :any, {:u, 8}, opts)
  end

  @doc """
  Returns a scalar tensor of value 1 if all element-wise values
  are within tolerance of b. Otherwise returns value 0.

  You may set the absolute tolerance, `:atol` and relative tolerance
  `:rtol`. Given tolerances, this functions returns 1 if

      absolute(a - b) <= (atol + rtol * absolute(b))

  is true for all elements of a and b.

  ## Examples

      iex> Nx.all_close(Nx.tensor([1.0e10, 1.0e-7]), Nx.tensor([1.00001e10, 1.0e-8]))
      #Nx.Tensor<
        u8
        0
      >

      iex> Nx.all_close(Nx.tensor([1.0e-8, 1.0e-8]), Nx.tensor([1.0e-8, 1.0e-9]))
      #Nx.Tensor<
        u8
        1
      >

  """
  @doc type: :aggregation
  def all_close(a, b, opts \\ []) do
    opts = keyword!(opts, rtol: 1.0e-5, atol: 1.0e-8)
    rtol = opts[:rtol]
    atol = opts[:atol]

    # TODO: deal with non_finite entries by adding is_infinity and is_nan
    all(less_equal(Nx.abs(subtract(a, b)), add(atol, multiply(rtol, Nx.abs(b)))))
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
        f32
        10.0
      >

  Giving a tensor with low precision casts it to a higher
  precision to make sure the sum does not overflow:

      iex> Nx.sum(Nx.tensor([[101, 102], [103, 104]], type: {:s, 8}, names: [:x, :y]))
      #Nx.Tensor<
        s64
        410
      >

      iex> Nx.sum(Nx.tensor([[101, 102], [103, 104]], type: {:s, 16}, names: [:x, :y]))
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

  Same tensor over different axes combinations:

      iex> t = Nx.tensor(
      ...>   [
      ...>     [
      ...>       [1, 2, 3],
      ...>       [4, 5, 6]
      ...>     ],
      ...>     [
      ...>       [7, 8, 9],
      ...>       [10, 11, 12]
      ...>     ]
      ...>   ],
      ...>   names: [:x, :y, :z]
      ...> )
      iex> Nx.sum(t, axes: [:x])
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >
      iex> Nx.sum(t, axes: [:y])
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [5, 7, 9],
          [17, 19, 21]
        ]
      >
      iex> Nx.sum(t, axes: [:z])
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [6, 15],
          [24, 33]
        ]
      >
      iex> Nx.sum(t, axes: [:x, :z])
      #Nx.Tensor<
        s64[y: 2]
        [30, 48]
      >
      iex> Nx.sum(t, axes: [:z])
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [6, 15],
          [24, 33]
        ]
      >
      iex> Nx.sum(t, axes: [-3])
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [8, 10, 12],
          [14, 16, 18]
        ]
      >

  ### Keeping axes

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.sum(t, axes: [:z], keep_axes: true)
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
  @doc type: :aggregation
  def sum(tensor, opts \\ []) do
    tensor = to_tensor(tensor)
    type = Nx.Type.to_aggregate(tensor.type)
    aggregate_axes_op(tensor, :sum, type, opts)
  end

  @doc """
  Returns the mean for the tensor.

  If the `:axes` option is given, it aggregates over
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
        f32
        42.0
      >

      iex> Nx.mean(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32
        2.0
      >

  ### Aggregating over an axis

      iex> Nx.mean(Nx.tensor([1, 2, 3], names: [:x]), axes: [0])
      #Nx.Tensor<
        f32
        2.0
      >

      iex> Nx.mean(Nx.tensor([1, 2, 3], type: {:u, 8}, names: [:x]), axes: [:x])
      #Nx.Tensor<
        f32
        2.0
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.mean(t, axes: [:x])
      #Nx.Tensor<
        f32[y: 2][z: 3]
        [
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.mean(t, axes: [:x, :z])
      #Nx.Tensor<
        f32[y: 2]
        [5.0, 8.0]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.mean(t, axes: [-1])
      #Nx.Tensor<
        f32[x: 2][y: 2]
        [
          [2.0, 5.0],
          [8.0, 11.0]
        ]
      >

  ### Keeping axes

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.mean(t, axes: [-1], keep_axes: true)
      #Nx.Tensor<
        f32[x: 2][y: 2][z: 1]
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
  @doc type: :aggregation, from_backend: false
  def mean(tensor, opts \\ []) do
    %T{shape: shape, names: names} = tensor = to_tensor(tensor)

    mean_den =
      if axes = opts[:axes] do
        mean_den(shape, Nx.Shape.normalize_axes(shape, axes, names))
      else
        size(shape)
      end

    divide(sum(tensor, opts), mean_den)
  end

  defp mean_den(_shape, []), do: 1

  defp mean_den(shape, [axis | axes]) when axis >= 0,
    do: elem(shape, axis) * mean_den(shape, axes)

  @doc """
  Returns the product for the tensor.

  If the `:axes` option is given, it aggregates over
  the given dimensions, effectively removing them.
  `axes: [0]` implies aggregating over the highest order
  dimension and so forth. If the axis is negative, then
  counts the axis from the back. For example, `axes: [-1]`
  will always aggregate all rows.

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the multiplied
  axes to size 1.

  ## Examples

      iex> Nx.product(Nx.tensor(42))
      #Nx.Tensor<
        s64
        42
      >

      iex> Nx.product(Nx.tensor([1, 2, 3], names: [:x]))
      #Nx.Tensor<
        s64
        6
      >

      iex> Nx.product(Nx.tensor([[1.0, 2.0], [3.0, 4.0]], names: [:x, :y]))
      #Nx.Tensor<
        f32
        24.0
      >

  Giving a tensor with low precision casts it to a higher
  precision to make sure the sum does not overflow:

      iex> Nx.product(Nx.tensor([[10, 20], [30, 40]], type: {:u, 8}, names: [:x, :y]))
      #Nx.Tensor<
        u64
        240000
      >

      iex> Nx.product(Nx.tensor([[10, 20], [30, 40]], type: {:s, 8}, names: [:x, :y]))
      #Nx.Tensor<
        s64
        240000
      >

  ### Aggregating over an axis

      iex> Nx.product(Nx.tensor([1, 2, 3], names: [:x]), axes: [0])
      #Nx.Tensor<
        s64
        6
      >

  Same tensor over different axes combinations:

      iex> t = Nx.tensor(
      ...>   [
      ...>     [
      ...>       [1, 2, 3],
      ...>       [4, 5, 6]
      ...>     ],
      ...>     [
      ...>       [7, 8, 9],
      ...>       [10, 11, 12]
      ...>     ]
      ...>   ],
      ...>   names: [:x, :y, :z]
      ...> )
      iex> Nx.product(t, axes: [:x])
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [7, 16, 27],
          [40, 55, 72]
        ]
      >
      iex> Nx.product(t, axes: [:y])
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [4, 10, 18],
          [70, 88, 108]
        ]
      >
      iex> Nx.product(t, axes: [:x, :z])
      #Nx.Tensor<
        s64[y: 2]
        [3024, 158400]
      >
      iex> Nx.product(t, axes: [:z])
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [6, 120],
          [504, 1320]
        ]
      >
      iex> Nx.product(t, axes: [-3])
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [7, 16, 27],
          [40, 55, 72]
        ]
      >

  ### Keeping axes

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], names: [:x, :y, :z])
      iex> Nx.product(t, axes: [:z], keep_axes: true)
      #Nx.Tensor<
        s64[x: 2][y: 2][z: 1]
        [
          [
            [6],
            [120]
          ],
          [
            [504],
            [1320]
          ]
        ]
      >

  ### Errors

      iex> Nx.product(Nx.tensor([[1, 2]]), axes: [2])
      ** (ArgumentError) given axis (2) invalid for shape with rank 2
  """
  @doc type: :aggregation
  def product(tensor, opts \\ []) do
    tensor = to_tensor(tensor)
    type = Nx.Type.to_aggregate(tensor.type)
    aggregate_axes_op(tensor, :product, type, opts)
  end

  @doc """
  Returns the maximum values of the tensor.

  If the `:axes` option is given, it aggregates over
  the given dimensions, effectively removing them.
  `axes: [0]` implies aggregating over the highest order
  dimension and so forth. If the axis is negative, then
  counts the axis from the back. For example, `axes: [-1]`
  will always aggregate all rows.

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the reduced
  axes to size 1.

  ## Examples

      iex> Nx.reduce_max(Nx.tensor(42))
      #Nx.Tensor<
        s64
        42
      >

      iex> Nx.reduce_max(Nx.tensor(42.0))
      #Nx.Tensor<
        f32
        42.0
      >

      iex> Nx.reduce_max(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        s64
        3
      >

  ### Aggregating over an axis

      iex> t = Nx.tensor([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      iex> Nx.reduce_max(t, axes: [:x])
      #Nx.Tensor<
        s64[y: 3]
        [3, 1, 4]
      >

      iex> t = Nx.tensor([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      iex> Nx.reduce_max(t, axes: [:y])
      #Nx.Tensor<
        s64[x: 2]
        [4, 2]
      >

      iex> t = Nx.tensor([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      iex> Nx.reduce_max(t, axes: [:x, :z])
      #Nx.Tensor<
        s64[y: 2]
        [4, 8]
      >

  ### Keeping axes

      iex> t = Nx.tensor([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      iex> Nx.reduce_max(t, axes: [:x, :z], keep_axes: true)
      #Nx.Tensor<
        s64[x: 1][y: 2][z: 1]
        [
          [
            [4],
            [8]
          ]
        ]
      >

  """
  @doc type: :aggregation
  def reduce_max(tensor, opts \\ []) do
    %{type: type} = tensor = to_tensor(tensor)
    Nx.Shared.raise_complex_not_supported(type, :reduce_max, 2)
    aggregate_axes_op(tensor, :reduce_max, type, opts)
  end

  @doc """
  Returns the minimum values of the tensor.

  If the `:axes` option is given, it aggregates over
  the given dimensions, effectively removing them.
  `axes: [0]` implies aggregating over the highest order
  dimension and so forth. If the axis is negative, then
  counts the axis from the back. For example, `axes: [-1]`
  will always aggregate all rows.

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the reduced
  axes to size 1.

  ## Examples

      iex> Nx.reduce_min(Nx.tensor(42))
      #Nx.Tensor<
        s64
        42
      >

      iex> Nx.reduce_min(Nx.tensor(42.0))
      #Nx.Tensor<
        f32
        42.0
      >

      iex> Nx.reduce_min(Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        s64
        1
      >

  ### Aggregating over an axis

      iex> t = Nx.tensor([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      iex> Nx.reduce_min(t, axes: [:x])
      #Nx.Tensor<
        s64[y: 3]
        [2, 1, 1]
      >

      iex> t = Nx.tensor([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      iex> Nx.reduce_min(t, axes: [:y])
      #Nx.Tensor<
        s64[x: 2]
        [1, 1]
      >

      iex> t = Nx.tensor([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      iex> Nx.reduce_min(t, axes: [:x, :z])
      #Nx.Tensor<
        s64[y: 2]
        [1, 3]
      >

  ### Keeping axes

      iex> t = Nx.tensor([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      iex> Nx.reduce_min(t, axes: [:x, :z], keep_axes: true)
      #Nx.Tensor<
        s64[x: 1][y: 2][z: 1]
        [
          [
            [1],
            [3]
          ]
        ]
      >

  """
  @doc type: :aggregation
  def reduce_min(tensor, opts \\ []) do
    %{type: type} = tensor = to_tensor(tensor)
    Nx.Shared.raise_complex_not_supported(type, :reduce_min, 2)
    aggregate_axes_op(tensor, :reduce_min, type, opts)
  end

  defp aggregate_axes_op(%T{shape: shape, names: names} = tensor, op, type, opts) do
    opts = keyword!(opts, [:axes, keep_axes: false])
    keep_axes = opts[:keep_axes]

    {shape, names, axes} =
      cond do
        axes = opts[:axes] ->
          axes = Nx.Shape.normalize_axes(shape, axes, names)
          {new_shape, new_names} = Nx.Shape.contract(shape, axes, names, keep_axes)
          {new_shape, new_names, axes}

        keep_axes ->
          shape = Tuple.duplicate(1, Nx.rank(shape))
          {shape, names, nil}

        true ->
          {{}, [], nil}
      end

    apply(impl!(tensor), op, [
      %{tensor | type: type, shape: shape, names: names},
      tensor,
      [axes: axes, keep_axes: keep_axes]
    ])
  end

  @doc """
  Returns the indices of the maximum values.

  ## Options

    * `:axis` - the axis to aggregate on. If no axis is given,
      returns the index of the absolute maximum value in the tensor.

    * `:keep_axis` - whether or not to keep the reduced axis with
      a size of 1. Defaults to `false`.

    * `:tie_break` - how to break ties. one of `:high`, or `:low`.
      default behavior is to always return the lower index.

  ## Examples

      iex> Nx.argmax(4)
      #Nx.Tensor<
        s64
        0
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])
      iex> Nx.argmax(t)
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

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmax(t, axis: :x)
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [1, 0, 0],
          [1, 1, 0]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmax(t, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [0, 0, 0],
          [0, 1, 0]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmax(t, axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [0, 2],
          [0, 1]
        ]
      >

  ### Tie breaks

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmax(t, tie_break: :low, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [0, 0, 0],
          [0, 1, 0]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmax(t, tie_break: :high, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [0, 0, 1],
          [0, 1, 1]
        ]
      >

  ### Keep axis

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmax(t, axis: :y, keep_axis: true)
      #Nx.Tensor<
        s64[x: 2][y: 1][z: 3]
        [
          [
            [0, 0, 0]
          ],
          [
            [0, 1, 0]
          ]
        ]
      >
  """
  @doc type: :aggregation
  def argmax(tensor, opts \\ []) do
    argmin_or_max(tensor, :argmax, opts)
  end

  @doc """
  Returns the indices of the minimum values.

  ## Options

    * `:axis` - the axis to aggregate on. If no axis is given,
      returns the index of the absolute minimum value in the tensor.

    * `:keep_axis` - whether or not to keep the reduced axis with
      a size of 1. Defaults to `false`.

    * `:tie_break` - how to break ties. one of `:high`, or `:low`.
      Default behavior is to always return the lower index.

  ## Examples

      iex> Nx.argmin(4)
      #Nx.Tensor<
        s64
        0
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])
      iex> Nx.argmin(t)
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

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmin(t, axis: :x)
      #Nx.Tensor<
        s64[y: 2][z: 3]
        [
          [0, 0, 0],
          [0, 0, 0]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmin(t, axis: 1)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [1, 1, 0],
          [1, 0, 0]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmin(t, axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 2]
        [
          [1, 1],
          [1, 2]
        ]
      >

  ### Tie breaks

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmin(t, tie_break: :low, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [1, 1, 0],
          [1, 0, 0]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmin(t, tie_break: :high, axis: :y)
      #Nx.Tensor<
        s64[x: 2][z: 3]
        [
          [1, 1, 1],
          [1, 0, 1]
        ]
      >

  ### Keep axis

      iex> t = Nx.tensor([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      iex> Nx.argmin(t, axis: :y, keep_axis: true)
      #Nx.Tensor<
        s64[x: 2][y: 1][z: 3]
        [
          [
            [1, 1, 0]
          ],
          [
            [1, 0, 0]
          ]
        ]
      >
  """
  @doc type: :aggregation
  def argmin(tensor, opts \\ []) do
    argmin_or_max(tensor, :argmin, opts)
  end

  defp argmin_or_max(tensor, op, opts) do
    opts = keyword!(opts, [:axis, tie_break: :low, keep_axis: false])

    tie_break =
      case opts[:tie_break] do
        :high ->
          :high

        :low ->
          :low

        other ->
          raise ArgumentError,
                "unknown value for :tie_break, expected :high or :low, got: #{inspect(other)}"
      end

    %{shape: shape, names: names, type: type} = tensor = to_tensor(tensor)
    Nx.Shared.raise_complex_not_supported(type, op, 2)

    {shape, names, axis} =
      if axis = opts[:axis] do
        axis = Nx.Shape.normalize_axis(shape, axis, names)
        {new_shape, new_names} = Nx.Shape.contract(shape, [axis], names, opts[:keep_axis])
        {new_shape, new_names, axis}
      else
        {{}, [], nil}
      end

    out = %{tensor | type: {:s, 64}, shape: shape, names: names}
    opts = [tie_break: tie_break, axis: axis, keep_axis: opts[:keep_axis]]
    apply(impl!(tensor), op, [out, tensor, opts])
  end

  defp aggregate_window_op(tensor, window_dimensions, opts, op) when is_list(opts) do
    opts = keyword!(opts, [:window_dilations, padding: :valid, strides: 1])
    Nx.Shape.validate!(window_dimensions, :window_dimensions)
    %{shape: shape} = tensor = to_tensor(tensor)

    strides = opts[:strides]
    padding = opts[:padding]
    dilations = opts[:window_dilations] || List.duplicate(1, rank(shape))

    strides =
      if is_integer(strides),
        do: List.duplicate(strides, rank(shape)),
        else: strides

    dilations =
      if is_integer(dilations),
        do: List.duplicate(dilations, rank(shape)),
        else: dilations

    {output_shape, padding_config} =
      Nx.Shape.pool(shape, window_dimensions, strides, padding, dilations)

    out = %{tensor | shape: output_shape}
    opts = [padding: padding_config, strides: strides, window_dilations: dilations]
    apply(impl!(tensor), op, [out, tensor, window_dimensions, opts])
  end

  @doc """
  Sums over each window of size `window_dimensions` in the
  given tensor, producing a tensor that contains the same
  number of elements as valid positions of the window.

  You may optionally specify `:strides` which is a tuple
  of non-zero steps to take along each axis between
  each window.

  You may also optionally specify `:padding` which is either
  one of `:valid` (no padding) or `:same` (pad so output shape
  is the same as input shape) or a general padding configuration
  for each dimension in the input tensor. Your padding configuration
  cannot include any negative pad values. You may only specify
  padding for the high and low edges of the given dimension. Pads
  with `0`.

  ## Examples

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      iex> Nx.window_sum(t, {1, 2, 1})
      #Nx.Tensor<
        s64[2][1][3]
        [
          [
            [5, 7, 9]
          ],
          [
            [5, 7, 9]
          ]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      iex> Nx.window_sum(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])
      #Nx.Tensor<
        s64[2][2][2]
        [
          [
            [0, 0],
            [0, 18]
          ],
          [
            [0, 0],
            [0, 9]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      iex> Nx.window_sum(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])
      #Nx.Tensor<
        f32[2][2][5]
        [
          [
            [0.0, 4.0, 2.0, 3.0, 0.0],
            [0.0, 2.0, 5.0, 6.5, 0.0]
          ],
          [
            [0.0, 1.2000000476837158, 2.200000047683716, 3.200000047683716, 0.0],
            [0.0, 4.0, 5.0, 6.199999809265137, 0.0]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 1]]
      iex> Nx.window_sum(t, {1, 1, 2}, opts)
      #Nx.Tensor<
        s64[1][2][3]
        [
          [
            [6, 3, 4],
            [6, 3, 8]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 2]]
      iex> Nx.window_sum(t, {1, 1, 2}, opts)
      #Nx.Tensor<
        s64[1][2][2]
        [
          [
            [5, 5],
            [5, 9]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: [{2, 1}, {3, 1}, {1, 0}], window_dilations: [1, 2, 2]]
      iex> Nx.window_sum(t, {2, 1, 2}, opts)
      #Nx.Tensor<
        s64[2][6][3]
        [
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
          ],
          [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [4, 11, 14],
            [10, 15, 19],
            [0, 0, 0]
          ]
        ]
      >
  """
  @doc type: :window
  def window_sum(tensor, window_dimensions, opts \\ []),
    do: aggregate_window_op(tensor, window_dimensions, opts, :window_sum)

  @doc """
  Averages over each window of size `window_dimensions` in the
  given tensor, producing a tensor that contains the same
  number of elements as valid positions of the window.

  You may optionally specify `:strides` which is a tuple
  of non-zero steps to take along each axis between
  each window.

  You may also optionally specify `:padding` which is either
  one of `:valid` (no padding) or `:same` (pad so output shape
  is the same as input shape) or a general padding configuration
  for each dimension in the input tensor. Your padding configuration
  cannot include any negative pad values. You may only specify
  padding for the high and low edges of the given dimension. Pads
  with `0`.

  ## Examples

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      iex> Nx.window_mean(t, {1, 2, 1})
      #Nx.Tensor<
        f32[2][1][3]
        [
          [
            [2.5, 3.5, 4.5]
          ],
          [
            [2.5, 3.5, 4.5]
          ]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      iex> Nx.window_mean(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])
      #Nx.Tensor<
        f32[2][2][2]
        [
          [
            [0.0, 0.0],
            [0.0, 4.5]
          ],
          [
            [0.0, 0.0],
            [0.0, 2.25]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      iex> Nx.window_mean(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])
      #Nx.Tensor<
        f32[2][2][5]
        [
          [
            [0.0, 2.0, 1.0, 1.5, 0.0],
            [0.0, 1.0, 2.5, 3.25, 0.0]
          ],
          [
            [0.0, 0.6000000238418579, 1.100000023841858, 1.600000023841858, 0.0],
            [0.0, 2.0, 2.5, 3.0999999046325684, 0.0]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 1]]
      iex> Nx.window_mean(t, {1, 1, 2}, opts)
      #Nx.Tensor<
        f32[1][2][3]
        [
          [
            [3.0, 1.5, 2.0],
            [3.0, 1.5, 4.0]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 2]]
      iex> Nx.window_mean(t, {1, 1, 2}, opts)
      #Nx.Tensor<
        f32[1][2][2]
        [
          [
            [2.5, 2.5],
            [2.5, 4.5]
          ]
        ]
      >
  """
  @doc type: :window
  def window_mean(tensor, window_dimensions, opts \\ []) do
    divide(window_sum(tensor, window_dimensions, opts), size(window_dimensions))
  end

  @doc """
  Returns the maximum over each window of size `window_dimensions`
  in the given tensor, producing a tensor that contains the same
  number of elements as valid positions of the window.

  You may optionally specify `:strides` which is a tuple
  of non-zero steps to take along each axis between
  each window.

  You may also optionally specify `:padding` which is either
  one of `:valid` (no padding) or `:same` (pad so output shape
  is the same as input shape) or a general padding configuration
  for each dimension in the input tensor. Your padding configuration
  cannot include any negative pad values. You may only specify
  padding for the high and low edges of the given dimension. Pads
  with the minimum value for the type of the given tensor.

  ## Examples

      iex> Nx.window_max(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]), {1, 2, 1})
      #Nx.Tensor<
        s64[2][1][3]
        [
          [
            [4, 5, 6]
          ],
          [
            [4, 5, 6]
          ]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      iex> Nx.window_max(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])
      #Nx.Tensor<
        s64[2][2][2]
        [
          [
            [-9223372036854775808, -9223372036854775808],
            [-9223372036854775808, 6]
          ],
          [
            [-9223372036854775808, -9223372036854775808],
            [-9223372036854775808, 6]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      iex> Nx.window_max(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])
      #Nx.Tensor<
        f32[2][2][5]
        [
          [
            [-3.4028234663852886e38, 4.0, 2.0, 3.0, -3.4028234663852886e38],
            [-3.4028234663852886e38, 2.0, 5.0, 6.5, -3.4028234663852886e38]
          ],
          [
            [-3.4028234663852886e38, 1.2000000476837158, 2.200000047683716, 3.200000047683716, -3.4028234663852886e38],
            [-3.4028234663852886e38, 4.0, 5.0, 6.199999809265137, -3.4028234663852886e38]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 2]]
      iex> Nx.window_max(t, {1, 1, 2}, opts)
      #Nx.Tensor<
        s64[1][2][2]
        [
          [
            [4, 3],
            [4, 7]
          ]
        ]
      >
  """
  @doc type: :window
  def window_max(tensor, window_dimensions, opts \\ []) do
    tensor = to_tensor(tensor)
    Nx.Shared.raise_complex_not_supported(tensor.type, :window_max, 3)
    aggregate_window_op(tensor, window_dimensions, opts, :window_max)
  end

  @doc """
  Returns the minimum over each window of size `window_dimensions`
  in the given tensor, producing a tensor that contains the same
  number of elements as valid positions of the window.

  You may optionally specify `:strides` which is a tuple
  of non-zero steps to take along each axis between
  each window.

  You may also optionally specify `:padding` which is either
  one of `:valid` (no padding) or `:same` (pad so output shape
  is the same as input shape) or a general padding configuration
  for each dimension in the input tensor. Your padding configuration
  cannot include any negative pad values. You may only specify
  padding for the high and low edges of the given dimension. Pads
  with the maximum value for the type of the given tensor.

  ## Examples

      iex> Nx.window_min(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]), {1, 2, 1})
      #Nx.Tensor<
        s64[2][1][3]
        [
          [
            [1, 2, 3]
          ],
          [
            [1, 2, 3]
          ]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      iex> Nx.window_min(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])
      #Nx.Tensor<
        s64[2][2][2]
        [
          [
            [9223372036854775807, 9223372036854775807],
            [9223372036854775807, 3]
          ],
          [
            [9223372036854775807, 9223372036854775807],
            [9223372036854775807, 3]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      iex> Nx.window_min(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])
      #Nx.Tensor<
        f32[2][2][5]
        [
          [
            [3.4028234663852886e38, 4.0, 2.0, 3.0, 3.4028234663852886e38],
            [3.4028234663852886e38, 2.0, 5.0, 6.5, 3.4028234663852886e38]
          ],
          [
            [3.4028234663852886e38, 1.2000000476837158, 2.200000047683716, 3.200000047683716, 3.4028234663852886e38],
            [3.4028234663852886e38, 4.0, 5.0, 6.199999809265137, 3.4028234663852886e38]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 2]]
      iex> Nx.window_min(t, {1, 1, 2}, opts)
      #Nx.Tensor<
        s64[1][2][2]
        [
          [
            [1, 2],
            [1, 2]
          ]
        ]
      >
  """
  @doc type: :window
  def window_min(tensor, window_dimensions, opts \\ []) do
    tensor = to_tensor(tensor)
    Nx.Shared.raise_complex_not_supported(tensor.type, :window_min, 3)
    aggregate_window_op(tensor, window_dimensions, opts, :window_min)
  end

  @doc """
  Returns the product over each window of size `window_dimensions`
  in the given tensor, producing a tensor that contains the same
  number of elements as valid positions of the window.

  The rank of the input tensor and the window dimensions must
  match.

  You may optionally specify `:strides` which is a tuple
  of non-zero steps to take along each axis between
  each window.

  You may also optionally specify `:padding` which is either
  one of `:valid` (no padding) or `:same` (pad so output shape
  is the same as input shape) or a general padding configuration
  for each dimension in the input tensor. Your padding configuration
  cannot include any negative pad values. You may only specify
  padding for the high and low edges of the given dimension. Pads
  with 1.

  ## Examples

      iex> Nx.window_product(Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]), {1, 2, 1})
      #Nx.Tensor<
        s64[2][1][3]
        [
          [
            [4, 10, 18]
          ],
          [
            [4, 10, 18]
          ]
        ]
      >

      iex> t = Nx.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
      iex> Nx.window_product(t, {2, 2, 1}, strides: [1, 2, 3], padding: [{0, 1}, {2, 0}, {1, 1}])
      #Nx.Tensor<
        s64[2][2][2]
        [
          [
            [1, 1],
            [1, 324]
          ],
          [
            [1, 1],
            [1, 18]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4.0, 2.0, 3.0], [2.0, 5.0, 6.5]], [[1.2, 2.2, 3.2], [4.0, 5.0, 6.2]]])
      iex> Nx.window_product(t, {2, 1, 1}, strides: [2, 1, 1], padding: [{1, 1}, {0, 0}, {1, 1}])
      #Nx.Tensor<
        f32[2][2][5]
        [
          [
            [1.0, 4.0, 2.0, 3.0, 1.0],
            [1.0, 2.0, 5.0, 6.5, 1.0]
          ],
          [
            [1.0, 1.2000000476837158, 2.200000047683716, 3.200000047683716, 1.0],
            [1.0, 4.0, 5.0, 6.199999809265137, 1.0]
          ]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [strides: [2, 1, 1], padding: :valid, window_dilations: [1, 2, 2]]
      iex> Nx.window_product(t, {1, 1, 2}, opts)
      #Nx.Tensor<
        s64[1][2][2]
        [
          [
            [4, 6],
            [4, 14]
          ]
        ]
      >
  """
  @doc type: :window
  def window_product(tensor, window_dimensions, opts \\ []),
    do: aggregate_window_op(tensor, window_dimensions, opts, :window_product)

  @doc """
  Returns the cumulative sum of elements along an axis.

  ## Options

    * `:axis` - the axis to sum elements along. Defaults to `0`

  ## Examples

      iex> Nx.cumulative_sum(Nx.tensor([1, 2, 3, 4]))
      #Nx.Tensor<
        s64[4]
        [1, 3, 6, 10]
      >

      iex> Nx.cumulative_sum(Nx.iota({3, 3}), axis: 0)
      #Nx.Tensor<
        s64[3][3]
        [
          [0, 1, 2],
          [3, 5, 7],
          [9, 12, 15]
        ]
      >

      iex> Nx.cumulative_sum(Nx.iota({3, 3}), axis: 1)
      #Nx.Tensor<
        s64[3][3]
        [
          [0, 1, 3],
          [3, 7, 12],
          [6, 13, 21]
        ]
      >
  """
  @doc type: :cumulative
  def cumulative_sum(tensor, opts \\ []),
    do: cumulative_op(tensor, opts, :cumulative_sum, :window_sum)

  @doc """
  Returns the cumulative product of elements along an axis.

  ## Options

    * `:axis` - the axis to multiply elements along. Defaults to `0`

  ## Examples

      iex> Nx.cumulative_product(Nx.tensor([1, 2, 3, 4]))
      #Nx.Tensor<
        s64[4]
        [1, 2, 6, 24]
      >

      iex> Nx.cumulative_product(Nx.iota({3, 3}), axis: 0)
      #Nx.Tensor<
        s64[3][3]
        [
          [0, 1, 2],
          [0, 4, 10],
          [0, 28, 80]
        ]
      >

      iex> Nx.cumulative_product(Nx.iota({3, 3}), axis: 1)
      #Nx.Tensor<
        s64[3][3]
        [
          [0, 0, 0],
          [3, 12, 60],
          [6, 42, 336]
        ]
      >
  """
  @doc type: :cumulative
  def cumulative_product(tensor, opts \\ []),
    do: cumulative_op(tensor, opts, :cumulative_product, :window_product)

  @doc """
  Returns the cumulative minimum of elements along an axis.

  ## Options

    * `:axis` - the axis to compare elements along. Defaults to `0`

  ## Examples

      iex> Nx.cumulative_min(Nx.tensor([3, 4, 2, 1]))
      #Nx.Tensor<
        s64[4]
        [3, 3, 2, 1]
      >

      iex> Nx.cumulative_min(Nx.tensor([[2, 3, 1], [1, 3, 2], [2, 1, 3]]), axis: 0)
      #Nx.Tensor<
        s64[3][3]
        [
          [2, 3, 1],
          [1, 3, 1],
          [1, 1, 1]
        ]
      >

      iex> Nx.cumulative_min(Nx.tensor([[2, 3, 1], [1, 3, 2], [2, 1, 3]]), axis: 1)
      #Nx.Tensor<
        s64[3][3]
        [
          [2, 2, 1],
          [1, 1, 1],
          [2, 1, 1]
        ]
      >
  """
  @doc type: :cumulative
  def cumulative_min(tensor, opts \\ []),
    do: cumulative_op(tensor, opts, :cumulative_min, :window_min)

  @doc """
  Returns the cumulative maximum of elements along an axis.

  ## Options

    * `:axis` - the axis to compare elements along. Defaults to `0`

  ## Examples

      iex> Nx.cumulative_max(Nx.tensor([3, 4, 2, 1]))
      #Nx.Tensor<
        s64[4]
        [3, 4, 4, 4]
      >

      iex> Nx.cumulative_max(Nx.tensor([[2, 3, 1], [1, 3, 2], [2, 1, 3]]), axis: 0)
      #Nx.Tensor<
        s64[3][3]
        [
          [2, 3, 1],
          [2, 3, 2],
          [2, 3, 3]
        ]
      >

      iex> Nx.cumulative_max(Nx.tensor([[2, 3, 1], [1, 3, 2], [2, 1, 3]]), axis: 1)
      #Nx.Tensor<
        s64[3][3]
        [
          [2, 3, 3],
          [1, 3, 3],
          [2, 2, 3]
        ]
      >
  """
  @doc type: :cumulative
  def cumulative_max(tensor, opts \\ []),
    do: cumulative_op(tensor, opts, :cumulative_max, :window_max)

  defp cumulative_op(tensor, opts, op, window_op) do
    opts = keyword!(opts, axis: 0)
    tensor = to_tensor(tensor)
    axis = Nx.Shape.normalize_axis(tensor.shape, opts[:axis], tensor.names)

    Nx.Shared.optional(op, [tensor, axis], tensor, fn tensor, axis ->
      shape = shape(tensor)
      axis_size = elem(shape, axis)
      rank = rank(shape)

      padding =
        List.duplicate({0, 0}, rank)
        |> List.replace_at(axis, {axis_size - 1, 0})

      window_shape =
        List.duplicate(1, rank)
        |> List.to_tuple()
        |> put_elem(axis, axis_size)

      aggregate_window_op(tensor, window_shape, [padding: padding], window_op)
    end)
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
  cast to float, as done by most binary operators. You can
  also pass a `:type` option to change this behaviour.

  You may optionally set `:keep_axes` to true, which will
  retain the rank of the input tensor by setting the reduced
  axes to size 1.

  ## Limitations

  Given this function relies on anonymous functions, it
  may not be available or efficient on all Nx backends.
  Therefore, you should avoid using `reduce/4` whenever
  possible. Instead, use functions `sum/2`, `reduce_max/2`,
  `all/1`, and so forth.

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
        f32
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
  @doc type: :aggregation
  def reduce(tensor, acc, opts \\ [], fun) when is_function(fun, 2) do
    opts = keyword!(opts, [:axes, :type, keep_axes: false])
    type = Nx.Type.normalize!(opts[:type] || binary_type(tensor, acc))
    keep_axes = opts[:keep_axes]

    %{shape: shape, names: names} = tensor = to_tensor(tensor)
    acc = to_tensor(acc)

    {shape, names, axes} =
      if axes = opts[:axes] do
        axes = Nx.Shape.normalize_axes(shape, axes, names)
        {new_shape, new_names} = Nx.Shape.contract(shape, axes, names, keep_axes)
        {new_shape, new_names, axes}
      else
        if keep_axes do
          shape = Tuple.duplicate(1, Nx.rank(shape))
          {shape, names, nil}
        else
          {{}, [], nil}
        end
      end

    out = %{tensor | type: type, shape: shape, names: names}
    impl!(tensor).reduce(out, tensor, acc, [axes: axes, keep_axes: keep_axes], fun)
  end

  @doc """
  Reduces over each window of size `dimensions`
  in the given tensor, producing a tensor that contains the same
  number of elements as valid positions of the window.

  The rank of the input tensor and the window dimensions must
  match.

  You may optionally specify `:strides` which is a tuple
  of non-zero steps to take along each axis between
  each window.

  You may also optionally specify `:padding` which is either
  one of `:valid` (no padding) or `:same` (pad so output shape
  is the same as input shape) or a general padding configuration
  for each dimension in the input tensor. Your padding configuration
  cannot include any negative pad values. You may only specify
  padding for the high and low edges of the given dimension. The
  padding value is equal to the initial value passed to `acc`.

  The initial value must be a number or a scalar shaped tensor.

  ### Examples

      iex> init_value = Nx.Constants.min_finite({:s, 64})
      iex> t = Nx.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [11, 12, 13, 14]])
      iex> Nx.window_reduce(t, init_value, {2, 2}, fn x, acc -> Nx.max(x, acc) end)
      #Nx.Tensor<
        s64[3][3]
        [
          [5, 6, 7],
          [8, 9, 10],
          [12, 13, 14]
        ]
      >

      iex> init_value = Nx.Constants.min_finite({:s, 64})
      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      iex> opts = [padding: :same, strides: [1, 1]]
      iex> Nx.window_reduce(t, init_value, {2, 2}, opts, fn x, acc -> Nx.max(x, acc) end)
      #Nx.Tensor<
        s64[3][3]
        [
          [5, 6, 6],
          [8, 9, 9],
          [8, 9, 9]
        ]
      >

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> opts = [padding: :same, strides: [1, 1]]
      iex> Nx.window_reduce(t, 0, {1, 2}, opts, fn x, acc -> Nx.add(x, acc) end)
      #Nx.Tensor<
        s64[2][3]
        [
          [3, 5, 3],
          [9, 11, 6]
        ]
      >

      iex> t = Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]])
      iex> opts = [padding: :valid, strides: [2, 1, 1], window_dilations: [1, 1, 2]]
      iex> Nx.window_reduce(t, 0, {1, 1, 2}, opts, fn x, acc -> Nx.add(x, acc) end)
      #Nx.Tensor<
        s64[1][2][2]
        [
          [
            [5, 5],
            [5, 9]
          ]
        ]
      >
  """
  @doc type: :window
  def window_reduce(tensor, acc, window_dimensions, opts \\ [], fun)
      when is_tuple(window_dimensions) do
    opts = keyword!(opts, [:window_dilations, :strides, padding: :valid])
    %T{shape: shape} = tensor = to_tensor(tensor)
    acc = to_tensor(acc)

    padding = opts[:padding]
    strides = opts[:strides] || List.duplicate(1, rank(tensor.shape))
    dilations = opts[:window_dilations] || List.duplicate(1, rank(tensor.shape))

    dilations =
      if is_integer(dilations),
        do: List.duplicate(dilations, rank(tensor.shape)),
        else: dilations

    strides =
      if is_integer(strides),
        do: List.duplicate(strides, rank(tensor.shape)),
        else: strides

    {output_shape, padding_config} =
      Nx.Shape.pool(shape, window_dimensions, strides, padding, dilations)

    out = %{tensor | shape: output_shape}
    opts = [padding: padding_config, strides: strides, window_dilations: dilations]
    impl!(tensor).window_reduce(out, tensor, acc, window_dimensions, opts, fun)
  end

  @doc """
  Maps the given scalar function over the entire
  tensor.

  The type of the returned tensor will be of the same type
  as the input tensor, unless the `:type` option is given.
  Therefore, you may need to explicitly cast the tensor to
  avoid errors. For example, if you have an integer tensor
  and you convert it to a float, as below, it will fail:

      tensor = Nx.tensor([[1, 2, 3], [4, 5, 6]]),
      Nx.map(tensor, fn x -> Nx.multiply(x, 1.0) end)

  You need to explicitly pass the output type in such cases:

      iex> tensor = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.map(tensor, [type: {:f, 32}], fn x -> Nx.multiply(x, 1.0) end)
      #Nx.Tensor<
        f32[2][3]
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]
        ]
      >

  ## Limitations

  Given this function relies on anonymous functions, it
  may not be available or efficient on all Nx backends.
  Therefore, you should avoid using `map/2` whenever possible
  and use other functions in the `Nx` module to achieve the
  desired result.

  ### Examples

      iex> Nx.map(Nx.tensor([[1, 2, 3], [4, 5, 6]]), fn x -> Nx.add(x, 1) end)
      #Nx.Tensor<
        s64[2][3]
        [
          [2, 3, 4],
          [5, 6, 7]
        ]
      >

      iex> Nx.map(Nx.tensor(1), fn x -> Nx.add(x, 1) end)
      #Nx.Tensor<
        s64
        2
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
  @doc type: :element
  def map(tensor, opts \\ [], fun) do
    %T{type: type} = tensor = to_tensor(tensor)
    opts = keyword!(opts, type: type)
    output_type = Nx.Type.normalize!(opts[:type])
    out = %{tensor | type: output_type}
    impl!(tensor).map(out, tensor, opts, fun)
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
        f32
        -10.0
      >

      iex> Nx.dot(2, 2.0)
      #Nx.Tensor<
        f32
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
        f32
        39.0
      >

      iex> Nx.dot(Nx.tensor([1.0, 2.0, 3.0]), Nx.tensor([1, 2, 3]))
      #Nx.Tensor<
        f32
        14.0
      >

  ### Dot product of matrices

      iex> left = Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:i, :j])
      iex> right = Nx.tensor([[7, 8], [9, 10], [11, 12]], names: [:x, :y])
      iex> Nx.dot(left, right)
      #Nx.Tensor<
        s64[i: 2][y: 2]
        [
          [58, 64],
          [139, 154]
        ]
      >

      iex> left = Nx.tensor([[10.0, 13.0, 14.0, 15.0], [59.0, 20.0, 10.0, 30.0]], names: [:i, :j])
      iex> right = Nx.tensor([[2.0, 4.0], [5.0, 1.0], [6.0, 8.0], [9.0, 10.0]], names: [:x, :y])
      iex> Nx.dot(left, right)
      #Nx.Tensor<
        f32[i: 2][y: 2]
        [
          [304.0, 315.0],
          [548.0, 636.0]
        ]
      >

      iex> left = Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:i, :j])
      iex> right = Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], names: [:x, :y])
      iex> Nx.dot(left, right)
      #Nx.Tensor<
        f32[i: 2][y: 2]
        [
          [58.0, 64.0],
          [139.0, 154.0]
        ]
      >

  ### Dot product of vector and n-d tensor

      iex> left = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], names: [:i, :j, :k])
      iex> right = Nx.tensor([5, 10], names: [:x])
      iex> Nx.dot(left, right)
      #Nx.Tensor<
        s64[i: 2][j: 2]
        [
          [25, 55],
          [85, 115]
        ]
      >

      iex> left = Nx.tensor([5, 10], names: [:x])
      iex> right = Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:i, :j])
      iex> Nx.dot(left, right)
      #Nx.Tensor<
        s64[j: 3]
        [45, 60, 75]
      >

      iex> left = Nx.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], names: [:shard, :batch, :x, :y, :z])
      iex> right = Nx.tensor([2.0, 2.0], names: [:data])
      iex> Nx.dot(left, right)
      #Nx.Tensor<
        f32[shard: 1][batch: 1][x: 2][y: 2]
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

      iex> left = Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:x, :y, :z])
      iex> right = Nx.tensor([[[1, 2, 3], [3, 4, 5], [5, 6, 7]]], names: [:i, :j, :k])
      iex> Nx.dot(left, right)
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
  @doc type: :ndim
  def dot(t1, t2) do
    %T{shape: s1} = t1 = to_tensor(t1)
    %T{shape: s2} = t2 = to_tensor(t2)

    case {tuple_size(s1), tuple_size(s2)} do
      {0, _} -> multiply(t1, t2)
      {_, 0} -> multiply(t1, t2)
      {n, 1} -> dot(t1, [n - 1], [], t2, [0], [])
      {1, m} -> dot(t1, [0], [], t2, [m - 2], [])
      {n, m} when n >= 2 and m >= 2 -> dot(t1, [n - 1], [], t2, [m - 2], [])
    end
  end

  @doc """
  Computes the generalized dot product between two tensors, given
  the contracting axes.

  This is equivalent to calling `Nx.dot/6` with no batching dimensions:

      Nx.dot(t1, contract_axes1, [], t2, contract_axes2, [])

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

      iex> t1 = Nx.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
      iex> t2 = Nx.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
      iex> Nx.dot(t1, [0, 1], t2, [1, 0])
      #Nx.Tensor<
        f32
        50.0
      >

  """
  @doc type: :ndim
  def dot(t1, contract_axes1, t2, contract_axes2) do
    dot(t1, contract_axes1, [], t2, contract_axes2, [])
  end

  @doc """
  Computes the generalized dot product between two tensors, given
  the contracting and batch axes.

  The dot product is computed by multiplying the values from `t1`
  given by `contract_axes1` against the values from `t2` given by
  `contract_axes2`, considering batch axes of `batch_axes1` and
  `batch_axes2`. For instance, the first axis in `contract_axes1`
  will be matched against the first axis in `contract_axes2` and
  so on. The axes given by `contract_axes1` and `contract_axes2`
  are effectively removed from the final tensor, which is why they
  are often called the contraction axes.

  If no contracting axes are given, the final product works like
  `Nx.outer/2`.

  Specifying batch axes will compute a vectorized dot product
  along the given batch dimensions. The length of `batch_axes1`
  and `batch_axes2` must match. Additionally, `batch_axes1` and
  `batch_axes2` must be a list of successive dimension numbers,
  where each batch axis matches the dimension of the corresponding
  batch axis in the other input.

  The contracting axes must be dot-product compatible and the
  batch dimensions must always have the same number of elements.

  ## Examples

  ### Contracting along axes

      iex> t1 = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      iex> t2 = Nx.tensor([[10, 20], [30, 40]], names: [:height, :width])
      iex> Nx.dot(t1, [0], [], t2, [0], [])
      #Nx.Tensor<
        s64[y: 2][width: 2]
        [
          [100, 140],
          [140, 200]
        ]
      >
      iex> Nx.dot(t1, [0], [], t2, [1], [])
      #Nx.Tensor<
        s64[y: 2][height: 2]
        [
          [70, 150],
          [100, 220]
        ]
      >
      iex> Nx.dot(t1, [1], [], t2, [0], [])
      #Nx.Tensor<
        s64[x: 2][width: 2]
        [
          [70, 100],
          [150, 220]
        ]
      >
      iex> Nx.dot(t1, [1], [], t2, [1], [])
      #Nx.Tensor<
        s64[x: 2][height: 2]
        [
          [50, 110],
          [110, 250]
        ]
      >
      iex> Nx.dot(t1, [0, 1], [], t2, [0, 1], [])
      #Nx.Tensor<
        s64
        300
      >

  If no axes are given, it works like `outer/2`:

      iex> t1 = Nx.tensor([[1, 2], [3, 4]])
      iex> t2 = Nx.tensor([[10, 20], [30, 40]])
      iex> Nx.dot(t1, [], [], t2, [], [])
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

  ### Dot product between two batched tensors

      iex> u = Nx.tensor([[[1]], [[2]]])
      iex> v = Nx.tensor([[[3]], [[4]]])
      iex> Nx.dot(u, [2], [0], v, [2], [0])
      #Nx.Tensor<
        s64[2][1][1]
        [
          [
            [3]
          ],
          [
            [8]
          ]
        ]
      >

      iex> u = Nx.tensor([[[1, 1]], [[2, 2]]])
      iex> v = Nx.tensor([[[3], [3]], [[4], [4]]])
      iex> Nx.dot(u, [2], [0], v, [1], [0])
      #Nx.Tensor<
        s64[2][1][1]
        [
          [
            [6]
          ],
          [
            [16]
          ]
        ]
      >

  ### Error cases

      iex> u = Nx.tensor([[[1, 1]], [[2, 2]]])
      iex> v = Nx.tensor([[[3], [3]], [[4], [4]]])
      iex> Nx.dot(u, [2], [0], v, [1], [])
      ** (ArgumentError) right tensor must be batched if left tensor is batched

      iex> u = Nx.tensor([[[1, 1]], [[2, 2]]])
      iex> v = Nx.tensor([[[3], [3]], [[4], [4]]])
      iex> Nx.dot(u, [2], [], v, [1], [0])
      ** (ArgumentError) left tensor must be batched if right tensor is batched

      iex> u = Nx.tensor([[[1, 1]], [[2, 2]]])
      iex> v = Nx.tensor([[[3], [3]], [[4], [4]]])
      iex> Nx.dot(u, [2], [1], v, [1], [0])
      ** (ArgumentError) invalid dot batch axis for the left tensor, batch axes must be successive dimensions starting from 0, got [1]

      iex> u = Nx.tensor([[[1, 1]], [[2, 2]]])
      iex> v = Nx.tensor([[[3], [3]], [[4], [4]]])
      iex> Nx.dot(u, [2], [0], v, [1], [1])
      ** (ArgumentError) invalid dot batch axis for the right tensor, batch axes must be successive dimensions starting from 0, got [1]

      iex> u = Nx.tensor([[[1, 1]], [[2, 2]]])
      iex> v = Nx.tensor([[[3], [3]], [[4], [4]]])
      iex> Nx.dot(u, [0], [0], v, [1], [0])
      ** (ArgumentError) dot batch axes for left tensor ([0]) cannot be in contract axes ([0])

      iex> u = Nx.tensor([[[1, 1]], [[2, 2]]])
      iex> v = Nx.tensor([[[3], [3]], [[4], [4]]])
      iex> Nx.dot(u, [2], [0], v, [0], [0])
      ** (ArgumentError) dot batch axes for right tensor ([0]) cannot be in contract axes ([0])
  """
  @doc type: :ndim
  def dot(t1, contract_axes1, batch_axes1, t2, contract_axes2, batch_axes2) do
    %{shape: s1, names: names1} = t1 = to_tensor(t1)
    %{shape: s2, names: names2} = t2 = to_tensor(t2)

    output_type = binary_type(t1, t2)

    # Axes normalization
    c1 = Nx.Shape.normalize_axes(s1, contract_axes1, names1)
    c2 = Nx.Shape.normalize_axes(s2, contract_axes2, names2)
    b1 = Nx.Shape.normalize_axes(s1, batch_axes1, names1)
    b2 = Nx.Shape.normalize_axes(s2, batch_axes2, names2)

    {output_shape, output_names} = Nx.Shape.dot(s1, c1, names1, b1, s2, c2, names2, b2)

    out = %{t1 | type: output_type, names: output_names, shape: output_shape}
    impl!(t1, t2).dot(out, t1, c1, b1, t2, c2, b2)
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
  @doc type: :ndim
  def outer(t1, t2) do
    %{names: n1} = t1 = to_tensor(t1)
    %{names: n2} = t2 = to_tensor(t2)

    names =
      case {n1, n2} do
        {[], rhs} -> [nil, List.last(rhs)]
        {lhs, rhs} -> [hd(lhs), List.last(rhs)]
      end

    %{multiply(reshape(t1, {size(t1), 1}), reshape(t2, {1, size(t2)})) | names: names}
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
  @doc type: :shape
  def transpose(tensor, opts \\ []) do
    opts = keyword!(opts, [:axes])
    %{shape: shape, names: names} = tensor = to_tensor(tensor)
    axes = opts[:axes] || Nx.Shape.transpose_axes(shape)
    axes = Nx.Shape.normalize_axes(shape, axes, names)

    if axes == Nx.axes(shape) do
      tensor
    else
      {shape, names} = Nx.Shape.transpose(shape, axes, names)
      impl!(tensor).transpose(%{tensor | shape: shape, names: names}, tensor, axes)
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
  @doc type: :ndim
  def reverse(tensor, opts \\ []) do
    opts = keyword!(opts, [:axes])
    %{shape: shape, names: names} = tensor = to_tensor(tensor)
    axes = opts[:axes] || axes(shape)

    case Nx.Shape.normalize_axes(shape, axes, names) do
      [] -> tensor
      axes -> impl!(tensor).reverse(tensor, tensor, Enum.sort(axes))
    end
  end

  ## Conv

  @doc """
  Computes an n-D convolution (where `n >= 3`) as used in neural networks.

  This function can be thought of as sliding an n-D
  kernel across the input, producing a new tensor that
  has the same number of elements as the number of valid
  windows in the input tensor. Each element is the result
  of summing the element-wise products in the window across
  each input channel.

  The ranks of both `input` and `kernel` must match. By
  default, both `input` and `kernel` are expected to have shapes
  of the following form:

    * `input` - `{batch_size, input_channels, input_d0, ..., input_dn}`
    * `kernel` - `{output_channels, input_channels, kernel_d0, ..., kernel_dn}`

  Where `input_d0...input_dn` and `kernel_d0...kernel_dn` represent
  an arbitrary number of spatial dimensions. You can alter this configuration
  using one of the `*_permutation` configuration options. Permutations
  are input, kernel, and output specifications for the layout of the
  convolution. For example, if your input tensor is configured with
  "channels last", you can specify the input permutation with:

      Nx.conv(img, kernel, input_permutation: [0, 3, 1, 2])

  Permutations expect configurations that specify the location of
  dimensions in the following orders:

    * `input_permutation` - `[batch_dim, input_channel_dim, ...spatial_dims...]`
    * `kernel_permutation` - `[output_channel_dim, input_channel_dim, ...spatial_dims...]`
    * `output_permutation` - `[batch_dim, output_channel_dim, ...spatial_dims...]`

  Using named tensors, it's a bit easier to see how permutations
  help you configure the convolution. Given input tensor with names
  `[:batch, :height, :width, :channels]` (channels last) and kernel
  tensor with names `[:input, :output, :height, :width]`, you can
  configure the convolution with the following permutations:

      Nx.conv(img, kernel,
        input_permutation: [:batch, :channels, :height, :width],
        kernel_permutation: [:output, :input, :height, :width],
        output_permutation: [:batch, :channels, :height, :width]
      )

  Notice that `output_permutation` is normalized with respect to
  the input permutation names. We cannot guarantee that every
  permutation is supported in every backend or compiler.

  To configure how the window slides along the input tensor, you
  can specify `:strides`. `:strides` must be a positive integer
  or tuple of positive integers for each spatial dimension
  in the input and kernel. For each spatial dimension, the
  window will slide by the configuration specified in `:strides`.
  As an example, for a 2-D convolution with `strides: [2, 1]`,
  the window will slide 2 positions along the first spatial
  dimension until it reaches the end of the dimension and then
  1 position along the second spatial dimension.

  You may specify a padding configuration using `:padding`,
  which will zero-pad the input tensor. Acceptable padding
  configurations are:

    * `:valid` - no padding
    * `:same` - pad input spatial dimensions such that they
    will remain unchanged in the output tensor
    * `[{d0_hi, d0_lo}, ..., {dn_hi, dn_lo}]` - a general padding
    configuration of edge high and edge low padding values. You
    may only specify padding for the edges of spatial dimensions
    of the input tensor. Padding values may be negative.

  You can dilate convolutions by setting `:input_dilation` or
  `:kernel_dilation`. Both `:input_dilation` and `:kernel_dilation`
  must either be positive integers or tuples of positive integers
  for each spatial dimension in the input and kernel tensors. Dilations
  can be thought of as applying `dilation - 1` interior padding to the
  input or kernel tensor.

  You can split both the input and kernel tensor into feature groups
  using `:feature_group_size`. This will split both the input and kernel
  tensor channels and compute a grouped convolution. The size of the
  kernel input feature channels times the size of the feature group must
  match the size of the input tensor feature channels. Additionally,
  the size of the kernel output feature channels must be evenly divisible
  by the group size.

  You can also split the input tensor along the batch dimension by
  specifying `:batch_group_size`. This will compute a grouped convolution
  in the same way as with `:feature_group_size`, however, the input
  tensor will be split into groups along the batch dimension.

  ## Examples

      iex> left = Nx.iota({9})
      iex> left = Nx.reshape(left, {1, 1, 3, 3})
      iex> right = Nx.iota({4})
      iex> right = Nx.reshape(right, {4, 1, 1, 1})
      iex> Nx.conv(left, right, strides: [1, 1])
      #Nx.Tensor<
        f32[1][4][3][3]
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

      iex> left = Nx.iota({9})
      iex> left = Nx.reshape(left, {1, 1, 3, 3})
      iex> right = Nx.iota({8})
      iex> right = Nx.reshape(right, {4, 1, 2, 1})
      iex> Nx.conv(left, right, strides: 2, padding: :same, kernel_dilation: [2, 1])
      #Nx.Tensor<
        f32[1][4][2][2]
        [
          [
            [
              [3.0, 5.0],
              [0.0, 0.0]
            ],
            [
              [9.0, 15.0],
              [6.0, 10.0]
            ],
            [
              [15.0, 25.0],
              [12.0, 20.0]
            ],
            [
              [21.0, 35.0],
              [18.0, 30.0]
            ]
          ]
        ]
      >

  """
  @doc type: :ndim
  def conv(tensor, kernel, opts \\ []) when is_list(opts) do
    opts =
      keyword!(opts, [
        :input_permutation,
        :kernel_permutation,
        :output_permutation,
        padding: :valid,
        strides: 1,
        input_dilation: 1,
        kernel_dilation: 1,
        feature_group_size: 1,
        batch_group_size: 1
      ])

    type = binary_type(tensor, kernel) |> Nx.Type.to_floating()
    padding = opts[:padding]
    strides = opts[:strides]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    feature_group_count = opts[:feature_group_size]
    batch_group_count = opts[:batch_group_size]

    %{shape: input_shape, names: input_names} = tensor = to_tensor(tensor)
    %{shape: kernel_shape, names: kernel_names} = kernel = to_tensor(kernel)
    Nx.Shape.validate_conv!(input_shape, kernel_shape)

    input_permutation = opts[:input_permutation] || axes(input_shape)
    input_permutation = Nx.Shape.normalize_axes(input_shape, input_permutation, input_names)
    kernel_permutation = opts[:kernel_permutation] || axes(kernel_shape)
    kernel_permutation = Nx.Shape.normalize_axes(kernel_shape, kernel_permutation, kernel_names)
    output_permutation = opts[:output_permutation] || axes(input_shape)
    output_permutation = Nx.Shape.normalize_axes(input_shape, output_permutation, input_names)

    strides =
      if is_integer(strides),
        do: List.duplicate(strides, Nx.rank(input_shape) - 2),
        else: strides

    cond do
      !is_integer(input_dilation) and !is_list(input_dilation) ->
        raise ArgumentError,
              "input dilation must be a positive integer or list of positive integers, got " <>
                inspect(input_dilation)

      !is_integer(kernel_dilation) and !is_list(kernel_dilation) ->
        raise ArgumentError,
              "kernel dilation must be a positive integer or list of positive integers, got " <>
                inspect(kernel_dilation)

      true ->
        :ok
    end

    input_dilation =
      if is_list(input_dilation),
        do: input_dilation,
        else: for(_ <- 1..(Nx.rank(input_shape) - 2), do: input_dilation)

    kernel_dilation =
      if is_list(kernel_dilation),
        do: kernel_dilation,
        else: for(_ <- 1..(Nx.rank(kernel_shape) - 2), do: kernel_dilation)

    {shape, names, padding_config} =
      Nx.Shape.conv(
        input_shape,
        input_names,
        kernel_shape,
        kernel_names,
        strides,
        padding,
        feature_group_count,
        batch_group_count,
        input_dilation,
        kernel_dilation,
        input_permutation,
        kernel_permutation,
        output_permutation
      )

    out = %{tensor | type: type, shape: shape, names: names}

    impl!(tensor).conv(
      out,
      tensor,
      kernel,
      strides: strides,
      padding: padding_config,
      input_dilation: input_dilation,
      kernel_dilation: kernel_dilation,
      feature_group_size: feature_group_count,
      batch_group_size: batch_group_count,
      input_permutation: input_permutation,
      kernel_permutation: kernel_permutation,
      output_permutation: output_permutation
    )
  end

  @doc """
  Clips the values of the tensor on the closed
  interval `[min, max]`.

  You can pass a tensor to `min` or `max` as long
  as the tensor has a scalar shape.

  ### Examples

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y])
      iex> Nx.clip(t, 2, 4)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [2, 2, 3],
          [4, 4, 4]
        ]
      >

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y])
      iex> Nx.clip(t, 2.0, 3)
      #Nx.Tensor<
        f32[x: 2][y: 3]
        [
          [2.0, 2.0, 3.0],
          [3.0, 3.0, 3.0]
        ]
      >

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]], names: [:x, :y])
      iex> Nx.clip(t, Nx.tensor(2.0), Nx.max(1.0, 3.0))
      #Nx.Tensor<
        f32[x: 2][y: 3]
        [
          [2.0, 2.0, 3.0],
          [3.0, 3.0, 3.0]
        ]
      >

      iex> t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names: [:x, :y])
      iex> Nx.clip(t, 2, 6.0)
      #Nx.Tensor<
        f32[x: 2][y: 3]
        [
          [2.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]
        ]
      >

      iex> t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: {:f, 32}, names: [:x, :y])
      iex> Nx.clip(t, 1, 4)
      #Nx.Tensor<
        f32[x: 2][y: 3]
        [
          [1.0, 2.0, 3.0],
          [4.0, 4.0, 4.0]
        ]
      >
  """
  @doc type: :element
  def clip(tensor, min, max) do
    %T{type: type} = tensor = to_tensor(tensor)
    %T{type: min_type, shape: min_shape} = min = to_tensor(min)
    %T{type: max_type, shape: max_shape} = max = to_tensor(max)

    if min_shape != {} do
      raise ArgumentError, "min value must be a scalar shape, got: #{inspect(min_shape)}"
    end

    if max_shape != {} do
      raise ArgumentError, "max value must be a scalar shape, got: #{inspect(max_shape)}"
    end

    output_type = Nx.Type.merge(type, Nx.Type.merge(min_type, max_type))

    Nx.Shared.raise_complex_not_supported(output_type, :clip, 2)

    impl!(tensor).clip(%{tensor | type: output_type}, tensor, min, max)
  end

  @doc """
  Slices a tensor from `start_indices` with `lengths`.

  You can optionally provide a `stride` to specify the amount
  of stride in each dimension.

  Both start indices and lengths must match the rank of the
  input tensor shape. All start indexes must be greater than
  or equal to zero. All lengths must be strictly greater than
  zero. `start_index + length` must not exceed the respective
  tensor dimension.

  It is possible for `start_indices` to be a list of tensors.
  However, `lengths` must always be a list of integers. If you
  want to specify a tensor as the list of indices, see `take/3`.

  If the `:strides` is given, it must be strictly greater than zero.
  The resulting tensor will have the shape of `length` unless
  `:strides` are given.

  It is not possible to slice in reverse. See `gather/2`,
  `slice_along_axis/4`, `take/3`, and `take_along_axis/3` for other ways
  to retrieve values from a tensor.

  ### Examples

      iex> Nx.slice(Nx.tensor([1, 2, 3, 4, 5, 6]), [0], [3])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.slice(Nx.tensor([1, 2, 3, 4, 5, 6]), [0], [6], strides: [2])
      #Nx.Tensor<
        s64[3]
        [1, 3, 5]
      >

      iex> Nx.slice(Nx.tensor([[1, 2], [3, 4], [5, 6]]), [0, 0], [3, 2], strides: [2, 1])
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [5, 6]
        ]
      >

  Strides can also be a number that applies to all dimensions:

      iex> t = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      iex> Nx.slice(t, [0, 0], [3, 2], strides: 2)
      #Nx.Tensor<
        s64[2][1]
        [
          [1],
          [5]
        ]
      >

  A more complex example:

      iex> t = Nx.iota({900})
      iex> t = Nx.reshape(t, {2, 15, 30})
      iex> Nx.slice(t, [0, 4, 11], [2, 3, 9], strides: [2, 1, 3])
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

  The `start_indices` list can be made of scalar tensors:

      iex> Nx.slice(Nx.tensor([[1, 2, 3], [4, 5, 6]]), [Nx.tensor(1), Nx.tensor(2)], [1, 1])
      #Nx.Tensor<
        s64[1][1]
        [
          [6]
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
      iex> Nx.slice(t, [Nx.tensor(0), Nx.tensor(0)], [6, 7], strides: [5, 3])
      #Nx.Tensor<
        f32[2][3]
        [
          [0.0, 0.0, 0.0],
          [1.0, 1.0, 1.0]
        ]
      >

  ### Error cases

      iex> Nx.slice(Nx.tensor([[1, 2, 3], [4, 5, 6]]), [Nx.tensor([1, 2]), Nx.tensor(1)], [1, 1])
      ** (ArgumentError) index must be scalar, got shape {2} for axis 0

      iex> Nx.slice(Nx.tensor([[1, 2, 3], [4, 5, 6]]), [Nx.tensor(1.0), Nx.tensor(0)], [1, 1])
      ** (ArgumentError) index must be integer type, got {:f, 32} for axis 0
  """
  @doc type: :indexed
  def slice(tensor, start_indices, lengths, opts \\ [])
      when is_list(start_indices) and is_list(lengths) and is_list(opts) do
    opts = keyword!(opts, strides: 1)
    %T{shape: shape} = tensor = to_tensor(tensor)
    strides = opts[:strides]

    start_indices = to_indices(start_indices)

    strides =
      if is_integer(strides),
        do: List.duplicate(strides, rank(shape)),
        else: strides

    output_shape = Nx.Shape.slice(shape, start_indices, lengths, strides)
    out = %{tensor | shape: output_shape}
    impl!(tensor).slice(out, tensor, start_indices, lengths, strides)
  end

  @doc """
  Slices a tensor along the given axis.

  You can optionally provide a `stride` to specify the amount
  of stride in along the given dimension.

  Start index must be greater than or equal to zero. It can be an
  integer or a scalar tensor. Length must be strictly greater than
  zero. `start_index + length` must not exceed the respective tensor
  dimension.

  The axis will be normalized with the dimensions and names of the
  given tensor.

  If the `:strides` is given, it must be strictly greater than zero.

  It is not possible to slice in reverse. See `gather/2`, `slice/3`,
  `take/3`, and `take_along_axis/3` for other ways to retrieve values
  from a tensor.

  ## Options

    * `:axis` - The axis along which to take the values from. Defaults to `0`.
    * `:strides` - The stride to slice the axis along of. Defaults to `1`.

  ## Examples

      iex> Nx.slice_along_axis(Nx.iota({5, 2}), 1, 2, axis: 0)
      #Nx.Tensor<
        s64[2][2]
        [
          [2, 3],
          [4, 5]
        ]
      >

      iex> Nx.slice_along_axis(Nx.iota({2, 5}), 1, 2, axis: 1)
      #Nx.Tensor<
        s64[2][2]
        [
          [1, 2],
          [6, 7]
        ]
      >

      iex> Nx.slice_along_axis(Nx.iota({2, 5}, names: [:x, :y]), 0, 1, axis: :x)
      #Nx.Tensor<
        s64[x: 1][y: 5]
        [
          [0, 1, 2, 3, 4]
        ]
      >

      iex> Nx.slice_along_axis(Nx.iota({2, 5}, names: [:x, :y]), Nx.tensor(0), 1, axis: :x)
      #Nx.Tensor<
        s64[x: 1][y: 5]
        [
          [0, 1, 2, 3, 4]
        ]
      >

      iex> Nx.slice_along_axis(Nx.iota({2, 5}), 0, 3, axis: -1, strides: 2)
      #Nx.Tensor<
        s64[2][2]
        [
          [0, 2],
          [5, 7]
        ]
      >

  """
  @doc type: :indexed, from_backend: false
  def slice_along_axis(tensor, start_index, len, opts \\ []) when is_integer(len) do
    opts = keyword!(opts, strides: 1, axis: 0)
    axis = Keyword.fetch!(opts, :axis)
    strides = Keyword.fetch!(opts, :strides)
    %T{shape: shape, names: names} = tensor = to_tensor(tensor)
    axis = Nx.Shape.normalize_axis(shape, axis, names)
    rank = rank(shape)

    start_indices = List.duplicate(0, rank) |> List.replace_at(axis, start_index)
    lengths = shape |> put_elem(axis, len) |> Tuple.to_list()
    strides = List.duplicate(1, rank) |> List.replace_at(axis, strides)
    slice(tensor, start_indices, lengths, strides: strides)
  end

  @doc false
  @deprecated "Use slice_along_axis/4 instead"
  def slice_axis(tensor, start_index, len, axis, opts \\ []) when is_integer(len) do
    slice_along_axis(tensor, start_index, len, [axis: axis] ++ opts)
  end

  @doc false
  @deprecated "Use sigmoid/1 instead"
  def logistic(tensor) do
    sigmoid(tensor)
  end

  @doc """
  Puts the given slice into the given tensor at the given
  start indices.

  The given slice shape must be less than or equal to the
  shape of the given tensor. All start indices must be less
  than their respective dimensions.

  ## Examples

      iex> t = Nx.tensor([0, 1, 2, 3, 4])
      iex> Nx.put_slice(t, [2], Nx.tensor([5, 6]))
      #Nx.Tensor<
        s64[5]
        [0, 1, 5, 6, 4]
      >

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.put_slice(t, [1, 2], Nx.tensor([[7, 8], [9, 10]]))
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 7, 8],
          [4, 9, 10]
        ]
      >

      iex> t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      iex> Nx.put_slice(t, [2, 2], Nx.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0]
        ]
      >

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.put_slice(t, [Nx.tensor(0), Nx.tensor(2)], Nx.tensor([[10.0, 11.0]]))
      #Nx.Tensor<
        f32[2][3]
        [
          [1.0, 10.0, 11.0],
          [4.0, 5.0, 6.0]
        ]
      >
  """
  @doc type: :indexed
  def put_slice(tensor, start_indices, slice) when is_list(start_indices) do
    %T{shape: shape, names: names, type: type} = tensor = to_tensor(tensor)
    %T{shape: slice_shape, names: slice_names, type: slice_type} = slice = to_tensor(slice)

    output_type = binary_type(type, slice_type)

    start_indices = to_indices(start_indices)

    {shape, names} = Nx.Shape.put_slice(shape, names, slice_shape, slice_names, start_indices)

    impl!(tensor).put_slice(
      %{tensor | shape: shape, names: names, type: output_type},
      tensor,
      start_indices,
      slice
    )
  end

  @doc """
  Takes and concatenates slices along an axis.

  Intuitively speaking, `take/3` reorders tensor slices along
  the given axis based on the given indices, possibly duplicating
  and removing slices.

  Passing a multi-dimensional indices tensor only affects the
  resulting shape. Specif
  ically, the given axis in the input shape
  gets replaced with the indices shape.

  See `gather/2`, `slice/3`, `slice_along_axis/4`, and `take_along_axis/3`
  for other ways to retrieve values from a tensor.

  ## Options

    * `:axis` - an axis to take tensor slices over. Defaults to 0.

  ## Examples

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> Nx.take(t, Nx.tensor([1, 0, 1]))
      #Nx.Tensor<
        s64[3][2]
        [
          [3, 4],
          [1, 2],
          [3, 4]
        ]
      >

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> Nx.take(t, Nx.tensor([1, 0, 1]), axis: 1)
      #Nx.Tensor<
        s64[2][3]
        [
          [2, 1, 2],
          [4, 3, 4]
        ]
      >


      iex> t = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      iex> Nx.take(t, Nx.tensor([1, 0, 1]), axis: :y)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [2, 1, 2],
          [4, 3, 4]
        ]
      >

      iex> t = Nx.tensor([[[1, 2], [11, 12]], [[101, 102], [111, 112]]])
      iex> Nx.take(t, Nx.tensor([1, 0, 1]), axis: 1)
      #Nx.Tensor<
        s64[2][3][2]
        [
          [
            [11, 12],
            [1, 2],
            [11, 12]
          ],
          [
            [111, 112],
            [101, 102],
            [111, 112]
          ]
        ]
      >

  Multi-dimensional indices tensor:

      iex> t = Nx.tensor([[1, 2], [11, 12]])
      iex> Nx.take(t, Nx.tensor([[0, 0], [1, 1], [0, 0]]), axis: 1)
      #Nx.Tensor<
        s64[2][3][2]
        [
          [
            [1, 1],
            [2, 2],
            [1, 1]
          ],
          [
            [11, 11],
            [12, 12],
            [11, 11]
          ]
        ]
      >

      iex> t = Nx.tensor([[[1, 2], [11, 12]], [[101, 102], [111, 112]]])
      iex> Nx.take(t, Nx.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), axis: 1)
      #Nx.Tensor<
        s64[2][3][3][2]
        [
          [
            [
              [1, 2],
              [1, 2],
              [1, 2]
            ],
            [
              [11, 12],
              [11, 12],
              [11, 12]
            ],
            [
              [1, 2],
              [1, 2],
              [1, 2]
            ]
          ],
          [
            [
              [101, 102],
              [101, 102],
              [101, 102]
            ],
            [
              [111, 112],
              [111, 112],
              [111, 112]
            ],
            [
              [101, 102],
              [101, 102],
              [101, 102]
            ]
          ]
        ]
      >

  ### Error cases

      iex> Nx.take(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([1, 0, 1], type: {:f, 32}))
      ** (ArgumentError) indices must be an integer tensor, got {:f, 32}
  """
  @doc type: :indexed
  def take(tensor, indices, opts \\ []) when is_list(opts) do
    tensor = to_tensor(tensor)
    indices = to_tensor(indices)

    unless Nx.Type.integer?(indices.type) do
      raise ArgumentError, "indices must be an integer tensor, got #{inspect(indices.type)}"
    end

    opts = keyword!(opts, axis: 0)
    axis = Nx.Shape.normalize_axis(tensor.shape, opts[:axis], tensor.names)

    {shape, names} = Nx.Shape.take(tensor.shape, tensor.names, indices.shape, indices.names, axis)

    impl!(tensor).take(%{tensor | shape: shape, names: names}, tensor, indices, axis)
  end

  @doc """
  Takes the values from a tensor given an `indices` tensor, along the specified axis.

  The `indices` shape must be the same as the `tensor`'s shape, with the exception for
  the `axis` dimension, which can have arbitrary size. The returned tensor will have the
  same shape as the `indices` tensor.

  See `gather/2`, `slice/3`, `slice_along_axis/4`, and `take/3` for other ways to retrieve
  values from a tensor.

  ## Options

    * `:axis` - The axis along which to take the values from. Defaults to `0`.

  ## Examples

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.take_along_axis(t, Nx.tensor([[0, 0, 2, 2, 1, 1], [2, 2, 1, 1, 0, 0]]), axis: 1)
      #Nx.Tensor<
        s64[2][6]
        [
          [1, 1, 3, 3, 2, 2],
          [6, 6, 5, 5, 4, 4]
        ]
      >

      iex> t = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      iex> Nx.take_along_axis(t, Nx.tensor([[0, 1, 1], [1, 0, 0], [0, 1, 0]]), axis: 0)
      #Nx.Tensor<
        s64[3][3]
        [
          [1, 5, 6],
          [4, 2, 3],
          [1, 5, 3]
        ]
      >

  The indices returned from `Nx.argsort/2` can be used with `Nx.take_along_axis/3` to
  produce the sorted tensor (or to sort more tensors according to the same criteria).

      iex> tensor = Nx.tensor([[[1, 2], [3, 4], [5, 6]]])
      #Nx.Tensor<
        s64[1][3][2]
        [
          [
            [1, 2],
            [3, 4],
            [5, 6]
          ]
        ]
      >
      iex> idx1 = Nx.argsort(tensor, axis: 1, direction: :desc)
      #Nx.Tensor<
        s64[1][3][2]
        [
          [
            [2, 2],
            [1, 1],
            [0, 0]
          ]
        ]
      >
      iex> Nx.take_along_axis(tensor, idx1, axis: 1)
      #Nx.Tensor<
        s64[1][3][2]
        [
          [
            [5, 6],
            [3, 4],
            [1, 2]
          ]
        ]
      >
      iex> idx2 = Nx.argsort(tensor, axis: 2, direction: :desc)
      #Nx.Tensor<
        s64[1][3][2]
        [
          [
            [1, 0],
            [1, 0],
            [1, 0]
          ]
        ]
      >
      iex> Nx.take_along_axis(tensor, idx2, axis: 2)
      #Nx.Tensor<
        s64[1][3][2]
        [
          [
            [2, 1],
            [4, 3],
            [6, 5]
          ]
        ]
      >

  ### Error cases

      iex> tensor = Nx.iota({3, 3})
      iex> idx = Nx.tensor([[2.0], [1.0], [2.0]], type: {:f, 32})
      iex> Nx.take_along_axis(tensor, idx, axis: 1)
      ** (ArgumentError) indices must be an integer tensor, got {:f, 32}
  """
  @doc type: :indexed
  def take_along_axis(tensor, indices, opts \\ []) when is_list(opts) do
    tensor = to_tensor(tensor)
    indices = to_tensor(indices)

    unless Nx.Type.integer?(indices.type) do
      raise ArgumentError, "indices must be an integer tensor, got #{inspect(indices.type)}"
    end

    opts = keyword!(opts, axis: 0)
    axis = Nx.Shape.normalize_axis(tensor.shape, opts[:axis], tensor.names)

    shape = Nx.Shape.take_along_axis(tensor.shape, indices.shape, axis)

    impl!(tensor).take_along_axis(%{tensor | shape: shape}, tensor, indices, axis)
  end

  @doc """
  Builds a new tensor by taking individual values from the original
  tensor at the given indices.

  The last dimension in indices must have the same size as the tensor
  rank, think of it as one value per axis.

  ## Examples

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> Nx.gather(t, Nx.tensor([[1, 1], [0, 1], [1, 0]]))
      #Nx.Tensor<
        s64[3]
        [4, 2, 3]
      >

      iex> t = Nx.tensor([[1, 2], [3, 4]])
      iex> Nx.gather(t, Nx.tensor([[[1, 1], [0, 0]], [[1, 0], [0, 1]]]))
      #Nx.Tensor<
        s64[2][2]
        [
          [4, 1],
          [3, 2]
        ]
      >

      iex> t = Nx.tensor([[[1, 2], [11, 12]], [[101, 102], [111, 112]]])
      iex> Nx.gather(t, Nx.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]]))
      #Nx.Tensor<
        s64[3]
        [1, 12, 112]
      >

  ### Error cases

      iex> Nx.gather(Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[0, 0]], type: {:f, 32}))
      ** (ArgumentError) indices must be an integer tensor, got {:f, 32}
  """
  @doc type: :indexed
  def gather(tensor, indices) do
    tensor = to_tensor(tensor)
    indices = to_tensor(indices)

    unless Nx.Type.integer?(indices.type) do
      raise ArgumentError, "indices must be an integer tensor, got #{inspect(indices.type)}"
    end

    {shape, names} = Nx.Shape.gather(tensor.shape, indices.shape)

    impl!(tensor).gather(%{tensor | shape: shape, names: names}, tensor, indices)
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
        f32[x: 4][y: 2][z: 2]
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
  @doc type: :ndim
  def concatenate(tensors, opts \\ []) when is_list(tensors) do
    opts = keyword!(opts, axis: 0)
    axis = opts[:axis]

    case tensors do
      [] ->
        raise ArgumentError, "empty list passed to concatenate"

      [t] ->
        t

      [t1 | _] = tensors ->
        {tensors, [type1 | rest], [s1 | _] = shapes, [n1 | _] = names} =
          tensors
          |> Enum.map(fn t ->
            %T{type: type, shape: shape, names: names} = t = to_tensor(t)

            {t, type, shape, names}
          end)
          |> unzip4()

        axis = Nx.Shape.normalize_axis(s1, axis, n1)
        {output_shape, output_names} = Nx.Shape.concatenate(shapes, names, axis)

        output_type =
          rest
          |> Enum.reduce(type1, fn t1, t2 -> Nx.Type.merge(t1, t2) end)

        out = %{t1 | type: output_type, shape: output_shape, names: output_names}
        list_impl!(tensors).concatenate(out, tensors, axis)
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
  Joins a list of tensors with the same shape along a new axis.

  ### Options

    * `:axis` - optional index of the axis along which the tensors are stacked. Defaults to 0.
    * `:name` - optional name for the added dimension. Defaults to an unnamed axis.

  ### Examples

      iex> Nx.stack([1, 2, 3])
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> Nx.stack([Nx.tensor([1, 2, 3]), Nx.tensor([4, 5, 6])])
      #Nx.Tensor<
        s64[2][3]
        [
          [1, 2, 3],
          [4, 5, 6]
        ]
      >

      iex> t1 = Nx.iota({2, 1, 4})
      iex> t2 = Nx.iota({2, 1, 4})
      iex> t3 = Nx.iota({2, 1, 4})
      iex> Nx.stack([t1, t2, t3], axis: -1)
      #Nx.Tensor<
        s64[2][1][4][3]
        [
          [
            [
              [0, 0, 0],
              [1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]
            ]
          ],
          [
            [
              [4, 4, 4],
              [5, 5, 5],
              [6, 6, 6],
              [7, 7, 7]
            ]
          ]
        ]
      >

      iex> t1 = Nx.iota({2, 1, 4})
      iex> t2 = Nx.iota({2, 1, 4})
      iex> t3 = Nx.iota({2, 1, 4})
      iex> Nx.stack([t1, t2, t3], axis: 1)
      #Nx.Tensor<
        s64[2][3][1][4]
        [
          [
            [
              [0, 1, 2, 3]
            ],
            [
              [0, 1, 2, 3]
            ],
            [
              [0, 1, 2, 3]
            ]
          ],
          [
            [
              [4, 5, 6, 7]
            ],
            [
              [4, 5, 6, 7]
            ],
            [
              [4, 5, 6, 7]
            ]
          ]
        ]
      >

      iex> Nx.stack([Nx.tensor(1), Nx.tensor(2)], name: :x)
      #Nx.Tensor<
        s64[x: 2]
        [1, 2]
      >
  """
  @doc type: :ndim, from_backend: false
  def stack(tensors, opts \\ []) when is_list(tensors) do
    opts = keyword!(opts, axis: 0, name: nil)
    axis = opts[:axis]
    name = opts[:name]

    tensors
    |> Enum.map(&Nx.new_axis(&1, axis, name))
    |> Nx.concatenate(axis: axis)
  end

  @doc """
  Sorts the tensor along the given axis according
  to the given direction.

  If no axis is given, defaults to `0`.

  ### Options

    * `:axis` - The name or number of the corresponding axis on which the sort
      should be applied
    * `:direction` - Can be `:asc` or `:desc`. Defaults to `:asc`

  ### Examples

      iex> Nx.sort(Nx.tensor([16, 23, 42, 4, 8, 15]))
      #Nx.Tensor<
        s64[6]
        [4, 8, 15, 16, 23, 42]
      >

      iex> t = Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y])
      iex> Nx.sort(t, axis: :x)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [2, 1, 4],
          [3, 5, 7]
        ]
      >

      iex> t = Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y])
      iex> Nx.sort(t, axis: :y)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [1, 3, 7],
          [2, 4, 5]
        ]
      >

      iex> t = Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y])
      iex> Nx.sort(t, axis: :y, direction: :asc)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [1, 3, 7],
          [2, 4, 5]
        ]
      >

      iex> t = Nx.tensor(
      ...>   [
      ...>     [[4, 5], [2, 5], [5, 0]],
      ...>     [[1, 9], [2, 1], [2, 1]],
      ...>     [[0, -1], [-1, 0], [0, -1]],
      ...>     [[-1, 0], [0, -1], [-1, 0]]
      ...>   ],
      ...>   names: [:x, :y, :z]
      ...> )
      iex> Nx.sort(t, axis: :x)
      #Nx.Tensor<
        s64[x: 4][y: 3][z: 2]
        [
          [
            [-1, -1],
            [-1, -1],
            [-1, -1]
          ],
          [
            [0, 0],
            [0, 0],
            [0, 0]
          ],
          [
            [1, 5],
            [2, 1],
            [2, 0]
          ],
          [
            [4, 9],
            [2, 5],
            [5, 1]
          ]
        ]
      >

  Same tensor sorted over different axes:

      iex> t = Nx.tensor(
      ...>   [
      ...>     [
      ...>       [4, 5, 2],
      ...>       [2, 5, 3],
      ...>       [5, 0, 2]
      ...>     ],
      ...>     [
      ...>       [1, 9, 8],
      ...>       [2, 1, 3],
      ...>       [2, 1, 4]
      ...>     ]
      ...>   ],
      ...>   names: [:x, :y, :z]
      ...> )
      iex> Nx.sort(t, axis: :x)
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
      iex> Nx.sort(t, axis: :y)
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
      iex> Nx.sort(t, axis: :z)
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
  """
  @doc type: :ndim
  def sort(tensor, opts \\ []) do
    opts = keyword!(opts, axis: 0, direction: :asc)

    direction =
      case opts[:direction] do
        :asc ->
          :asc

        :desc ->
          :desc

        other ->
          raise ArgumentError,
                "unknown value for :direction, expected :asc or :desc, got: #{inspect(other)}"
      end

    %T{shape: shape, names: names, type: type} = tensor = to_tensor(tensor)
    Nx.Shared.raise_complex_not_supported(type, :sort, 2)
    axis = Nx.Shape.normalize_axis(shape, opts[:axis], names)

    impl!(tensor).sort(
      tensor,
      tensor,
      axis: axis,
      direction: direction
    )
  end

  @doc """
  Sorts the tensor along the given axis according
  to the given direction and returns the corresponding indices
  of the original tensor in the new sorted positions.

  If no axis is given, defaults to `0`.

  ## Options

    * `:axis` - The name or number of the corresponding axis on which the sort
      should be applied
    * `:direction` - Can be `:asc` or `:desc`. Defaults to `:asc`

  ## Examples

      iex> Nx.argsort(Nx.tensor([16, 23, 42, 4, 8, 15]))
      #Nx.Tensor<
        s64[6]
        [3, 4, 5, 0, 1, 2]
      >

      iex> t = Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y])
      iex> Nx.argsort(t, axis: :x)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [1, 0, 1],
          [0, 1, 0]
        ]
      >

      iex> t = Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y])
      iex> Nx.argsort(t, axis: :y)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [1, 0, 2],
          [0, 2, 1]
        ]
      >

      iex> t = Nx.tensor([[3, 1, 7], [2, 5, 4]], names: [:x, :y])
      iex> Nx.argsort(t, axis: :y, direction: :asc)
      #Nx.Tensor<
        s64[x: 2][y: 3]
        [
          [1, 0, 2],
          [0, 2, 1]
        ]
      >

  Same tensor sorted over different axes:

      iex> t = Nx.tensor(
      ...>   [
      ...>     [
      ...>       [4, 5, 2],
      ...>       [2, 5, 3],
      ...>       [5, 0, 2]
      ...>     ],
      ...>     [
      ...>       [1, 9, 8],
      ...>       [2, 1, 3],
      ...>       [2, 1, 4]
      ...>     ]
      ...>   ],
      ...>   names: [:x, :y, :z]
      ...> )
      iex> Nx.argsort(t, axis: :x)
      #Nx.Tensor<
        s64[x: 2][y: 3][z: 3]
        [
          [
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0]
          ],
          [
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1]
          ]
        ]
      >
      iex> Nx.argsort(t, axis: :y)
      #Nx.Tensor<
        s64[x: 2][y: 3][z: 3]
        [
          [
            [1, 2, 0],
            [0, 0, 2],
            [2, 1, 1]
          ],
          [
            [0, 1, 1],
            [1, 2, 2],
            [2, 0, 0]
          ]
        ]
      >
      iex> Nx.argsort(t, axis: :z)
      #Nx.Tensor<
        s64[x: 2][y: 3][z: 3]
        [
          [
            [2, 0, 1],
            [0, 2, 1],
            [1, 2, 0]
          ],
          [
            [0, 2, 1],
            [1, 0, 2],
            [1, 0, 2]
          ]
        ]
      >
  """
  @doc type: :ndim
  def argsort(tensor, opts \\ []) do
    opts = keyword!(opts, axis: 0, direction: :asc)

    direction =
      case opts[:direction] do
        :asc ->
          :asc

        :desc ->
          :desc

        other ->
          raise ArgumentError,
                "unknown value for :direction, expected :asc or :desc, got: #{inspect(other)}"
      end

    %T{type: type, shape: shape, names: names} = tensor = to_tensor(tensor)
    axis = Nx.Shape.normalize_axis(shape, opts[:axis], names)

    Nx.Shared.raise_complex_not_supported(type, :argsort, 2)

    impl!(tensor).argsort(
      %{tensor | type: {:s, 64}},
      tensor,
      axis: axis,
      direction: direction
    )
  end

  ## Utilities

  @doc """
  Serializes the given tensor or container of tensors to iodata.

  You may pass a tensor, tuple, or map to serialize.

  `opts` controls the serialization options. For example, you can choose
  to compress the given tensor or container of tensors by passing a
  compression level:

      Nx.serialize(tensor, compressed: 9)

  Compression level corresponds to compression options in `:erlang.term_to_iovec/2`.

  `iodata` is a list of binaries that can be written to any io device,
  such as a file or a socket. You can ensure the result is a binary by
  calling `IO.iodata_to_binary/1`.  

  ## Examples

      iex> a = Nx.tensor([1, 2, 3])
      iex> serialized_a = Nx.serialize(a)
      iex> Nx.deserialize(serialized_a)
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> container = {Nx.tensor([1, 2, 3]), %{b: Nx.tensor([4, 5, 6])}}
      iex> serialized_container = Nx.serialize(container)
      iex> {a, %{b: b}} = Nx.deserialize(serialized_container)
      iex> a
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >
      iex> b
      #Nx.Tensor<
        s64[3]
        [4, 5, 6]
      >
  """
  @doc type: :conversion
  def serialize(tensor_or_container, opts \\ []) do
    data_term = to_term(tensor_or_container)
    term = {@file_version, System.endianness(), data_term}
    :erlang.term_to_iovec(term, opts)
  end

  defp to_term(tensor_or_container) do
    case tensor_or_container do
      number when is_number(number) ->
        type = Nx.Type.infer(number)
        {:tensor, {}, type, [], number_to_binary(number, type)}

      %T{} = tensor ->
        shape = shape(tensor)
        type = type(tensor)
        names = names(tensor)
        binary = to_binary(tensor)
        {:tensor, shape, type, names, binary}

      %_{} = value ->
        bad_serialize!(value)

      container when is_tuple(container) or is_map(container) ->
        {serialized, :ok} =
          Nx.Container.traverse(container, :ok, fn container_elem, :ok ->
            {to_term(container_elem), :ok}
          end)

        {:container, serialized}

      value ->
        bad_serialize!(value)
    end
  end

  defp bad_serialize!(value) do
    raise ArgumentError,
          "unable to serialize #{inspect(value)}. Only tensors, tuples and " <>
            "maps are supported. If you are attempting to serialize a custom " <>
            "container, you will need to serialize fields in the container manually"
  end

  @doc """
  Deserializes a serialized representation of a tensor or a container
  with the given options.

  It is the opposite of `Nx.serialize/2`.

  ## Examples

      iex> a = Nx.tensor([1, 2, 3])
      iex> serialized_a = Nx.serialize(a)
      iex> Nx.deserialize(serialized_a)
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >

      iex> container = {Nx.tensor([1, 2, 3]), %{b: Nx.tensor([4, 5, 6])}}
      iex> serialized_container = Nx.serialize(container)
      iex> {a, %{b: b}} = Nx.deserialize(serialized_container)
      iex> a
      #Nx.Tensor<
        s64[3]
        [1, 2, 3]
      >
      iex> b
      #Nx.Tensor<
        s64[3]
        [4, 5, 6]
      >
  """
  @doc type: :conversion
  def deserialize(data, opts \\ []) do
    data
    |> IO.iodata_to_binary()
    |> :erlang.binary_to_term(opts)
    |> from_term()
  end

  defp from_term({1, endianness, term}) do
    case term do
      {:tensor, shape, {_, size} = type, names, binary} ->
        binary
        |> new_byte_order(size, endianness)
        |> from_binary(type)
        |> reshape(shape, names: names)

      {:container, container} ->
        {deserialized, :ok} =
          Nx.Container.traverse(container, :ok, fn container_elem, :ok ->
            {from_term({1, endianness, container_elem}), :ok}
          end)

        deserialized

      _ ->
        raise ArgumentError, "unable to deserialize binary term to tensor"
    end
  end

  defp from_term(_) do
    raise ArgumentError, "unable to deserialize binary term to tensor"
  end

  @doc """
  Loads a `.npy` file into a tensor.

  An `.npy` file stores a single array created from Python's
  NumPy library. This function can be useful for loading data
  originally created or intended to be loaded from NumPy into
  Elixir.
  """
  @doc type: :conversion
  def from_numpy(file) do
    file
    |> File.read!()
    |> parse_numpy()
  end

  @doc """
  Loads a `.npz` archive into a list of tensors.

  An `.npz` file is a zipped, possibly compressed archive containing
  multiple `.npy` files.
  """
  @doc type: :conversion
  def from_numpy_archive(archive) do
    archive = File.read!(archive)

    case :zip.unzip(archive, [:memory]) do
      {:ok, files} ->
        files
        |> Enum.map(fn {_, data} -> parse_numpy(data) end)

      _ ->
        raise ArgumentError,
              "unable to parse NumPy archive, it may be corrupted" <>
                " or invalid"
    end
  end

  defp parse_numpy(<<"\x93NUMPY"::binary, major::size(8), minor::size(8), rest::binary>>) do
    parse_numpy(rest, major, minor)
  end

  defp parse_numpy(_) do
    raise ArgumentError,
          "unable to parse NumPy file, it may be corrupted" <>
            " or invalid"
  end

  defp parse_numpy(<<header_size::size(16)-little-unsigned, rest::binary>>, 1, 0) do
    do_numpy_to_tensor(rest, header_size)
  end

  defp parse_numpy(<<header_size::size(32)-little-unsigned, rest::binary>>, _, _) do
    do_numpy_to_tensor(rest, header_size)
  end

  defp do_numpy_to_tensor(rest, header_size) when is_binary(rest) do
    <<header::size(header_size)-binary, array::binary>> = rest
    {byte_order, {_, size} = type, shape} = parse_header(header)
    byte_size_of_array = div(size, 8) * Nx.size(shape)

    <<data::size(byte_size_of_array)-binary>> = array

    data
    |> new_byte_order(size, byte_order)
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
  end

  defp parse_header(header) do
    header = header |> String.trim("{") |> String.trim("}") |> String.trim(", ")

    case header do
      "'descr': " <> <<dtype::size(5)-binary>> <> ", 'fortran_order': False, 'shape': " <> shape ->
        {byte_order, type} = parse_type(dtype)
        {byte_order, type, parse_shape(shape)}

      "'descr': " <> <<dtype::size(5)-binary>> <> ", 'fortran_order': True, 'shape': " <> shape ->
        {byte_order, type} = parse_type(dtype)
        {byte_order, type, parse_shape(shape)}
    end
  end

  defp parse_type(dtype) do
    [byte_order, type, size] =
      dtype
      |> String.trim("'")
      |> String.split("", trim: true)

    byte_order =
      case byte_order do
        ">" ->
          :big

        "<" ->
          :little

        # We can't just infer native endianness matches our native endianness
        endianness ->
          raise ArgumentError, "Numpy tensor has unsupported endianness: #{endianness}"
      end

    type =
      case type do
        "u" ->
          :u

        "i" ->
          :s

        "f" ->
          :f

        _ ->
          raise "unsupported type"
      end

    size = size |> String.to_integer() |> Kernel.*(8)

    {byte_order, {type, size}}
  end

  defp parse_shape(shape) do
    shape
    |> String.trim()
    |> String.trim("), }")
    |> String.trim("(")
    |> String.split(",", trim: true)
    |> Enum.map(&(String.trim(&1) |> String.to_integer()))
    |> List.to_tuple()
  end

  defp new_byte_order(binary, size, endianness) do
    if System.endianness() == endianness do
      binary
    else
      data =
        for <<data::size(size)-binary <- binary>> do
          data
          |> :binary.decode_unsigned()
          |> :binary.encode_unsigned(endianness)
        end

      IO.iodata_to_binary(data)
    end
  end

  @doc """
  Finds the variance of a tensor.

  The variance is the average of the squared deviations from the mean.
  The mean is typically calculated as `sum(tensor) / n`, where `n` is the total
  of elements. If, however, `:ddof` (delta degrees of freedom) is specified, the
  divisor `n - ddof` is used instead.

  ## Examples

      iex> Nx.variance(Nx.tensor([[1, 2], [3, 4]]))
      #Nx.Tensor<
        f32
        1.25
      >

      iex> Nx.variance(Nx.tensor([[1, 2], [3, 4]]), ddof: 1)
      #Nx.Tensor<
        f32
        1.6666666269302368
      >

      iex> Nx.variance(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      #Nx.Tensor<
        f32[2]
        [1.0, 1.0]
      >

      iex> Nx.variance(Nx.tensor([[1, 2], [3, 4]]), axes: [1])
      #Nx.Tensor<
        f32[2]
        [0.25, 0.25]
      >

      iex> Nx.variance(Nx.tensor([[1, 2], [3, 4]]), axes: [0], ddof: 1)
      #Nx.Tensor<
        f32[2]
        [2.0, 2.0]
      >

      iex> Nx.variance(Nx.tensor([[1, 2], [3, 4]]), axes: [1], ddof: 1)
      #Nx.Tensor<
        f32[2]
        [0.5, 0.5]
      >

  ### Keeping axes

      iex> Nx.variance(Nx.tensor([[1, 2], [3, 4]]), axes: [1], keep_axes: true)
      #Nx.Tensor<
        f32[2][1]
        [
          [0.25],
          [0.25]
        ]
      >
  """
  @doc type: :aggregation
  @spec variance(tensor :: Nx.Tensor.t(), opts :: Keyword.t()) :: Nx.Tensor.t()
  def variance(tensor, opts \\ []) do
    %T{shape: shape, names: names} = tensor = to_tensor(tensor)
    opts = keyword!(opts, [:axes, ddof: 0, keep_axes: false])
    axes = opts[:axes]
    {ddof, opts} = Keyword.pop!(opts, :ddof)

    total =
      if axes do
        mean_den(shape, Nx.Shape.normalize_axes(shape, axes, names))
      else
        size(shape)
      end

    mean = mean(tensor, Keyword.put(opts, :keep_axes, true))

    tensor
    |> subtract(mean)
    |> power(2)
    |> sum(opts)
    |> divide(total - ddof)
  end

  @doc """
  Finds the standard deviation of a tensor.

  The standard deviation is taken as the square root of the variance.
  If the `:ddof` (delta degrees of freedom) option is given, the divisor
  `n - ddof` is used to calculate the variance. See `variance/2`.

  ## Examples

      iex> Nx.standard_deviation(Nx.tensor([[1, 2], [3, 4]]))
      #Nx.Tensor<
        f32
        1.1180340051651
      >

      iex> Nx.standard_deviation(Nx.tensor([[1, 2], [3, 4]]), ddof: 1)
      #Nx.Tensor<
        f32
        1.29099440574646
      >

      iex> Nx.standard_deviation(Nx.tensor([[1, 2], [3, 4]]), axes: [0])
      #Nx.Tensor<
        f32[2]
        [1.0, 1.0]
      >

      iex> Nx.standard_deviation(Nx.tensor([[1, 2], [3, 4]]), axes: [1])
      #Nx.Tensor<
        f32[2]
        [0.5, 0.5]
      >

      iex> Nx.standard_deviation(Nx.tensor([[1, 2], [3, 4]]), axes: [0], ddof: 1)
      #Nx.Tensor<
        f32[2]
        [1.4142135381698608, 1.4142135381698608]
      >

      iex> Nx.standard_deviation(Nx.tensor([[1, 2], [3, 4]]), axes: [1], ddof: 1)
      #Nx.Tensor<
        f32[2]
        [0.7071067690849304, 0.7071067690849304]
      >

  ### Keeping axes

      iex> Nx.standard_deviation(Nx.tensor([[1, 2], [3, 4]]), keep_axes: true)
      #Nx.Tensor<
        f32[1][1]
        [
          [1.1180340051651]
        ]
      >
  """
  @doc type: :aggregation
  @spec standard_deviation(tensor :: Nx.Tensor.t(), opts :: Keyword.t()) :: Nx.Tensor.t()
  def standard_deviation(tensor, opts \\ []) do
    sqrt(variance(tensor, opts))
  end

  @doc """
  Calculates the DFT of the given 1D tensor.

  ## Options

    * `:eps` - Threshold which backends can use for cleaning-up results. Defaults to `1.0e-10`.
    * `:length` - Either a positive integer or `:power_of_two`. Will pad or slice the tensor
      accordingly. `:power_of_two` will automatically pad to the next power of two.

  ## Examples

      iex> Nx.fft(Nx.tensor([1, 1, 0, 0]))
      #Nx.Tensor<
        c64[4]
        [2.0+0.0i, 1.0-1.0i, 0.0+0.0i, 1.0+1.0i]
      >

      iex> Nx.fft(Nx.tensor([1, 1, 0, 0, 0]))
      #Nx.Tensor<
        c64[5]
        [2.0+0.0i, 1.3090169429779053-0.9510565400123596i, 0.19098301231861115-0.5877852439880371i, 0.19098301231861115+0.5877852439880371i, 1.3090169429779053+0.9510565400123596i]
      >

      iex> Nx.fft(Nx.tensor([1, 1, 1, 0, 1, 1]))
      #Nx.Tensor<
        c64[6]
        [5.0+0.0i, 1.0+0.0i, -1.0+0.0i, 1.0+0.0i, -1.0+0.0i, 1.0+0.0i]
      >

  Padding and slicing can be introduced through `:length`:

      iex> Nx.fft(Nx.tensor([1, 1]), length: 4)
      #Nx.Tensor<
        c64[4]
        [2.0+0.0i, 1.0-1.0i, 0.0+0.0i, 1.0+1.0i]
      >

      iex> Nx.fft(Nx.tensor([1, 1, 0]), length: :power_of_two)
      #Nx.Tensor<
        c64[4]
        [2.0+0.0i, 1.0-1.0i, 0.0+0.0i, 1.0+1.0i]
      >

      iex> Nx.fft(Nx.tensor([1, 1, 0, 0, 2, 3]), length: 4)
      #Nx.Tensor<
        c64[4]
        [2.0+0.0i, 1.0-1.0i, 0.0+0.0i, 1.0+1.0i]
      >

  ## Error Cases

      iex> Nx.fft(Nx.tensor([1, 1]), length: :invalid)
      ** (RuntimeError) expected an integer or :power_of_two as length, got: :invalid
  """
  @doc type: :signal
  def fft(tensor, opts \\ []) do
    tensor = to_tensor(tensor)

    {n} = Nx.Shape.fft(tensor.shape)

    opts = Keyword.validate!(opts, length: n, eps: 1.0e-10)

    length =
      case opts[:length] do
        :power_of_two ->
          2 ** Kernel.ceil(:math.log2(n))

        n when is_integer(n) and n > 0 ->
          n

        length ->
          raise "expected an integer or :power_of_two as length, got: #{inspect(length)}"
      end

    opts = Keyword.put(opts, :length, length)

    out = to_template(%{tensor | shape: {length}, type: Nx.Type.to_complex(tensor.type)})
    impl!(tensor).fft(out, tensor, opts)
  end

  ## Sigils

  @doc """
  A convenient `~M` sigil for building matrices (two-dimensional tensors).

  ## Examples

  Before using sigils, you must first import them:

      import Nx, only: :sigils

  Then you use the sigil to create matrices. The sigil:

      ~M<
        -1 0 0 1
        0 2 0 0
        0 0 3 0
        0 0 0 4
      >

  Is equivalent to:

      Nx.tensor([
        [-1, 0, 0, 1],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4]
      ])

  If the tensor has any complex type, it defaults to c64.
  If the tensor has any float type, it defaults to f32.
  Otherwise, it is s64. You can specify the tensor type
  as a sigil modifier:

      iex> import Nx, only: :sigils
      iex> ~M[0.1 0.2 0.3 0.4]f16
      #Nx.Tensor<
        f16[1][4]
        [
          [0.0999755859375, 0.199951171875, 0.300048828125, 0.39990234375]
        ]
      >
      iex> ~M[1+1i 2-2.0i -3]
      #Nx.Tensor<
        c64[1][3]
        [
          [1.0+1.0i, 2.0-2.0i, -3.0+0.0i]
        ]
      >
      iex> ~M[1 Inf NaN]
      #Nx.Tensor<
        f32[1][3]
        [
          [1.0, Inf, NaN]
        ]
      >
      iex> ~M[1i Inf NaN]
      #Nx.Tensor<
        c64[1][3]
        [
          [0.0+1.0i, Inf+0.0i, NaN+0.0i]
        ]
      >
      iex> ~M[1i Inf+2i NaN-Infi]
      #Nx.Tensor<
        c64[1][3]
        [
          [0.0+1.0i, Inf+2.0i, NaN-Infi]
        ]
      >

  """
  @doc type: :creation
  defmacro sigil_M({:<<>>, _meta, [string]}, modifiers) do
    {numbers, type} = string |> String.trim() |> binary_to_numbers()
    numbers_to_tensor(numbers, type, modifiers)
  end

  @doc """
  A convenient `~V` sigil for building vectors (one-dimensional tensors).

  ## Examples

  Before using sigils, you must first import them:

      import Nx, only: :sigils

  Then you use the sigil to create vectors. The sigil:

      ~V[-1 0 0 1]

  Is equivalent to:

      Nx.tensor([-1, 0, 0, 1])

  If the tensor has any complex type, it defaults to c64.
  If the tensor has any float type, it defaults to f32.
  Otherwise, it is s64. You can specify the tensor type
  as a sigil modifier:

      iex> import Nx, only: :sigils
      iex> ~V[0.1 0.2 0.3 0.4]f16
      #Nx.Tensor<
        f16[4]
        [0.0999755859375, 0.199951171875, 0.300048828125, 0.39990234375]
      >
      iex> ~V[1+1i 2-2.0i -3]
      #Nx.Tensor<
        c64[3]
        [1.0+1.0i, 2.0-2.0i, -3.0+0.0i]
      >
      iex> ~V[1 Inf NaN]
      #Nx.Tensor<
        f32[3]
        [1.0, Inf, NaN]
      >
      iex> ~V[1i Inf NaN]
      #Nx.Tensor<
        c64[3]
        [0.0+1.0i, Inf+0.0i, NaN+0.0i]
      >
      iex> ~V[1i Inf+2i NaN-Infi]
      #Nx.Tensor<
        c64[3]
        [0.0+1.0i, Inf+2.0i, NaN-Infi]
      >
  """
  @doc type: :creation
  defmacro sigil_V({:<<>>, _meta, [string]}, modifiers) do
    string
    |> String.trim()
    |> binary_to_numbers()
    |> case do
      {[numbers], type} ->
        numbers_to_tensor(numbers, type, modifiers)

      _ ->
        raise ArgumentError, "must be one-dimensional"
    end
  end

  defp numbers_to_tensor(numbers, type, modifiers) do
    type =
      case modifiers do
        [unit | size] ->
          Nx.Type.normalize!({List.to_atom([unit]), List.to_integer(size)})

        [] ->
          type
      end

    {shape, binary} = flatten_list(numbers, type)

    quote do
      unquote(binary)
      |> Nx.from_binary(unquote(type))
      |> Nx.reshape(unquote(Macro.escape(shape)))
    end
  end

  defp binary_to_numbers(string) do
    string
    |> String.split(["\n", "\r\n"], trim: true)
    |> Enum.map_reduce({:s, 64}, fn row, type ->
      row
      |> String.split(" ", trim: true)
      |> Enum.map_reduce(type, fn str, type ->
        {module, type} =
          cond do
            elem(type, 0) == :c -> {Complex, type}
            String.contains?(str, ["Inf", "NaN"]) -> {Complex, type}
            String.contains?(str, "i") -> {Complex, {:c, 64}}
            String.contains?(str, ".") -> {Float, {:f, 32}}
            :otherwise -> {Integer, type}
          end

        parse_string_to_number(module, str, type)
      end)
    end)
  end

  defp parse_string_to_number(Complex, str, {type_class, _} = type) do
    apply_parse = fn fun ->
      case apply(fun, [str]) do
        :error -> false
        val -> val
      end
    end

    result = Enum.find_value([&Complex.parse/1, &Float.parse/1, &Integer.parse/1], apply_parse)

    case result do
      {%Complex{re: re, im: im}, ""}
      when re in [:nan, :neg_infinity, :infinity] and im == 0 and type_class != :c ->
        {re, Nx.Type.merge(type, {:f, 32})}

      {%Complex{} = num, ""} ->
        {num, Nx.Type.merge(type, {:c, 64})}

      {num, ""} ->
        {Complex.new(num), Nx.Type.merge(type, {:c, 64})}

      _ ->
        raise ArgumentError, "expected a numerical value for tensor, got #{str}"
    end
  end

  defp parse_string_to_number(module, str, type) do
    case module.parse(str) do
      {number, ""} ->
        {number, type}

      _ ->
        raise ArgumentError, "expected a numerical value for tensor, got #{str}"
    end
  end

  ## Helpers

  defp backend!(backend) when is_atom(backend),
    do: {backend, []}

  defp backend!({backend, options}) when is_atom(backend) and is_list(options),
    do: {backend, options}

  defp backend!(other) do
    raise ArgumentError,
          "backend must be an atom or a tuple {backend, options}, got: #{inspect(other)}"
  end

  defp number_to_binary(number, type), do: match_types([type], do: <<write!(number, 0)>>)

  defp names!(%T{names: names}), do: names
  defp names!(_), do: nil

  defp to_indices(start_indices) do
    all_static? = Enum.all?(start_indices, &is_integer/1)

    if all_static? do
      start_indices
    else
      Enum.with_index(start_indices, fn index, i ->
        %T{shape: idx_shape, type: idx_type} = t = to_tensor(index)

        unless idx_shape == {} do
          raise ArgumentError,
                "index must be scalar, got shape #{inspect(idx_shape)}" <>
                  " for axis #{i}"
        end

        unless Nx.Type.integer?(idx_type) do
          raise ArgumentError,
                "index must be integer type, got #{inspect(idx_type)} for axis #{i}"
        end

        t
      end)
    end
  end
end

defmodule Nx.Constants do
  @moduledoc """
  Common constants used in computations.
  """

  import Nx.Shared
  import Nx.Defn.Kernel, only: [keyword!: 2]

  @doc """
  Returns infinity in f32.
  """
  def nan, do: nan({:f, 32}, [])

  @doc """
  Returns NaN (Not a Number).

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.nan({:bf, 16})
      #Nx.Tensor<
        bf16
        NaN
      >

      iex> Nx.Constants.nan({:f, 16})
      #Nx.Tensor<
        f16
        NaN
      >

      iex> Nx.Constants.nan({:f, 32})
      #Nx.Tensor<
        f32
        NaN
      >

      iex> Nx.Constants.nan({:f, 64})
      #Nx.Tensor<
        f64
        NaN
      >

  """
  def nan(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.nan_binary(type), type, opts)
  end

  @doc """
  Returns infinity in f32.
  """
  def infinity, do: infinity({:f, 32}, [])

  @doc """
  Returns infinity.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.infinity({:bf, 16})
      #Nx.Tensor<
        bf16
        Inf
      >

      iex> Nx.Constants.infinity({:f, 16})
      #Nx.Tensor<
        f16
        Inf
      >

      iex> Nx.Constants.infinity({:f, 32})
      #Nx.Tensor<
        f32
        Inf
      >

      iex> Nx.Constants.infinity({:f, 64})
      #Nx.Tensor<
        f64
        Inf
      >

  """
  def infinity(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.infinity_binary(type), type, opts)
  end

  @doc """
  Returns infinity in f32.
  """
  def neg_infinity, do: neg_infinity({:f, 32}, [])

  @doc """
  Returns negative infinity.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.neg_infinity({:bf, 16})
      #Nx.Tensor<
        bf16
        -Inf
      >

      iex> Nx.Constants.neg_infinity({:f, 16})
      #Nx.Tensor<
        f16
        -Inf
      >

      iex> Nx.Constants.neg_infinity({:f, 32})
      #Nx.Tensor<
        f32
        -Inf
      >

      iex> Nx.Constants.neg_infinity({:f, 64})
      #Nx.Tensor<
        f64
        -Inf
      >

  """
  def neg_infinity(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.neg_infinity_binary(type), type, opts)
  end

  @doc """
  Returns a scalar tensor with the maximum finite value for the given type.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.max_finite({:u, 8})
      #Nx.Tensor<
        u8
        255
      >

      iex> Nx.Constants.max_finite({:s, 16})
      #Nx.Tensor<
        s16
        32677
      >

      iex> Nx.Constants.max_finite({:f, 32})
      #Nx.Tensor<
        f32
        3.4028234663852886e38
      >

  """
  def max_finite(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.max_finite_binary(type), type, opts)
  end

  @doc """
  Returns a scalar tensor with the maximum value for the given type.

  It is infinity for floating point tensors.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.max({:u, 8})
      #Nx.Tensor<
        u8
        255
      >

      iex> Nx.Constants.max({:f, 32})
      #Nx.Tensor<
        f32
        Inf
      >

  """
  def max(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.max_binary(type), type, opts)
  end

  @doc """
  Returns a scalar tensor with the minimum finite value for the given type.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.min_finite({:u, 8})
      #Nx.Tensor<
        u8
        0
      >

      iex> Nx.Constants.min_finite({:s, 16})
      #Nx.Tensor<
        s16
        -32678
      >

      iex> Nx.Constants.min_finite({:f, 32})
      #Nx.Tensor<
        f32
        -3.4028234663852886e38
      >

  """
  def min_finite(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.min_finite_binary(type), type, opts)
  end

  @doc """
  Returns a scalar tensor with the minimum value for the given type.

  It is negative infinity for floating point tensors.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.min({:u, 8})
      #Nx.Tensor<
        u8
        0
      >

      iex> Nx.Constants.min({:f, 32})
      #Nx.Tensor<
        f32
        -Inf
      >

  """
  def min(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.min_binary(type), type, opts)
  end

  @doc """
  Returns the imaginary constant.

  Accepts the same options as `Nx.tensor/2`

  ### Examples

      iex> Nx.Constants.i()
      #Nx.Tensor<
        c64
        0.0+1.0i
      >

      iex> Nx.Constants.i(type: {:c, 128})
      #Nx.Tensor<
        c128
        0.0+1.0i
      >

  ### Error cases

      iex> Nx.Constants.i(type: {:f, 32})
      ** (ArgumentError) invalid type for complex number. Expected {:c, 64} or {:c, 128}, got: {:f, 32}
  """
  def i(opts \\ []) do
    Nx.tensor(Complex.new(0, 1), opts)
  end

  defp from_binary(binary, type, opts) do
    opts = keyword!(opts, [:backend])
    {backend, backend_options} = backend_from_options!(opts) || Nx.default_backend()
    backend.from_binary(%Nx.Tensor{type: type, shape: {}, names: []}, binary, backend_options)
  end
end

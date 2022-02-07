defmodule Nx.Constants do
  @moduledoc """
  Common constants used in computations.
  """

  import Nx.Shared
  import Nx.Defn.Kernel, only: [keyword!: 2]

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

  defp from_binary(binary, type, opts) do
    opts = keyword!(opts, [:backend])
    {backend, backend_options} = backend_from_options!(opts) || Nx.default_backend()
    backend.from_binary(%Nx.Tensor{type: type, shape: {}, names: []}, binary, backend_options)
  end
end

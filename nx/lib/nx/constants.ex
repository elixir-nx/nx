defmodule Nx.Constants do
  @moduledoc """
  Common constants used in computations.

  This module can be used in `defn`.
  """

  import Nx.Shared
  import Nx.Defn.Kernel, only: [keyword!: 2]

  @doc """
  Returns NaN in f32.
  """
  def nan, do: nan({:f, 32}, [])

  @doc """
  Returns NaN (Not a Number).

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.nan({:f, 8})
      #Nx.Tensor<
        f8
        NaN
      >

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

      iex> Nx.Constants.infinity({:f, 8})
      #Nx.Tensor<
        f8
        Inf
      >

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
  Returns negative infinity in f32.
  """
  def neg_infinity, do: neg_infinity({:f, 32}, [])

  @doc """
  Returns negative infinity.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.neg_infinity({:f, 8})
      #Nx.Tensor<
        f8
        -Inf
      >

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
        32767
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
        -32768
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
  Returns the imaginary constant in c64.
  """
  def i, do: i(:c64)

  @doc """
  Returns the imaginary constant.

  Accepts the same options as `Nx.tensor/2`

  ## Examples

      iex> Nx.Constants.i()
      #Nx.Tensor<
        c64
        0.0+1.0i
      >

      iex> Nx.Constants.i(:c128)
      #Nx.Tensor<
        c128
        0.0+1.0i
      >

  ## Error cases

      iex> Nx.Constants.i({:f, 32})
      ** (ArgumentError) invalid type for complex number. Expected {:c, 64} or {:c, 128}, got: {:f, 32}
  """
  def i(type, opts \\ []) do
    if Keyword.has_key?(opts, :type) do
      raise "type must not be passed as an option"
    end

    Nx.tensor(Complex.new(0, 1), Keyword.put(opts, :type, type))
  end

  @doc """
  Returns a scalar tensor with the smallest positive value for the given type.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.smallest_positive_normal({:f, 64})
      #Nx.Tensor<
        f64
        2.2250738585072014e-308
      >

      iex> Nx.Constants.smallest_positive_normal({:f, 32})
      #Nx.Tensor<
        f32
        1.1754943508222875e-38
      >

      iex> Nx.Constants.smallest_positive_normal({:f, 16})
      #Nx.Tensor<
        f16
        6.103515625e-5
      >

      iex> Nx.Constants.smallest_positive_normal(:bf16)
      #Nx.Tensor<
        bf16
        1.1754943508222875e-38
      >

      iex> Nx.Constants.smallest_positive_normal(:f8)
      #Nx.Tensor<
        f8
        6.103515625e-5
      >

      iex> Nx.Constants.smallest_positive_normal({:s, 32})
      ** (ArgumentError) only floating types are supported, got: {:s, 32}
  """
  def smallest_positive_normal(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.smallest_positive_normal_binary(type), type, opts)
  end

  @doc """
  Returns a scalar with the machine epsilon for the given type.

  The values are compatible with a IEEE 754 floating point standard.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.epsilon({:f, 64})
      #Nx.Tensor<
        f64
        2.220446049250313e-16
      >

      iex> Nx.Constants.epsilon({:f, 32})
      #Nx.Tensor<
        f32
        1.1920928955078125e-7
      >

      iex> Nx.Constants.epsilon({:f, 16})
      #Nx.Tensor<
        f16
        9.765625e-4
      >

      iex> Nx.Constants.epsilon(:bf16)
      #Nx.Tensor<
        bf16
        0.0078125
      >

      iex> Nx.Constants.epsilon(:f8)
      #Nx.Tensor<
        f8
        0.25
      >

      iex> Nx.Constants.epsilon({:s, 32})
      ** (ArgumentError) only floating types are supported, got: {:s, 32}
  """
  def epsilon(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.epsilon_binary(type), type, opts)
  end

  @doc ~S"""
  Returns $\pi$ in f32.
  """
  def pi, do: pi({:f, 32}, [])

  @doc ~S"""
  Returns a scalar tensor with the value of $\pi$ for the given type.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.pi({:f, 64})
      #Nx.Tensor<
        f64
        3.141592653589793
      >

      iex> Nx.Constants.pi({:f, 32})
      #Nx.Tensor<
        f32
        3.1415927410125732
      >

      iex> Nx.Constants.pi({:f, 16})
      #Nx.Tensor<
        f16
        3.140625
      >

      iex> Nx.Constants.pi({:bf, 16})
      #Nx.Tensor<
        bf16
        3.140625
      >

      iex> Nx.Constants.pi({:f, 8})
      #Nx.Tensor<
        f8
        3.0
      >

      iex> Nx.Constants.pi({:s, 32})
      ** (ArgumentError) only floating types are supported, got: {:s, 32}
  """
  def pi(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.pi_binary(type), type, opts)
  end

  @doc """
  Returns $e$ in f32.
  """
  def e, do: e({:f, 32}, [])

  @doc """
  Returns a scalar tensor with the value of $e$ for the given type.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.e({:f, 64})
      #Nx.Tensor<
        f64
        2.718281828459045
      >

      iex> Nx.Constants.e({:f, 32})
      #Nx.Tensor<
        f32
        2.7182817459106445
      >

      iex> Nx.Constants.e({:f, 16})
      #Nx.Tensor<
        f16
        2.71875
      >

      iex> Nx.Constants.e({:bf, 16})
      #Nx.Tensor<
        bf16
        2.703125
      >

      iex> Nx.Constants.e({:f, 8})
      #Nx.Tensor<
        f8
        2.5
      >

      iex> Nx.Constants.e({:s, 32})
      ** (ArgumentError) only floating types are supported, got: {:s, 32}
  """
  def e(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.e_binary(type), type, opts)
  end

  @doc ~S"""
  Returns $\gamma$ (Euler-Mascheroni constant) in f32.
  """
  def euler_gamma, do: euler_gamma({:f, 32}, [])

  @doc ~S"""
  Returns a scalar tensor with the value of $\gamma$ (Euler-Mascheroni constant) for the given type.

  ## Options

    * `:backend` - a backend to allocate the tensor on.

  ## Examples

      iex> Nx.Constants.euler_gamma({:f, 64})
      #Nx.Tensor<
        f64
        0.5772156649015329
      >

      iex> Nx.Constants.euler_gamma({:f, 32})
      #Nx.Tensor<
        f32
        0.5772156715393066
      >

      iex> Nx.Constants.euler_gamma({:f, 16})
      #Nx.Tensor<
        f16
        0.5771484375
      >

      iex> Nx.Constants.euler_gamma({:bf, 16})
      #Nx.Tensor<
        bf16
        0.57421875
      >

      iex> Nx.Constants.euler_gamma({:f, 8})
      #Nx.Tensor<
        f8
        0.5
      >

      iex> Nx.Constants.euler_gamma({:s, 32})
      ** (ArgumentError) only floating types are supported, got: {:s, 32}
  """
  def euler_gamma(type, opts \\ []) do
    type = Nx.Type.normalize!(type)
    from_binary(Nx.Type.euler_gamma_binary(type), type, opts)
  end

  defp from_binary(binary, type, opts) do
    opts = keyword!(opts, [:backend])
    {backend, backend_options} = backend_from_options!(opts) || Nx.default_backend()
    backend.from_binary(%Nx.Tensor{type: type, shape: {}, names: []}, binary, backend_options)
  end
end

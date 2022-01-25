defmodule Nx.Constants do
  @moduledoc """
  Common constants used in computations.
  """

  import Nx.Shared
  import Nx.Defn.Kernel, only: [keyword!: 2]

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
defmodule EXLA.Sharding do
  @moduledoc """
  Helper module for defining Shardy device meshes and tensor sharding specifications.
  """

  defmodule DeviceMesh do
    @moduledoc """
    Represents a device mesh configuration.
    """
    @enforce_keys [:name, :axes]
    defstruct [:name, :axes]

    @type axis :: {name :: String.t(), size :: pos_integer()}
    @type t :: %__MODULE__{
            name: String.t(),
            axes: [axis()]
          }
  end

  defmodule TensorSharding do
    @moduledoc """
    Represents a sharding specification for a tensor.
    """
    @enforce_keys [:mesh_name, :axes]
    defstruct [:mesh_name, :axes]

    @type dim_sharding :: [String.t()]
    @type t :: %__MODULE__{
            mesh_name: String.t(),
            axes: [dim_sharding()]
          }
  end

  @doc """
  Creates a device mesh definition.

  ## Examples

      iex> EXLA.Sharding.mesh(:my_mesh, x: 2, y: 4)
      %EXLA.Sharding.DeviceMesh{name: "my_mesh", axes: [{"x", 2}, {"y", 4}]}
  """
  def mesh(name, axes) when (is_atom(name) or is_binary(name)) and is_list(axes) do
    normalized_axes =
      Enum.map(axes, fn {k, v} -> {to_string(k), v} end)

    %DeviceMesh{name: to_string(name), axes: normalized_axes}
  end

  @doc """
  Creates a sharding specification for a tensor.

  The `dim_shardings` list must match the rank of the tensor.
  Each element is a list of axis names that the corresponding dimension is sharded on.

  ## Examples

      # Rank 2 tensor, dim 0 sharded on "x", dim 1 sharded on "y"
      iex> EXLA.Sharding.sharding(:my_mesh, [["x"], ["y"]])
      %EXLA.Sharding.TensorSharding{mesh_name: "my_mesh", axes: [["x"], ["y"]]}

      # Rank 2 tensor, dim 0 sharded on "x", dim 1 replicated
      iex> EXLA.Sharding.sharding(:my_mesh, [["x"], []])
      %EXLA.Sharding.TensorSharding{mesh_name: "my_mesh", axes: [["x"], []]}
  """
  def sharding(mesh_name, dim_shardings) do
    %TensorSharding{mesh_name: to_string(mesh_name), axes: dim_shardings}
  end

  @doc """
  Creates a fully replicated sharding specification (empty list for all dims).
  """
  def replicated(mesh_name, rank) do
    %TensorSharding{mesh_name: to_string(mesh_name), axes: List.duplicate([], rank)}
  end
end

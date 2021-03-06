defmodule Nx.BinaryBackend.Weights do

  @moduledoc """
  Weight is the per element count of scalar items for an axis of a
  tensor.

  For example, given a tensor with the shape {3, 4, 5} the dimensional
  length of axis 2 is 5 and each of the 5 elements in axis 2 represents 1
  scalar value. Therefore the weight of axis 2 is 1 - the last axis of a
  non-scalar tensor is always 1. For axis 1, the dimensional length is
  4 and each element in axis 1 represents 5 scalar elements so the
  weight of axis 1 is 4. Finally for axis 0, the dimensional length is
  3 and each of the 3 elements in axis 0 holds 20 scalar elements and
  so the weight of axis 0 is 20.

  Furthermore, if a tensor of shape {3, 4, 5} was concatenated side-by-side
  with another tensor of shape {3, 4, 5} the result would be a tensor of
  shape {2, 3, 4, 5}. Shape {2, 3, 4, 5} has a weight of 60 at axis 0 and
  the size of the shape {3, 4, 5} is also 60! Each element in axis 0 of a
  tensor of shape {2, 3, 4, 5} contains a {3, 4, 5} tensor and the size
  of a tensor is the sum total count of the scalar elements of a tensor:

      iex> t = Nx.iota({3, 4, 5})
      iex> t2 = Nx.concatenate([t, t]) |> Nx.reshape({2, 3, 4, 5})
      iex> t2.shape
      {2, 3, 4, 5}

      iex> {2, 3, 4, 5} |> Weights.build() |> Weights.weight_of_axis(0)
      60

      iex> Nx.size({3, 4, 5})
      60

  TL;DR - The weight of an axis is the size of that axis's sub-shape.

  For convenience the size of the shape is kept along side the weights -
  for this reason it is recommended to use the functions in this module
  to work with weights instead of treating weights like a collection of
  tuples.
  """

  @doc """
  Builds weights from a shape.

  ## Examples

      iex> Weights.build({10, 10, 10})
      {1000, {100, 10, 1}}

      iex> Weights.build({3, 2, 3, 2})
      {36, {12, 6, 2, 1}}

      iex> Weights.build({3, 3, 3, 3, 3})
      {243, {81, 27, 9, 3, 1}}

      iex> Weights.build({})
      {0, {}}

      iex> Weights.build({100})
      {100, {1}}

      iex> Weights.build({2, 3})
      {6, {3, 1}}
  """
  def build({}) do
   {0, {}}
  end

  def build({d0}) do
   {d0, {1}}
  end

  def build({d0, d1}) do
    {d1 * d0, {d1, 1}}
  end

  def build({d0, d1, d2}) do
    w0 = d1 * d2
    {d0 * w0, {w0, d2, 1}}
  end

  def build({d0, d1, d2, d3}) do
    w1 = d2 * d3
    w0 = w1 * d1
    {d0 * w0, {w0, w1, d3, 1}}
  end

  def build(shape) when tuple_size(shape) > 0 do
    # just a little slower for rank 5 or greater
    {[size], weights} =
      shape
      |> Tuple.to_list()
      |> Enum.reduce([1], fn d, [prev_w | _] = acc ->
        [d * prev_w | acc]
      end)
      |> Enum.split(1)

    {size, List.to_tuple(weights)}
  end

  @doc """
  The size of the original shape used
  to build the weights.

  ## Examples

      iex> Weights.size(Weights.build({3, 5}))
      15

      iex> Nx.size({10, 2}) == Weights.size(Weights.build({10, 2}))
      true
  """
  def size({size, _}) do
    size
  end

  @doc """
  Returns the weight of the given axis.

  ## Examples

      iex> Weights.weight_of_axis(Weights.build({3, 4, 5}), 0)
      20

      iex> Weights.weight_of_axis(Weights.build({4, 5}), 0)
      5

      iex> Weights.weight_of_axis(Weights.build({4, 5}), 1)
      1

      iex> Weights.weight_of_axis(Weights.build({10}), 0)
      1
  """
  def weight_of_axis({_, weights}, axis) do
    elem(weights, axis)
  end
end
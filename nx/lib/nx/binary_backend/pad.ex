defmodule Nx.BinaryBackend.Pad do
  @moduledoc false

  alias Nx.BinaryBackend.Bits
  alias Nx.BinaryBackend.Index

  def run(%Nx.Tensor{} = t, %Nx.Tensor{} = pad_value, padding_config) do
    %{type: type, shape: shape, data: %{state: data}} = t

    pad_value =
      pad_value
      |> Nx.to_scalar()
      |> Bits.from_number(type)

    padding_config
    |> build_steps(shape)
    |> run_steps("", 0, pad_value, type, data)
  end

   defp build_steps(padding_config, shape) do
    shape_out = Nx.Shape.pad(shape, padding_config)
    weights_out = Nx.Shape.weights(shape_out)
    weights_in = Nx.Shape.weights(shape)

    padding_config
    |> Enum.with_index()
    |> Enum.map(fn {cfg, axis} ->
      build_step(cfg, axis, shape, weights_in, weights_out)
    end)
  end


  defp build_step({lo, hi, mid}, axis, shape_in, weights_in, weights_out) do

    max_axis? = axis == Nx.Shape.rank(shape_in) - 1

    weight_out = Bits.weight_of_axis(weights_out, axis)
    weight_in = Bits.weight_of_axis(weights_in, axis)
    
    remove_lo = abs(min(0, lo))
    remove_hi = abs(min(0, hi))
    
    add_lo = max(0, lo)
    add_lo_wo = add_lo * weight_out
    add_hi = max(0, hi)
    add_hi_wo = add_hi * weight_out

    dim_in = elem(shape_in, axis)

    interior = (dim_in - 1) * mid
    repeat_count = (dim_in + interior) - (remove_lo + remove_hi)

    start = remove_lo
    stop = (start + repeat_count) - 1
    repeat_iter =
      if start < 0 || stop < 0 || stop < start do
        Index.range(0)
      else
        start..stop
      end

    cycle_size = 1 + mid

    List.flatten([
      pad_step(add_lo_wo),
      interior_step(max_axis?, repeat_iter, cycle_size, weight_in, weight_out),
      pad_step(add_hi_wo)
    ])
  end

  defp pad_step(n) when n > 0, do: {:pad, n}
  defp pad_step(_), do: []

  defp interior_step(_, [], _, _, _) do
    # nothing to do
    []
  end

  defp interior_step(true, offset_by..stop, 1, _offset_w, _pad_n) do
    # weight not needed for max_axis
    # cycle_size == 1 means no interior padding
    {:slice_binary, offset_by, Enum.count(offset_by..stop)}
  end

  defp interior_step(true, iter, cycle_size, _offset_w, pad_n) do
    # weight not needed for max_axis
    {:slice_and_pad, iter, cycle_size, pad_n}
  end

  defp interior_step(false, iter, 1, offset_weight, _pad_n) do
    # cycle_size == 1 means no interior padding
    {:iter_axis, iter, offset_weight}
  end

  defp interior_step(false, iter, cycle_size, offset_weight, pad_n) do
    {:iter_axis_and_pad, iter, cycle_size, offset_weight, pad_n}
  end

  defp run_steps([steps | rest], acc, offset, pad_v, type, data) do
    for step <- steps, reduce: acc do
      acc ->
        case step do
          {:pad, pad_n} ->
            pad_n_times(acc, pad_v, pad_n)

          {:slice_binary, offset_by, len} ->
            acc <> Bits.slice(data, type, offset + offset_by, len)
          
          {:slice_and_pad, range, cycle_size, pad_n} ->
            slice_and_pad(acc, offset, range, cycle_size, pad_n, pad_v, type, data)

          {:iter_axis, iter, offset_weight} ->
            iter_axis(rest, acc, offset, pad_v, type, data, iter, offset_weight)

          {:iter_axis_and_pad, iter, cycle_size, offset_weight, pad_n} ->
            iter_axis_and_pad(rest, acc, offset, pad_v, type, data, iter, cycle_size, offset_weight, pad_n)
        end
    end
  end

  defp iter_axis_and_pad(steps, acc, offset, pad_v, type, data, iter, cycle_size, offset_weight, pad_n) do
    for i <- iter, reduce: acc do
      acc ->
        if rem(i, cycle_size) == 0 do
          # normalize the index
          i = div(i, cycle_size)
          run_steps(steps, acc, offset + (offset_weight * i), pad_v, type, data)
        else
          pad_n_times(acc, pad_v, pad_n)
        end
    end
  end

  defp iter_axis(steps, acc, offset, pad_v, type, data, iter, offset_weight) do
    for i <- iter, reduce: acc do
      acc ->
        run_steps(steps, acc, offset + (offset_weight * i), pad_v, type, data)
    end
  end

  defp slice_and_pad(acc, offset, start..stop, cycle_size, pad_n, pad_v, type, data) do
    for i <- start..stop, reduce: acc do
      acc ->
        if rem(i, cycle_size) == 0 do
          # normalize the index
          i = div(i, cycle_size)
          acc <> Bits.slice(data, type, offset + i, 1)
        else
          pad_n_times(acc, pad_v, pad_n)
        end
    end
  end

  defp pad_n_times(acc, pad_v, n) when n > 0 do
    pad_n_times(acc <> pad_v, pad_v, n - 1)
  end
  
  defp pad_n_times(acc, _pad_v, _n) do
    acc
  end
end
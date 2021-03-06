defmodule Nx.BinaryBackend.Pad do
  alias Nx.BinaryBackend.Bits

  def run(%Nx.Tensor{} = t, %Nx.Tensor{} = pad_value, padding_config) do
    %{type: type, shape: shape, data: %{state: data}} = t

    pad_value =
      pad_value
      |> Nx.to_scalar()
      |> Bits.from_number(type)

    axis_ctxs = build_axis_ctxs(padding_config, shape)

    run_ctx("", pad_value, 0, type, data, axis_ctxs)
  end

  defp build_axis_ctx({lo, hi, mid} = cfg, axis, shape_in, weights_in, weights_out) do
    if mid < 0 do
      raise ArgumentError, "pad/3 cannot handle negative interior value" <>
            " #{inspect(mid)} in padding config #{inspect(cfg)} at axis" <>
            " #{axis}"
    end
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
        iter(0)
      else
        start..stop
      end

    cycle_size = 1 + mid
    
    {add_lo_wo, {repeat_iter, cycle_size, weight_in, weight_out}, add_hi_wo}
  end


  defp run_ctx(acc, pad_value, offset, type, data, [ctx | rest]) do
    {_, sizeof} = type
    {add_lo_wo, {repeat_iter, cycle_size, weight_in, weight_out}, add_hi_wo} = ctx
    on_max_axis? = rest == []

    acc = add_n_pad_values(acc, add_lo_wo, pad_value)

    acc =
      for i <- repeat_iter, reduce: acc do
        acc ->

          slice? = rem(i, cycle_size) == 0
          cond do
            slice? && on_max_axis? ->
              # normalize the index
              slice_i = div(i, cycle_size)

              offset = offset + slice_i * weight_in

              bits_offset = offset * sizeof
              <<_::size(bits_offset)-bitstring, target::size(sizeof)-bitstring, _::bitstring>> = data
              acc <> target

            slice? ->
              # normalize the index
              slice_i = div(i, cycle_size)
              offset = offset + slice_i * weight_in
              run_ctx(acc, pad_value, offset, type, data, rest)

            true ->
              # not slicing and not removed and not last iter means interior padding.
              add_n_pad_values(acc, weight_out, pad_value)
          end
      end

    acc = add_n_pad_values(acc, add_hi_wo, pad_value)

    acc
  end


  defp build_axis_ctxs(padding_config, shape) do
    shape_out = Nx.Shape.pad(shape, padding_config)
    weights_out = Nx.Shape.weights(shape_out)
    weights_in = Nx.Shape.weights(shape)
 
    {_, ctxs} = Enum.reduce(padding_config, {0, []}, fn cfg, {axis, acc} ->
      ctx = build_axis_ctx(cfg, axis, shape, weights_in, weights_out)
      {axis + 1, [ctx | acc]}
    end)

    Enum.reverse(ctxs)
  end

  defp iter(0), do: []
  defp iter(n) when n > 0, do: 0..(n - 1)


  defp add_n_pad_values(acc, n, pad_value) when n > 0 do
    for _ <- 1..n, into: acc do
      pad_value
    end
  end

  defp add_n_pad_values(acc, _n, _pad_value) do
    acc
  end
end
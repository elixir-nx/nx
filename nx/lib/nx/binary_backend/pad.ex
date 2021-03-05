defmodule Nx.BinaryBackend.Pad do
  alias Nx.BinaryBackend.Bits

  def run(%Nx.Tensor{} = t, %Nx.Tensor{} = pad_value, padding_configs) do
    %{type: type, shape: shape, data: %{state: data}} = t

    pad_value =
      pad_value
      |> Nx.to_scalar()
      |> Bits.from_number(type)

    axis_ctxs = build_axis_ctxs(padding_configs, shape)

    run_ctx("", pad_value, 0, type, data, axis_ctxs)
  end

  defp build_axis_ctx({lo, hi, mid} = cfg, axis, shape_in, shape_out, weights_in, weights_out) do
    if mid < 0 do
      raise ArgumentError, "invalid padding config #{cfg} for axis #{axis}"
    end
    weight_out = Bits.weight_of_axis(weights_out, axis)
    weight_in = Bits.weight_of_axis(weights_in, axis)
    
    IO.inspect({weights_in, weights_out}, label: :weights)
    
    remove_lo = abs(min(0, lo))
    remove_hi = abs(min(0, hi))
    
    add_lo = max(0, lo)
    add_lo_wo = add_lo * weight_out
    add_hi = max(0, hi)
    add_hi_wo = add_hi * weight_out

    # mid_wi = mid * weight_in
    mid_wo = mid * weight_out

    dim_in = elem(shape_in, axis)
    dim_out = elem(shape_out, axis)
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
    # IO.inspect(binding(), label: :CTX)
    
    {add_lo_wo, remove_lo, {repeat_iter, cycle_size, mid, mid_wo}, remove_hi, add_hi_wo, weight_in, weight_out}
  end


  defp run_ctx(acc, pad_value, offset, type, data, [ctx | rest]) do
    {_, sizeof} = type
    {add_lo_wo, remove_lo, {repeat_iter, cycle_size, mid, mid_wo}, remove_hi, add_hi_wo, weight_in, weight_out}= ctx
    on_max_axis? = rest == []
    height = length(rest)
    
    print(:start_height, height, acc, offset, height)
    print(:weight_in, weight_in, acc, offset, height)
    
    acc = add_n_pad_values(acc, add_lo_wo, pad_value)

    print(:add_lo_wo, add_lo_wo, acc, offset, height)

    # offset = offset + remove_lo * weight_in

    # print(:move_offset, remove_lo * weight_in, acc, offset, height)
    
    # last_iter = repeat_count - 1

    # print(:last_iter, last_iter, acc, offset)

    # acc =
    #   if repeat_count <= 0 do
    #     # acc = add_n_pad_values(acc, mid_wi, pad_value)
    #     raise "WHERE?"
    #     # print(:solo_mid_pad, repeat_count, acc, offset)
    #     acc
    #   else
    #     # print(:no_solo_mid_pad, repeat_count, acc, offset)
    #     acc 
    #   end

    
    # remove_lo_range = iter(remove_lo)

    print(:repeat_iter, repeat_iter, acc, offset, height)
    print(:cycle_size, cycle_size, acc, offset, height)

    acc =
      for i <- repeat_iter, reduce: acc do
        acc ->
          print(:start_iter, i, acc, offset, i)

          slice? = rem(i, cycle_size) == 0
          # removed? = i in remove_lo_range
          # last_iter? = i == last_iter

          # acc =
            cond do
              # removed? ->
              #   print(:ignored_lo_i, remove_lo_range, acc, offset, i)

              #   acc

              slice? && on_max_axis? ->
                # normalize the index
                slice_i = div(i, cycle_size)
                print(:bin_slice_i, slice_i, acc, offset, i, height)
                
                # we don't need weight_in because on the max axis weight_in == 1
                offset = offset + slice_i * weight_in
                # offset = offset + slice_i
          
                print(:bin_offset_i, slice_i, acc, offset, i, height)

                bits_offset = offset * sizeof
                <<_::size(bits_offset)-bitstring, target::size(sizeof)-bitstring, _::bitstring>> = data
                acc = acc <> target

                print(:bin_done, slice_i, acc, offset, i, height)
                
                acc

              slice? ->
                # normalize the index
                slice_i = div(i, cycle_size)
                # print(:recurse_start, i, acc, offset, i)
                offset = offset + slice_i * weight_in
                # print(:recurse_i_offset, i * weight_in, acc, offset, i)

                print(:recurse_down, height, acc, offset, i, height)
                acc = run_ctx(acc, pad_value, offset, type, data, rest)
                print(:recurse_up, height, acc, offset, i, height)
                acc
              # last_iter? ->

              #   acc

              true ->
                # not slicing and not removed and not last iter means interior padding.
                acc = add_n_pad_values(acc, weight_out, pad_value)
                print(:pad_mid, weight_out, acc, offset, i, height)
                acc
            end
          
          # print(:past_inner, height, acc, offset, i)
          
          # acc =
          #   if  do
          #     print(:last_iter, last_iter, acc, offset, i)
              
          #   else
          #     acc = add_n_pad_values(acc, mid_wo, pad_value)
          #     print(:add_mid_wo, mid_wo, acc, offset, i)
          #     acc
          #   end
          # acc
      end
    
    print(:done_repeating, height, acc, offset, height)
    
    acc = add_n_pad_values(acc, add_hi_wo, pad_value)
    
    print(:add_hi_wo, add_hi_wo, acc, offset, height)
    
    acc
  end


  defp build_axis_ctxs(padding_configs, shape) do
    shape_out = Nx.Shape.pad(shape, padding_configs)
    weights_out = Nx.Shape.weights(shape_out)
    weights = Nx.Shape.weights(shape)
 
    {_, ctxs} = Enum.reduce(padding_configs, {0, []}, fn cfg, {axis, acc} ->
      ctx = build_axis_ctx(cfg, axis, shape, shape_out, weights, weights_out)
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

  def print(key, value, acc, offset, height, i \\ nil, padder \\ " ", seq \\ nil, indent \\ "   ") do
    :ok
  end

  def fmt(nil, padding, padder) do
    List.duplicate(padder, padding)
  end

  def fmt({:label, str}, padding, padder) do
    String.pad_trailing(str, padding, padder)
  end

  def fmt(item, padding, padder) do
    str = inspect(item, binaries: :as_binaries)
    String.pad_trailing(str, padding, padder)
  end
end
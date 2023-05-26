defmodule Nx.FFT do
  @moduledoc false
  # Default implementation of FFT that calculates radix-N FFTs
  # using Bluestein's algorithm. Relies on a base radix-2 implementation.
  import Nx.Defn

  import Nx.Constants, only: [i: 1, pi: 1]

  deftransform bluesteins_fft(tensor, opts) do
    n = Nx.axis_size(tensor, -1)

    # we use bitwise AND (&) to test if n and (n-1) have no common set bits.
    # If this condition holds, it means that there is only one set bit in
    # the binary representation of n, which is true for all powers of 2.
    is_power_of_two = Bitwise.band(n, n - 1)

    if is_power_of_two do
      fft_2(tensor, [{:levels, fft_2_levels(n)} | opts])
    else
      m = 2 ** ceil(:math.log2(2 * n - 2))
      bluesteins_fft_n(tensor, [{:m, m} | opts])
    end
  end

  # direct-to-integer calculation of trunc(:math.log2(n)) to avoid rounding errors
  def fft_2_levels(n, l \\ 0)
  def fft_2_levels(1, l), do: l
  def fft_2_levels(n, l), do: fft_2_levels(div(n, 2), l + 1)

  defnp bluesteins_fft_n(x, opts) do
    # radix-n
    n = Nx.axis_size(x, -1)
    m = opts[:m]

    t = Nx.type(x)
    t_cpx = Nx.Type.to_complex(t)

    iota = Nx.iota({n})

    exp_coefs = Nx.exp(-1 * pi(t) * i(t_cpx) * iota ** 2 / n)
    a = x * exp_coefs
    b = Nx.exp(pi(t) * i(t_cpx) * iota ** 2 / n)

    pad = m - n
    a_pad = bluesteins_fft_n_pad_config(Nx.rank(a), {0, pad, 0})

    a = Nx.pad(a, 0, a_pad)
    b = Nx.pad(b, 0, [{pad, 0, 0}])

    c_fft = fft_2(a, opts) * fft_2(b, opts)
    c = ifft(c_fft, opts)

    exp_coefs * Nx.slice_along_axis(c, 0, n, axis: -1)
  end

  deftransformp bluesteins_fft_n_pad_config(r, c) do
    List.duplicate({0, 0, 0}, r - 1) ++ [c]
  end

  defn fft_2(x, opts) do
    # base case for radix-2
    n = Nx.axis_size(x, -1)
    levels = opts[:levels]
    t = Nx.type(x)
    t_cpx = Nx.Type.to_complex(t)

    iota = Nx.iota({n}, type: :s64)
    y = Nx.take(x, bit_reverse(iota, levels: levels), axis: -1)

    starts = fft_2_starts(levels)
    js = fft_2_js(levels)

    {result, _} =
      while {y, {size = 2, half_size = 1}}, size <= n do
        w_base = Nx.exp(-2 * i(t_cpx) * pi(t) / size)
      end

    result
  end

  defn bit_reverse_order(x) do
    n = Nx.axis_size(x, -1)
    indices = Nx.iota({n})

    bit_reversed_indices =
      Nx.bitwise_xor(
        indices,
        Nx.bitwise_and(Nx.right_shift(indices, 1), Nx.quotient(n, 2) - 1 + Nx.remainder(n, 2))
      )

    Nx.take(x, bit_reversed_indices, axis: -1)
  end

  defn single_step_fft_2(x, twiddle_factors, opts \\ []) do
    # this considers x to be rank 1 (vectorization helps here)
    stride = opts[:stride]
    num_pairs = Nx.size(twiddle_factors)
    even_idx = Nx.iota({num_pairs}) * 2 * stride
    odd_idx = even_idx + stride

    even_terms = x[even_idx]
    odd_terms = x[odd_idx]
    odd_terms_times_twiddles = odd_terms * twiddle_factors

    s = even_terms + odd_terms_times_twiddles
    d = even_terms - odd_terms_times_twiddles

    0
    |> Nx.broadcast(x.shape)
    |> Nx.indexed_put(s_idx, s)
    |> Nx.indexed_put(d_idx, d)

    # TODO: this needs iota with strides
    jnp.arange(0, 2 * num_pairs * stride, 2 * stride)
  end

  defnp ifft(tensor, opts) do
    # ifft written in terms of the fft
  end
end

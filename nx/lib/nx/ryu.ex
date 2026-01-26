# Copyright Ericsson AB 2017-2019. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference implementation: https://github.com/erlang/otp/pull/2960

defmodule Nx.Ryu do
  @moduledoc false

  import Bitwise

  # Lookup tables - sized for f32 maximum (f16 also supported)
  @pow5_bitcount 61
  @pow5_inv_bitcount 59

  @doc """
  Convert IEEE 754 binary representation to shortest decimal string.

  Can handle any F32 format and below.

  Parameters:
    - bits: the raw bit pattern as an integer
    - mantissa_bits: number of mantissa bits (e.g., 10 for f16, 23 for f32, 52 for f64)
    - exponent_bits: number of exponent bits (e.g., 5 for f16, 8 for f32, 11 for f64)
    - modifier: optional modifier atom (e.g., `:fn` for formats with no infinities
      and NaN only when all mantissa bits are 1)

  Returns the shortest decimal string representation.
  """
  def bits_to_decimal(bits, mantissa_bits, exponent_bits, modifier \\ nil) do
    {sign, mantissa, exponent} = decode_ieee754(bits, mantissa_bits, exponent_bits)

    bias = (1 <<< (exponent_bits - 1)) - 1
    max_exponent = (1 <<< exponent_bits) - 1

    cond do
      exponent == 0 and mantissa == 0 ->
        if sign == 1, do: "-0.0", else: "0.0"

      exponent == max_exponent ->
        cond do
          modifier == :fn and mantissa == (1 <<< mantissa_bits) - 1 ->
            "NaN"

          modifier == :fn ->
            normal_number(sign, mantissa, exponent, mantissa_bits, exponent_bits, bias)

          mantissa != 0 ->
            "NaN"

          sign == 1 ->
            "-Inf"

          true ->
            "Inf"
        end

      true ->
        normal_number(sign, mantissa, exponent, mantissa_bits, exponent_bits, bias)
    end
  end

  defp normal_number(sign, mantissa, exponent, mantissa_bits, exponent_bits, bias) do
    case check_small_int(mantissa, exponent, mantissa_bits, bias) do
      {:int, m, e} ->
        {place, digits} = compute_shortest_int(m, e)
        digit_list = insert_decimal(place, digits, nil, mantissa_bits, exponent_bits)
        insert_minus(sign, digit_list)

      :not_int ->
        {place, digits} = fwrite_g_1(mantissa, exponent, mantissa_bits, bias)
        digit_list = insert_decimal(place, digits, nil, mantissa_bits, exponent_bits)
        insert_minus(sign, digit_list)
    end
  end

  # Decode IEEE 754 representation
  defp decode_ieee754(bits, mantissa_bits, exponent_bits) do
    sign_bit = bits >>> (mantissa_bits + exponent_bits)
    exponent_mask = (1 <<< exponent_bits) - 1
    mantissa_mask = (1 <<< mantissa_bits) - 1

    exponent = bits >>> mantissa_bits &&& exponent_mask
    mantissa = bits &&& mantissa_mask

    {sign_bit, mantissa, exponent}
  end

  # Check if the number is a small integer
  defp check_small_int(mantissa, exponent, mantissa_bits, bias) do
    big_pow = 1 <<< mantissa_bits
    m2 = big_pow ||| mantissa
    decode_correction = bias + mantissa_bits
    e2 = exponent - decode_correction

    cond do
      e2 > 0 or e2 < -mantissa_bits ->
        :not_int

      true ->
        mask = (1 <<< -e2) - 1
        fraction = m2 &&& mask

        if fraction == 0 do
          {:int, m2 >>> -e2, 0}
        else
          :not_int
        end
    end
  end

  # Remove trailing zeros from integer mantissa
  defp compute_shortest_int(m, e) when rem(m, 10) == 0 do
    compute_shortest_int(div(m, 10), e + 1)
  end

  defp compute_shortest_int(m, e) do
    {e, Integer.to_string(m)}
  end

  # Main Ryu algorithm
  defp fwrite_g_1(mantissa, exponent, mantissa_bits, bias) do
    {mf, ef} = decode(mantissa, exponent, mantissa_bits, bias)
    shift = mmshift(mantissa, exponent)
    mv = 4 * mf

    {q, vm, vr, vp, e10} = convert_to_decimal(ef, mv, shift)

    accept = rem(mantissa, 2) == 0
    {vm_is_trailing_zero, vr_is_trailing_zero, vp1} = bounds(mv, q, vp, accept, ef, shift)

    {d1, e1} = compute_shortest(vm, vr, vp1, vm_is_trailing_zero, vr_is_trailing_zero, accept)

    {e1 + e10, Integer.to_string(d1)}
  end

  # Decode mantissa and exponent
  defp decode(mantissa, 0, mantissa_bits, bias) do
    {mantissa, 1 - bias - mantissa_bits - 2}
  end

  defp decode(mantissa, exponent, mantissa_bits, bias) do
    big_pow = 1 <<< mantissa_bits
    {mantissa + big_pow, exponent - bias - mantissa_bits - 2}
  end

  # Mantissa shift for boundary calculations
  defp mmshift(0, e) when e > 1, do: 0
  defp mmshift(_m, _e), do: 1

  # Convert to decimal using lookup tables
  defp convert_to_decimal(e2, mv, shift) when e2 >= 0 do
    q = max(0, ((e2 * 78913) >>> 18) - 1)
    mul = inv_table_value(q)
    k = @pow5_inv_bitcount + pow5bits(q) - 1
    i = -e2 + q + k

    {vm, vr, vp} = mul_shift_all(mv, shift, i, mul)
    {q, vm, vr, vp, q}
  end

  defp convert_to_decimal(e2, mv, shift) when e2 < 0 do
    q = max(0, ((-e2 * 732_923) >>> 20) - 1)
    i = -e2 - q
    k = pow5bits(i) - @pow5_bitcount
    from_file = table_value(i)
    j = q - k

    {vm, vr, vp} = mul_shift_all(mv, shift, j, from_file)
    e10 = e2 + q

    {q, vm, vr, vp, e10}
  end

  defp pow5bits(e) do
    ((e * 1_217_359) >>> 19) + 1
  end

  # Multiply and shift for three boundary values
  defp mul_shift_all(mv, shift, j, mul) do
    a = mul_shift_64(mv - 1 - shift, mul, j)
    b = mul_shift_64(mv, mul, j)
    c = mul_shift_64(mv + 2, mul, j)
    {a, b, c}
  end

  defp mul_shift_64(m, mul, j) do
    (m * mul) >>> j
  end

  # Determine boundary conditions and trailing zeros
  defp bounds(mv, q, vp, _accept, e2, _shift) when e2 >= 0 and q <= 21 and rem(mv, 5) == 0 do
    {false, multiple_of_power_of_5(mv, q), vp}
  end

  defp bounds(mv, q, vp, true, e2, shift) when e2 >= 0 and q <= 21 do
    {multiple_of_power_of_5(mv - 1 - shift, q), false, vp}
  end

  defp bounds(mv, q, vp, _accept, e2, _shift) when e2 >= 0 and q <= 21 do
    modifier = if multiple_of_power_of_5(mv + 2, q), do: 1, else: 0
    {false, false, vp - modifier}
  end

  defp bounds(_mv, q, vp, true, e2, shift) when e2 < 0 and q <= 1 do
    {shift == 1, true, vp}
  end

  defp bounds(_mv, q, vp, false, e2, _shift) when e2 < 0 and q <= 1 do
    {false, true, vp - 1}
  end

  defp bounds(mv, q, vp, _accept, e2, _shift) when e2 < 0 and q < 63 do
    {false, (mv &&& (1 <<< q) - 1) == 0, vp}
  end

  defp bounds(_mv, _q, vp, _accept, _e2, _shift) do
    {false, false, vp}
  end

  defp multiple_of_power_of_5(value, q) do
    pow5factor(value) >= q
  end

  defp pow5factor(val) do
    pow5factor(div(val, 5), 0)
  end

  defp pow5factor(val, count) when rem(val, 5) != 0 do
    count
  end

  defp pow5factor(val, count) do
    pow5factor(div(val, 5), count + 1)
  end

  # Compute shortest representation
  defp compute_shortest(vm, vr, vp, false, false, _accept) do
    {vm1, vr1, removed, round_up} = general_case(vm, vr, vp, 0, false)
    output = vr1 + handle_normal_output_mod(vr1, vm1, round_up)
    {output, removed}
  end

  defp compute_shortest(vm, vr, vp, vm_is_trailing_zero, vr_is_trailing_zero, accept) do
    {vm1, vr1, removed, last_removed_digit} =
      handle_trailing_zeros(vm, vr, vp, vm_is_trailing_zero, vr_is_trailing_zero, 0, 0)

    output =
      vr1 + handle_zero_output_mod(vr1, vm1, accept, vm_is_trailing_zero, last_removed_digit)

    {output, removed}
  end

  # General case - no special trailing zero handling
  defp general_case(vm, vr, vp, removed, round_up) when div(vp, 100) <= div(vm, 100) do
    general_case_10(vm, vr, vp, removed, round_up)
  end

  defp general_case(vm, vr, vp, removed, _ru) do
    vm_d100 = div(vm, 100)
    vr_d100 = div(vr, 100)
    vp_d100 = div(vp, 100)
    round_up = rem(vr, 100) >= 50
    general_case_10(vm_d100, vr_d100, vp_d100, removed + 2, round_up)
  end

  defp general_case_10(vm, vr, vp, removed, round_up) when div(vp, 10) <= div(vm, 10) do
    {vm, vr, removed, round_up}
  end

  defp general_case_10(vm, vr, vp, removed, _ru) do
    vm_d10 = div(vm, 10)
    vr_d10 = div(vr, 10)
    vp_d10 = div(vp, 10)
    round_up = rem(vr, 10) >= 5
    general_case_10(vm_d10, vr_d10, vp_d10, removed + 1, round_up)
  end

  defp handle_normal_output_mod(vr, vm, round_up) when vr == vm or round_up, do: 1
  defp handle_normal_output_mod(_vr, _vm, _round_up), do: 0

  # Handle trailing zeros
  defp handle_trailing_zeros(vm, vr, vp, vm_tz, vr_tz, removed, last_removed_digit)
       when div(vp, 10) <= div(vm, 10) do
    vm_is_trailing_zero(vm, vr, vp, vm_tz, vr_tz, removed, last_removed_digit)
  end

  defp handle_trailing_zeros(
         vm,
         vr,
         vp,
         vm_is_trailing_zero,
         vr_is_trailing_zero,
         removed,
         last_removed_digit
       ) do
    vm_tz = vm_is_trailing_zero and rem(vm, 10) == 0
    vr_tz = vr_is_trailing_zero and last_removed_digit == 0

    handle_trailing_zeros(
      div(vm, 10),
      div(vr, 10),
      div(vp, 10),
      vm_tz,
      vr_tz,
      removed + 1,
      rem(vr, 10)
    )
  end

  defp vm_is_trailing_zero(vm, vr, _vp, false = _vm_tz, vr_tz, removed, last_removed_digit) do
    handle_50_dotdot_0(vm, vr, vr_tz, removed, last_removed_digit)
  end

  defp vm_is_trailing_zero(vm, vr, _vp, _vm_tz, vr_tz, removed, last_removed_digit)
       when rem(vm, 10) != 0 do
    handle_50_dotdot_0(vm, vr, vr_tz, removed, last_removed_digit)
  end

  defp vm_is_trailing_zero(vm, vr, vp, vm_tz, vr_tz, removed, last_removed_digit) do
    vm_is_trailing_zero(
      div(vm, 10),
      div(vr, 10),
      div(vp, 10),
      vm_tz,
      last_removed_digit == 0 and vr_tz,
      removed + 1,
      rem(vr, 10)
    )
  end

  defp handle_50_dotdot_0(vm, vr, true, removed, 5) when rem(vr, 2) == 0 do
    {vm, vr, removed, 4}
  end

  defp handle_50_dotdot_0(vm, vr, _vr_tz, removed, last_removed_digit) do
    {vm, vr, removed, last_removed_digit}
  end

  defp handle_zero_output_mod(_vr, _vm, _accept, _vm_tz, last_removed_digit)
       when last_removed_digit >= 5,
       do: 1

  defp handle_zero_output_mod(vr, vm, accept, vm_tz, _last_removed_digit)
       when vr == vm and (not accept or not vm_tz),
       do: 1

  defp handle_zero_output_mod(_vr, _vm, _accept, _vm_tz, _last_removed_digit), do: 0

  # Format output with decimal point - shortest format (decimal vs scientific)
  defp insert_decimal(place, s, _bits, _mantissa_bits, _exponent_bits) do
    l = byte_size(s)
    exp = place + l - 1
    exp_l = Integer.to_string(exp)
    exp_cost = byte_size(exp_l) + 2

    cond do
      place < 0 ->
        cond do
          exp >= 0 ->
            # Split mantissa: "123" with place=-1 -> "12.3"
            split_pos = l + place

            if split_pos > 0 and split_pos < l do
              s0 = binary_part(s, 0, split_pos)
              s1 = binary_part(s, split_pos, l - split_pos)
              s0 <> "." <> s1
            else
              insert_exp(exp_l, s)
            end

          2 - place - l <= exp_cost ->
            # Use "0.000...digits" format
            "0." <> :binary.copy("0", -place - l) <> s

          true ->
            insert_exp(exp_l, s)
        end

      true ->
        # place >= 0
        dot = if l == 1, do: 1, else: 0

        if exp_cost + dot >= place + 2 do
          # Use "digits000.0" format
          s <> :binary.copy("0", place) <> ".0"
        else
          insert_exp(exp_l, s)
        end
    end
  end

  defp insert_exp(exp_l, <<c::binary-size(1)>>) do
    c <> "e" <> exp_l
  end

  defp insert_exp(exp_l, <<c::binary-size(1), rest::binary>>) do
    c <> "." <> rest <> "e" <> exp_l
  end

  defp insert_minus(0, digits), do: digits
  defp insert_minus(1, digits), do: "-" <> digits

  defp table_value(0), do: 1_152_921_504_606_846_976
  defp table_value(1), do: 1_441_151_880_758_558_720
  defp table_value(2), do: 1_801_439_850_948_198_400
  defp table_value(3), do: 2_251_799_813_685_248_000
  defp table_value(4), do: 1_407_374_883_553_280_000
  defp table_value(5), do: 1_759_218_604_441_600_000
  defp table_value(6), do: 2_199_023_255_552_000_000
  defp table_value(7), do: 1_374_389_534_720_000_000
  defp table_value(8), do: 1_717_986_918_400_000_000
  defp table_value(9), do: 2_147_483_648_000_000_000
  defp table_value(10), do: 1_342_177_280_000_000_000
  defp table_value(11), do: 1_677_721_600_000_000_000
  defp table_value(12), do: 2_097_152_000_000_000_000
  defp table_value(13), do: 1_310_720_000_000_000_000
  defp table_value(14), do: 1_638_400_000_000_000_000
  defp table_value(15), do: 2_048_000_000_000_000_000
  defp table_value(16), do: 1_280_000_000_000_000_000
  defp table_value(17), do: 1_600_000_000_000_000_000
  defp table_value(18), do: 2_000_000_000_000_000_000
  defp table_value(19), do: 1_250_000_000_000_000_000
  defp table_value(20), do: 1_562_500_000_000_000_000
  defp table_value(21), do: 1_953_125_000_000_000_000
  defp table_value(22), do: 1_220_703_125_000_000_000
  defp table_value(23), do: 1_525_878_906_250_000_000
  defp table_value(24), do: 1_907_348_632_812_500_000
  defp table_value(25), do: 1_192_092_895_507_812_500
  defp table_value(26), do: 1_490_116_119_384_765_625
  defp table_value(27), do: 1_862_645_149_230_957_031
  defp table_value(28), do: 1_164_153_218_269_348_144
  defp table_value(29), do: 1_455_191_522_836_685_180
  defp table_value(30), do: 1_818_989_403_545_856_475
  defp table_value(31), do: 2_273_736_754_432_320_594
  defp table_value(32), do: 1_421_085_471_520_200_371
  defp table_value(33), do: 1_776_356_839_400_250_464
  defp table_value(34), do: 2_220_446_049_250_313_080
  defp table_value(35), do: 1_387_778_780_781_445_675
  defp table_value(36), do: 1_734_723_475_976_807_094
  defp table_value(37), do: 2_168_404_344_971_008_868
  defp table_value(38), do: 1_355_252_715_606_880_542
  defp table_value(39), do: 1_694_065_894_508_600_678
  defp table_value(40), do: 2_117_582_368_135_750_847
  defp table_value(41), do: 1_323_488_980_084_844_279
  defp table_value(42), do: 1_654_361_225_106_055_349
  defp table_value(43), do: 2_067_951_531_382_569_187
  defp table_value(44), do: 1_292_469_707_114_105_741
  defp table_value(45), do: 1_615_587_133_892_632_177
  defp table_value(46), do: 2_019_483_917_365_790_221
  defp table_value(47), do: 1_262_177_448_353_618_888

  # F32 inv_table_value (inverse powers of 5) - 55 entries
  defp inv_table_value(0), do: 576_460_752_303_423_489
  defp inv_table_value(1), do: 461_168_601_842_738_791
  defp inv_table_value(2), do: 368_934_881_474_191_033
  defp inv_table_value(3), do: 295_147_905_179_352_826
  defp inv_table_value(4), do: 472_236_648_286_964_522
  defp inv_table_value(5), do: 377_789_318_629_571_618
  defp inv_table_value(6), do: 302_231_454_903_657_294
  defp inv_table_value(7), do: 483_570_327_845_851_670
  defp inv_table_value(8), do: 386_856_262_276_681_336
  defp inv_table_value(9), do: 309_485_009_821_345_069
  defp inv_table_value(10), do: 495_176_015_714_152_110
  defp inv_table_value(11), do: 396_140_812_571_321_688
  defp inv_table_value(12), do: 316_912_650_057_057_351
  defp inv_table_value(13), do: 507_060_240_091_291_761
  defp inv_table_value(14), do: 405_648_192_073_033_409
  defp inv_table_value(15), do: 324_518_553_658_426_727
  defp inv_table_value(16), do: 519_229_685_853_482_763
  defp inv_table_value(17), do: 415_383_748_682_786_211
  defp inv_table_value(18), do: 332_306_998_946_228_969
  defp inv_table_value(19), do: 531_691_198_313_966_350
  defp inv_table_value(20), do: 425_352_958_651_173_080
  defp inv_table_value(21), do: 340_282_366_920_938_464
  defp inv_table_value(22), do: 544_451_787_073_501_542
  defp inv_table_value(23), do: 435_561_429_658_801_234
  defp inv_table_value(24), do: 348_449_143_727_040_987
  defp inv_table_value(25), do: 557_518_629_963_265_579
  defp inv_table_value(26), do: 446_014_903_970_612_463
  defp inv_table_value(27), do: 356_811_923_176_489_971
  defp inv_table_value(28), do: 570_899_077_082_383_953
  defp inv_table_value(29), do: 456_719_261_665_907_162
  defp inv_table_value(30), do: 365_375_409_332_725_730
  defp inv_table_value(31), do: 292_300_327_466_180_584
  defp inv_table_value(32), do: 467_680_523_945_888_934
  defp inv_table_value(33), do: 374_144_419_156_711_148
  defp inv_table_value(34), do: 299_315_535_325_368_918
  defp inv_table_value(35), do: 478_904_856_520_590_269
  defp inv_table_value(36), do: 383_123_885_216_472_215
  defp inv_table_value(37), do: 306_499_108_173_177_772
  defp inv_table_value(38), do: 490_398_573_077_084_435
  defp inv_table_value(39), do: 392_318_858_461_667_548
  defp inv_table_value(40), do: 313_855_086_769_334_039
  defp inv_table_value(41), do: 502_168_138_830_934_462
  defp inv_table_value(42), do: 401_734_511_064_747_569
  defp inv_table_value(43), do: 321_387_608_851_798_056
  defp inv_table_value(44), do: 514_220_174_162_876_889
  defp inv_table_value(45), do: 411_376_139_330_301_511
  defp inv_table_value(46), do: 329_100_911_464_241_209
  defp inv_table_value(47), do: 526_561_458_342_785_934
  defp inv_table_value(48), do: 421_249_166_674_228_747
  defp inv_table_value(49), do: 336_999_333_339_382_998
  defp inv_table_value(50), do: 539_198_933_343_012_796
  defp inv_table_value(51), do: 431_359_146_674_410_237
  defp inv_table_value(52), do: 345_087_317_339_528_190
  defp inv_table_value(53), do: 552_139_707_743_245_103
  defp inv_table_value(54), do: 441_711_766_194_596_083
end

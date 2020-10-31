defmodule Exla do
  @on_load :load_nifs

  app = Mix.Project.config()[:app]
  @doc false
  def load_nifs do
    path = :filename.join(:code.priv_dir(unquote(app)), 'libexla')
    :erlang.load_nif(path, 0)
  end

  def add(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.add/3.")

  def sub(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.sub/3.")

  def mul(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.mul/3.")

  def div(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.div/3.")

  def rem(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.rem/3.")

  def min(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.min/3.")

  def max(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.max/3.")

  def logical_and(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.and/3.")

  def logical_or(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.or/3.")

  def logical_xor(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.xor/3.")

  def shift_left(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.shift_left/3.")

  def shift_right_arithmetic(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.shift_right_arithmetic/3.")

  def shift_right_logical(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.shift_right_logical/3.")

  def eq(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.eq/3.")

  def eq_total_order(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.eq_total_order/3.")

  def ne(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.ne/3.")

  def ne_total_order(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.ne_total_order/3.")

  def ge(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.ge/3.")

  def ge_total_order(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.ge_total_order/3.")

  def gt(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.gt/3.")

  def gt_total_order(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.gt_total_order/3.")

  def le(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.le/3.")

  def le_total_order(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.le_total_order/3.")

  def lt(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.lt/3.")

  def lt_total_order(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.lt/3.")

  def pow(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.pow/3.")

  def complex(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.complex/3.")

  def atan2(_a, _b, _broadcast_dims \\ []),
    do: raise("Failed to load implementation of #{__MODULE__}.atan2/3.")

  def abs(_a), do: raise("Failed to load implementation of #{__MODULE__}.abs/1.")
  def exp(_a), do: raise("Failed to load implementation of #{__MODULE__}.exp/1.")
  def expm1(_a), do: raise("Failed to load implementation of #{__MODULE__}.expm1/1.")
  def floor(_a), do: raise("Failed to load implementation of #{__MODULE__}.floor/1.")
  def ceil(_a), do: raise("Failed to load implementation of #{__MODULE__}.ceil/1.")
  def round(_a), do: raise("Failed to load implementation of #{__MODULE__}.round/1.")
  def log(_a), do: raise("Failed to load implementation of #{__MODULE__}.log/1.")
  def log1p(_a), do: raise("Failed to load implementation of #{__MODULE__}.log1p/1.")
  def logistic(_a), do: raise("Failed to load implementation of #{__MODULE__}.logistic/1.")
  def sign(_a), do: raise("Failed to load implementation of #{__MODULE__}.sign/1.")
  def clz(_a), do: raise("Failed to load implementation of #{__MODULE__}.clz/1.")
  def cos(_a), do: raise("Failed to load implementation of #{__MODULE__}.cos/1.")
  def sin(_a), do: raise("Failed to load implementation of #{__MODULE__}.sin/1.")
  def tanh(_a), do: raise("Failed to load implementation of #{__MODULE__}.tanh/1.")
  def real(_a), do: raise("Failed to load implementation of #{__MODULE__}.real/1.")
  def imag(_a), do: raise("Failed to load implementation of #{__MODULE__}.imag/1.")
  def sqrt(_a), do: raise("Failed to load implementation of #{__MODULE__}.sqrt/1.")
  def rsqrt(_a), do: raise("Failed to load implementation of #{__MODULE__}.rsqrt/1.")
  def cbrt(_a), do: raise("Failed to load implementation of #{__MODULE__}.cbrt/1.")
  def is_finite(_a), do: raise("Failed to load implementation of #{__MODULE__}.is_finite/1.")
  def logical_not(_a), do: raise("Failed to load implementation of #{__MODULE__}.not/1.")
  def neg(_a), do: raise("Failed to load implementation of #{__MODULE__}.neg/1.")
  def conj(_a), do: raise("Failed to load implementation of #{__MODULE__}.conj/1.")

  def population_count(_a),
    do: raise("Failed to load implementation of #{__MODULE__}.population_count/1.")

  def dot(_a, _b), do: raise("Failed to load implementation of #{__MODULE__}.dot/2.")

  def constant_r1(_length, _value),
    do: raise("Failed to load implementation of #{__MODULE__}.constant_r1/2.")

  def get_or_create_local_client,
    do: raise("Failed to load implementation of #{__MODULE__}.get_or_create_local_client/0.")

  def get_computation_hlo_proto,
    do: raise("Failed to load implementation of #{__MODULE__}.get_computation_hlo_proto/0.")

  def run, do: raise("Failed to load implementation of #{__MODULE__}.run/0.")
end

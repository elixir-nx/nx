defmodule Exla.Op do
  @doc """
  References to XLA Ops. See: https://www.tensorflow.org/xla/operation_semantics
  """
  def after_all, do: :ok
  def all_reduce, do: :ok
  def all_to_all, do: :ok
  def batch_norm_grad, do: :ok
  def batch_norm_inference, do: :ok
  def batch_norm_training, do: :ok
  def bitcast_convert_type, do: :ok
  def broadcast, do: :ok
  def broadcast_in_dim, do: :ok
  def call, do: :ok
  def cholesky, do: :ok
  def clamp, do: :ok
  def collapse, do: :ok
  def collective_permute, do: :ok
  def concatenate, do: :ok
  def conditional, do: :ok
  def conv, do: :ok
  def conv_with_general_padding, do: :ok
  def convert_element_type, do: :ok
  def cross_replica_sum, do: :ok
  def custom_call, do: :ok
  def dot, do: :ok
  def dot_general, do: :ok
  def dynamic_slice, do: :ok
  def dynamic_update_slice, do: :ok
  def fft, do: :ok
  def gather, do: :ok
  def get_tuple_element, do: :ok
  def infeed, do: :ok
  def iota, do: :ok
  def map, do: :ok
  def pad, do: :ok
  def recv, do: :ok
  def reduce, do: :ok
  def reduce_precision, do: :ok
  def reduce_window, do: :ok
  def replica_id, do: :ok
  def reshape, do: :ok
  def rev, do: :ok
  def rng_normal, do: :ok
  def rng_uniform, do: :ok
  def rng_bit_generator, do: :ok
  def scatter, do: :ok
  def select, do: :ok
  def select_and_scatter, do: :ok
  def send, do: :ok
  def slice, do: :ok
  def sort, do: :ok
  def transpose, do: :ok
  def triangular_solve, do: :ok
  def tuple, do: :ok
  def while, do: :ok

  # Element-Wise Binary Ops
  def add(lhs, rhs), do: :ok
  def sub(lhs, rhs), do: :ok
  def mul(lhs, rhs), do: :ok
  def div(lhs, rhs), do: :ok
  def rem(lhs, rhs), do: :ok
  def max(lhs, rhs), do: :ok
  def min(lhs, rhs), do: :ok
  def and(lhs, rhs), do: :ok
  def or(lhs, rhs), do: :ok
  def xor(lhs, rhs), do: :ok
  def shift_left(lhs, rhs), do: :ok
  def shift_right_arithmetic(lhs, rhs), do: :ok
  def shift_right_logical(lhs, rhs), do: :ok
  def atan2(lhs, rhs), do: :ok
  def pow(lhs, rhs), do: :ok
  def complex(lhs, rhs), do: :ok

  # Element-Wise Comparison Ops
  def eq(lhs, rhs), do: :ok
  def ne(lhs, rhs), do: :ok
  def ge(lhs, rhs), do: :ok
  def gt(lhs, rhs), do: :ok
  def le(lhs, rhs), do: :ok
  def lt(lhs, rhs), do: :ok

  # Element-Wise Unary Ops
  def abs, do: :ok
  def acos, do: :ok
  def asin, do: :ok
  def atan, do: :ok
  def acosh, do: :ok
  def asinh, do: :ok
  def atanh, do: :ok
  def bessel_i0e, do: :ok
  def bessel_i1e, do: :ok
  def ceil, do: :ok
  def conj, do: :ok
  def cos, do: :ok
  def cosh, do: :ok
  def clz, do: :ok
  def digamma, do: :ok
  def erfc, do: :ok
  def erf, do: :ok
  def erfinv, do: :ok
  def exp, do: :ok
  def expm1, do: :ok
  def floor, do: :ok
  def imag, do: :ok
  def is_finite, do: :ok
  def lgamma, do: :ok
  def log, do: :ok
  def log1p, do: :ok
  def not, do: :ok
  def logistic, do: :ok
  def population_count, do: :ok
  def neg, do: :ok
  def real, do: :ok
  def reciprocal, do: :ok
  def round, do: :ok
  def rsqrt, do: :ok
  def sign, do: :ok
  def sin, do: :ok
  def sinh, do: :ok
  def sqrt, do: :ok
  def square, do: :ok
  def cbrt, do: :ok
  def tan, do: :ok
  def tanh, do: :ok
end
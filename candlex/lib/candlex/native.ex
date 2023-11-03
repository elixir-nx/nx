defmodule Candlex.Native do
  @moduledoc false

  mix_config = Mix.Project.config()
  version = mix_config[:version]
  source_url = mix_config[:package][:links]["GitHub"]
  mode = if Mix.env() in [:dev, :test], do: :debug, else: :release

  use RustlerPrecompiled,
    otp_app: :candlex,
    features: Application.compile_env(:candlex, :crate_features, []),
    base_url: "#{source_url}/releases/download/v#{version}",
    force_build: System.get_env("CANDLEX_NIF_BUILD") in ["1", "true"],
    mode: mode,
    version: version,
    nif_versions: ["2.16"],
    targets: [
      "aarch64-apple-darwin",
      "aarch64-unknown-linux-gnu",
      "x86_64-apple-darwin",
      "x86_64-unknown-linux-gnu"
    ]

  # Rustler will override all the below stub functions with real NIFs
  def from_binary(_binary, _dtype, _shape, _device), do: error()
  def to_binary(_tensor), do: error()
  def all(_tensor), do: error()
  def all_within_dims(_tensor, _dims, _keep_dims), do: error()
  def any(_tensor), do: error()
  def any_within_dims(_tensor, _dims, _keep_dims), do: error()
  def where_cond(_tensor, _on_true, _on_false), do: error()
  def narrow(_tensor, _dim, _start, _length), do: error()
  def gather(_tensor, _indexes, _dim), do: error()
  def index_select(_tensor, _indexes, _dim), do: error()
  def index_add(_tensor, _indexes, _source, _dim), do: error()
  def chunk(_tensor, _num_chunks), do: error()
  def squeeze(_tensor, _dim), do: error()
  def arange(_start, _end, _dtype, _shape, _device), do: error()
  def broadcast_to(_tensor, _shape), do: error()
  def reshape(_tensor, _shape), do: error()
  def to_type(_tensor, _dtype), do: error()
  def dtype(_tensor), do: error()
  def t_shape(_tensor), do: error()
  def concatenate(_tensors, _axis), do: error()
  def conv1d(_tensor, _kernel), do: error()
  def conv2d(_tensor, _kernel), do: error()
  def slice_scatter(_tensor, _src, _dim, _start), do: error()
  def pad_with_zeros(_tensor, _left, _right), do: error()
  def clamp(_tensor, _min, _max), do: error()

  for op <- [
        :abs,
        :acos,
        :acosh,
        :asin,
        :asinh,
        :atan,
        :atanh,
        :bitwise_not,
        :cbrt,
        :ceil,
        :cos,
        :cosh,
        :erf,
        :erfc,
        :erf_inv,
        :exp,
        :expm1,
        :floor,
        :is_infinity,
        :is_nan,
        :log,
        :log1p,
        :negate,
        :round,
        :rsqrt,
        :sigmoid,
        :sign,
        :sin,
        :sinh,
        :sqrt,
        :tan,
        :tanh
      ] do
    def unquote(op)(_tensor), do: error()
  end

  for op <- [
        :add,
        :atan2,
        :bitwise_and,
        :bitwise_or,
        :bitwise_xor,
        :divide,
        :dot,
        :equal,
        :greater,
        :greater_equal,
        :left_shift,
        :less,
        :less_equal,
        :logical_and,
        :logical_or,
        :logical_xor,
        :matmul,
        :max,
        :min,
        :multiply,
        :not_equal,
        :pow,
        :quotient,
        :remainder,
        :right_shift,
        :subtract
      ] do
    def unquote(op)(_left, _right), do: error()
  end

  def sum(_tensor, _dims, _keep_dims), do: error()
  def permute(_tensor, _dims), do: error()

  for op <- [:argmax, :argmin, :reduce_max, :reduce_min] do
    def unquote(op)(_tensor, _dim, _keep_dim), do: error()
  end

  def is_cuda_available(), do: error()
  def to_device(_tensor, _device), do: error()

  defp error(), do: :erlang.nif_error(:nif_not_loaded)
end

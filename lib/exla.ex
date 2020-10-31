defmodule Exla do
  @on_load :load_nifs

  app = Mix.Project.config()[:app]
  @doc false
  def load_nifs do
    path = :filename.join(:code.priv_dir(unquote(app)), 'bazel-bin/exla/libexla')
    :erlang.load_nif(path, 0)
  end

  def add(_a, _b), do: raise "Failed to load implementation of #{__MODULE__}.add/2."
  def constant_r1(_length, _value), do: raise "Failed to load implementation of #{__MODULE__}.constant_r1/2."
  def get_or_create_local_client, do: raise "Failed to load implementation of #{__MODULE__}.get_or_create_local_client/0."
  def run, do: raise "Failed to load implementation of #{__MODULE__}.run/0."
end

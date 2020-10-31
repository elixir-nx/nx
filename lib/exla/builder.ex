defmodule Exla.Builder do
  @moduledoc """
  Wrapper around the xla::XlaBuilder class.
  """

  @on_load :load_nifs

  app = Mix.Project.config()[:app]
  @doc false
  def load_nifs do
    path = :filename.join(:code.priv_dir(unquote(app)), 'bazel-bin/exla/libbuilder')
    :erlang.load_nif(path, 0)
  end

  def add(_a, _b), do: raise "BAD"
  def constant_r1(_length, _value), do: raise "BAD"
  def get_or_create_local_client, do: raise "BAD"
  def run, do: raise "BAD"
end

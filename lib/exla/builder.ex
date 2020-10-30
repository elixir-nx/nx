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
  def build, do: raise "BAD"
  def constant_from_array(_array), do: raise "BAD"
end

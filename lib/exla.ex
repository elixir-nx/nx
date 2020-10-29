defmodule Exla do
  @on_load :load_nifs

  app = Mix.Project.config[:app]
  @doc false
  def load_nifs do
    path = :filename.join(:code.priv_dir(unquote(app)), 'bazel-bin/exla/libexla')
    :erlang.load_nif(path, 0)
  end

  def create_builder, do: raise "Failed to load implementation of #{__MODULE__}.create_builder/0"
end

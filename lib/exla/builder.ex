defmodule Exla.Builder do
  @moduledoc """
  Wrapper around the xla::XlaBuilder class.

  The Builder object is created on application startup and maintained inside a GenServer.
  """

  @on_load :load_nifs

  app = Mix.Project.config()[:app]
  @doc false
  def load_nifs do
    path = :filename.join(:code.priv_dir(unquote(app)), 'bazel-bin/exla/libexla')
    :erlang.load_nif(path, 0)
  end

  alias __MODULE__, as: Builder

  defstruct [:name, :ref]

  use GenServer

  def start_link(name) do
    GenServer.start_link(__MODULE__, name, name: __MODULE__)
  end

  def build do
    GenServer.call(__MODULE__, :build)
  end

  def get do
    GenServer.call(__MODULE__, :get)
  end

  @impl true
  def init(name) do
    builder = Builder.create_builder(name)
    {:ok, struct(Builder, builder)}
  end

  @impl true
  def handle_call(:get, _from, %Builder{} = builder) do
    {:reply, builder.ref, builder}
  end

  @impl true
  def handle_call(:build, _from, %Builder{} = builder) do
    {:reply, _build(builder.ref), builder}
  end

  @doc false
  def _build(_ref),
    do: raise("Failed to load implementation #{__MODULE__}._build/0")

  @doc false
  def create_builder(_name),
    do: raise("Failed to load implementation #{__MODULE__}.create_builder/1")
end

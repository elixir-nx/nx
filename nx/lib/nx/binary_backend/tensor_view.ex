defmodule Nx.BinaryBackend.TensorView do
  alias Nx.BinaryBackend.Traverser
  alias Nx.BinaryBackend.View
  alias Nx.Tensor, as: T
  alias Nx.BinaryBackend, as: B

  def update(t, fun) do
    t = resolve_if_required(t)
    view = get_or_create_view(t)
    put(t, fun.(view))
  end

  def put(%T{data: data} = t, view) do
    %T{t | data: %B{data | view: view}}
  end

  def resolve(%T{data: %B{view: nil}} = t), do: t

  def resolve(t), do: do_resolve(t)

  def resolve_if_required(%T{data: %{view: view}} = t) do
    if view && View.must_be_resolved?(view) do
      do_resolve(t)
    else
      t
    end
  end

  def get_or_create_view(%T{shape: shape, data: %B{view: view}}) do
    view || View.build(shape)
  end

  def raw_binary(%T{data: %B{state: state}}), do: state

  defp do_resolve(%T{type: type, data: %B{state: bin, view: view}} = t) do
    data = %B{
      state: resolve_binary(view, type, bin),
      view: nil
    }
    %T{t | data: data}
  end

  defp resolve_binary(view, {_, sizeof} = type, bin) do
    view
    |> View.with_type(type)
    |> View.build_traverser()
    |> Traverser.reduce([], fn offset, acc ->
      <<_::size(offset)-bitstring, target::size(sizeof)-bitstring, _::bitstring>> = bin
      [acc | target]
    end)
    |> IO.iodata_to_binary()
  end
end
defmodule Nx.Defn.TemplateDiff do
  @moduledoc false
  import Nx, only: [is_tensor: 1]
  defstruct [:left, :right, :left_title, :right_title, :compatible]

  def build(left, right, left_title, right_title, _compatibility_fn \\ &Nx.compatible?/2)

  def build(left, right, left_title, right_title, compatibility_fn)
      when is_tensor(left) and is_tensor(right) do
    %__MODULE__{
      left: left,
      left_title: left_title,
      right: right,
      right_title: right_title,
      compatible: compatibility_fn.(left, right)
    }
  end

  def build(left, right, left_title, right_title, compatibility_fn) do
    left_impl = Nx.Container.impl_for(left)
    right_impl = Nx.Container.impl_for(right)

    if left_impl == right_impl and left_impl != nil do
      flatten = right |> Nx.Container.reduce([], &[&1 | &2]) |> Enum.reverse()

      {diff, acc} =
        Nx.Container.traverse(left, flatten, fn
          left, [] ->
            {%__MODULE__{left: left}, :incompatible_sizes}

          left, [right | acc] ->
            {build(left, right, left_title, right_title, compatibility_fn), acc}
        end)

      if acc == [] and compatible_keys?(left_impl, left, right) do
        diff
      else
        %__MODULE__{
          left: left,
          left_title: left_title,
          right: right,
          right_title: right_title,
          compatible: false
        }
      end
    else
      %__MODULE__{
        left: left,
        left_title: left_title,
        right: right,
        right_title: right_title,
        compatible: false
      }
    end
  end

  defp compatible_keys?(Nx.Container.Map, left, right),
    do: Enum.all?(Map.keys(left), &is_map_key(right, &1))

  defp compatible_keys?(_, _, _),
    do: true

  def build_and_inspect(
        left,
        right,
        left_title,
        right_title,
        compatibility_fn \\ &Nx.compatible?/2
      ) do
    left
    |> build(right, left_title, right_title, compatibility_fn)
    |> inspect()
  end

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(%Nx.Defn.TemplateDiff{left: left, right: nil}, opts) do
      inspect_as_template(left, opts)
    end

    def inspect(%Nx.Defn.TemplateDiff{left: left, compatible: true}, opts) do
      inspect_as_template(left, opts)
    end

    def inspect(
          %Nx.Defn.TemplateDiff{
            left: left,
            left_title: left_title,
            right: right,
            right_title: right_title
          },
          opts
        ) do
      {left_title, right_title} = centralize_titles(left_title, right_title)

      concat([
        IO.ANSI.green(),
        line(),
        "<<<<< #{left_title} <<<<<",
        line(),
        inspect_as_template(left, opts),
        line(),
        "==========",
        line(),
        IO.ANSI.red(),
        inspect_as_template(right, opts),
        line(),
        ">>>>> #{right_title} >>>>>",
        line(),
        IO.ANSI.reset()
      ])
    end

    defp centralize_titles(l, r) do
      l_len = String.length(l)
      r_len = String.length(r)
      max_len = max(l_len, r_len)

      {centralize_string(l, l_len, max_len), centralize_string(r, r_len, max_len)}
    end

    defp centralize_string(s, n, n), do: s

    defp centralize_string(s, l, n) do
      pad = div(n - l, 2)

      s
      |> String.pad_leading(l + pad)
      |> String.pad_trailing(n)
    end

    defp inspect_as_template(data, opts) do
      if Nx.Container.impl_for(data) != nil do
        data
        |> Nx.to_template()
        |> to_doc(
          update_in(opts.custom_options, &Keyword.put(&1, :skip_template_backend_header, true))
        )
      else
        to_doc(data, opts)
      end
    end
  end
end

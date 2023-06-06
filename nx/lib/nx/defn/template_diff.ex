defmodule Nx.Defn.TemplateDiff do
  @moduledoc false
  defstruct [:left, :right, :left_title, :right_title, :compatible]

  defp is_valid_container?(impl) do
    not is_nil(impl) and impl != Nx.Container.Any
  end

  def build(left, right, left_title, right_title, compatibility_fn \\ &Nx.compatible?/2) do
    left_impl = Nx.Container.impl_for(left)
    right_impl = Nx.Container.impl_for(right)

    l = is_valid_container?(left_impl)
    r = is_valid_container?(right_impl)

    cond do
      not l and not r ->
        %__MODULE__{
          left: left,
          left_title: left_title,
          right: right,
          right_title: right_title,
          compatible: left == right
        }

      not l or not r ->
        %__MODULE__{
          left: left,
          left_title: left_title,
          right: right,
          right_title: right_title,
          compatible: false
        }

      left_impl != right_impl ->
        %__MODULE__{
          left: left,
          left_title: left_title,
          right: right,
          right_title: right_title,
          compatible: false
        }

      l and r ->
        {diff, acc} =
          Nx.Defn.Composite.traverse(left, Nx.Defn.Composite.flatten_list([right]), fn
            left, [] ->
              {%__MODULE__{left: left}, :incompatible_sizes}

            left, [right | acc] ->
              {
                %__MODULE__{
                  left: left,
                  right: right,
                  left_title: left_title,
                  right_title: right_title,
                  compatible: compatibility_fn.(left, right)
                },
                acc
              }
          end)

        if acc == :incompatible_sizes do
          %__MODULE__{
            left: left,
            left_title: left_title,
            right: right,
            right_title: right_title,
            compatible: false
          }
        else
          diff
        end
    end
  end

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
      if is_number(data) or is_tuple(data) or
           (is_map(data) and Nx.Container.impl_for(data) != Nx.Container.Any) do
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

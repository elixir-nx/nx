defmodule Nx.Defn.ShardingCompiler.Shard do
  import Inspect.Algebra
  defstruct [:id, :axis, :input_id, :start, :length, :parents]

  def inspect(%__MODULE__{start: start, length: length}, inspect_opts)
      when is_nil(start) or is_nil(length) do
    color("Shard<>", :map, inspect_opts)
  end

  def inspect(
        %__MODULE__{id: id, axis: axis, start: start, length: length, input_id: input_id},
        inspect_opts
      ) do
    single_line = inspect_opts.custom_options[:single_line]
    print_axis = inspect_opts.custom_options[:print_axis]

    range_doc = "#{start}..#{start + length - 1}"
    input_id_doc = if(input_id, do: "(#{inspect(input_id)})", else: "")

    if single_line do
      concat([
        color("Shard<", :map, inspect_opts),
        if(print_axis && axis, do: "#{axis}: ", else: ""),
        range_doc,
        " (#{inspect(id)})",
        input_id_doc,
        color(">", :map, inspect_opts)
      ])
    else
      concat([
        color("Shard<", :map, inspect_opts),
        nest(
          concat([
            line(),
            if(print_axis && axis, do: "#{axis}: ", else: ""),
            range_doc,
            line(),
            "(#{inspect(id)})",
            line(),
            input_id_doc
          ]),
          2
        ),
        line(),
        color(">", :map, inspect_opts)
      ])
    end
  end

  defimpl Inspect do
    def inspect(mod, opts), do: Nx.Defn.ShardingCompiler.Shard.inspect(mod, opts)
  end

  @doc """
  Config is a map of axis index or name -> slices
  """
  def from_config(tensor, config, opts \\ []) do
    input_id = opts[:input_id]

    shards =
      Map.new(config, fn
        {axis_or_name, slices} ->
          axis =
            if is_atom(axis_or_name) do
              Nx.axis_index(tensor, axis_or_name)
            else
              axis_or_name
            end

          shards =
            Enum.map(slices, fn start..finish//1 ->
              id = make_ref()

              %__MODULE__{
                id: id,
                axis: axis,
                start: start,
                length: finish - start + 1,
                input_id: input_id,
                parents: []
              }
            end)

          {axis, shards}
      end)

    Enum.reduce(Nx.axes(tensor), shards, fn axis, shards_by_axis ->
      if Map.has_key?(shards_by_axis, axis) do
        shards_by_axis
      else
        # If no shards are given, assume a fully independent axis by default.
        # We can group shards as needed later.

        shards =
          Enum.map(0..(Nx.axis_size(tensor, axis) - 1), fn start ->
            id = make_ref()

            %__MODULE__{
              id: id,
              axis: axis,
              start: start,
              length: 1,
              input_id: input_id,
              parents: []
            }
          end)

        Map.put(shards_by_axis, axis, shards)
      end
    end)
  end
end

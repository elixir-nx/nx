f_rands = fn size -> for(_ <- 1..size, do: :rand.uniform()) end
i_rands = fn size -> for(_ <- 1..size, do: :rand.uniform(100)) end

rand = f_rands.(10_000)
md_f32 = rand |> Nx.tensor(type: {:f, 32}) |> Nx.reshape({100, 100})
md_f64 = rand |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({100, 100})

rand = i_rands.(10_000)
md_s64 = rand |> Nx.tensor(type: {:s, 64}) |> Nx.reshape({100, 100})
md_u8 = rand |> Nx.tensor(type: {:u, 8}) |> Nx.reshape({100, 100})

rand = f_rands.(16)
sm_f64 = rand |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({4, 4})
sm_f32 = rand |> Nx.tensor(type: {:f, 32}) |> Nx.reshape({4, 4})
rand = i_rands.(16)
sm_u8 = rand |> Nx.tensor(type: {:u, 8}) |> Nx.reshape({4, 4})

old_sort = fn t, axis, comparator ->
  opts = [axis: axis, comparator: comparator]
  fn ->
    Nx.BinaryBackend.old_sort(t, t, opts)
  end
end

new_sort = fn t, axis, comparator ->
  opts = [axis: axis, comparator: comparator]
  fn ->
    Nx.BinaryBackend.sort(t, t, opts)
  end
end

run = fn benches ->
  Benchee.run(
    benches,
    time: 10,
    memory_time: 2,
    print: %{
      configuration: false,
      fast_warning: false,
      benchmarking: true
    }
  )
end

run.(%{
  "old sort md_f32 axis: 0" => old_sort.(md_f32, 0, :asc),
  "new sort md_f32 axis: 0" => new_sort.(md_f32, 0, :asc),
})

run.(%{
  "old sort md_f64 axis: 0" => old_sort.(md_f64, 0, :asc),
  "new sort md_f64 axis: 0" => new_sort.(md_f64, 0, :asc),
})

run.(%{
  "old sort md_s64 axis: 0" => old_sort.(md_s64, 0, :asc),
  "new sort md_s64 axis: 0" => new_sort.(md_s64, 0, :asc),
})

run.(%{
  "old sort md_u8 axis: 0" => old_sort.(md_u8, 0, :asc),
  "new sort md_u8 axis: 0" => new_sort.(md_u8, 0, :asc),
})

run.(%{
  "old sort md_f32 axis: 1" => old_sort.(md_f32, 1, :asc),
  "new sort md_f32 axis: 1" => new_sort.(md_f32, 1, :asc),
})

run.(%{
  "old sort md_f64 axis: 1" => old_sort.(md_f64, 1, :asc),
  "new sort md_f64 axis: 1" => new_sort.(md_f64, 1, :asc),
})

run.(%{
  "old sort md_s64 axis: 1" => old_sort.(md_s64, 1, :asc),
  "new sort md_s64 axis: 1" => new_sort.(md_s64, 1, :asc),
})

run.(%{
  "old sort md_u8 axis: 1" => old_sort.(md_u8, 1, :asc),
  "new sort md_u8 axis: 1" => new_sort.(md_u8, 1, :asc),
})

run.(%{
  "old sort sm_f32 axis: 0" => old_sort.(sm_f32, 0, :asc),
  "new sort sm_f32 axis: 0" => new_sort.(sm_f32, 0, :asc),
})

run.(%{
  "old sort sm_f64 axis: 0" => old_sort.(sm_f64, 0, :asc),
  "new sort sm_f64 axis: 0" => new_sort.(sm_f64, 0, :asc),
})

run.(%{
  "old sort sm_u8 axis: 0" => old_sort.(sm_u8, 0, :asc),
  "new sort sm_u8 axis: 0" => new_sort.(sm_u8, 0, :asc),
})

run.(%{
  "old sort sm_f32 axis: 1" => old_sort.(sm_f32, 1, :asc),
  "new sort sm_f32 axis: 1" => new_sort.(sm_f32, 1, :asc),
})

run.(%{
  "old sort sm_f64 axis: 1" => old_sort.(sm_f64, 1, :asc),
  "new sort sm_f64 axis: 1" => new_sort.(sm_f64, 1, :asc),
})

run.(%{
  "old sort sm_u8 axis: 1" => old_sort.(sm_u8, 1, :asc),
  "new sort sm_u8 axis: 1" => new_sort.(sm_u8, 1, :asc),
})

f_rands = fn size -> for(_ <- 1..size, do: :rand.uniform()) end
i_rands = fn size -> for(_ <- 1..size, do: :rand.uniform(100)) end

rand = f_rands.(10_000)
md_f32 = rand |> Nx.tensor(type: {:f, 32}) |> Nx.reshape({100, 100})
md_f64 = rand |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({100, 100})

rand = f_rands.(16)
sm_f64 = rand |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({4, 4})
sm_f32 = rand |> Nx.tensor(type: {:f, 32}) |> Nx.reshape({4, 4})
rand = i_rands.(16)
sm_u8 = rand |> Nx.tensor(type: {:u, 8}) |> Nx.reshape({4, 4})

rand = f_rands.(1600)
tall_f64 = rand |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({400, 4})
wide_f64 = rand |> Nx.tensor(type: {:f, 64}) |> Nx.reshape({4, 400})

rand = i_rands.(1600)
tall_u8 = rand |> Nx.tensor(type: {:u, 8}) |> Nx.reshape({400, 4})
wide_u8 = rand |> Nx.tensor(type: {:u, 8}) |> Nx.reshape({4, 400})

defmodule Dots do
  alias Nx.BinaryBackend.BinReducer
  alias Nx.BinaryBackend.TraverserReducer
  alias Nx.BinaryBackend
  
  def dot_bin(out, %{type: t1} = left, axes1, %{type: t2} = right, axes2) do
    BinReducer.bin_zip_reduce(out, left, axes1, right, axes2, 0, fn lhs, rhs, acc ->
      res = BinaryBackend.binary_to_number(lhs, t1) * BinaryBackend.binary_to_number(rhs, t2) + acc
      {res, res}
    end)
  end

  def dot_trav(out, %{type: t1} = left, axes1, %{type: t2} = right, axes2) do
    TraverserReducer.bin_zip_reduce(out, left, axes1, right, axes2, 0, fn lhs, rhs, acc ->
      res = BinaryBackend.binary_to_number(lhs, t1) * BinaryBackend.binary_to_number(rhs, t2) + acc
      {res, res}
    end)
  end
end

dot4_bin = fn t -> 
  fn -> Dots.dot_bin(t, t, [1], t, [0]) end
end

dot4_trav = fn t -> 
  fn -> Dots.dot_trav(t, t, [1], t, [0]) end
end

dot4_bin2 = fn t1, t2 ->
  {d0, _} = t1.shape
  {_, d1} = t2.shape
  t_out = %{t1 | shape: {d0, d1}}
  fn -> Dots.dot_bin(t_out, t1, [1], t2, [0]) end
end

dot4_trav2 = fn t1, t2 ->
  {d0, _} = t1.shape
  {_, d1} = t2.shape
  t_out = %{t1 | shape: {d0, d1}}
  fn -> Dots.dot_trav(t_out, t1, [1], t2, [0]) end
end

md_benches = %{
  "dot4_bin md_f32" => dot4_bin.(md_f32),
  "dot4_trav md_f32" => dot4_trav.(md_f32),

  "dot4_bin md_f64" => dot4_bin.(md_f64),
  "dot4_trav md_f64" => dot4_trav.(md_f64),
}

sm_benches = %{
  "dot4_bin sm_f64" => dot4_bin.(sm_f64),
  "dot4_trav sm_f64" => dot4_trav.(sm_f64),

  "dot4_bin sm_f32" => dot4_bin.(sm_f32),
  "dot4_trav sm_f32" => dot4_trav.(sm_f32),

  "dot4_bin sm_u8" => dot4_bin.(sm_u8),
  "dot4_trav sm_u8" => dot4_trav.(sm_u8),
}

lg_rect_benches = %{
  "dot4_bin lg_rect_f64" => dot4_bin2.(tall_f64, wide_f64),
  "dot4_trav lg_rect_f64" => dot4_trav2.(tall_f64, wide_f64),
}

sm_rect_benches = %{
  "dot4_bin sm_rect_f64" => dot4_bin2.(wide_f64, tall_f64),
  "dot4_trav sm_rect_f64" => dot4_trav2.(wide_f64, tall_f64),
}

IO.puts("md benches {100, 100}")
Benchee.run(
  md_benches,
  time: 10,
  memory_time: 2
)

IO.puts("sm benches {4, 4}")
Benchee.run(
  sm_benches,
  time: 10,
  memory_time: 2
)

IO.puts("lg rect benches dot({400, 4}, {4, 400})")
Benchee.run(
  lg_rect_benches,
  time: 10,
  memory_time: 2
)

IO.puts("sm rect benches dot({4, 400}, {400, 4})")
Benchee.run(
  sm_rect_benches,
  time: 10,
  memory_time: 2
)

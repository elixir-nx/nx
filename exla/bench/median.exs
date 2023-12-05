# mix run bench/median.exs
Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA)

key = Nx.Random.key(System.os_time())

inputs_median = %{
  "10000" => elem(Nx.Random.shuffle(key, Nx.iota({10000})), 0),
  #"100000" => elem(Nx.Random.shuffle(key, Nx.iota({100000})), 0),
  #"1000000" => elem(Nx.Random.shuffle(key, Nx.iota({1000000})), 0)
}


Benchee.run(
  %{
    "sort" => fn x ->
      Nx.sort(x)[2222]
     end,
    "lazy_select" => fn x ->
      Nx.lazy_select(x, k: 2222)
     end
  },
  time: 10,
  memory_time: 2,
  inputs: inputs_median
)

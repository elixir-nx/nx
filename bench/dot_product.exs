Exla.get_or_create_local_client()

# For small lists, Elixir wins big which is expected because of overhead associated with XLA
# The BEAM Crashes on my machine at 100_000_000 elements, XLA handles 100_000_000 and even 1_000_000_000 elements
# with ease AND faster than Elixir with 10_000_000 elements. This isn't even running on a GPU yet...

t1 = for _ <- 1..10_000_000, do: 1000

elixir_dot =
  fn a, b ->
    a
    |> Enum.zip(b)
    |> Enum.reduce(0, fn {x, y}, acc -> acc + (x*y) end)
  end

Benchee.run(%{
  "elixir dot" => fn _ -> elixir_dot.(t1, t1) end,
  "xla dot" => fn exec -> Exla.run(exec, {}, %{}) end
},
  time: 10,
  memory_time: 2,
  before_each:
    fn _ ->
      Exla.dot(Exla.constant_r1(10_000_000, 1000), Exla.constant_r1(10_000_000, 1000))
      comp = Exla.build()
      exec = Exla.compile(comp, {}, %{})
      hd(exec)
    end)
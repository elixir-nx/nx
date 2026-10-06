# Contemporary CPU timings for Nx.fft/1 and Nx.ifft/1.
#
# The host client runs the executable to completion before returning.
# It does not take a thread count. schedulers_online/0 below is the BEAM
# scheduler count, not an XLA pool size.
#
# Cold call, then the median of the next five. From exla/:
#
#     mix run bench/fft.exs
#
# Torchx and NxEigen have no compile step. FFTW is not linked here.
# The NxEigen numbers are whatever FFT that project was built with.
#
# From nx/torchx:
#
#     FFT_BENCH=torchx mix run ../exla/bench/fft.exs
#
# From a checkout of nx_eigen next to this repo:
#
#     FFT_BENCH=eigen NX_EIGEN_FFT_LIB=eigen mix run ../nx/exla/bench/fft.exs

defmodule FFTBench do
  @cases [
    {:f32, {1, 400}},
    {:f32, {1, 512}},
    {:f32, {1024}},
    {:f32, {1000}},
    {:f32, {3000, 400}},
    {:f32, {3000, 512}},
    {:f32, {4096, 512}},
    {:f64, {1, 512}},
    {:f64, {1024}}
  ]

  @ops [:fft, :ifft]

  def run(mode) do
    Process.put(:fft_bench_mode, mode)
    IO.puts("schedulers_online: #{System.schedulers_online()}")
    IO.puts("mode: #{mode}")

    case mode do
      "exla" ->
        EXLA.Client.fetch!(:host)
        attach()
        prime(&exla_tensor/2)
        Enum.each(cases(), fn {type, shape} -> exla_case(type, shape) end)
        exla_pad_case()

      "torchx" ->
        prime(&torchx_tensor/2)

        Enum.each(cases(), fn {type, shape} ->
          eager_case("torchx", &torchx_tensor/2, type, shape)
        end)

      "eigen" ->
        prime(&eigen_tensor/2)

        Enum.each(cases(), fn {type, shape} ->
          eager_case("eigen", &eigen_tensor/2, type, shape)
        end)
    end
  end

  defp cases, do: @cases
  defp ops, do: @ops

  defp prime(build) do
    tensor = build.(:f32, {8})
    Enum.each(ops(), fn op -> time(fn -> apply(Nx, op, [tensor]) end) end)
    release(tensor)
    take()
  end

  defp exla_case(type, shape) do
    tensor = exla_tensor(type, shape)
    # iota has its own compilation. Drop it before timing the transform.
    take()

    Enum.each(ops(), fn op ->
      cold_us = time(fn -> apply(Nx, op, [tensor]) end)
      cold_events = take()
      warm = warm_times(fn -> apply(Nx, op, [tensor]) end)
      warm_events = take()

      IO.puts(
        "exla cold #{op} #{inspect(shape)} #{type} #{cold_us} us compile #{compile_time(cold_events)} us events #{length(cold_events)}"
      )

      IO.puts(
        "exla warm #{op} #{inspect(shape)} #{type} median #{median(warm)} us events #{length(warm_events)}"
      )
    end)

    release(tensor)
  end

  # length: 1024 on a 1000-vector is the pad inside the EXLA lowering.
  # A plain 1024-vector does not take that pad.
  defp exla_pad_case do
    short = exla_tensor(:f32, {1000})
    take()
    cold_us = time(fn -> Nx.fft(short, length: 1024) end)
    cold_events = take()
    warm = warm_times(fn -> Nx.fft(short, length: 1024) end)
    warm_events = take()

    IO.puts(
      "exla cold fft {1000} length 1024 f32 #{cold_us} us compile #{compile_time(cold_events)} us events #{length(cold_events)}"
    )

    IO.puts(
      "exla warm fft {1000} length 1024 f32 median #{median(warm)} us events #{length(warm_events)}"
    )

    release(short)
  end

  defp eager_case(name, build, type, shape) do
    tensor = build.(type, shape)

    Enum.each(ops(), fn op ->
      time(fn -> apply(Nx, op, [tensor]) end)
      warm = warm_times(fn -> apply(Nx, op, [tensor]) end)
      IO.puts("#{name} warm #{op} #{inspect(shape)} #{type} median #{median(warm)} us")
    end)

    release(tensor)
  end

  defp exla_tensor(type, shape), do: Nx.iota(shape, type: nx_type(type), backend: EXLA.Backend)

  defp torchx_tensor(type, shape),
    do: Nx.iota(shape, type: nx_type(type), backend: {Torchx.Backend, device: :cpu})

  defp eigen_tensor(type, shape),
    do: Nx.iota(shape, type: nx_type(type), backend: NxEigen.Backend)

  defp nx_type(:f32), do: {:f, 32}
  defp nx_type(:f64), do: {:f, 64}

  defp time(fun) do
    {microseconds, out} = :timer.tc(fun)
    release(out)
    microseconds
  end

  # NxEigen.backend_deallocate/1 matches the data struct, not the tensor.
  defp release(tensor) do
    if Process.get(:fft_bench_mode) == "eigen" do
      :ok
    else
      Nx.backend_deallocate(tensor)
    end
  end

  defp warm_times(fun), do: Enum.map(1..5, fn _ -> time(fun) end)

  defp median(times) do
    times |> Enum.sort() |> Enum.at(2)
  end

  defp attach do
    :telemetry.detach("exla-fft-bench")

    :telemetry.attach(
      "exla-fft-bench",
      [:exla, :compilation],
      fn _, measurements, _, _ ->
        Agent.update(__MODULE__, &[measurements | &1])
      end,
      nil
    )
  end

  defp take do
    Agent.get_and_update(__MODULE__, fn events -> {Enum.reverse(events), []} end)
  end

  defp compile_time(events) do
    Enum.reduce(events, 0, fn
      %{compile_time: microseconds}, acc -> acc + microseconds
      _, acc -> acc
    end)
  end
end

{:ok, _pid} = Agent.start_link(fn -> [] end, name: FFTBench)
FFTBench.run(System.get_env("FFT_BENCH", "exla"))

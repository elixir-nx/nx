defmodule EXLA.MLIR.ContextPool do
  @behaviour NimblePool

  def checkout(fun) when is_function(fun, 1) do
    NimblePool.checkout!(
      __MODULE__,
      :checkout,
      fn _pool, context -> {fun.(context), :ok} end,
      :infinity
    )
  end

  @impl NimblePool
  def init_worker(pool_state) do
    {:ok, context} = EXLA.NIF.new_mlir_context()
    {:ok, context, pool_state}
  end

  @impl NimblePool
  # Transfer the port to the caller
  def handle_checkout(:checkout, _from, context, pool_state) do
    {:ok, context, context, pool_state}
  end

  @impl NimblePool
  def handle_checkin(:ok, _from, context, pool_state) do
    {:ok, context, pool_state}
  end

  @impl NimblePool
  def terminate_worker(_reason, _context, pool_state) do
    # GC will clean it up
    {:ok, pool_state}
  end
end

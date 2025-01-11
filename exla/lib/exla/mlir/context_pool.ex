defmodule EXLA.MLIR.ContextPool do
  @moduledoc false
  # Internal pool for MLIRContext reference management
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
  def init_pool(%{pool_size: pool_size}) do
    {:ok, thread_pool} = EXLA.NIF.mlir_new_thread_pool(pool_size)

    {:ok, %{thread_pool: thread_pool}}
  end

  @impl NimblePool
  def init_worker(%{thread_pool: thread_pool} = pool_state) do
    {:ok, context} = EXLA.NIF.mlir_new_context(thread_pool)
    {:ok, context, pool_state}
  end

  @impl NimblePool
  def handle_checkout(:checkout, _from, context, pool_state) do
    {:ok, context, context, pool_state}
  end

  @impl NimblePool
  def handle_checkin(:ok, _from, context, pool_state) do
    # We just keep the references around and let them die out upon worker termination/GC
    {:ok, context, pool_state}
  end

  @impl NimblePool
  def terminate_worker(_reason, _context, pool_state) do
    # GC will clean it up
    {:ok, pool_state}
  end
end

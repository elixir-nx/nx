defmodule EXLA.MLIR.IREE.InstancePool do
  @moduledoc false
  # Internal pool for MLIRContext reference management
  # @behaviour NimblePool

  def checkout(fun) when is_function(fun, 1) do
    {t, r} =
      :timer.tc(fn ->
        fun.(:persistent_term.get({EXLA.MLIR.IREE, :instance}))
      end)

    :telemetry.execute(
      [:exla, :mlir, :iree, :instance_pool, :checkout],
      %{duration: t}
    )

    r

    # NimblePool.checkout!(
    #   __MODULE__,
    #   :checkout,
    #   fn _pool, instance -> {fun.(instance), :ok} end,
    #   :infinity
    # )
  end

  # @impl NimblePool
  # def init_worker(pool_state) do
  #   {:ok, instance} = EXLA.MLIR.IREE.create_instance()
  #   {:ok, instance, pool_state}
  # end

  # @impl NimblePool
  # def handle_checkout(:checkout, _from, instance, pool_state) do
  #   {:ok, instance, instance, pool_state}
  # end

  # @impl NimblePool
  # def handle_checkin(:ok, _from, instance, pool_state) do
  #   # We just keep the references around and let them die out upon worker termination/GC
  #   {:ok, instance, pool_state}
  # end

  # @impl NimblePool
  # def terminate_worker(_reason, _instance, pool_state) do
  #   # GC will clean it up
  #   {:ok, pool_state}
  # end
end

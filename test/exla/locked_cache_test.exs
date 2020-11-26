defmodule Exla.LockedCacheTest do
  use ExUnit.Case, async: true

  alias Exla.LockedCache, as: LC

  test "caches keys", config do
    assert LC.run(config.test, fn -> :this_is_cached end) == :this_is_cached
    assert LC.run(config.test, fn -> flunk() end) == :this_is_cached
  end

  test "locks cache keys", config do
    %Task{pid: first_pid, ref: first_ref} = task_run(config.test, fn -> :this_is_cached end)
    assert_receive {:running, ^first_pid}

    %Task{pid: second_pid, ref: second_ref} = task_run(config.test, fn -> flunk() end)
    send(first_pid, :run)

    assert_receive {:DOWN, ^first_ref, _, _, _}
    assert_receive {:DOWN, ^second_ref, _, _, _}

    assert Process.info(self(), :messages) ==
             {:messages,
              [
                {first_ref, :this_is_cached},
                {second_ref, :this_is_cached}
              ]}

    refute_received {:running, ^second_pid}
  end

  test "allows cache to be recomputed if cache fails", config do
    assert catch_error(LC.run(config.test, fn -> raise "oops" end))
    assert LC.run(config.test, fn -> :this_is_cached end) == :this_is_cached
  end

  @tag :capture_log
  test "allows cache to be recomputed if cache fails when locked", config do
    Process.flag(:trap_exit, true)
    %Task{pid: error_pid, ref: error_ref} = task_run(config.test, fn -> raise "oops" end)
    assert_receive {:running, ^error_pid}

    %Task{pid: ok_pid} = ok_task = task_run(config.test, fn -> :this_is_cached end)
    refute_received {:running, ^ok_pid}

    send(error_pid, :run)
    assert_receive {:DOWN, ^error_ref, _, _, _}
    assert_receive {:running, ^ok_pid}

    send(ok_pid, :run)
    assert Task.await(ok_task) == :this_is_cached
    assert LC.run(config.test, fn -> flunk() end) == :this_is_cached
  end

  @tag :capture_log
  test "allows cache to be recomputed if cache exits", config do
    Process.flag(:trap_exit, true)
    %Task{pid: error_pid, ref: error_ref} = task_run(config.test, fn -> raise "oops" end)
    assert_receive {:running, ^error_pid}
    Process.exit(error_pid, :kill)
    assert_receive {:DOWN, ^error_ref, _, _, :killed}

    assert LC.run(config.test, fn -> :this_is_cached end) == :this_is_cached
  end

  @tag :capture_log
  test "allows cache to be recomputed if cache exiots when locked", config do
    Process.flag(:trap_exit, true)
    %Task{pid: error_pid, ref: error_ref} = task_run(config.test, fn -> raise "oops" end)
    assert_receive {:running, ^error_pid}

    %Task{pid: ok_pid} = ok_task = task_run(config.test, fn -> :this_is_cached end)
    refute_received {:running, ^ok_pid}

    Process.exit(error_pid, :kill)
    assert_receive {:DOWN, ^error_ref, _, _, :killed}
    assert_receive {:running, ^ok_pid}

    send(ok_pid, :run)
    assert Task.await(ok_task) == :this_is_cached
    assert LC.run(config.test, fn -> flunk() end) == :this_is_cached
  end

  defp task_run(key, fun) do
    parent = self()

    Task.async(fn ->
      LC.run(key, fn ->
        send(parent, {:running, self()})
        assert_receive :run
        fun.()
      end)
    end)
  end
end

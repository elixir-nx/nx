defmodule EXLA.Defn.LockeTest do
  use ExUnit.Case, async: true

  alias EXLA.Defn.Lock, as: L

  test "locks a given key while process is alive", config do
    parent = self()

    task1 =
      Task.async(fn ->
        L.lock(config.test)
        send(parent, :locked1)
        assert_receive :done
      end)

    assert_receive :locked1

    task2 =
      Task.async(fn ->
        L.lock(config.test)
        send(parent, :locked2)
      end)

    assert Process.alive?(task2.pid)
    send(task1.pid, :done)
    assert_receive :locked2
    assert Task.await(task1)
    assert Task.await(task2)
  end

  test "locks a given key until unlocked", config do
    parent = self()
    ref = L.lock(config.test)

    task =
      Task.async(fn ->
        L.lock(config.test)
        send(parent, :locked)
      end)

    :ok = L.unlock(ref)
    assert_receive :locked
    assert Task.await(task)
  end

  test "relocks a given key", config do
    parent = self()

    task1 =
      Task.async(fn ->
        assert_receive :done
      end)

    task2 =
      Task.async(fn ->
        ref = L.lock(config.test)
        send(parent, :locked)

        L.relock(
          ref,
          fn ->
            send(parent, :relocked)
          end,
          fn ->
            send(parent, :unlocked)
            {:lock, task1.pid, fn -> :unlock end}
          end
        )

        assert_receive :done
      end)

    assert_receive :locked

    task3 =
      Task.async(fn ->
        L.lock(config.test)
      end)

    send(task2.pid, :done)
    assert_receive :unlocked
    assert Task.await(task2)

    assert Process.alive?(task3.pid)
    send(task1.pid, :done)
    assert Task.await(task1)
    assert Task.await(task3)
  end
end

defmodule Nx.Defn.RuntimeCallEvaluatorTest.RemoteCallback do
  def callback(t, _opts) do
    Nx.add(Nx.as_type(t, :f32), 10.0)
  end
end

defmodule Nx.Defn.RuntimeCallEvaluatorTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  alias Nx.Defn.RuntimeCallEvaluatorTest.RemoteCallback, as: AliasMod

  setup do
    Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)
    :ok
  end

  def add_offset_callback(t, _opts) do
    Nx.add(Nx.as_type(t, :f32), 10.0)
  end

  defn add_offset(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, &add_offset_callback/2)
  end

  test "runtime_call with single output" do
    x = Nx.iota({5})
    y = add_offset(x)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert Nx.all_close(y, expected) |> Nx.to_number() == 1
  end

  defn add_offset_remote_unaliased(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, &Nx.Defn.RuntimeCallEvaluatorTest.RemoteCallback.callback/2)
  end

  defn add_offset_remote_aliased(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, &AliasMod.callback/2)
  end

  test "runtime_call with fully-qualified unaliased module (&Mod.fun/2)" do
    x = Nx.iota({5})
    y = add_offset_remote_unaliased(x)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert Nx.all_close(y, expected) |> Nx.to_number() == 1
  end

  test "runtime_call with fully-qualified aliased module (&Alias.fun/2)" do
    x = Nx.iota({5})
    y = add_offset_remote_aliased(x)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert Nx.all_close(y, expected) |> Nx.to_number() == 1
  end

  def split_and_sum_callback(t, _opts) do
    {Nx.multiply(t, 2.0), Nx.add(t, 1.0)}
  end

  defn split_and_sum(x) do
    fx = Nx.as_type(x, :f32)

    out0 = fx
    out1 = fx
    out_template = {out0, out1}

    {a, b} = Nx.runtime_call(out_template, fx, &split_and_sum_callback/2)

    Nx.add(a, b)
  end

  test "runtime_call with tuple output" do
    x = Nx.tensor([1, 2, 3])
    y = split_and_sum(x)

    fx = Nx.as_type(x, :f32)
    expected = Nx.add(Nx.multiply(fx, 2.0), Nx.add(fx, 1.0))
    assert expected == y
  end

  defp return_as_container_tuple_callback({x, y}, opts) do
    send(opts[:pid], {:container_fun, opts[:ref]})
    {x, y}
  end

  defn return_as_container_tuple(x, y, opts) do
    Nx.runtime_call({x, y}, {x, y}, opts, &return_as_container_tuple_callback/2)
  end

  defp return_as_container_map_callback({x, y}, opts) do
    send(opts[:pid], {:container_fun, opts[:ref]})
    %{x: x, y: {%{key: y}, Nx.s32(1)}}
  end

  defn return_as_container_map(x, y, opts) do
    Nx.runtime_call(
      %{x: x, y: {%{key: y}, Nx.s32(1)}},
      {x, y},
      opts,
      &return_as_container_map_callback/2
    )
  end

  test "runtime_call with container output" do
    x = Nx.tensor([1, 2, 3])
    y = Nx.tensor([4, 5, 6])

    ref = make_ref()
    pid = self()

    assert {x_res, y_res} = return_as_container_tuple(x, y, ref: ref, pid: pid)
    assert x_res == x
    assert y_res == y
    assert_receive {:container_fun, ^ref}

    ref = make_ref()

    assert result = return_as_container_map(x, y, ref: ref, pid: pid)
    assert %{x: _, y: {%{key: _}, _}} = result
    assert result.x == x
    assert result.y == {%{key: y}, Nx.s32(1)}
    assert_receive {:container_fun, ^ref}
  end

  describe "invalid callback" do
    test "rejects anonymous function (fn x, y -> x + y end)" do
      assert_raise CompileError, ~r/anonymous functions are not allowed/, fn ->
        defmodule BadAnonFn do
          import Nx.Defn

          defn bad(x) do
            Nx.runtime_call(x, x, fn x, y -> x + y end)
          end
        end
      end
    end

    test "rejects anonymous capture (&(&1 / 2))" do
      assert_raise CompileError, ~r/requires a named capture/, fn ->
        defmodule BadAnonCapture do
          import Nx.Defn

          defn bad(x) do
            Nx.runtime_call(x, x, &(&1 / 2))
          end
        end
      end
    end

    test "rejects wrong arity (&fun/1)" do
      assert_raise CompileError, ~r/requires a named capture with arity 2.*got arity 1/, fn ->
        defmodule BadArity do
          import Nx.Defn

          def callback(_t), do: Nx.tensor(0)

          defn bad(x) do
            Nx.runtime_call(x, x, &callback/1)
          end
        end
      end
    end
  end
end

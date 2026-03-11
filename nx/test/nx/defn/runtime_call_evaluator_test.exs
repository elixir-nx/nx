defmodule Nx.Defn.RuntimeCallEvaluatorTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  setup do
    Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)
    :ok
  end

  defn add_offset(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t ->
      Nx.add(Nx.as_type(t, :f32), 10.0)
    end)
  end

  test "runtime_call with single output" do
    x = Nx.iota({5})
    y = add_offset(x)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert Nx.all_close(y, expected) |> Nx.to_number() == 1
  end

  defn split_and_sum(x) do
    fx = Nx.as_type(x, :f32)

    out0 = fx
    out1 = fx
    out_template = {out0, out1}

    {a, b} =
      Nx.runtime_call(out_template, fx, fn t ->
        {Nx.multiply(t, 2.0), Nx.add(t, 1.0)}
      end)

    Nx.add(a, b)
  end

  test "runtime_call with tuple output" do
    x = Nx.tensor([1, 2, 3])
    y = split_and_sum(x)

    fx = Nx.as_type(x, :f32)
    expected = Nx.add(Nx.multiply(fx, 2.0), Nx.add(fx, 1.0))
    assert expected == y
  end

  # Issue #1684: runtime_call callbacks should be able to call
  # regular defp functions without requiring deftransform.
  defp do_add_offset(t, opts) do
    t
    |> Nx.as_type(:f32)
    |> Nx.add(opts[:offset])
  end

  defn add_offset_via_defp(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t ->
      do_add_offset(t, offset: 10.0)
    end)
  end

  test "runtime_call callback can call regular defp functions" do
    x = Nx.iota({5})
    y = add_offset_via_defp(x)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert Nx.all_close(y, expected) |> Nx.to_number() == 1
  end

  defn callback_with_remote_call(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t ->
      [t] |> Enum.map(&Nx.as_type(&1, :f32)) |> hd()
    end)
  end

  test "runtime_call callback can call functions from non-Nx modules" do
    x = Nx.tensor([1, 2, 3])
    result = callback_with_remote_call(x)

    expected = Nx.as_type(x, :f32)
    assert Nx.all_close(result, expected) |> Nx.to_number() == 1
  end

  defn return_as_container(x, y, template_fun, container_fun) do
    Nx.runtime_call(template_fun.(x, y), {x, y}, container_fun)
  end

  test "runtime_call with container output" do
    x = Nx.tensor([1, 2, 3])
    y = Nx.tensor([4, 5, 6])

    ref = make_ref()
    pid = self()

    container_fun = fn {x, y} ->
      send(pid, {:container_fun, ref})
      {x, y}
    end

    template_fun = fn x, y -> {x, y} end

    assert {x_res, y_res} = return_as_container(x, y, template_fun, container_fun)
    assert x_res == x
    assert y_res == y
    assert_receive {:container_fun, ^ref}

    ref = make_ref()

    container_fun = fn {x, y} ->
      send(pid, {:container_fun, ref})
      %{x: x, y: {%{key: y}, Nx.s32(1)}}
    end

    template_fun = fn x, y -> %{x: x, y: {%{key: y}, Nx.s32(1)}} end

    assert result = return_as_container(x, y, template_fun, container_fun)
    assert %{x: _, y: {%{key: _}, _}} = result
    assert result.x == x
    assert result.y == {%{key: y}, Nx.s32(1)}
    assert_receive {:container_fun, ^ref}
  end
end

defmodule EXLA.FFI.TermTensorTest do
  use ExUnit.Case, async: true

  alias EXLA.FFI.TermTensor

  test "pid_to_tensor encodes current process pid" do
    t = TermTensor.pid_to_tensor(self())
    assert %Nx.Tensor{type: {:u, 8}, shape: {size}} = t
    assert size == byte_size(:erlang.term_to_binary(self()))

    assert self() == t |> Nx.to_binary() |> :erlang.binary_to_term()
  end

  test "tag_to_tensor encodes arbitrary term" do
    ref = make_ref()
    pid = self()
    tag = {ref, pid}
    t = TermTensor.tag_to_tensor(tag)
    assert %Nx.Tensor{type: {:u, 8}, shape: {size}} = t
    assert size == byte_size(:erlang.term_to_binary(tag))

    assert {ref, pid} == t |> Nx.to_binary() |> :erlang.binary_to_term()
  end
end

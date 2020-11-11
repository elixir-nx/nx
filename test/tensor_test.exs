defmodule TensorTest do
  use ExUnit.Case
  alias Exla.Tensor
  alias Exla.Client

  setup_all do
    {:ok, client: Client.create_client()}
  end

  test "scalar/3 creates a new scalar tensor" do
    assert %Tensor{} = Tensor.scalar(1, :int32)
  end

  test "to_device/3 creates a reference tensor", state do
    tensor = Tensor.scalar(1, :int32)
    assert %Tensor{data: {:ref, _}} = Tensor.to_device(state[:client], tensor)
  end
end

defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client
  alias Exla.Options.LocalClientOptions

  test "create_client/1 succeeds on host device" do
    assert {:ok, client = %Client{}} = Client.create_client(%LocalClientOptions{})
  end
end

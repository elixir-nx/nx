defmodule EXLA.ClientTest do
  use ExUnit.Case, async: true

  doctest EXLA.Client

  describe "get_supported_platforms/0" do
    test "returns supported platforms with device information" do
      %{host: _} = EXLA.Client.get_supported_platforms()
    end
  end

  describe "build_client/2" do
    test "unknown platform names the client and the supported platforms" do
      exception =
        assert_raise ArgumentError, fn ->
          EXLA.Client.build_client(:diag, platform: :not_a_platform)
        end

      assert exception.message =~ "client :diag"
      assert exception.message =~ ":not_a_platform"
      assert exception.message =~ ":host"
    end

    test "invalid default device names the client and platform" do
      exception =
        assert_raise ArgumentError, fn ->
          EXLA.Client.build_client(:diag, platform: :host, default_device_id: -1)
        end

      assert exception.message =~ "client :diag"
      assert exception.message =~ ":host"
      assert exception.message =~ "-1"
    end
  end
end

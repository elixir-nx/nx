defmodule EXLA.Plugin do
  @moduledoc """
  Plugin system for registering custom calls.
  """

  def register(library_path) do
    unless File.exists?(library_path) do
      raise ArgumentError, "#{library_path} does not exist"
    end

    ref =
      library_path
      |> EXLA.NIF.load_custom_call_plugin_library()
      |> unwrap!()

    # we need to keep the ref from getting garbage collected so
    # we can use the symbols within it at anytime 
    :persistent_term.put({__MODULE__, library_path}, ref)
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise "#{reason}"
end
defmodule EXLA.Defn.Disk do
  @moduledoc false
  @version 1

  require Logger

  def cache(cache, _client, _key, _debug?, callback) when is_boolean(cache) do
    callback.()
  end

  def cache(cache, client, keys, debug?, callback) when is_binary(cache) do
    cached =
      case File.read(cache) do
        {:ok, <<"EXLA", @version, blob::binary>>} ->
          case :erlang.binary_to_term(blob) do
            {^keys, executable, value} ->
              debug? && Logger.debug("EXLA disk cache found at #{inspect(cache)}")
              {EXLA.Executable.load(client, executable), value}

            {disk_keys, _executable, _value} ->
              mismatched = for {key, value} <- disk_keys, keys[key] != value, do: key

              Logger.warning("""
              EXLA disk cache does not match configuration.

              Expected: #{inspect(Map.take(keys, mismatched))}

              Found: #{inspect(Map.take(disk_keys, mismatched))}
              """)

              nil
          end

        {:ok, <<"EXLA", _::binary>>} ->
          Logger.warning(
            "Discarding EXLA disk cache at #{inspect(cache)} because it is for an older EXLA version"
          )

          nil

        {:error, _} ->
          debug? && Logger.debug("EXLA disk cache not found at #{inspect(cache)}")
          nil
      end

    if cached do
      cached
    else
      {executable, value} = callback.()
      blob = :erlang.term_to_binary({keys, EXLA.Executable.dump(executable), value})
      File.mkdir_p!(Path.dirname(cache))
      File.write!(cache, <<"EXLA", @version, blob::binary>>)
      {executable, value}
    end
  end
end

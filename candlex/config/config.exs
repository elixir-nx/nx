import Config

enable_cuda =
  case System.get_env("CUDA") do
    nil -> System.find_executable("nvcc") && System.find_executable("nvidia-smi")
    "false" -> false
    _ -> true
  end

crate_features =
  if enable_cuda do
    [:cuda]
  else
    []
  end

config :candlex, crate_features: crate_features

IO.puts("Testing EXLA with CUDA 12.8...")

# Check supported platforms
IO.puts("\nSupported platforms:")
IO.inspect(EXLA.Client.get_supported_platforms())

# Set EXLA as default backend with CUDA device
Nx.default_backend({EXLA.Backend, client: :cuda})

# Simple test
result = Nx.tensor([1, 2, 3]) |> Nx.add(1)
IO.puts("\nResult:")
IO.inspect(result)

# Verify it's on CUDA
backend_info = inspect(result)
if String.contains?(backend_info, "cuda") do
  IO.puts("\n✓ EXLA is working correctly with CUDA 12.8 on GPU!")
else
  IO.puts("\n✗ Warning: Tensor is not on CUDA device!")
  IO.puts("Backend: #{backend_info}")
end

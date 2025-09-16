# Changelog

## v0.10.0 (2025-06-17)

### Enhancements

  * NIFs now use Fine for wrapping the C++ code
  * Nx.to_pointer/2 and Nx.from_pointer/5 now raise on errors
  * LU decomposition is now supported in all devices
  * New EXLA_CPU_ONLY compilation flag for disabling the CUDA-specific EXLA files
  * Update XLA to latest version

### Bug fixes

  * Device id is now respected when automatic transfers are disabled
  * Improve vectorized gather implementation
  * Fix hook order inside while loop

## v0.9.2 (2024-11-16)

### Enhancements

  * Support cross-compilation for use with Nerves
  * Optimize LU with a custom call

## v0.9.1 (2024-10-08)

### Enhancements

  * Improve compilation times of native code

### Bug fixes

  * Fix encoding of binary floats

## v0.9.0 (2024-09-26)

### Enhancements

  * Overall improvements to the Nx.Defn compiler
  * Compiled functions now work across BEAM nodes
  * Add `cache: "path/to/file"` for disk caching JIT/compiled functions

### Bug fixes

  * Use a single thread pool for MLIR contexts

## v0.8.0 (2024-08-19)

  * Add `EXLA.to_mlir_module/2`
  * The precompiled XLA CUDA binaries now require CUDA 12.1+ and cuDNN 9.1+
  * Renamed `XLA_TARGET` value "cuda120" to "cuda12"
  * `XLA_TARGET` automatically defaults to "cuda12" when CUDA installation is detected
  * Allow NIF modules to be upgradable

## v0.7.1 (2024-02-27)

  * Add CustomCallOp for QR decomposition
  * Minor improvements to the MLIR modules generated
  * MLIR Context pooling for better concurrency

## v0.7.0 (2024-02-22)

  * Update to latest Nx
  * Introduce a `:mlir` based compiler and use it by default. The previous `:xla` based compiler is deprecatead. You can temporarily revert to the previous compiler by setting `config :exla, :compiler_mode, :xla`

## v0.6.4 (2023-11-13)

  * Update to latest Nx
  * Allow `:automatic_transfers` configuration on client
  * Do not discard client/device in `EXLA.Backend` when it is host
  * Always sort NaN last
  * Improve the `:axes` option in `gather`, `indexed_add`, and `indexed_put`

## v0.6.3 (2023-11-09)

  * Update to latest Nx
  * Fix mixed device usage on EXLA.Backend

## v0.6.1 (2023-09-12)

  * Update to latest Nx

## v0.6.0 (2023-08-15)

  * Allow cross-device transfers on host
  * Update dependencies to OpenXLA
  * Update to latest Nx

## v0.5.3 (2023-04-14)

  * Fix compilation issue on certain macOS caused by O3
  * Fix optimization which would cause EXLA to return a complete tuple instead of a subset

## v0.5.2 (2023-03-21)

  * Automatically transfer tensors between nodes

## v0.5.1 (2023-02-18)

  * Support `top_k`

## v0.5.0 (2023-02-10)

  * Optimize backend_transfer/backend_copy within EXLA backends
  * Use relative symlinks on compilation whenever possible

## v0.4.2 (2023-01-13)

### Enhancements

  * Automatically transfer from `:host` to other devices
  * Support `lazy_transfers: :always` on `EXLA.jit/3` and `EXLA.compile/2`
  * Run hooks concurrently once they have received the data

### Bug fixes

  * Respect default `EXLA.Backend` client when jitting argumentless operations
  * Do not pick client without devices when loading initial client
  * Consider the first conditional of a `cond` part of the current scope

## v0.4.1 (2022-12-07)

### Enhancements

  * [EXLA] Require Nx ~> 0.4.1
  * [EXLA] Update `XLA` to TF2.11
  * [EXLA.Defn] Send telemetry event after XLA compilation
  * [EXLA.Op] Add optimization barriers as operations

### Bug fixes

  * [EXLA] Validate backend options
  * [EXLA.Backend] Fix `Nx.{any,all}` with `:keep_axes`
  * [EXLA.Backend] Make SVD return `V` instead of `transpose(V)`
  * [EXLA.Backend] Preserve NaNs in `window` and `reduce` operations

## v0.4.0 (2022-10-25)

### Enhancements

  * Support zero copy binaries
  * Redirect group leader for EXLA hooks

### Bug fixes

  * Always hoist `cond` expressions
  * Fix conditional inside `Nx.map`

## v0.3.0 (2022-08-13)

### Enhancements

  * Support `debug: true` option on `defn` compiler
  * Allow specifying preferred clients via the application environment
  * Support new callbacks added in Nx v0.3.0

### Deprecations

  * Deprecate `set_as_nx_default`

## v0.2.3 (2022-07-05)

### Bug fixes

  * Fix predicate handling inside `cond`/`while`
  * Set Nx backend globally

## v0.2.2 (2022-06-15)

### Bug fixes

  * Fix invalid cache expiration when defn received functions as arguments

## v0.2.1 (2022-06-04)

### Enhancements

  * Implement `EXLA.Backend.to_batched_list/3`

### Bug fixes

  * Improve support for non-finite values in `EXLA` compiler
  * Fix segmentation fault while deallocating tensors

## v0.2.0 (2022-04-28)

First release.

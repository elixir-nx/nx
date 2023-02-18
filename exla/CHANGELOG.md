# Changelog

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

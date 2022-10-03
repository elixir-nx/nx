# Changelog

## v0.4.0

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

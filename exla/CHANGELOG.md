# Changelog

## v0.2.2 (2022-06-15)

### Bug fixes

  * Consider input structure as cache keys for EXLA
  * Pass tensors as arguments on `EXLA.Backend` during `indexed_add`, `slice`, and `put_slice`

## v0.2.1 (2022-06-04)

### Enhancements

  * Implement `EXLA.Backend.to_batched_list/3`

### Bug fixes

  * Improve support for non-finite values in `EXLA` compiler
  * Fix segmentation fault while deallocating tensors

## v0.2.0 (2022-04-28)

First release.

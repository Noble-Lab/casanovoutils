# Changelog

## Unreleased

### Added

- `casanovoutils mgf pipeline`: chain shuffle, downsample, and purge-redundant
  in a single command with each stage independently optional.
- `casanovoutils mgf purge-redundant`: remove near-duplicate peaks within each
  spectrum based on a configurable m/z epsilon threshold.
- `casanovoutils denovo`: subcommands for loading MGF and mzTab files into
  Polars DataFrames and joining them into a ground-truth table.
- `casanovoutils dump-residues dump`: export the default residue mass table.
- Module-level `COMMANDS` constants in each submodule, assembled into a single
  nested CLI via `casanovoutils.main`.
- `configure_logging` in package `__init__` is a no-op if already called,
  preventing duplicate handler registration when functions are composed.

### Changed

- All CLI entry points consolidated into a single `casanovoutils` command with
  nested subcommands (`mgf`, `denovo`, `dump-residues`). The previous separate
  entry points (`graph-prec-cov`, `downsample-ms`, `mgf-utils`,
  `casanovo-utils`) are removed.
- `write_spectra` now converts `PathLike` paths to `str` before passing to
  `pyteomics.mgf.write`, fixing a `PosixPath` compatibility bug.
- `shuffle` no longer writes the output file twice.

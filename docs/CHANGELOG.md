# Changelog

## Unreleased

### Added

- `casanovoutils visualize_errors`: creates mirror plots of the top scoring
  incorrect predictions at the peptide level
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

- `casanovoutils preccov get_pc_df` now decides whether a prediction is correct
  by mass-based alignment (`--cum_mass_threshold`, default 0.5 Da, and
  `--ind_mass_threshold`, default 0.1 Da) instead of exact token equality, so
  mass-equivalent residues such as I/L and N[Deamidated]/D count as matches.
  The matching helpers are ported from Casanovo, which is no longer a
  dependency.
- Boolean CLI flags accept `true` and `false` in any case; other strings are
  rejected instead of being treated as true.
- All CLI entry points consolidated into a single `casanovoutils` command with
  nested subcommands (`mgf`, `denovo`, `dump-residues`). The previous separate
  entry points (`graph-prec-cov`, `downsample-ms`, `mgf-utils`,
  `casanovo-utils`) are removed.
- `write_spectra` now converts `PathLike` paths to `str` before passing to
  `pyteomics.mgf.write`, fixing a `PosixPath` compatibility bug.
- `shuffle` no longer writes the output file twice.

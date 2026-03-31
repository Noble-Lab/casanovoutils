# casanovoutils

**casanovoutils** is an open-source collection of python and command-line utilities for
evaluating, visualizing, and manipulating peptide-spectrum match (PSM) data.
It is designed to complement [Casanovo](https://casanovo.readthedocs.io/), the
state-of-the-art de novo peptide sequencing tool, and works directly with
**mzTab** and **MGF** file formats.

## Key capabilities

- **MGF processing pipeline** — shuffle, downsample by peptide sequence, and
  purge near-duplicate peaks, either as individual steps or chained together
  via the `casanovoutils mgf pipeline` command.

- **PSM data loading** — parse MGF and mzTab files into Polars DataFrames,
  join predicted and ground-truth annotations, and export to Parquet, CSV,
  or TSV via `casanovoutils denovo`.

- **Residue mass tables** — export and customize the amino acid mass table
  used for evaluation via `casanovoutils dump-residues dump`.

```{toctree}
:maxdepth: 2
:caption: User Guide

getting_started
cli_reference
file_formats
faq
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api/mgfutils
api/denovoutils
api/align
api/preccov
api/residues
api/constants
api/types
api/main
```

```{toctree}
:maxdepth: 1
:caption: Reference

CHANGELOG
```

```{toctree}
:maxdepth: 1
:caption: Community

CONTRIBUTING
CODE_OF_CONDUCT
```

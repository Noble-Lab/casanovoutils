# casanovoutils

**casanovoutils** is an open-source collection of command-line utilities for
evaluating, visualizing, and manipulating peptide-spectrum match (PSM) data.
It is designed to complement [Casanovo](https://casanovo.readthedocs.io/), the
state-of-the-art de novo peptide sequencing tool, and works directly with
**mzTab** and **MGF** file formats.

## Key capabilities

- **Precision–Coverage plotting** — visualize and compare de novo sequencing
  accuracy across models or conditions using precision–coverage (Prec–Cov)
  curves with area-under-curve (AUPC) values reported in the legend.

- **MGF downsampling** — balance datasets by limiting the number of spectra
  retained per peptide sequence, with optional shuffling and reproducible
  random seeds.

- **MGF utilities** — merge, downsample, and write MGF files from the command
  line using a chainable interface.

- **Residue mass tables** — export and customize the amino acid mass table used
  for evaluation.

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
:caption: Reference

autoapi/index
CHANGELOG
```

```{toctree}
:maxdepth: 1
:caption: Community

CONTRIBUTING
CODE_OF_CONDUCT
```

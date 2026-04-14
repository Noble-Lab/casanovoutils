# casanovoutils

Utility tools for **evaluating, visualizing, and manipulating peptide-spectrum
match (PSM) data**, designed to work cleanly with
[Casanovo](https://casanovo.readthedocs.io/), **mzTab**, and **MGF** files.

## [Documentation](https://casanovo.readthedocs.io/en/latest/)

## Features

- **MGF processing pipeline** — shuffle, downsample by peptide sequence, and
  purge near-duplicate peaks, individually or chained in a single command
- **PSM data loading** — parse MGF and mzTab files into Polars DataFrames and
  join them into a ground-truth table
- **Precision–coverage evaluation** — compute and plot Prec–Cov curves with
  AUPC at the peptide and amino-acid level
- **Residue mass tables** — export and customize the amino acid mass vocabulary
  used for evaluation

## Installation

```bash
pip install casanovoutils
```

Requires Python 3.13 or later.

## Quick start

### Process an MGF file

Shuffle, downsample to at most 2 spectra per peptide, and remove near-duplicate
peaks in one pass:

```bash
casanovoutils mgf pipeline input.mgf \
  --outfile out.mgf \
  --downsample_k 2 \
  --purge_epsilon 0.001
```

Run a single stage:

```bash
casanovoutils mgf shuffle input.mgf --outfile shuffled.mgf
casanovoutils mgf downsample input.mgf --outfile sampled.mgf --k 3
casanovoutils mgf purge-redundant input.mgf --outfile purged.mgf
```

### Load PSM data

Join MGF metadata with mzTab predictions into a single DataFrame:

```bash
casanovoutils denovo get_groundtruth input.mgf results.mztab \
  --out_path groundtruth.parquet
```

Load either source individually:

```bash
casanovoutils denovo get_mgf_psms input.mgf --out_path psms.parquet
casanovoutils denovo get_mztab results.mztab --out_path matches.parquet
```

### Export the residue mass table

```bash
casanovoutils dump-residues dump residues.yaml
```

Edit `residues.yaml` to add custom modifications or non-standard residues,
then pass it back to other tools via `--residues_path`.

## CLI reference

All commands live under the single `casanovoutils` entry point:

```text
casanovoutils
├── mgf
│   ├── pipeline        shuffle → downsample → purge-redundant in one pass
│   ├── shuffle         randomise spectrum order
│   ├── downsample      limit spectra per peptide sequence
│   └── purge-redundant remove near-duplicate peaks by m/z
├── denovo
│   ├── get_mgf_psms    load MGF metadata into a DataFrame
│   ├── get_mztab       load mzTab PSMs into a DataFrame
│   └── get_groundtruth join MGF + mzTab into a ground-truth table
└── dump-residues
    └── dump            copy the default residue mass YAML to a path
```

Pass `--help` to any subcommand for full argument details:

```bash
casanovoutils mgf pipeline --help
casanovoutils denovo get_groundtruth --help
```

### `casanovoutils mgf pipeline`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file |
| `--outfile` | path | `None` | Output MGF file |
| `--do_shuffle` | bool | `True` | Shuffle spectra |
| `--downsample_k` | int | `None` | Max spectra per peptide (skipped if omitted) |
| `--purge_epsilon` | float | `None` | Min m/z gap in Da to keep a peak (skipped if omitted) |
| `--random_seed` | int | `42` | Seed for shuffle and downsample |

### `casanovoutils mgf downsample`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file |
| `--k` | int | `1` | Max spectra per peptide sequence |
| `--outfile` | path | `None` | Output MGF file |
| `--random_seed` | int | `42` | Random seed |

### `casanovoutils mgf purge-redundant`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file |
| `--epsilon` | float | `0.001` | Min m/z separation in Da |
| `--outfile` | path | `None` | Output MGF file |

### `casanovoutils denovo get_groundtruth`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_path` | path | required | Input MGF file |
| `mztab_path` | path | required | Input mzTab file |
| `--out_path` | path | `None` | Output path (`.parquet`, `.csv`, `.tsv`) |

## Logging

All processing commands write log messages to stdout. If `--outfile` is
provided, a `.log` file is written alongside it with the same base name:

```text
out.mgf   → out.log
```

## Development

```bash
git clone https://github.com/Noble-Lab/casanovoutils.git
cd casanovoutils
uv sync
pytest
```

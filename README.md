# casanovoutils

Utility tools for **evaluating, visualizing, and manipulating peptide-spectrum
match (PSM) data**, designed to work cleanly with
[Casanovo](https://casanovo.readthedocs.io/), **mzTab**, and **MGF** files.

## [Documentation](https://casanovoutils.readthedocs.io/en/latest/)

## Features

- **MGF processing pipeline** — shuffle, downsample by peptide sequence, and
  purge near-duplicate peaks, individually or chained in a single command
- **mzML sampling** — stream-sample a proportion of spectra from an mzML file
  in a single pass and write the result as MGF
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
casanovoutils mgfutils pipeline input.mgf \
  --outfile out.mgf \
  --downsample_k 2 \
  --purge_epsilon 0.001
```

Run a single stage:

```bash
casanovoutils mgfutils shuffle input.mgf --outfile shuffled.mgf
casanovoutils mgfutils downsample input.mgf --outfile sampled.mgf --k 3
casanovoutils mgfutils purge-redundant input.mgf --outfile purged.mgf
```

### Load PSM data

Join MGF metadata with mzTab predictions into a single DataFrame:

```bash
casanovoutils denovoutils get_groundtruth input.mgf results.mztab \
  --out_path groundtruth.parquet
```

Load either source individually:

```bash
casanovoutils denovoutils get_mgf_psms input.mgf --out_path psms.parquet
casanovoutils denovoutils get_mztab results.mztab --out_path matches.parquet
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
├── mgfutils            MGF file processing
│   ├── pipeline        shuffle → downsample → purge-redundant in one pass
│   ├── shuffle         randomise spectrum order
│   ├── downsample      limit spectra per peptide sequence
│   ├── spectra-per-peptide  reservoir-sample k spectra per peptide
│   ├── downsample-spectra   downsample to a target count or proportion
│   └── purge-redundant remove near-duplicate peaks by m/z
├── mzmlutils           sample a proportion of spectra from an mzML file
├── denovoutils         load PSM data into Polars DataFrames
│   ├── get_mgf_psms    load MGF metadata into a DataFrame
│   ├── get_mztab       load mzTab PSMs into a DataFrame
│   └── get_groundtruth join MGF + mzTab into a ground-truth table
├── preccov             precision-coverage evaluation
│   ├── get_pc_df       build a precision-coverage DataFrame
│   └── graph_prec_cov  plot precision-coverage curves
├── summarize_mgf       MGF file statistics and HTML reports
│   ├── summarize       full HTML summary report
│   ├── charge-distribution
│   ├── fragment-coverage
│   ├── peak-counts
│   └── peptide-lengths
├── datasets            create train/val/test splits from MGF files
├── graphloss           plot Casanovo training/validation loss curves
└── residues            copy the bundled residue mass YAML to a path
```

Pass `--help` to any subcommand for full argument details:

```bash
casanovoutils mgfutils pipeline --help
casanovoutils mzmlutils --help
casanovoutils denovoutils get_groundtruth --help
```

### `casanovoutils mzmlutils`

Sample a proportion of spectra from an mzML file and write the result as MGF.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `input_file` | path | required | Input mzML file |
| `k` | float | required | Proportion to sample; must be in (0, 1) |
| `outfile` | path | required | Output MGF file path |
| `--buffer_size` | int | `1000` | Spectra read per I/O chunk |
| `--random_seed` | int | `42` | Random seed for reproducibility |

```bash
casanovoutils mzmlutils input.mzML 0.1 sampled.mgf
```

### `casanovoutils mgfutils pipeline`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file |
| `--outfile` | path | `None` | Output MGF file |
| `--do_shuffle` | bool | `True` | Shuffle spectra |
| `--downsample_k` | int | `None` | Max spectra per peptide (skipped if omitted) |
| `--purge_epsilon` | float | `None` | Min m/z gap in Da to keep a peak (skipped if omitted) |
| `--random_seed` | int | `42` | Seed for shuffle and downsample |

### `casanovoutils denovoutils get_groundtruth`

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

# Getting Started

## Requirements

casanovoutils requires **Python 3.13 or later**.

## Installation

### From PyPI

```bash
pip install casanovoutils
```

### From source

Clone the repository and install with pip:

```bash
git clone https://github.com/Noble-Lab/casanovoutils.git
cd casanovoutils
pip install .
```

If you use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Noble-Lab/casanovoutils.git
cd casanovoutils
uv sync
```

## Verifying the installation

After installation, the `casanovoutils` command should be available in your
shell:

```bash
casanovoutils --help
casanovoutils mgfutils --help
casanovoutils mzmlutils --help
casanovoutils denovoutils --help
```

## Quick start examples

### Shuffle and downsample an MGF file

Shuffle spectra and retain at most 2 per peptide sequence:

```bash
casanovoutils mgfutils pipeline input.mgf --outfile out.mgf --downsample_k 2
```

### Downsample only (no shuffle)

```bash
casanovoutils mgfutils downsample input.mgf --outfile sampled.mgf --k 2
```

### Remove near-duplicate peaks

Remove peaks separated by less than 0.001 Da:

```bash
casanovoutils mgfutils purge-redundant input.mgf --outfile purged.mgf
```

### Sample spectra from an mzML file

Sample 10% of spectra and write to MGF:

```bash
casanovoutils mzmlutils input.mzML 0.1 sampled.mgf
```

### Load PSM data into a DataFrame

```bash
casanovoutils denovoutils get_groundtruth input.mgf results.mztab \
  --out_path groundtruth.parquet
```

### Compute precision-coverage metrics

```bash
casanovoutils preccov get_pc_df \
  --mgf_df psms.parquet --mztab_df matches.parquet \
  --out_path pc.parquet

casanovoutils preccov graph_prec_cov pc.parquet --out_path pc_curve.png
```

### Summarise an MGF file

```bash
casanovoutils summarize_mgf summarize input.mgf --output_root my_report
```

### Create train/val/test splits

```bash
casanovoutils datasets input.mgf --output_root splits/run1
```

### Export the residue mass table

```bash
casanovoutils residues residues.yaml
```

Edit `residues.yaml` to add custom modifications or non-standard residues,
then pass it back to other tools via `--residues_path`.

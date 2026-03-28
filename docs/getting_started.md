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
casanovoutils mgf --help
casanovoutils denovo --help
```

## Quick start examples

### Shuffle and downsample an MGF file

Shuffle spectra and retain at most 2 per peptide sequence:

```bash
casanovoutils mgf pipeline input.mgf --outfile out.mgf --downsample_k 2
```

### Downsample only (no shuffle)

```bash
casanovoutils mgf downsample input.mgf --outfile sampled.mgf --k 2
```

### Remove near-duplicate peaks

Remove peaks separated by less than 0.001 Da:

```bash
casanovoutils mgf purge-redundant input.mgf --outfile purged.mgf
```

### Load PSM data into a DataFrame

```bash
casanovoutils denovo get_groundtruth input.mgf results.mztab \
  --out_path groundtruth.parquet
```

### Export the residue mass table

```bash
casanovoutils dump-residues dump residues.yaml
```

Edit `residues.yaml` to add custom modifications or non-standard residues,
then pass it back to other tools via `--residues_path`.

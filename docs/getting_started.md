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

After installation, the following commands should be available in your shell:

```bash
graph-prec-cov --help
downsample-ms --help
mgf-utils --help
casanovo-utils --help
```

## Quick start examples

### Plot a precision–coverage curve

Given a Casanovo mzTab output file and the MGF file that was used as input,
plot a precision–coverage curve:

```bash
graph-prec-cov \
  add-peptides results.mztab ground_truth.mgf "My Model" \
  save prec_cov.png
```

The AUPC (area under the precision–coverage curve) is printed in the legend
automatically.

### Compare two models on one plot

```bash
graph-prec-cov \
  --fig_width 6 \
  --fig_height 4 \
  add-peptides modelA.mztab truth.mgf "Model A" \
  add-peptides modelB.mztab truth.mgf "Model B" \
  save comparison.png
```

### Downsample an MGF file

Retain at most 2 spectra per peptide sequence and write the result to a new
MGF file:

```bash
downsample-ms input.mgf --outfile sampled.mgf --k 2
```

### Chain MGF operations

Use `mgf-utils` to downsample and write multiple input MGF files in one
command:

```bash
mgf-utils file1.mgf file2.mgf \
  downsample --k 5 \
  write --outfile combined_downsampled.mgf
```

### Export the residue mass table

```bash
casanovo-utils dump-residues residues.yaml
```

Edit `residues.yaml` to add custom modifications or non-standard residues,
then pass it back to other tools via `--residues_path`.

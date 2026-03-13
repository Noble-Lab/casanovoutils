# CLI Reference

casanovoutils installs four command-line entrypoints. Each is built with
[Python Fire](https://github.com/google/python-fire), which means:

- Boolean flags can be passed as `--flag` (True) or `--noflag` (False).
- Positional arguments can also be passed as keyword arguments.
- Chained method calls are separated by a space on the command line.

---

## `graph-prec-cov`

Plot and compare precision–coverage (Prec–Cov) curves for one or more
datasets.

**Entry point:** `casanovoutils.preccov:main`

### Global options

These options are set at instantiation time, before any chained method calls.

| Option | Type | Default | Description |
|---|---|---|---|
| `--fig_width` | float | `3.0` | Figure width in inches |
| `--fig_height` | float | `3.0` | Figure height in inches |
| `--fig_dpi` | int | `150` | Figure resolution (dots per inch) |
| `--legend_border` | bool | `False` | Draw a border around the legend |
| `--legend_location` | str | `"lower left"` | Legend position (matplotlib string) |
| `--ax_x_label` | str | `"Coverage"` | X-axis label |
| `--ax_y_label` | str | `"Precision"` | Y-axis label |
| `--ax_title` | str | `""` | Plot title prefix |

### Methods

#### `add-peptides`

Add a peptide-level precision–coverage curve to the current plot.

| Argument | Type | Default | Description |
|---|---|---|---|
| `mztab_path` | path | required | mzTab file with PSM predictions |
| `mgf_path` | path | required | MGF file with ground-truth sequences |
| `name` | str | required | Dataset label shown in the legend |
| `--replace_i_l` | bool | `True` | Treat I and L as equivalent |

#### `clear`

Reset the figure to a blank plot.

#### `save`

Save the current plot to a file. The format is inferred from the file extension
(`.png`, `.pdf`, `.svg`, etc.).

| Argument | Type | Default | Description |
|---|---|---|---|
| `save_path` | path | required | Output file path |

#### `show`

Display the current plot interactively.

### Examples

**Single dataset:**

```bash
graph-prec-cov \
  add-peptides results.mztab ground_truth.mgf "Casanovo" \
  save prec_cov.png
```

**Multiple datasets on one figure:**

```bash
graph-prec-cov \
  --fig_width 6 \
  --fig_height 4 \
  --legend_location "upper right" \
  add-peptides modelA.mztab truth.mgf "Model A" \
  add-peptides modelB.mztab truth.mgf "Model B" \
  save comparison.png
```

**Without I/L equivalence:**

```bash
graph-prec-cov \
  add-peptides results.mztab truth.mgf "Strict" --noreplace_i_l \
  save strict.png
```

---

## `downsample-ms`

Downsample one or more MGF files by limiting the number of spectra retained
per peptide sequence.

**Entry point:** `casanovoutils.downsample:main`

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `spectra` | path | required | Path to the input MGF file |
| `--outfile` | path | `"out.mgf"` | Output MGF file path |
| `--k` | int | `1` | Maximum spectra per peptide sequence |
| `--shuffle` | bool | `False` | Shuffle output spectra |
| `--random_seed` | int | `42` | Random seed for reproducibility |

### Examples

**Keep at most 1 spectrum per peptide (default):**

```bash
downsample-ms input.mgf --outfile sampled.mgf
```

**Keep up to 5 spectra per peptide, shuffled:**

```bash
downsample-ms input.mgf --outfile sampled.mgf --k 5 --shuffle
```

**Reproducible sampling:**

```bash
downsample-ms input.mgf --outfile sampled.mgf --k 2 --random_seed 0
```

---

## `mgf-utils`

A chainable utility for working with one or more MGF files. Accepts multiple
input files, which are streamed together as a single dataset.

**Entry point:** `casanovoutils.mgfutils:main`

### Initialization arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `*mgf_files` | paths | required | One or more input MGF file paths |
| `--random_seed` | int | `42` | Seed for shuffling and sampling |

### Methods

#### `shuffle`

Load all spectra into memory and shuffle them in place.

#### `downsample`

Downsample spectra by peptide sequence.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--k` | int | `1` | Maximum spectra per peptide sequence |

#### `write`

Write the current spectra to an MGF file.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--outfile` | path | `"out.mgf"` | Output file path |

### Examples

**Merge two MGF files and write to a new file:**

```bash
mgf-utils file1.mgf file2.mgf write --outfile merged.mgf
```

**Downsample then write:**

```bash
mgf-utils input.mgf downsample --k 3 write --outfile downsampled.mgf
```

**Merge, downsample, and write:**

```bash
mgf-utils file1.mgf file2.mgf \
  downsample --k 5 \
  write --outfile merged_downsampled.mgf
```

---

## `casanovo-utils`

Miscellaneous casanovoutils utilities.

**Entry point:** `casanovoutils:main`

### Sub-commands

#### `dump-residues`

Copy the default residue mass table (a YAML file) to a specified path. This
file can then be edited and passed back to other tools via `--residues_path`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `destination_path` | path | required | Destination path for the YAML file |

**Example:**

```bash
casanovo-utils dump-residues my_residues.yaml
```

Edit `my_residues.yaml` to add custom modifications, then use it with:

```bash
graph-prec-cov --residues_path my_residues.yaml ...
```

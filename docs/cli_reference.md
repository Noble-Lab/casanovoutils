# CLI Reference

casanovoutils installs a single `casanovoutils` command with nested
subcommands. All subcommands are built with
[Python Fire](https://github.com/google/python-fire), which means:

- Boolean flags can be passed as `--flag` (True) or `--noflag` (False).
- Positional arguments can also be passed as keyword arguments.

The top-level key for each group is the bare module name. The full structure is:

```text
casanovoutils
├── mgfutils        — MGF file processing
│   ├── pipeline
│   ├── shuffle
│   ├── downsample
│   ├── spectra-per-peptide
│   ├── downsample-spectra
│   └── purge-redundant
├── mzmlutils       — mzML file sampling (writes MGF)
├── denovoutils     — Load PSM data into DataFrames
│   ├── get_mgf_psms
│   ├── get_mztab
│   └── get_groundtruth
├── preccov         — Precision-coverage evaluation
│   ├── get_pc_df
│   └── graph_prec_cov
├── summarize_mgf   — MGF file statistics and HTML reports
│   ├── summarize
│   ├── charge-distribution
│   ├── fragment-coverage
│   ├── peak-counts
│   └── peptide-lengths
├── datasets        — Create train/val/test splits from MGF files
├── graphloss       — Plot Casanovo training/validation loss curves
└── residues        — Residue mass table utilities
```

---

## `casanovoutils mgfutils`

Process MGF spectrum files.

### `pipeline`

Run spectra through an optional chain of processing stages in order:
shuffle → downsample → purge redundant peaks. Each stage is skipped when
its enabling parameter is omitted.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--outfile` | path | `None` | Output MGF file path |
| `--do_shuffle` | bool | `True` | Shuffle spectra |
| `--downsample_k` | int | `None` | Max spectra per peptide (skip if omitted) |
| `--purge_epsilon` | float | `None` | Min m/z gap to keep a peak in Da (skip if omitted) |
| `--random_seed` | int | `42` | Random seed for shuffle and downsample |

**Examples:**

```bash
# Shuffle only
casanovoutils mgfutils pipeline input.mgf --outfile out.mgf --nodo_shuffle False

# Downsample to 2 spectra per peptide, no shuffle
casanovoutils mgfutils pipeline input.mgf --outfile out.mgf --nodo_shuffle --downsample_k 2

# Full pipeline
casanovoutils mgfutils pipeline input.mgf --outfile out.mgf \
  --downsample_k 3 --purge_epsilon 0.001
```

---

### `shuffle`

Read all spectra and return them in a shuffled order.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--outfile` | path | `None` | Output MGF file path |
| `--random_seed` | int | `42` | Random seed for reproducibility |

**Example:**

```bash
casanovoutils mgfutils shuffle input.mgf --outfile shuffled.mgf
```

---

### `downsample`

Limit the number of spectra retained per peptide sequence.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--k` | int | `1` | Maximum spectra per peptide |
| `--outfile` | path | `None` | Output MGF file path |
| `--random_seed` | int | `42` | Random seed for reproducibility |

**Example:**

```bash
casanovoutils mgfutils downsample input.mgf --outfile sampled.mgf --k 5
```

---

### `spectra-per-peptide`

Reservoir-sample up to `k` spectra per peptide in a single streaming pass.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--k` | int | `1` | Maximum spectra per peptide |
| `--outfile` | path | `None` | Output MGF file path |
| `--random_seed` | int | `42` | Random seed for reproducibility |

**Example:**

```bash
casanovoutils mgfutils spectra-per-peptide input.mgf --outfile sampled.mgf --k 3
```

---

### `downsample-spectra`

Downsample an MGF file to a target number or proportion of spectra using an
adaptive two-pass streaming approach that guarantees exactly `k` spectra.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `input_file` | path | required | Input MGF file |
| `output_file` | path | required | Output MGF file (must differ from input) |
| `--downsample_type` | str | `"number"` | `"number"` (exact count) or `"proportion"` |
| `--downsample_rate` | float | `100` | Target count (integer) or proportion in `(0, 1]` |
| `--random_seed` | int | `42` | Random seed for reproducibility |

**Examples:**

```bash
# Keep exactly 1000 spectra
casanovoutils mgfutils downsample-spectra input.mgf out.mgf \
  --downsample_type number --downsample_rate 1000

# Keep 20 % of spectra
casanovoutils mgfutils downsample-spectra input.mgf out.mgf \
  --downsample_type proportion --downsample_rate 0.2
```

---

### `purge-redundant`

Sort peaks by m/z and remove any peak whose m/z differs from the previous
peak by less than `epsilon`.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--epsilon` | float | `~1.19e-7` | Minimum m/z separation in Da to keep a peak |
| `--outfile` | path | `None` | Output MGF file path |

**Example:**

```bash
casanovoutils mgfutils purge-redundant input.mgf --outfile purged.mgf --epsilon 0.005
```

---

## `casanovoutils mzmlutils`

Sample a proportion of spectra from an mzML file and write the result as MGF.

Reads the file in chunks of `buffer_size` spectra and draws `round(k ×
chunk_size)` spectra from each chunk at random, without replacement, in a
single streaming pass.  Precursor m/z, charge state, and retention time are
carried through to the output MGF when present in the source file.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `input_file` | path | required | Input mzML file |
| `k` | float | required | Proportion of spectra to sample; must be in (0, 1) |
| `outfile` | path | required | Output MGF file path (must have `.mgf` extension) |
| `--buffer_size` | int | `1000` | Spectra read per I/O chunk |
| `--random_seed` | int | `42` | Random seed for reproducibility |

> **Note on count accuracy:** the final sample count equals
> `sum(round(k × b) for b in buffers)`, which can differ slightly from
> `round(k × total)` due to per-buffer rounding.  Use a `buffer_size` large
> relative to `1 / k` to minimise this effect.
>
> **mzML output:** not supported directly.  If you need mzML output, convert
> the MGF result with [msConvert](https://proteowizard.sourceforge.io/).

**Examples:**

```bash
# Sample 10 % of spectra
casanovoutils mzmlutils input.mzML 0.1 sampled.mgf

# Sample 25 % with a 5 000-spectrum buffer
casanovoutils mzmlutils input.mzML 0.25 sampled.mgf --buffer_size 5000

# Reproducible run with a fixed seed
casanovoutils mzmlutils input.mzML 0.5 sampled.mgf --random_seed 123
```

---

## `casanovoutils denovoutils`

Load and join PSM data from MGF and mzTab files into Polars DataFrames.

### `get_mgf_psms`

Load spectrum metadata from an MGF file.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_path` | path | required | Input MGF file |
| `--out_path` | path | `None` | Output file path (`.parquet`, `.csv`, or `.tsv`) |
| `--meta_data_only` | bool | `True` | Exclude m/z and intensity arrays from output |

**Example:**

```bash
casanovoutils denovoutils get_mgf_psms input.mgf --out_path psms.parquet
```

---

### `get_mztab`

Load the spectrum match table from an mzTab file.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mztab_path` | path | required | Input mzTab file |
| `--out_path` | path | `None` | Output file path (`.parquet`, `.csv`, or `.tsv`) |

**Example:**

```bash
casanovoutils denovoutils get_mztab results.mztab --out_path matches.parquet
```

---

### `get_groundtruth`

Join MGF PSM metadata with mzTab predictions into a single DataFrame.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_path` | path | required | Input MGF file |
| `mztab_path` | path | required | Input mzTab file |
| `--out_path` | path | `None` | Output file path (`.parquet`, `.csv`, or `.tsv`) |

**Example:**

```bash
casanovoutils denovoutils get_groundtruth input.mgf results.mztab \
  --out_path groundtruth.parquet
```

---

## `casanovoutils preccov`

Compute and plot precision-coverage curves from PSM predictions.

### `get_pc_df`

Build a precision-coverage DataFrame from predicted and ground-truth PSMs.
Accepts a pre-built ground-truth DataFrame or the raw MGF and mzTab paths.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--ground_truth_df` | path | `None` | Pre-built ground-truth DataFrame |
| `--mgf_df` | path | `None` | MGF PSM DataFrame (required if `ground_truth_df` is omitted) |
| `--mztab_df` | path | `None` | mzTab DataFrame (required if `ground_truth_df` is omitted) |
| `--residues_path` | path | `None` | Custom residue mass YAML; uses bundled file if omitted |
| `--replace_isoleucine_with_leucine` | bool | `True` | Treat I and L as equivalent |
| `--aa_level` | bool | `False` | Compute per-amino-acid rather than per-peptide metrics |
| `--out_path` | path | `None` | Output file path for the resulting DataFrame |

**Example:**

```bash
casanovoutils preccov get_pc_df \
  --mgf_df psms.parquet --mztab_df matches.parquet \
  --out_path pc.parquet
```

---

### `graph_prec_cov`

Plot precision-coverage curves from one or more pre-computed DataFrames.
Each file is plotted as a separate series labelled by its file stem.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `*pc_df_paths` | path(s) | required | One or more precision-coverage DataFrames |
| `--out_path` | path | `None` | Save the figure to this path (e.g. `.png`, `.pdf`) |

**Example:**

```bash
casanovoutils preccov graph_prec_cov run1.parquet run2.parquet \
  --out_path comparison.png
```

---

## `casanovoutils summarize_mgf`

Generate per-file statistics and visualisations for MGF files.

### `summarize`

Produce a self-contained HTML report for an MGF file covering charge
distribution, peak counts, peptide lengths, and fragment ion coverage.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_file` | path | required | Input MGF file |
| `--output_root` | path | `"mgf_summary"` | Output directory; HTML file shares this basename |
| `--tolerance` | float | `0.05` | Fragment mass tolerance |
| `--tolerance_unit` | str | `"Da"` | Tolerance unit: `"ppm"` or `"Da"` |
| `--workers` | int | `1` | Parallel worker processes for coverage annotation |
| `--max_charge` | str | `"1less"` | Max fragment charge: `"max"` or `"1less"` |
| `--neutral_losses` | bool | `True` | Include neutral losses in annotation |

**Example:**

```bash
casanovoutils summarize_mgf summarize input.mgf --output_root my_report \
  --tolerance 10 --tolerance_unit ppm --workers 4
```

---

### `charge-distribution`

Count and plot the charge state distribution across all spectra.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_file` | path | required | Input MGF file |
| `--output_tsv` | path | `"charge_distribution.tsv"` | Output counts TSV |
| `--output_plot` | path | `"charge_distribution.png"` | Output bar chart |

**Example:**

```bash
casanovoutils summarize_mgf charge-distribution input.mgf \
  --output_tsv charges.tsv --output_plot charges.png
```

---

### `fragment-coverage`

Annotate spectra with b/y ions and report the fraction of total intensity
covered by matched fragments.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_file` | path | required | Annotated MGF file (requires `SEQ=` in ProForma notation) |
| `--tolerance` | float | `0.05` | Mass tolerance |
| `--tolerance_unit` | str | `"Da"` | Tolerance unit: `"ppm"` or `"Da"` |
| `--output_tsv` | path | `"fragment_coverage.tsv"` | Summary TSV |
| `--output_full_tsv` | path | `"fragment_coverage.full.tsv"` | Per-spectrum TSV |
| `--output_plot` | path | `"fragment_coverage.png"` | Coverage histogram |
| `--workers` | int | `1` | Parallel worker processes |
| `--max_charge` | str | `"1less"` | Max fragment charge: `"max"` or `"1less"` |
| `--neutral_losses` | bool | `True` | Include neutral losses |

**Example:**

```bash
casanovoutils summarize_mgf fragment-coverage input.mgf \
  --tolerance 10 --tolerance_unit ppm --workers 4
```

---

### `peak-counts`

Histogram of the number of peaks per spectrum.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_file` | path | required | Input MGF file |
| `--output_tsv` | path | `"peak_counts.tsv"` | Output counts TSV |
| `--output_plot` | path | `"peak_counts.png"` | Output histogram |

**Example:**

```bash
casanovoutils summarize_mgf peak-counts input.mgf
```

---

### `peptide-lengths`

Histogram of peptide sequence lengths for annotated spectra (requires `SEQ=`).

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_file` | path | required | Input MGF file |
| `--output_tsv` | path | `"peptide_lengths.tsv"` | Output counts TSV |
| `--output_plot` | path | `"peptide_lengths.png"` | Output histogram |

**Example:**

```bash
casanovoutils summarize_mgf peptide-lengths input.mgf
```

---

## `casanovoutils datasets`

Create peptide-level train/validation/test splits from annotated MGF files.
Peptides are split 80 / 10 / 10 by unique sequence to prevent leakage between
splits. Outputs three MGF files: `<output_root>.train.mgf`, `.val.mgf`, and
`.test.mgf`.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `*mgf_files` | path(s) | required | One or more annotated MGF files |
| `--output_root` | str | required | Base path for output files |
| `--spectra_per_peptide` | int | `None` | Cap spectra per peptide from new input files |
| `--random_seed` | int | `42` | Random seed for reproducibility |
| `--overwrite` | bool | `False` | Overwrite existing output files |
| `--existing_splits` | paths | `None` | Tuple of existing (train, val, test) MGF paths to extend |
| `--combine_with_existing` | bool | `False` | Include existing spectra in output alongside new ones |

**Examples:**

```bash
# Basic split
casanovoutils datasets input.mgf --output_root splits/run1

# Multiple input files, cap at 3 spectra per peptide
casanovoutils datasets a.mgf b.mgf --output_root splits/combined \
  --spectra_per_peptide 3
```

---

## `casanovoutils graphloss`

Read Casanovo log files and/or `metrics.csv` files and plot training and
validation loss curves.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `root` | str | required | Output file root; plot saved as `<root>.png` |
| `inputs` | path(s) | required | One or more Casanovo log or `metrics.csv` files |
| `--max_y` | float | `None` | Optional y-axis maximum |

**Example:**

```bash
casanovoutils graphloss run1_plot run1.log run2_metrics.csv --max_y 2.0
```

---

## `casanovoutils residues`

Copy the bundled residue mass YAML file to a specified path.  The file can
then be edited to add custom modifications or non-standard residues and passed
back to other tools via `--residues_path`.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `destination_path` | path | required | Destination path for the YAML file |

**Example:**

```bash
casanovoutils residues my_residues.yaml
```

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
├── residues        — Residue mass table utilities
└── visualize_errors — Mirror plots of top-k incorrectly predicted spectra
```

---

## `casanovoutils mgfutils`

Process MGF spectrum files.

### `pipeline`

Run spectra through an optional chain of processing stages in order:
shuffle → spectra-per-peptide → purge redundant peaks. Each stage is skipped when
its enabling parameter is omitted.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--outfile` | path | `None` | Output MGF file path |
| `--do_shuffle` | bool | `True` | Shuffle spectra |
| `--downsample_k` | int | `None` | Max spectra per peptide (skip if omitted) |
| `--purge_epsilon` | float | `None` | Min m/z gap to keep a peak in Da (skip if omitted) |
| `--random_seed` | int | `42` | Random seed for shuffle and spectra-per-peptide |

**Examples:**

```bash
# Shuffle only (default; no extra flags needed)
casanovoutils mgfutils pipeline input.mgf --outfile out.mgf

# Cap at 2 spectra per peptide, no shuffle
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

### `spectra-per-peptide`

Reservoir-sample up to `k` spectra per peptide in a single streaming pass.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--k` | int | `1` | Maximum spectra per peptide |
| `--outfile` | path | `None` | Output MGF file path |
| `--precursor` | bool | `False` | Group by peptide sequence *and* charge state (same peptide in different charge states treated as separate groups) |
| `--ignore_mods` | bool | `False` | Strip ProForma bracketed modification annotations before grouping (modified and unmodified forms counted together) |
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
distribution, peak counts, peptide lengths, C-terminal amino acid distribution,
and fragment ion coverage.

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

### `count-cterm-aas`

Count C-terminal amino acid residues across annotated spectra (requires `SEQ=`).
Counts at PSM level (one tally per spectrum, not per unique peptide).  The full
residue token is reported, so a modified residue such as `K[+229.163]` is
counted separately from bare `K`.  Spectra without `SEQ=` are skipped.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_file` | path | required | Input MGF file (requires `SEQ=` in ProForma notation) |
| `--output_tsv` | path | `"cterm_aas.tsv"` | Output counts TSV (`amino_acid`, `count`, `percentage`) |
| `--output_plot` | path | `"cterm_aas.png"` | Output horizontal bar chart |

**Example:**

```bash
casanovoutils summarize_mgf count-cterm-aas input.mgf \
  --output_tsv cterm.tsv --output_plot cterm.png
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
splits.

The following output files are written:

- `<output_root>.{train,val,test}.mgf` — spectra assigned to each split
- `<output_root>.{train,val,test}.peptides.txt` — tab-separated modified and bare sequences
- `<output_root>.log.txt` — single run-level log with spectrum and peptide counts

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `*mgf_files` | path(s) | required | One or more annotated MGF files |
| `--output_root` | str | required | Base path for output files |
| `--spectra_per_precursor` | int | `None` | Cap spectra per (peptide, charge state) precursor from new input files |
| `--random_seed` | int | `42` | Random seed for reproducibility |
| `--overwrite` | bool | `False` | Overwrite existing output files |
| `--existing_splits` | paths | `None` | Tuple of existing (train, val, test) MGF paths to extend |
| `--combine_with_existing` | bool | `False` | Include existing spectra in output alongside new ones |
| `--mskb_format` | bool | `False` | Convert input sequences from MassIVE-KB PTM notation to ProForma before splitting |

**Examples:**

```bash
# Basic split
casanovoutils datasets input.mgf --output_root splits/run1

# Multiple input files, cap at 3 spectra per precursor
casanovoutils datasets a.mgf b.mgf --output_root splits/combined \
  --spectra_per_precursor 3

# Convert MassIVE-KB PTM notation to ProForma before splitting
casanovoutils datasets input.mgf --output_root splits/run1 --mskb_format
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

---

## `casanovoutils visualize_errors`

Plot the top-k incorrectly predicted spectra from a Casanovo de novo
sequencing run.

Loads an annotated MGF file (with `SEQ=` fields in ProForma notation) and a
Casanovo mzTab output file, joins them on the spectrum index, and identifies
incorrect predictions using mass-based matching.  The top-k incorrect spectra
by Casanovo score are plotted as **mirror plots**: the predicted ProForma
sequence annotates the top panel and the ground truth ProForma sequence
annotates the mirrored bottom panel.  A text header on each figure shows rank,
score, charge, precursor m/z, Δm/z in Da and ppm, and scan number.

By default, isoleucine (I) and leucine (L) are treated as equivalent when
deciding whether a prediction is correct — they share the same monoisotopic
mass and cannot be distinguished by standard CID/HCD fragmentation.  Use
`--distinct_il` to treat them as distinct amino acids.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `mgf_file` | path | required | Annotated MGF file (`SEQ=` fields must be in ProForma notation) |
| `mztab_file` | path | required | Casanovo mzTab output file |
| `--output_dir` | path | `"visualize_errors"` | Directory for output PNG files and the log file |
| `--k` | int | `10` | Maximum number of spectra to plot |
| `--fragment_tol` | float | `0.05` | Fragment ion mass tolerance for b/y ion annotation |
| `--fragment_tol_mode` | str | `"Da"` | Tolerance unit: `"Da"` or `"ppm"` |
| `--ion_types` | str | `"by"` | Ion series to annotate, e.g. `"by"` or `"abcxyz"` |
| `--neutral_losses` | bool | `False` | Annotate NH₃ and H₂O neutral loss ions |
| `--distinct_il` | bool | `False` | Treat I and L as distinct amino acids when assessing correctness (by default they are considered equivalent) |
| `--overwrite` | bool | `False` | Overwrite output PNGs from a previous run |
| `--residues_path` | path | `None` | Custom residue mass YAML file; if omitted the bundled `residues.yaml` is used |

Each output PNG is named `rank_NNNN_scan_S.png`, where `NNNN` is the rank
(1 = highest-scoring incorrect prediction) and `S` is the scan number.  A
`visualize_errors.log` file is also written to `output_dir`.

**Examples:**

```bash
# Plot top 10 incorrect spectra with default settings
casanovoutils visualize_errors predictions.mgf casanovo.mztab \
  --output_dir error_plots/

# Plot top 5, treating I and L as distinct, with ppm tolerance
casanovoutils visualize_errors predictions.mgf casanovo.mztab \
  --output_dir error_plots/ --k 5 --distinct_il \
  --fragment_tol 20 --fragment_tol_mode ppm

# Include neutral loss annotations
casanovoutils visualize_errors predictions.mgf casanovo.mztab \
  --output_dir error_plots/ --neutral_losses
```

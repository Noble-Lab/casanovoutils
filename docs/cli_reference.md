# CLI Reference

casanovoutils installs a single `casanovoutils` command with nested
subcommands. All subcommands are built with
[Python Fire](https://github.com/google/python-fire), which means:

- Boolean flags can be passed as `--flag` (True) or `--noflag` (False).
- Positional arguments can also be passed as keyword arguments.

The top-level structure is:

```text
casanovoutils
├── mgf          — MGF file processing
│   ├── pipeline
│   ├── shuffle
│   ├── downsample
│   └── purge-redundant
├── denovo       — Load and join PSM data
│   ├── get_mgf_psms
│   ├── get_mztab
│   └── get_groundtruth
└── dump-residues
    └── dump
```

---

## `casanovoutils mgf`

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
casanovoutils mgf pipeline input.mgf --outfile out.mgf --nodownsample_k

# Downsample to 2 spectra per peptide, no shuffle
casanovoutils mgf pipeline input.mgf --outfile out.mgf --nodo_shuffle --downsample_k 2

# Full pipeline
casanovoutils mgf pipeline input.mgf --outfile out.mgf \
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
casanovoutils mgf shuffle input.mgf --outfile shuffled.mgf
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

**Examples:**

```bash
# Keep at most 1 spectrum per peptide (default)
casanovoutils mgf downsample input.mgf --outfile sampled.mgf

# Keep up to 5 spectra per peptide
casanovoutils mgf downsample input.mgf --outfile sampled.mgf --k 5
```

---

### `purge-redundant`

Sort peaks by m/z and remove any peak whose m/z differs from the previous
peak by less than `epsilon`.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `spectra` | path | required | Input MGF file path |
| `--epsilon` | float | `0.001` | Minimum m/z separation in Da to keep a peak |
| `--outfile` | path | `None` | Output MGF file path |

**Example:**

```bash
casanovoutils mgf purge-redundant input.mgf --outfile purged.mgf --epsilon 0.005
```

---

## `casanovoutils denovo`

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
casanovoutils denovo get_mgf_psms input.mgf --out_path psms.parquet
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
casanovoutils denovo get_mztab results.mztab --out_path matches.parquet
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
casanovoutils denovo get_groundtruth input.mgf results.mztab \
  --out_path groundtruth.parquet
```

---

## `casanovoutils dump-residues`

### `dump`

Copy the default residue mass table (a YAML file) to a specified path. This
file can then be edited and passed back to other tools via `--residues_path`.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `destination_path` | path | required | Destination path for the YAML file |

**Example:**

```bash
casanovoutils dump-residues dump my_residues.yaml
```

Edit `my_residues.yaml` to add custom modifications, then use it with other
tools via `--residues_path`.

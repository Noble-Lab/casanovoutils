# Casanovo Dataset Split Command-Line Utility

This module exposes a small command-line interface (CLI) for constructing the
training, validation, and test splits used to train and evaluate **Casanovo**.
The CLI is implemented using [`fire`](https://github.com/google/python-fire),
which automatically maps Python functions and arguments to shell commands.



## Installation / Invocation

> **Note:** The `casanovo-datasets` entrypoint is not yet registered in
> `pyproject.toml` and is not available after a standard `pip install
> casanovoutils`. The commands below document the intended interface for a
> future release.

```bash
casanovo-datasets <command> [arguments]
```

To see all available commands:

```bash
casanovo-datasets --help
```



## Available Commands

The CLI exposes three top-level commands:

* `multi_enzyme_simple`
* `mskb_final`
* `full_splits`

Each command corresponds directly to one function in the code.



## `multi_enzyme_simple`

Constructs the **multi-enzymic train/validation/test splits** used to train
Casanovo on non-tryptic data.

### Description

PSMs from the MassIVE-KB v2.0.15 dataset are filtered according to optional
allow/block peptide lists, then assigned to train/validation/test splits in an
approximately **8:1:1 ratio**, with splits disjoint at the peptide level.

### Command

```bash
casanovo-datasets multi_enzyme_simple \
  --mskb_v2_path <PATH_TO_MSKB_V2> \
  --output_dir <OUTPUT_DIR> \
  --peptide_allow_list <ALLOW_LIST.json> \
  --peptide_block_list <BLOCK_LIST.json>
```

### Outputs

* `multi-enzyme-simple.train.mgf`
* `multi-enzyme-simple.test.mgf`
* `multi-enzyme-simple.val.mgf`
* `peptides.json`



## `mskb_final`

Constructs the **tryptic MassIVE-KB v1 splits** used for the final Casanovo
training and evaluation.

### Description

From the MassIVE-KB v1 dataset, this command creates peptide-disjoint splits
consisting of:

* **2,000,000 PSMs** for training
* **200,000 PSMs** for validation
* **200,000 PSMs** for testing

Optional peptide allow/block lists are respected prior to splitting.

### Command

```bash
casanovo-datasets mskb_final \
  --mskb_v1_path <PATH_TO_MSKB_V1> \
  --output_dir <OUTPUT_DIR> \
  --peptide_allow_list <ALLOW_LIST.json> \
  --peptide_block_list <BLOCK_LIST.json>
```

### Outputs

* `mskb_final.train.mgf`
* `mskb_final.test.mgf`
* `mskb_final.val.mgf`
* `peptides.json`

casanovo-datasets

## `full_splits`

Convenience command that generates **both tryptic and multi-enzymic splits**
with guaranteed peptide-level isolation between them.

### Description

This command performs the following steps:

1. Runs `mskb_final` to generate tryptic splits.
2. Constructs a peptide block list from the tryptic splits to prevent
   peptide-level leakage.
3. Runs `multi_enzyme_simple` using the generated block list.

This mirrors the split construction used in [Melendez et al.](https://www.biorxiv.org/content/biorxiv/early/2024/05/21/2024.05.16.594602.full.pdf) when combining
tryptic and multi-enzyme training data.

### Command

```bash
casanovo-datasets full_splits \
  --mskb_v1_path <PATH_TO_MSKB_V1> \
  --mskb_v2_path <PATH_TO_MSKB_V2> \
  --output_dir <OUTPUT_DIR> \
  --peptide_allow_list <ALLOW_LIST.json> \
  --peptide_block_list <BLOCK_LIST.json>
```

### Outputs

```
output_dir/
├── mskb_final/
│   ├── mskb_final.train.mgf
│   ├── mskb_final.test.mgf
│   ├── mskb_final.val.mgf
│   └── peptides.json
├── multi_enzyme_simple/
│   ├── multi-enzyme-simple.train.mgf
│   ├── multi-enzyme-simple.test.mgf
│   ├── multi-enzyme-simple.val.mgf
│   └── peptides.json
└── mskb_block_list.json
```

# File Formats

casanovoutils reads and writes two file formats: **MGF** and **mzTab**. This
page describes the relevant fields and conventions expected by each tool.

---

## MGF (Mascot Generic Format)

MGF is a plain-text format for tandem mass spectrometry data. Each spectrum is
delimited by `BEGIN IONS` / `END IONS` blocks.

### Example

```
BEGIN IONS
TITLE=spectrum_001
PEPMASS=612.3456
CHARGE=2+
SEQ=PEPTIDEK
100.0 1234.5
200.0 5678.9
...
END IONS
```

### Fields used by casanovoutils

| Field | Description |
|---|---|
| `SEQ=` | Ground-truth peptide sequence. Required for evaluation with `graph-prec-cov`. |
| `PEPMASS=` | Precursor m/z (and optionally intensity). |
| `CHARGE=` | Precursor charge state. |
| `TITLE=` | Spectrum identifier (optional; used by some tools for logging). |

The m/z and intensity peak list follows the header fields, one peak per line,
space-separated.

### Notes

- casanovoutils reads MGF files using
  [Pyteomics](https://pyteomics.readthedocs.io/). Spectra are accessed in
  order of appearance in the file.
- Spectrum indices used in the mzTab `spectra_ref` column (see below) are
  **zero-based** positions within the MGF file.
- The `SEQ=` field is required for ground-truth evaluation. Casanovo writes
  this field when it generates annotated MGF output.

---

## mzTab

mzTab is a tab-delimited PSI standard format for reporting peptide-spectrum
matches. casanovoutils reads the **PSM section** of mzTab files, which is
produced by Casanovo as its primary output format.

### Required columns

| Column | Description |
|---|---|
| `sequence` | Predicted peptide sequence. |
| `search_engine_score[1]` | Per-PSM confidence score used to rank predictions. |
| `spectra_ref` | Reference to the originating spectrum. |

### `spectra_ref` format

casanovoutils expects `spectra_ref` values of the form:

```
ms_run[1]:index=<INT>
```

where `<INT>` is the **zero-based** index of the spectrum in the corresponding
MGF file. This is the format written by Casanovo.

### Example PSM section

```
PSH	sequence	PSM_ID	accession	unique	database	...	spectra_ref	search_engine_score[1]	...
PSM	PEPTIDEK	1	null	null	null	...	ms_run[1]:index=0	0.9982	...
PSM	ACDEFGHIK	2	null	null	null	...	ms_run[1]:index=1	0.8741	...
```

### Notes

- casanovoutils reads mzTab files using
  [Pyteomics](https://pyteomics.readthedocs.io/).
- Only spectra present in the mzTab are evaluated; spectra in the MGF that
  have no corresponding mzTab row are assigned a score of `-1.0` and marked as
  incorrect.
- Additional columns present in the mzTab (e.g., per-amino-acid score columns
  from Casanovo) are preserved in the internal DataFrame but are not used
  unless explicitly referenced.

# FAQ

## General

**What is the AUPC?**

The AUPC (Area Under the Precision–Coverage curve) is a single-number summary
of a model's precision–coverage trade-off. It is computed as the area under
the curve obtained by sorting predictions from highest to lowest confidence and
computing the running precision and coverage at each threshold. A perfect model
has an AUPC of 1.0; a random model will have an AUPC approximately equal to
the fraction of correct predictions.

**What is a precision–coverage curve?**

A precision–coverage curve plots precision (fraction of accepted predictions
that are correct) on the y-axis against coverage (fraction of all spectra
included) on the x-axis. As the confidence threshold is lowered, more spectra
are included (coverage increases) but precision may decrease. The curve
summarizes the accuracy–completeness trade-off across all possible thresholds.

---

## Installation

**What Python version is required?**

casanovoutils requires Python 3.13 or later.

**How do I install casanovoutils in a virtual environment?**

```bash
python -m venv .venv
source .venv/bin/activate
pip install casanovoutils
```

Or with uv:

```bash
uv venv
uv pip install casanovoutils
```

---

## Evaluation

**Are isoleucine (I) and leucine (L) treated as the same amino acid?**

By default, yes. Pass `--noreplace_i_l` to require an exact match.

**What happens to spectra that are missing from the mzTab output?**

Spectra present in the MGF but absent from the mzTab are assigned a prediction
score of `-1.0` and marked as incorrect. This means they appear at the bottom
of the ranked list and count against coverage, which accurately reflects that
the model did not return a prediction for those spectra.

**My mzTab and MGF files come from different tools. Will casanovoutils work?**

casanovoutils expects `spectra_ref` values of the form
`ms_run[1]:index=<INT>` where `<INT>` is the zero-based position of the
spectrum in the MGF file. This is the format written by Casanovo. If your
mzTab uses a different `spectra_ref` convention, you may need to reformat it
before using casanovoutils.

---

## MGF operations

**What does `pipeline` do compared to running stages individually?**

`casanovoutils mgf pipeline` chains shuffle → downsample → purge-redundant
in a single pass, writing one output file. Each stage is skipped when its
parameter is omitted. Running the stages individually via separate commands
produces identical results but requires intermediate files.

**What does downsampling do to peptides with fewer than `k` spectra?**

All spectra for that peptide are kept. Downsampling only removes spectra when a
peptide has *more* than `k` spectra; otherwise the full set is retained.

**Is the downsampling reproducible?**

Yes, set `--random_seed` to a fixed integer. The default seed is `42`.

**What does `purge-redundant` do exactly?**

Peaks within each spectrum are sorted by m/z. Any peak whose m/z differs from
the preceding peak by less than `epsilon` (default `0.001` Da) is discarded.
This removes near-duplicate peaks that can arise from instrument noise or
rounding.

---

## Residue mass tables

**What is the default residue mass table?**

The bundled `residues.yaml` file contains standard monoisotopic masses for the
20 canonical amino acids plus common modifications used by Casanovo. Export it
with:

```bash
casanovoutils dump-residues dump residues.yaml
```

**How do I add a custom modification?**

Export the default table, add your modification as a new key–value pair
(residue name: mass in daltons), and pass the edited file back via
`--residues_path`:

```bash
casanovoutils dump-residues dump my_residues.yaml
# edit my_residues.yaml ...
```

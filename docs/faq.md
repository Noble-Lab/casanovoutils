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

By default, yes. The `add-peptides` method of `graph-prec-cov` sets
`--replace_i_l True` by default, so I and L in ground-truth sequences are
replaced with L before comparing to predictions. To disable this and require
an exact match, pass `--noreplace_i_l`:

```bash
graph-prec-cov \
  add-peptides results.mztab truth.mgf "Strict" --noreplace_i_l \
  save strict.png
```

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

**Can I combine multiple MGF files before downsampling?**

Yes. `mgf-utils` accepts multiple input files and streams them together:

```bash
mgf-utils file1.mgf file2.mgf downsample --k 3 write --outfile out.mgf
```

**What does downsampling do to peptides with fewer than `k` spectra?**

All spectra for that peptide are kept. Downsampling only removes spectra when a
peptide has *more* than `k` spectra; otherwise the full set is retained.

**Is the downsampling reproducible?**

Yes, set `--random_seed` to a fixed integer. The default seed is `42`.

---

## Residue mass tables

**What is the default residue mass table?**

The bundled `residues.yaml` file contains standard monoisotopic masses for the
20 canonical amino acids plus common modifications used by Casanovo. You can
export it with:

```bash
casanovo-utils dump-residues residues.yaml
```

**How do I add a custom modification?**

Export the default table, add your modification as a new key–value pair
(residue name: mass in daltons), and pass the edited file back via
`--residues_path`:

```bash
casanovo-utils dump-residues my_residues.yaml
# edit my_residues.yaml ...
graph-prec-cov --residues_path my_residues.yaml ...
```

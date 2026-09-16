import csv

import numpy as np
import pytest

from casanovoutils.summarize_mgf import (
    _extract_cterm_token,
    charge_distribution,
    count_charge_states,
    count_cterm_aas,
    count_peaks,
    cterm_aa_distribution,
    fragment_coverage,
    measure_peptide_lengths,
    peak_counts,
    peptide_lengths,
    summarize_mgf,
)

# ---------------------------------------------------------------------------
# count_charge_states tests (pure function)
# ---------------------------------------------------------------------------


def _spectrum(charge):
    """Create a minimal spectrum dict with the given charge (as a list)."""
    return {"params": {"charge": [charge]}}


def test_count_charge_states_mixed():
    """Multiple spectra with mixed charges produce correct counts."""
    spectra = [_spectrum(2), _spectrum(3), _spectrum(2), _spectrum(4), _spectrum(3)]
    counts, n_skipped = count_charge_states(spectra)
    assert counts == {2: 2, 3: 2, 4: 1}
    assert n_skipped == 0


def test_count_charge_states_single():
    """All spectra with the same charge produce a single entry."""
    spectra = [_spectrum(2), _spectrum(2), _spectrum(2)]
    counts, n_skipped = count_charge_states(spectra)
    assert counts == {2: 3}
    assert n_skipped == 0


def test_count_charge_states_empty():
    """Empty input produces an empty dict."""
    counts, n_skipped = count_charge_states([])
    assert counts == {}
    assert n_skipped == 0


def test_count_charge_states_scalar_charge():
    """Charge returned as a single integer (not a list) is handled."""
    spectra = [
        {"params": {"charge": 2}},
        {"params": {"charge": 3}},
        {"params": {"charge": 2}},
    ]
    counts, n_skipped = count_charge_states(spectra)
    assert counts == {2: 2, 3: 1}
    assert n_skipped == 0


def test_count_charge_states_multiple_charges_skipped():
    """Spectra with multiple charge states are skipped."""
    spectra = [
        _spectrum(2),
        {"params": {"charge": [2, 3]}},  # ambiguous — skip
        _spectrum(3),
    ]
    counts, n_skipped = count_charge_states(spectra)
    assert counts == {2: 1, 3: 1}
    assert n_skipped == 1


def test_count_charge_states_empty_charge_list_skipped():
    """Spectra with an empty charge list are skipped."""
    spectra = [
        _spectrum(2),
        {"params": {"charge": []}},  # empty — skip
    ]
    counts, n_skipped = count_charge_states(spectra)
    assert counts == {2: 1}
    assert n_skipped == 1


def test_count_charge_states_missing_charge_skipped():
    """Spectra with no charge key are skipped."""
    spectra = [
        {"params": {}},  # no charge key — skip
        _spectrum(3),
    ]
    counts, n_skipped = count_charge_states(spectra)
    assert counts == {3: 1}
    assert n_skipped == 1


# ---------------------------------------------------------------------------
# charge_distribution integration test
# ---------------------------------------------------------------------------

SMALL_MGF = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.0
CHARGE=2+
100.0 10
200.0 20
END IONS

BEGIN IONS
TITLE=spec2
PEPMASS=600.0
CHARGE=3+
150.0 15
250.0 25
END IONS

BEGIN IONS
TITLE=spec3
PEPMASS=700.0
CHARGE=2+
120.0 12
220.0 22
END IONS
"""


def test_charge_distribution_integration(tmp_path):
    """CLI function writes expected TSV and PNG."""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(SMALL_MGF)

    tsv_path = tmp_path / "out.tsv"
    png_path = tmp_path / "out.png"

    charge_distribution(str(mgf_path), str(tsv_path), str(png_path))

    # Check TSV contents
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["charge"] == "2"
    assert rows[0]["count"] == "2"
    assert rows[1]["charge"] == "3"
    assert rows[1]["count"] == "1"

    # Check PNG exists and is non-empty
    assert png_path.exists()
    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Helpers for new tests
# ---------------------------------------------------------------------------


def _make_spectrum(charge, n_peaks, seq=None):
    """Build a minimal spectrum dict for testing."""
    mz = np.linspace(100.0, 1000.0, n_peaks)
    intensity = np.ones(n_peaks, dtype=float)
    params = {"charge": [charge]}
    if seq is not None:
        params["seq"] = seq
    return {
        "params": params,
        "m/z array": mz,
        "intensity array": intensity,
    }


# ---------------------------------------------------------------------------
# count_peaks tests (pure function)
# ---------------------------------------------------------------------------


def test_count_peaks_basic():
    """count_peaks returns the number of peaks for each spectrum."""
    spectra = [
        _make_spectrum(2, 5),
        _make_spectrum(3, 10),
        _make_spectrum(2, 3),
    ]
    result = count_peaks(spectra)
    assert result == [5, 10, 3]


def test_count_peaks_empty():
    """Empty input returns an empty list."""
    assert count_peaks([]) == []


# ---------------------------------------------------------------------------
# measure_peptide_lengths tests (pure function)
# ---------------------------------------------------------------------------


def test_measure_peptide_lengths_basic():
    """Known sequences return correct lengths."""
    spectra = [
        _make_spectrum(2, 5, seq="AGK"),  # 3 residues
        _make_spectrum(2, 5, seq="PEPTIDE"),  # 7 residues
    ]
    lengths, n_skipped = measure_peptide_lengths(spectra)
    assert lengths == [3, 7]
    assert n_skipped == 0


def test_measure_peptide_lengths_no_seq():
    """Spectra without SEQ= are counted as skipped."""
    spectra = [
        _make_spectrum(2, 5, seq="AGK"),
        _make_spectrum(2, 5),  # no seq
        _make_spectrum(2, 5),  # no seq
    ]
    lengths, n_skipped = measure_peptide_lengths(spectra)
    assert lengths == [3]
    assert n_skipped == 2


def test_measure_peptide_lengths_unknown_mod():
    """Unknown modification names do not cause a spectrum to be skipped."""
    spectra = [
        _make_spectrum(2, 5, seq="A[FakeMod]G"),  # unknown mod: still parses
        _make_spectrum(2, 5, seq="AGK"),
    ]
    lengths, n_skipped = measure_peptide_lengths(spectra)
    assert lengths == [2, 3]  # A[FakeMod]G has 2 residues
    assert n_skipped == 0


def test_measure_peptide_lengths_with_mod():
    """Known modifications do not affect the residue count."""
    spectra = [
        _make_spectrum(2, 5, seq="AC[Carbamidomethyl]G"),  # 3 residues
    ]
    lengths, n_skipped = measure_peptide_lengths(spectra)
    assert lengths == [3]
    assert n_skipped == 0


# ---------------------------------------------------------------------------
# peak_counts integration test
# ---------------------------------------------------------------------------

SMALL_MGF_PEAKS = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.0
CHARGE=2+
100.0 10
200.0 20
300.0 30
END IONS

BEGIN IONS
TITLE=spec2
PEPMASS=600.0
CHARGE=3+
150.0 15
250.0 25
END IONS

BEGIN IONS
TITLE=spec3
PEPMASS=700.0
CHARGE=2+
120.0 12
220.0 22
320.0 32
END IONS
"""


def test_peak_counts_integration(tmp_path):
    """peak_counts writes expected TSV and PNG."""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(SMALL_MGF_PEAKS)

    tsv_path = tmp_path / "out.tsv"
    png_path = tmp_path / "out.png"

    peak_counts(str(mgf_path), str(tsv_path), str(png_path))

    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)

    # spec1 has 3 peaks, spec2 has 2 peaks, spec3 has 3 peaks
    counts_by_n = {int(r["n_peaks"]): int(r["count"]) for r in rows}
    assert counts_by_n[2] == 1
    assert counts_by_n[3] == 2

    assert png_path.exists()
    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# peptide_lengths integration test
# ---------------------------------------------------------------------------

SMALL_MGF_SEQS = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.0
CHARGE=2+
SEQ=AGK
100.0 10
200.0 20
END IONS

BEGIN IONS
TITLE=spec2
PEPMASS=600.0
CHARGE=3+
100.0 10
200.0 20
END IONS

BEGIN IONS
TITLE=spec3
PEPMASS=700.0
CHARGE=2+
SEQ=PEPTIDE
120.0 12
220.0 22
END IONS
"""


def test_peptide_lengths_integration(tmp_path):
    """peptide_lengths writes expected TSV and PNG; spec without SEQ= skipped."""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(SMALL_MGF_SEQS)

    tsv_path = tmp_path / "out.tsv"
    png_path = tmp_path / "out.png"

    peptide_lengths(str(mgf_path), str(tsv_path), str(png_path))

    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)

    # AGK -> 3, PEPTIDE -> 7
    counts_by_len = {int(r["length"]): int(r["count"]) for r in rows}
    assert counts_by_len[3] == 1
    assert counts_by_len[7] == 1
    # spec2 without SEQ= should not contribute any entry
    assert 0 not in counts_by_len

    assert png_path.exists()
    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# summarize_mgf integration test
# ---------------------------------------------------------------------------


def test_summarize_mgf_integration(tmp_path):
    """summarize_mgf writes HTML linking to PNG/TSV files."""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(SMALL_MGF_SEQS)

    out_dir = tmp_path / "summary"
    summarize_mgf(str(mgf_path), str(out_dir))

    html_path = out_dir / "summary.html"
    assert html_path.exists()
    assert html_path.stat().st_size > 0

    content = html_path.read_text(encoding="utf-8")

    # Expected section headings
    assert "Charge State Distribution" in content
    assert "Peaks per Spectrum" in content
    assert "Peptide Lengths" in content
    assert "Fragment Ion Coverage" in content

    # HTML links to PNG files (not embedded base64)
    assert 'src="charge_distribution.png"' in content
    assert 'src="peak_counts.png"' in content
    assert 'src="peptide_lengths.png"' in content
    assert 'src="fragment_coverage.png"' in content
    assert "data:image/png;base64," not in content

    # PNG and TSV files exist
    for stem in (
        "charge_distribution",
        "peak_counts",
        "peptide_lengths",
        "fragment_coverage",
    ):
        assert (out_dir / f"{stem}.png").exists()
        assert (out_dir / f"{stem}.tsv").exists()

    # Log file exists
    assert (out_dir / "summary.log").exists()


# ---------------------------------------------------------------------------
# fragment_coverage integration test
# ---------------------------------------------------------------------------

SMALL_MGF_ANNOTATED = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.26 100.0
CHARGE=2+
SEQ=AGK
100.0 10
200.0 20
300.0 5
END IONS

BEGIN IONS
TITLE=spec2
PEPMASS=701.37 80.0
CHARGE=2+
SEQ=PEPTIDE
120.0 12
220.0 22
320.0 8
END IONS
"""


def test_fragment_coverage_integration(tmp_path):
    """fragment_coverage writes a summary TSV, a per-spectrum TSV, and a PNG."""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(SMALL_MGF_ANNOTATED)

    tsv_path = tmp_path / "coverage.tsv"
    full_tsv_path = tmp_path / "coverage.full.tsv"
    png_path = tmp_path / "coverage.png"

    fragment_coverage(
        str(mgf_path),
        output_tsv=str(tsv_path),
        output_full_tsv=str(full_tsv_path),
        output_plot=str(png_path),
    )

    # Summary TSV exists and has a row per scored spectrum, sorted by coverage
    assert tsv_path.exists()
    with open(tsv_path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 2
    proportions = [float(r["proportion_matched"]) for r in rows]
    assert proportions == sorted(proportions)  # sorted ascending

    # Full per-spectrum TSV has one row per spectrum
    assert full_tsv_path.exists()
    with open(full_tsv_path) as fh:
        full_rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(full_rows) == 2
    assert set(full_rows[0].keys()) == {"scan", "peptide", "charge", "coverage"}

    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_fragment_coverage_skips_missing_charge(tmp_path):
    """fragment_coverage skips spectra with missing or ambiguous charge."""
    mgf_text = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.26 100.0
CHARGE=2+
SEQ=AGK
100.0 10
200.0 20
END IONS

BEGIN IONS
TITLE=spec_no_charge
PEPMASS=600.0 80.0
SEQ=AGK
100.0 10
200.0 20
END IONS

BEGIN IONS
TITLE=spec_multi_charge
PEPMASS=700.0 60.0
CHARGE=2+, 3+
SEQ=AGK
100.0 10
200.0 20
END IONS
"""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(mgf_text)

    tsv_path = tmp_path / "coverage.tsv"
    full_tsv_path = tmp_path / "coverage.full.tsv"
    png_path = tmp_path / "coverage.png"

    fragment_coverage(
        str(mgf_path),
        output_tsv=str(tsv_path),
        output_full_tsv=str(full_tsv_path),
        output_plot=str(png_path),
    )

    with open(full_tsv_path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    # Only spec1 has a valid, unambiguous charge; the other two are skipped
    assert len(rows) == 1


def test_fragment_coverage_workers_match(tmp_path):
    """fragment_coverage with workers=2 produces the same results as workers=1."""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(SMALL_MGF_ANNOTATED)

    def _run(workers, suffix):
        fragment_coverage(
            str(mgf_path),
            output_tsv=str(tmp_path / f"cov{suffix}.tsv"),
            output_full_tsv=str(tmp_path / f"cov_full{suffix}.tsv"),
            output_plot=str(tmp_path / f"cov{suffix}.png"),
            workers=workers,
        )
        with open(tmp_path / f"cov_full{suffix}.tsv") as fh:
            return list(csv.DictReader(fh, delimiter="\t"))

    rows1 = _run(1, "_w1")
    rows2 = _run(2, "_w2")

    # Same set of scans and coverages regardless of worker count
    assert {r["scan"] for r in rows1} == {r["scan"] for r in rows2}
    cov1 = {r["scan"]: r["coverage"] for r in rows1}
    cov2 = {r["scan"]: r["coverage"] for r in rows2}
    assert cov1 == cov2


# ---------------------------------------------------------------------------
# _extract_cterm_token tests (pure function)
# ---------------------------------------------------------------------------


def test_extract_cterm_token_bare():
    """Plain sequence returns the last amino acid."""
    assert _extract_cterm_token("PEPTIDE") == "E"


def test_extract_cterm_token_with_mod():
    """Last residue with a modification returns the full token."""
    assert _extract_cterm_token("PEPTK[+229.163]") == "K[+229.163]"


def test_extract_cterm_token_named_mod():
    """Named modification on the last residue is included in the token."""
    assert _extract_cterm_token("PEPTIC[Carbamidomethyl]") == "C[Carbamidomethyl]"


def test_extract_cterm_token_nterm_mod_ignored():
    """N-terminal modification does not affect C-terminal extraction."""
    assert _extract_cterm_token("[Acetyl]-PEPTIDER") == "R"


def test_extract_cterm_token_cterm_tag_stripped():
    """ProForma C-terminal sequence tag is stripped; bare residue returned."""
    assert _extract_cterm_token("PEPTIDE-[Amidated]") == "E"


def test_extract_cterm_token_nterm_shift():
    """Leading mass shift is ignored; correct C-terminal residue returned."""
    assert (
        _extract_cterm_token("+229.163AC[Carbamidomethyl]GANHTLVLDSQK[+229.163]")
        == "K[+229.163]"
    )


def test_extract_cterm_token_empty():
    """Empty string returns None."""
    assert _extract_cterm_token("") is None


# ---------------------------------------------------------------------------
# count_cterm_aas tests (pure function)
# ---------------------------------------------------------------------------


def _seq_spectrum(seq):
    """Build a minimal spectrum dict with only a SEQ= field."""
    return {
        "params": {"seq": seq},
        "m/z array": np.array([]),
        "intensity array": np.array([]),
    }


def test_count_cterm_aas_basic():
    """Correct C-terminal tokens are counted at PSM level."""
    spectra = [
        _seq_spectrum("PEPTIDK"),
        _seq_spectrum("GFLAGGR"),
        _seq_spectrum("PEPTIDR"),
        _seq_spectrum("PEPTIDK"),
    ]
    counts, n_skipped = count_cterm_aas(spectra)
    assert counts["K"] == 2
    assert counts["R"] == 2
    assert n_skipped == 0


def test_count_cterm_aas_with_mod():
    """Residue token including modification is counted separately from bare residue."""
    spectra = [
        _seq_spectrum("PEPTIDK[+229.163]"),
        _seq_spectrum("PEPTIDK"),
        _seq_spectrum("PEPTIDK[+229.163]"),
    ]
    counts, n_skipped = count_cterm_aas(spectra)
    assert counts["K[+229.163]"] == 2
    assert counts["K"] == 1
    assert n_skipped == 0


def test_count_cterm_aas_no_seq_skipped():
    """Spectra without SEQ= are counted as skipped."""
    spectra = [
        _seq_spectrum("PEPTIDK"),
        {"params": {}, "m/z array": np.array([]), "intensity array": np.array([])},
    ]
    counts, n_skipped = count_cterm_aas(spectra)
    assert counts["K"] == 1
    assert n_skipped == 1


def test_count_cterm_aas_invalid_proforma_skipped():
    """Spectra with invalid ProForma sequences are skipped, not counted."""
    spectra = [
        _seq_spectrum("PEPTIDK"),
        _seq_spectrum("NOT[VALID[PROFORMA"),  # unclosed bracket — invalid ProForma
    ]
    counts, n_skipped = count_cterm_aas(spectra)
    assert counts["K"] == 1
    assert n_skipped == 1


def test_count_cterm_aas_il_counted_separately():
    """I and L at the C-terminus are counted as distinct tokens."""
    spectra = [
        _seq_spectrum("PEPTIDI"),
        _seq_spectrum("PEPTIDL"),
        _seq_spectrum("PEPTIDL"),
    ]
    counts, n_skipped = count_cterm_aas(spectra)
    assert counts["I"] == 1
    assert counts["L"] == 2
    assert n_skipped == 0


# ---------------------------------------------------------------------------
# cterm_aa_distribution integration test
# ---------------------------------------------------------------------------

SMALL_MGF_CTERM = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.0
CHARGE=2+
SEQ=PEPTIDK
100.0 10
END IONS

BEGIN IONS
TITLE=spec2
PEPMASS=600.0
CHARGE=2+
SEQ=PEPTIDEK[+229.163]
100.0 10
END IONS

BEGIN IONS
TITLE=spec3
PEPMASS=700.0
CHARGE=2+
SEQ=PEPTIDR
100.0 10
END IONS

BEGIN IONS
TITLE=spec4
PEPMASS=800.0
CHARGE=2+
100.0 10
END IONS
"""


def test_cterm_aa_distribution_integration(tmp_path):
    """cterm_aa_distribution writes expected TSV (sorted by count) and PNG."""
    mgf_path = tmp_path / "test.mgf"
    mgf_path.write_text(SMALL_MGF_CTERM)

    tsv_path = tmp_path / "out.tsv"
    png_path = tmp_path / "out.png"

    cterm_aa_distribution(str(mgf_path), str(tsv_path), str(png_path))

    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)

    # 3 spectra have SEQ=; 1 is skipped (no SEQ=)
    assert len(rows) == 3
    counts = {r["amino_acid"]: int(r["count"]) for r in rows}

    assert counts["K"] == 1
    assert counts["K[+229.163]"] == 1
    assert counts["R"] == 1

    # Verify percentage column is present and sums to ~100
    pcts = [float(r["percentage"]) for r in rows]
    assert abs(sum(pcts) - 100.0) < 0.1

    assert png_path.exists()
    assert png_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Graceful exit when all spectra are filtered out
# ---------------------------------------------------------------------------

SMALL_MGF_NO_SEQ = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.0
CHARGE=2+
100.0 10
200.0 20
END IONS

BEGIN IONS
TITLE=spec2
PEPMASS=600.0
CHARGE=3+
150.0 15
250.0 25
END IONS
"""


def test_all_spectra_filtered_exits(tmp_path):
    """Both peptide_lengths and fragment_coverage exit with a non-zero status
    when all spectra are filtered out (here: none have SEQ=)."""
    mgf_path = tmp_path / "no_seq.mgf"
    mgf_path.write_text(SMALL_MGF_NO_SEQ)

    with pytest.raises(SystemExit) as exc_info:
        peptide_lengths(
            str(mgf_path),
            output_tsv=str(tmp_path / "out.tsv"),
            output_plot=str(tmp_path / "out.png"),
        )
    assert exc_info.value.code != 0

    with pytest.raises(SystemExit) as exc_info:
        fragment_coverage(
            str(mgf_path),
            output_tsv=str(tmp_path / "cov.tsv"),
            output_full_tsv=str(tmp_path / "cov_full.tsv"),
            output_plot=str(tmp_path / "cov.png"),
        )
    assert exc_info.value.code != 0

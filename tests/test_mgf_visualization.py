import csv

import numpy as np

from casanovoutils.mgf_visualization import (
    AA_MASS,
    MOD_MASS,
    charge_distribution,
    count_charge_states,
    matched_proportion,
    parse_proforma,
    theoretical_mzs,
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
# parse_proforma tests
# ---------------------------------------------------------------------------


def test_parse_proforma_simple_sequence():
    """A plain amino acid sequence returns (0.0, list_of_residue_masses)."""
    result = parse_proforma("AG")
    assert result is not None
    n_term, masses = result
    assert n_term == 0.0
    assert len(masses) == 2
    assert np.isclose(masses[0], AA_MASS["A"])
    assert np.isclose(masses[1], AA_MASS["G"])


def test_parse_proforma_modified_residue():
    """A residue with a bracketed modification adds the mod delta."""
    result = parse_proforma("AC[Carbamidomethyl]G")
    assert result is not None
    n_term, masses = result
    assert len(masses) == 3
    assert np.isclose(masses[1], AA_MASS["C"] + MOD_MASS["Carbamidomethyl"])


def test_parse_proforma_nterm_mod():
    """An N-terminal modification is parsed into n_term_mod."""
    result = parse_proforma("[Acetyl]-AG")
    assert result is not None
    n_term, masses = result
    assert np.isclose(n_term, MOD_MASS["Acetyl"])
    assert len(masses) == 2


def test_parse_proforma_unknown_mod_returns_none():
    """An unrecognised modification causes None to be returned."""
    assert parse_proforma("A[FakeMod]G") is None


def test_parse_proforma_unknown_aa_returns_none():
    """An unrecognised amino acid letter causes None to be returned."""
    assert parse_proforma("AXG") is None


# ---------------------------------------------------------------------------
# theoretical_mzs tests
# ---------------------------------------------------------------------------


def test_theoretical_mzs_known_peptide_count():
    """A known dipeptide should produce the expected number of fragments."""
    parsed = parse_proforma("AG")
    n_term, masses = parsed
    mzs = theoretical_mzs(n_term, masses, precursor_charge=2)
    # dipeptide: 1 cleavage site, b+y ions, 3 neutral losses, 1 charge state
    # = 1 * 2 * 3 * 1 = 6 fragments
    assert len(mzs) == 6


def test_theoretical_mzs_single_residue_empty():
    """A single residue has no internal cleavage sites -> empty."""
    parsed = parse_proforma("A")
    n_term, masses = parsed
    mzs = theoretical_mzs(n_term, masses, precursor_charge=2)
    assert len(mzs) == 0


def test_theoretical_mzs_charge_increases_fragments():
    """Higher precursor charge produces more fragment m/z values."""
    parsed = parse_proforma("AGK")
    n_term, masses = parsed
    mzs_z2 = theoretical_mzs(n_term, masses, precursor_charge=2)
    mzs_z3 = theoretical_mzs(n_term, masses, precursor_charge=3)
    # z=3 has charge states 1 and 2 vs z=2 has only charge state 1
    assert len(mzs_z3) > len(mzs_z2)


# ---------------------------------------------------------------------------
# matched_proportion tests
# ---------------------------------------------------------------------------


def test_matched_proportion_exact_match():
    """When all observed peaks match theoretical, proportion is 1.0."""
    theo = np.array([100.0, 200.0, 300.0])
    obs_mz = np.array([100.0, 200.0, 300.0])
    obs_int = np.array([10.0, 20.0, 30.0])
    n_matched, prop = matched_proportion(obs_mz, obs_int, theo, 10.0, "ppm")
    assert n_matched == 3
    assert np.isclose(prop, 1.0)


def test_matched_proportion_no_overlap():
    """When no observed peaks match, proportion is 0.0."""
    theo = np.array([100.0, 200.0])
    obs_mz = np.array([500.0, 600.0])
    obs_int = np.array([10.0, 20.0])
    n_matched, prop = matched_proportion(obs_mz, obs_int, theo, 10.0, "ppm")
    assert n_matched == 0
    assert prop == 0.0


def test_matched_proportion_da_tolerance():
    """Da tolerance mode uses absolute tolerance."""
    theo = np.array([100.0])
    obs_mz = np.array([100.05, 200.0])
    obs_int = np.array([10.0, 20.0])
    # 0.1 Da tolerance should match 100.05
    n_matched, prop = matched_proportion(obs_mz, obs_int, theo, 0.1, "Da")
    assert n_matched == 1
    assert np.isclose(prop, 10.0 / 30.0)

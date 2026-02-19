import csv
import pathlib

import pytest

from casanovoutils.charge_distribution import count_charge_states, charge_distribution


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


# ---------------------------------------------------------------------------
# Integration test
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

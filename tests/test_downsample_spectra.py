"""Tests for the downsample-spectra command."""

import numpy as np
import pytest
import pyteomics.mgf

from casanovoutils.downsample_spectra import downsample_spectra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_mgf(path, n_spectra):
    """Write *n_spectra* minimal spectra to *path* and return the path."""
    spectra = [
        {
            "params": {
                "title": f"spectrum_{i}",
                "pepmass": (500.0 + i,),
                "charge": [2],
            },
            "m/z array": np.array([100.0 + i, 200.0 + i]),
            "intensity array": np.array([1000.0, 2000.0]),
        }
        for i in range(n_spectra)
    ]
    pyteomics.mgf.write(spectra, output=str(path))
    return path


def _read_titles(path):
    """Return list of spectrum titles from an MGF file."""
    with pyteomics.mgf.read(str(path), use_index=False) as reader:
        return [s["params"]["title"] for s in reader]


def _count_spectra(path):
    """Return the number of spectra in an MGF file."""
    with pyteomics.mgf.read(str(path), use_index=False) as reader:
        return sum(1 for _ in reader)


# ---------------------------------------------------------------------------
# number mode
# ---------------------------------------------------------------------------


def test_number_exact_count(tmp_path):
    """Downsampling to k=5 from 20 yields exactly 5 spectra."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=5)
    assert _count_spectra(out) == 5


def test_number_larger_than_total_keeps_all(tmp_path):
    """Requesting more spectra than exist returns all of them."""
    inp = _write_mgf(tmp_path / "in.mgf", 10)
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=100)
    assert _count_spectra(out) == 10


def test_number_equal_to_total_keeps_all(tmp_path):
    """Requesting exactly as many as exist returns all of them."""
    inp = _write_mgf(tmp_path / "in.mgf", 15)
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=15)
    assert _count_spectra(out) == 15


def test_number_reproducibility(tmp_path):
    """Same seed produces identical output for number mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 50)
    out1 = tmp_path / "out1.mgf"
    out2 = tmp_path / "out2.mgf"
    downsample_spectra(
        inp, out1, downsample_type="number", downsample_rate=20, random_seed=7
    )
    downsample_spectra(
        inp, out2, downsample_type="number", downsample_rate=20, random_seed=7
    )
    assert _read_titles(out1) == _read_titles(out2)


def test_number_different_seeds_differ(tmp_path):
    """Different seeds typically produce different subsets."""
    inp = _write_mgf(tmp_path / "in.mgf", 100)
    out1 = tmp_path / "out1.mgf"
    out2 = tmp_path / "out2.mgf"
    downsample_spectra(
        inp, out1, downsample_type="number", downsample_rate=10, random_seed=1
    )
    downsample_spectra(
        inp, out2, downsample_type="number", downsample_rate=10, random_seed=99
    )
    # Probability of identical subsets by chance is negligible (10 from 100)
    assert _read_titles(out1) != _read_titles(out2)


def test_number_output_is_subset(tmp_path):
    """All output spectra originate from the input."""
    inp = _write_mgf(tmp_path / "in.mgf", 30)
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=15)
    input_titles = set(_read_titles(inp))
    output_titles = set(_read_titles(out))
    assert output_titles.issubset(input_titles)


def test_number_float_rate_accepted(tmp_path):
    """A float like 5.0 is accepted as equivalent to integer 5."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=5.0)
    assert _count_spectra(out) == 5


def test_number_output_has_no_duplicate_spectra(tmp_path):
    """Sampled spectra are unique (no spectrum selected twice)."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=10)
    titles = _read_titles(out)
    assert len(titles) == len(set(titles))


# ---------------------------------------------------------------------------
# proportion mode
# ---------------------------------------------------------------------------


def test_proportion_exact_count(tmp_path):
    """proportion=0.5 from 20 yields exactly 10 spectra."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="proportion", downsample_rate=0.5
    )
    assert _count_spectra(out) == 10


def test_proportion_full(tmp_path):
    """proportion=1.0 retains all spectra."""
    inp = _write_mgf(tmp_path / "in.mgf", 15)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="proportion", downsample_rate=1.0
    )
    assert _count_spectra(out) == 15


def test_proportion_rounding(tmp_path):
    """proportion=0.1 from 15 spectra yields round(1.5)=2 spectra."""
    inp = _write_mgf(tmp_path / "in.mgf", 15)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="proportion", downsample_rate=0.1
    )
    assert _count_spectra(out) == round(15 * 0.1)


def test_proportion_reproducibility(tmp_path):
    """Same seed produces identical output for proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 40)
    out1 = tmp_path / "out1.mgf"
    out2 = tmp_path / "out2.mgf"
    downsample_spectra(
        inp, out1, downsample_type="proportion", downsample_rate=0.25,
        random_seed=3,
    )
    downsample_spectra(
        inp, out2, downsample_type="proportion", downsample_rate=0.25,
        random_seed=3,
    )
    assert _read_titles(out1) == _read_titles(out2)


def test_proportion_output_is_subset(tmp_path):
    """All output spectra originate from the input for proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 30)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="proportion", downsample_rate=0.4
    )
    input_titles = set(_read_titles(inp))
    output_titles = set(_read_titles(out))
    assert output_titles.issubset(input_titles)


def test_proportion_output_has_no_duplicate_spectra(tmp_path):
    """Sampled spectra are unique (no spectrum selected twice)."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="proportion", downsample_rate=0.5
    )
    titles = _read_titles(out)
    assert len(titles) == len(set(titles))


# ---------------------------------------------------------------------------
# approx-proportion mode
# ---------------------------------------------------------------------------


def test_approx_proportion_ballpark(tmp_path):
    """approx-proportion=0.5 on 1000 spectra gives roughly 400–600."""
    inp = _write_mgf(tmp_path / "in.mgf", 1000)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="approx-proportion",
        downsample_rate=0.5, random_seed=42,
    )
    n = _count_spectra(out)
    assert 400 <= n <= 600


def test_approx_proportion_rate_1_keeps_all(tmp_path):
    """approx-proportion=1.0 retains all spectra."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="approx-proportion", downsample_rate=1.0
    )
    assert _count_spectra(out) == 20


def test_approx_proportion_reproducibility(tmp_path):
    """Same seed yields identical output for approx-proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 200)
    out1 = tmp_path / "out1.mgf"
    out2 = tmp_path / "out2.mgf"
    downsample_spectra(
        inp, out1, downsample_type="approx-proportion",
        downsample_rate=0.3, random_seed=11,
    )
    downsample_spectra(
        inp, out2, downsample_type="approx-proportion",
        downsample_rate=0.3, random_seed=11,
    )
    assert _read_titles(out1) == _read_titles(out2)


def test_approx_proportion_output_is_subset(tmp_path):
    """All output spectra originate from the input for approx-proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 100)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="approx-proportion", downsample_rate=0.5
    )
    input_titles = set(_read_titles(inp))
    output_titles = set(_read_titles(out))
    assert output_titles.issubset(input_titles)


def test_approx_proportion_different_seeds_differ(tmp_path):
    """Different seeds typically produce different subsets."""
    inp = _write_mgf(tmp_path / "in.mgf", 200)
    out1 = tmp_path / "out1.mgf"
    out2 = tmp_path / "out2.mgf"
    downsample_spectra(
        inp, out1, downsample_type="approx-proportion",
        downsample_rate=0.5, random_seed=1,
    )
    downsample_spectra(
        inp, out2, downsample_type="approx-proportion",
        downsample_rate=0.5, random_seed=99,
    )
    # With 200 spectra at 50%, probability of identical output is negligible
    assert _read_titles(out1) != _read_titles(out2)


# ---------------------------------------------------------------------------
# Input/output path safety
# ---------------------------------------------------------------------------


def test_same_input_output_raises(tmp_path):
    """Passing the same path for input and output raises ValueError."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    with pytest.raises(ValueError, match="different paths"):
        downsample_spectra(inp, inp, downsample_type="number", downsample_rate=3)


def test_same_input_output_raises_approx(tmp_path):
    """Same-path check fires before any I/O for approx-proportion mode too."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    with pytest.raises(ValueError, match="different paths"):
        downsample_spectra(
            inp, inp, downsample_type="approx-proportion", downsample_rate=0.5
        )


# ---------------------------------------------------------------------------
# Output order preservation
# ---------------------------------------------------------------------------


def test_number_output_preserves_input_order(tmp_path):
    """Sampled spectra appear in the same relative order as in the input."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=10)
    input_titles = _read_titles(inp)
    output_titles = _read_titles(out)
    # Positions in the input must be strictly increasing
    positions = [input_titles.index(t) for t in output_titles]
    assert positions == sorted(positions)


def test_proportion_output_preserves_input_order(tmp_path):
    """Sampled spectra appear in the same relative order as in the input."""
    inp = _write_mgf(tmp_path / "in.mgf", 20)
    out = tmp_path / "out.mgf"
    downsample_spectra(
        inp, out, downsample_type="proportion", downsample_rate=0.5
    )
    input_titles = _read_titles(inp)
    output_titles = _read_titles(out)
    positions = [input_titles.index(t) for t in output_titles]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_type_raises(tmp_path):
    """An unrecognised downsample_type raises ValueError."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match="downsample-type"):
        downsample_spectra(inp, out, downsample_type="bad-type", downsample_rate=5)


def test_number_rate_zero_raises(tmp_path):
    """downsample_rate=0 raises ValueError for number mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match="positive integer"):
        downsample_spectra(inp, out, downsample_type="number", downsample_rate=0)


def test_number_rate_negative_raises(tmp_path):
    """Negative downsample_rate raises ValueError for number mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match="positive integer"):
        downsample_spectra(inp, out, downsample_type="number", downsample_rate=-3)


def test_number_rate_non_integer_raises(tmp_path):
    """A non-integer float raises ValueError for number mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match="positive integer"):
        downsample_spectra(inp, out, downsample_type="number", downsample_rate=2.5)


def test_number_rate_nan_raises(tmp_path):
    """NaN raises ValueError for number mode rather than a bare conversion error."""
    import math

    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match="positive integer"):
        downsample_spectra(inp, out, downsample_type="number", downsample_rate=math.nan)


def test_number_rate_inf_raises(tmp_path):
    """Infinity raises ValueError for number mode rather than OverflowError."""
    import math

    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match="positive integer"):
        downsample_spectra(inp, out, downsample_type="number", downsample_rate=math.inf)


def test_proportion_rate_zero_raises(tmp_path):
    """downsample_rate=0 raises ValueError for proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        downsample_spectra(
            inp, out, downsample_type="proportion", downsample_rate=0
        )


def test_proportion_rate_gt1_raises(tmp_path):
    """downsample_rate > 1 raises ValueError for proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        downsample_spectra(
            inp, out, downsample_type="proportion", downsample_rate=1.5
        )


def test_approx_proportion_rate_zero_raises(tmp_path):
    """downsample_rate=0 raises ValueError for approx-proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        downsample_spectra(
            inp, out, downsample_type="approx-proportion", downsample_rate=0
        )


def test_approx_proportion_rate_gt1_raises(tmp_path):
    """downsample_rate > 1 raises ValueError for approx-proportion mode."""
    inp = _write_mgf(tmp_path / "in.mgf", 5)
    out = tmp_path / "out.mgf"
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        downsample_spectra(
            inp, out, downsample_type="approx-proportion", downsample_rate=2.0
        )

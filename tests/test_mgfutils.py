import numpy as np
import pytest

import pyteomics.mgf

from casanovoutils.mgfutils import (
    downsample,
    downsample_spectra,
    get_pep_dict_mgf,
    iter_spectra,
    pipeline,
    purge_redundant,
    shuffle,
    spectra_per_peptide,
    write_spectra,
)


def make_spectrum(seq, mz, intensity):
    return {
        "params": {"seq": seq},
        "m/z array": np.array(mz, dtype=float),
        "intensity array": np.array(intensity, dtype=float),
    }


# ---------------------------------------------------------------------------
# iter_spectra
# ---------------------------------------------------------------------------


def test_iter_spectra_yields_all():
    spectra = [make_spectrum("PEP", [1.0, 2.0], [10.0, 20.0])] * 3
    assert len(list(iter_spectra(spectra))) == 3


def test_iter_spectra_empty():
    assert list(iter_spectra([])) == []


def test_iter_spectra_preserves_content():
    spec = make_spectrum("ABC", [1.0], [5.0])
    (result,) = iter_spectra([spec])
    assert result["params"]["seq"] == "ABC"


# ---------------------------------------------------------------------------
# get_pep_dict_mgf
# ---------------------------------------------------------------------------


def test_get_pep_dict_groups_by_seq():
    spectra = [
        make_spectrum("PEP", [1.0], [1.0]),
        make_spectrum("PEP", [2.0], [2.0]),
        make_spectrum("TID", [3.0], [3.0]),
    ]
    result = get_pep_dict_mgf(spectra)
    assert set(result.keys()) == {"PEP", "TID"}
    assert len(result["PEP"]) == 2
    assert len(result["TID"]) == 1


def test_get_pep_dict_single_peptide():
    spectra = [make_spectrum("AAA", [1.0], [1.0])] * 5
    result = get_pep_dict_mgf(spectra)
    assert list(result.keys()) == ["AAA"]
    assert len(result["AAA"]) == 5


# ---------------------------------------------------------------------------
# write_spectra
# ---------------------------------------------------------------------------


def test_write_spectra_noop_when_outfile_none():
    # should not raise
    write_spectra([make_spectrum("P", [1.0], [1.0])], None)


def test_write_spectra_creates_file(tmp_path):
    spectra = [make_spectrum("PEP", [100.0, 200.0], [500.0, 300.0])]
    outfile = tmp_path / "out.mgf"
    write_spectra(spectra, outfile)
    assert outfile.exists()


# ---------------------------------------------------------------------------
# purge_redundant
# ---------------------------------------------------------------------------


def test_purge_redundant_removes_close_peaks():
    spec = make_spectrum("P", [100.0, 100.0005, 200.0], [1.0, 2.0, 3.0])
    (result,) = purge_redundant([spec], epsilon=0.001)
    assert len(result["m/z array"]) == 2


def test_purge_redundant_keeps_distant_peaks():
    spec = make_spectrum("P", [100.0, 101.0, 102.0], [1.0, 2.0, 3.0])
    (result,) = purge_redundant([spec], epsilon=0.001)
    assert len(result["m/z array"]) == 3


def test_purge_redundant_sorts_by_mz():
    spec = make_spectrum("P", [300.0, 100.0, 200.0], [3.0, 1.0, 2.0])
    (result,) = purge_redundant([spec], epsilon=0.001)
    mz = result["m/z array"]
    assert list(mz) == sorted(mz)


def test_purge_redundant_intensity_follows_mz():
    # after sorting [300, 100, 200] -> [100, 200, 300], intensities should follow
    spec = make_spectrum("P", [300.0, 100.0, 200.0], [30.0, 10.0, 20.0])
    (result,) = purge_redundant([spec], epsilon=0.001)
    assert list(result["intensity array"]) == [10.0, 20.0, 30.0]


def test_purge_redundant_exact_epsilon_boundary():
    # diff == epsilon should be kept
    spec = make_spectrum("P", [100.0, 100.001], [1.0, 2.0])
    (result,) = purge_redundant([spec], epsilon=0.001)
    assert len(result["m/z array"]) == 2


def test_purge_redundant_preserves_other_keys():
    spec = make_spectrum("P", [100.0, 200.0], [1.0, 2.0])
    spec["params"]["charge"] = 2
    (result,) = purge_redundant([spec], epsilon=0.001)
    assert result["params"]["charge"] == 2


def test_purge_redundant_multiple_spectra():
    spectra = [
        make_spectrum("P", [100.0, 100.0005], [1.0, 2.0]),
        make_spectrum("Q", [200.0, 201.0], [1.0, 2.0]),
    ]
    results = purge_redundant(spectra, epsilon=0.001)
    assert len(results[0]["m/z array"]) == 1
    assert len(results[1]["m/z array"]) == 2


# ---------------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------------


def test_downsample_limits_to_k():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(5)]
    result = downsample(spectra, k=2)
    assert len(result) == 2


def test_downsample_keeps_all_when_k_exceeds_count():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(3)]
    result = downsample(spectra, k=10)
    assert len(result) == 3


def test_downsample_per_peptide():
    spectra = [make_spectrum("AAA", [float(i)], [1.0]) for i in range(4)] + [
        make_spectrum("BBB", [float(i)], [1.0]) for i in range(4)
    ]
    result = downsample(spectra, k=2)
    assert len(result) == 4


def test_downsample_reproducible():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(10)]
    r1 = downsample(spectra, k=3, random_seed=0)
    r2 = downsample(spectra, k=3, random_seed=0)
    assert [s["m/z array"][0] for s in r1] == [s["m/z array"][0] for s in r2]


def test_downsample_different_seeds_differ():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(10)]
    r1 = downsample(spectra, k=3, random_seed=0)
    r2 = downsample(spectra, k=3, random_seed=99)
    # Very unlikely to be identical across seeds with 10 items choosing 3
    assert [s["m/z array"][0] for s in r1] != [s["m/z array"][0] for s in r2]


# ---------------------------------------------------------------------------
# shuffle
# ---------------------------------------------------------------------------


def test_shuffle_preserves_all_spectra():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(5)]
    result = shuffle(spectra)
    assert len(result) == 5


def test_shuffle_reproducible():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(10)]
    r1 = shuffle(spectra, random_seed=7)
    r2 = shuffle(spectra, random_seed=7)
    assert [s["m/z array"][0] for s in r1] == [s["m/z array"][0] for s in r2]


def test_shuffle_different_seeds_differ():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(10)]
    r1 = shuffle(spectra, random_seed=1)
    r2 = shuffle(spectra, random_seed=2)
    assert [s["m/z array"][0] for s in r1] != [s["m/z array"][0] for s in r2]


def test_shuffle_writes_file(tmp_path):
    spectra = [make_spectrum("PEP", [100.0, 200.0], [1.0, 2.0])]
    outfile = tmp_path / "shuffled.mgf"
    shuffle(spectra, outfile=outfile)
    assert outfile.exists()


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def test_pipeline_noop_returns_all():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(5)]
    result = pipeline(spectra, do_shuffle=False)
    assert len(result) == 5


def test_pipeline_shuffle_stage():
    spectra = [make_spectrum("PEP", [float(i)], [1.0]) for i in range(10)]
    result = pipeline(spectra, do_shuffle=True, random_seed=42)
    assert len(result) == 10


def test_pipeline_downsample_stage():
    spectra = [make_spectrum("AAA", [float(i)], [1.0]) for i in range(5)] + [
        make_spectrum("BBB", [float(i)], [1.0]) for i in range(5)
    ]
    result = pipeline(spectra, do_shuffle=False, downsample_k=2)
    assert len(result) == 4


def test_pipeline_purge_stage():
    spec = make_spectrum("P", [100.0, 100.0005, 200.0], [1.0, 2.0, 3.0])
    result = pipeline([spec], do_shuffle=False, purge_epsilon=0.001)
    assert len(result[0]["m/z array"]) == 2


def test_pipeline_all_stages():
    spectra = [
        make_spectrum("AAA", [float(i), float(i) + 0.0001], [1.0, 2.0])
        for i in range(5)
    ] + [
        make_spectrum("BBB", [float(i), float(i) + 0.0001], [1.0, 2.0])
        for i in range(5)
    ]
    result = pipeline(spectra, do_shuffle=True, downsample_k=2, purge_epsilon=0.001)
    assert len(result) == 4
    assert all(len(s["m/z array"]) == 1 for s in result)


def test_pipeline_writes_file(tmp_path):
    spectra = [make_spectrum("PEP", [100.0, 200.0], [1.0, 2.0])]
    outfile = tmp_path / "out.mgf"
    pipeline(spectra, outfile=outfile, do_shuffle=False)
    assert outfile.exists()


# ---------------------------------------------------------------------------
# spectra_per_peptide
# ---------------------------------------------------------------------------


def test_spp_k1_limits_to_one_per_peptide():
    spectra = [make_spectrum("AAA", [float(i)], [1.0]) for i in range(5)] + [
        make_spectrum("BBB", [float(i)], [1.0]) for i in range(3)
    ]
    result = spectra_per_peptide(spectra, k=1)
    seqs = [s["params"]["seq"] for s in result]
    assert seqs.count("AAA") == 1
    assert seqs.count("BBB") == 1


def test_spp_k_greater_than_max_keeps_all():
    spectra = [make_spectrum("AAA", [float(i)], [1.0]) for i in range(3)] + [
        make_spectrum("BBB", [float(i)], [1.0]) for i in range(2)
    ]
    result = spectra_per_peptide(spectra, k=100)
    assert len(result) == 5


def test_spp_k_limits_per_peptide():
    spectra = [make_spectrum("AAA", [float(i)], [1.0]) for i in range(5)] + [
        make_spectrum("BBB", [float(i)], [1.0]) for i in range(4)
    ]
    result = spectra_per_peptide(spectra, k=2)
    seqs = [s["params"]["seq"] for s in result]
    assert seqs.count("AAA") == 2
    assert seqs.count("BBB") == 2


def test_spp_reproducible():
    spectra = [make_spectrum("AAA", [float(i)], [1.0]) for i in range(10)] + [
        make_spectrum("BBB", [float(i)], [1.0]) for i in range(10)
    ]
    r1 = spectra_per_peptide(list(spectra), k=3, random_seed=123)
    r2 = spectra_per_peptide(list(spectra), k=3, random_seed=123)
    assert [s["m/z array"][0] for s in r1] == [s["m/z array"][0] for s in r2]


def test_spp_different_seeds_differ():
    spectra = [make_spectrum("AAA", [float(i)], [1.0]) for i in range(20)]
    r1 = spectra_per_peptide(list(spectra), k=5, random_seed=1)
    r2 = spectra_per_peptide(list(spectra), k=5, random_seed=99)
    assert [s["m/z array"][0] for s in r1] != [s["m/z array"][0] for s in r2]


def test_spp_accepts_generator():
    def _gen():
        for i in range(3):
            yield make_spectrum("AAA", [float(i)], [1.0])

    result = spectra_per_peptide(_gen(), k=1)
    assert len(result) == 1
    assert result[0]["params"]["seq"] == "AAA"


# ---------------------------------------------------------------------------
# downsample_spectra
# ---------------------------------------------------------------------------


def _write_mgf_file(path, spectra):
    pyteomics.mgf.write(spectra, output=str(path))
    return path


def _count_mgf(path):
    with pyteomics.mgf.read(str(path), use_index=False) as r:
        return sum(1 for _ in r)


def test_ds_number_exact_count(tmp_path):
    inp = _write_mgf_file(
        tmp_path / "in.mgf",
        [make_spectrum("P", [float(i)], [1.0]) for i in range(20)],
    )
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=5)
    assert _count_mgf(out) == 5


def test_ds_number_larger_than_total_keeps_all(tmp_path):
    inp = _write_mgf_file(
        tmp_path / "in.mgf",
        [make_spectrum("P", [float(i)], [1.0]) for i in range(10)],
    )
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="number", downsample_rate=100)
    assert _count_mgf(out) == 10


def test_ds_proportion_exact_count(tmp_path):
    inp = _write_mgf_file(
        tmp_path / "in.mgf",
        [make_spectrum("P", [float(i)], [1.0]) for i in range(20)],
    )
    out = tmp_path / "out.mgf"
    downsample_spectra(inp, out, downsample_type="proportion", downsample_rate=0.5)
    assert _count_mgf(out) == 10


def test_ds_reproducible(tmp_path):
    inp = _write_mgf_file(
        tmp_path / "in.mgf",
        [make_spectrum("P", [float(i)], [1.0]) for i in range(50)],
    )
    out1 = tmp_path / "out1.mgf"
    out2 = tmp_path / "out2.mgf"
    downsample_spectra(
        inp, out1, downsample_type="number", downsample_rate=20, random_seed=7
    )
    downsample_spectra(
        inp, out2, downsample_type="number", downsample_rate=20, random_seed=7
    )
    with pyteomics.mgf.read(str(out1), use_index=False) as r1:
        titles1 = [s["params"].get("title") for s in r1]
    with pyteomics.mgf.read(str(out2), use_index=False) as r2:
        titles2 = [s["params"].get("title") for s in r2]
    assert titles1 == titles2


def test_ds_same_path_raises(tmp_path):
    inp = _write_mgf_file(
        tmp_path / "in.mgf",
        [make_spectrum("P", [1.0], [1.0])],
    )
    with pytest.raises(ValueError, match="different paths"):
        downsample_spectra(inp, inp)


def test_ds_invalid_type_raises(tmp_path):
    inp = _write_mgf_file(tmp_path / "in.mgf", [make_spectrum("P", [1.0], [1.0])])
    with pytest.raises(ValueError, match="downsample-type"):
        downsample_spectra(inp, tmp_path / "out.mgf", downsample_type="bad")


def test_ds_proportion_out_of_range_raises(tmp_path):
    inp = _write_mgf_file(tmp_path / "in.mgf", [make_spectrum("P", [1.0], [1.0])])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        downsample_spectra(
            inp, tmp_path / "out.mgf", downsample_type="proportion", downsample_rate=1.5
        )

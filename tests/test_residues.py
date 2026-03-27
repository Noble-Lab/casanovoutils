import pytest

from casanovoutils.residues import dump_residues, get_residues

# ---------------------------------------------------------------------------
# get_residues
# ---------------------------------------------------------------------------


def test_get_residues_returns_dict():
    assert isinstance(get_residues(), dict)


def test_get_residues_nonempty():
    assert len(get_residues()) > 0


def test_get_residues_values_are_floats():
    residues = get_residues()
    assert all(isinstance(v, float) for v in residues.values())


def test_get_residues_contains_standard_amino_acids():
    # C is stored as C[Carbamidomethyl] in the bundled file; check the others.
    residues = get_residues()
    for aa in "ADEFGHIKLMNPQRSTVWY":
        assert aa in residues, f"Missing standard amino acid: {aa}"


def test_get_residues_custom_path(tmp_path):
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text('"X": 123.456\n"Y": 789.012\n')
    result = get_residues(yaml_file)
    assert result == {"X": 123.456, "Y": 789.012}


def test_get_residues_custom_path_returns_correct_values(tmp_path):
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text('"A": 71.037114\n')
    result = get_residues(yaml_file)
    assert result["A"] == pytest.approx(71.037114)


# ---------------------------------------------------------------------------
# dump_residues
# ---------------------------------------------------------------------------


def test_dump_residues_creates_file(tmp_path):
    dest = tmp_path / "residues.yaml"
    dump_residues(dest)
    assert dest.exists()


def test_dump_residues_to_directory(tmp_path):
    dump_residues(tmp_path)
    assert (tmp_path / "residues.yaml").exists()


def test_dump_residues_content_is_valid_yaml(tmp_path):
    dest = tmp_path / "out.yaml"
    dump_residues(dest)
    result = get_residues(dest)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_dump_residues_matches_bundled(tmp_path):
    dest = tmp_path / "copy.yaml"
    dump_residues(dest)
    assert get_residues(dest) == get_residues()

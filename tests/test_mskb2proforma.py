"""Tests for mskb2proforma — MassIVE-KB to ProForma sequence converter."""

import pyteomics.mgf
import pytest

from casanovoutils.mskb2proforma import _convert_seq, convert

# ---------------------------------------------------------------------------
# _convert_seq — unit tests for the pure conversion function
# ---------------------------------------------------------------------------


class TestConvertSeqPlain:
    def test_no_modifications(self):
        """A bare amino-acid sequence is returned unchanged."""
        assert _convert_seq("PEPTIDE") == "PEPTIDE"

    def test_single_residue(self):
        """A single residue with no modifications is returned unchanged."""
        assert _convert_seq("A") == "A"


class TestConvertSeqNterminal:
    def test_acetyl(self):
        """+42.011 N-terminal shift maps to named [Acetyl]- token."""
        assert _convert_seq("+42.011PEPTIDE") == "[Acetyl]-PEPTIDE"

    def test_carbamyl(self):
        """+43.006 N-terminal shift maps to named [Carbamyl]- token."""
        assert _convert_seq("+43.006PEPTIDE") == "[Carbamyl]-PEPTIDE"

    def test_ammonia_loss(self):
        """-17.027 N-terminal shift maps to named [Ammonia-loss]- token."""
        assert _convert_seq("-17.027PEPTIDE") == "[Ammonia-loss]-PEPTIDE"

    def test_tmt_nterm(self):
        """+229.163 N-terminal shift (TMT label) maps to a numeric token."""
        result = _convert_seq("+229.163PEPTIDE")
        assert result == "[+229.163]-PEPTIDE"

    def test_unknown_nterm_positive(self):
        """An unknown positive N-terminal shift is formatted as [+mass]-."""
        result = _convert_seq("+28.031PEPTIDE")
        assert result == "[+28.031]-PEPTIDE"

    def test_unknown_nterm_negative(self):
        """An unknown negative N-terminal shift is formatted as [-mass]-."""
        result = _convert_seq("-18.011PEPTIDE")
        assert result == "[-18.011]-PEPTIDE"

    def test_zero_nterm_shift_omitted(self):
        """A net N-terminal shift of zero produces no N-terminal token."""
        assert _convert_seq("+17.027-17.027PEPTIDE") == "PEPTIDE"

    def test_combined_tmt_acetyl(self):
        """+229.163+42.011 sums to +271.174, producing one numeric token."""
        result = _convert_seq("+229.163+42.011PEPTIDE")
        assert result == "[+271.174]-PEPTIDE"

    def test_combined_tmt_carbamyl(self):
        """+229.163+43.006 sums to +272.169, producing one numeric token."""
        result = _convert_seq("+229.163+43.006PEPTIDE")
        assert result == "[+272.169]-PEPTIDE"

    def test_combined_tmt_ammonia_loss(self):
        """+229.163-17.027 sums to +212.136 — maps to named token if present,
        otherwise numeric."""
        result = _convert_seq("+229.163-17.027PEPTIDE")
        assert result == "[+212.136]-PEPTIDE"

    def test_combined_carbamyl_ammonia_loss_named(self):
        """+43.006-17.027 == +25.979 maps to the named token +25.979265."""
        result = _convert_seq("+43.006-17.027PEPTIDE")
        assert result == "[+25.979265]-PEPTIDE"

    def test_combined_labeling_with_residues(self):
        """Combined N-term mod summed correctly alongside per-residue mods."""
        result = _convert_seq("+229.163+42.011AC+57.021K")
        assert result == "[+271.174]-AC[Carbamidomethyl]K"


class TestConvertSeqResidues:
    def test_carbamidomethyl_cys(self):
        """C+57.021 maps to C[Carbamidomethyl]."""
        assert _convert_seq("AC+57.021G") == "AC[Carbamidomethyl]G"

    def test_oxidized_met(self):
        """M+15.995 maps to M[Oxidation]."""
        assert _convert_seq("AM+15.995G") == "AM[Oxidation]G"

    def test_deamidated_asn(self):
        """N+0.984 maps to N[Deamidated]."""
        assert _convert_seq("AN+0.984G") == "AN[Deamidated]G"

    def test_deamidated_gln(self):
        """Q+0.984 maps to Q[Deamidated]."""
        assert _convert_seq("AQ+0.984G") == "AQ[Deamidated]G"

    def test_phospho_ser(self):
        """S+79.966 maps to S[Phospho]."""
        assert _convert_seq("AS+79.966G") == "AS[Phospho]G"

    def test_phospho_thr(self):
        """T+79.966 maps to T[Phospho]."""
        assert _convert_seq("AT+79.966G") == "AT[Phospho]G"

    def test_phospho_tyr(self):
        """Y+79.966 maps to Y[Phospho]."""
        assert _convert_seq("AY+79.966G") == "AY[Phospho]G"

    def test_tmt_lys(self):
        """K+229.163 (TMT label on lysine) maps to K[+229.163]."""
        assert _convert_seq("AK+229.163G") == "AK[+229.163]G"

    def test_dimethyl_lys(self):
        """K+28.031 maps to K[+28.031]."""
        assert _convert_seq("AK+28.031G") == "AK[+28.031]G"

    def test_heavy_lys(self):
        """K+8.014 (heavy isotope label) maps to K[+8.014]."""
        assert _convert_seq("AK+8.014G") == "AK[+8.014]G"

    def test_heavy_arg(self):
        """R+10.008 (heavy isotope label) maps to R[+10.008]."""
        assert _convert_seq("AR+10.008G") == "AR[+10.008]G"

    def test_unknown_residue_mod_negative(self):
        """Unknown negative residue shift is formatted as AA[-mass]."""
        assert _convert_seq("AW-1.000G") == "AW[-1.000]G"

    def test_multiple_residue_mods(self):
        """Multiple modified residues in a sequence are all converted."""
        result = _convert_seq("C+57.021M+15.995PEPTIDE")
        assert result == "C[Carbamidomethyl]M[Oxidation]PEPTIDE"


class TestConvertSeqRealWorldExamples:
    """Representative sequences taken from the actual MassIVE-KB v2 dataset."""

    def test_tmt_cterm_lys(self):
        """+229.163 N-term and K+229.163 C-term, both converted."""
        result = _convert_seq("+229.163GANHTLVLDSQK+229.163")
        assert result == "[+229.163]-GANHTLVLDSQK[+229.163]"

    def test_tmt_acetyl_and_cam(self):
        """+229.163+42.011 N-term summed, C+57.021 named, K+229.163 numeric."""
        result = _convert_seq("+229.163+42.011AC+57.021GANHTLVLDSQK+229.163")
        assert result == "[+271.174]-AC[Carbamidomethyl]GANHTLVLDSQK[+229.163]"

    def test_no_nterm_with_phospho(self):
        """Phosphorylated sequence with no N-terminal modification."""
        result = _convert_seq("AAGS+79.966PEPTIDE")
        assert result == "AAGS[Phospho]PEPTIDE"


# ---------------------------------------------------------------------------
# convert — integration tests for the file-to-file converter
# ---------------------------------------------------------------------------

_SMALL_MGF = """\
BEGIN IONS
TITLE=spec1
PEPMASS=500.0
CHARGE=2+
SEQ=C+57.021PEPTIDE
100.0 10
200.0 20
END IONS

BEGIN IONS
TITLE=spec2
PEPMASS=600.0
CHARGE=3+
SEQ=+42.011AGK
150.0 15
250.0 25
END IONS

BEGIN IONS
TITLE=spec3
PEPMASS=700.0
CHARGE=2+
100.0 10
200.0 20
END IONS
"""


class TestConvert:
    def test_basic_conversion(self, tmp_path):
        """SEQ fields are converted to ProForma; non-SEQ fields pass through."""
        inp = tmp_path / "in.mgf"
        out = tmp_path / "out.mgf"
        inp.write_text(_SMALL_MGF)

        convert(inp, out)

        spectra = list(pyteomics.mgf.read(str(out), use_index=False))
        assert len(spectra) == 3
        assert spectra[0]["params"]["seq"] == "C[Carbamidomethyl]PEPTIDE"
        assert spectra[1]["params"]["seq"] == "[Acetyl]-AGK"

    def test_no_seq_passes_through(self, tmp_path):
        """Spectra without a SEQ field are written unchanged."""
        inp = tmp_path / "in.mgf"
        out = tmp_path / "out.mgf"
        inp.write_text(_SMALL_MGF)

        convert(inp, out)

        spectra = list(pyteomics.mgf.read(str(out), use_index=False))
        assert "seq" not in spectra[2]["params"]

    def test_output_file_exists_raises(self, tmp_path):
        """FileExistsError when output exists and overwrite=False."""
        inp = tmp_path / "in.mgf"
        out = tmp_path / "out.mgf"
        inp.write_text(_SMALL_MGF)
        out.write_text("existing")

        with pytest.raises(FileExistsError):
            convert(inp, out, overwrite=False)

    def test_overwrite_flag(self, tmp_path):
        """overwrite=True allows clobbering an existing output file."""
        inp = tmp_path / "in.mgf"
        out = tmp_path / "out.mgf"
        inp.write_text(_SMALL_MGF)
        out.write_text("existing")

        convert(inp, out, overwrite=True)

        spectra = list(pyteomics.mgf.read(str(out), use_index=False))
        assert len(spectra) == 3

    def test_same_path_raises(self, tmp_path):
        """ValueError when input and output resolve to the same path."""
        inp = tmp_path / "in.mgf"
        inp.write_text(_SMALL_MGF)

        with pytest.raises(ValueError):
            convert(inp, inp)

    def test_combined_nterm_summed(self, tmp_path):
        """Combined N-terminal shifts are summed into one ProForma token."""
        mgf_text = """\
BEGIN IONS
TITLE=s1
PEPMASS=900.0
CHARGE=2+
SEQ=+229.163+42.011PEPTIDEK+229.163
100.0 10
END IONS
"""
        inp = tmp_path / "in.mgf"
        out = tmp_path / "out.mgf"
        inp.write_text(mgf_text)

        convert(inp, out)

        spectra = list(pyteomics.mgf.read(str(out), use_index=False))
        assert spectra[0]["params"]["seq"] == "[+271.174]-PEPTIDEK[+229.163]"

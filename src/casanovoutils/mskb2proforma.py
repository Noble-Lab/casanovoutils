"""Convert MassIVE-KB annotated MGF files to ProForma sequence notation."""

import logging
import pathlib
import re
import tempfile
from os import PathLike

import fire
import pyteomics.mgf
import tqdm

from . import configure_logging
from .types import Commands

# Named N-terminal modifications keyed by their summed mass in daltons,
# rounded to 3 decimal places.  Entries with a string value starting with
# "+" or "-" are written as numeric ProForma tokens (e.g. "[+25.979265]-");
# all other entries are written as named tokens (e.g. "[Acetyl]-").
_NTERM_NAMES: dict[float, str] = {
    -17.027: "Ammonia-loss",
    25.979: "+25.979265",  # Carbamyl (+43.006) + Ammonia-loss (-17.027)
    42.011: "Acetyl",
    43.006: "Carbamyl",
}

# Named per-residue modifications.  Key is the MassIVE-KB token
# "AA+mass"; value is the ProForma token.
_RESIDUE_NAMES: dict[str, str] = {
    "C+57.021": "C[Carbamidomethyl]",
    "M+15.995": "M[Oxidation]",
    "N+0.984": "N[Deamidated]",
    "Q+0.984": "Q[Deamidated]",
    "S+79.966": "S[Phospho]",
    "T+79.966": "T[Phospho]",
    "Y+79.966": "Y[Phospho]",
}

# Matches one or more leading signed mass shifts before the first residue.
_LEADING_SHIFTS_RE = re.compile(r"^((?:[+-]\d+(?:\.\d+)?)+)(?=[A-Z])")

# Extracts individual signed mass shifts from a string.
_SHIFT_RE = re.compile(r"[+-]\d+(?:\.\d+)?")

# Splits a sequence string before each uppercase letter (after position 0).
_RESIDUE_SPLIT_RE = re.compile(r"(?<=.)(?=[A-Z])")

# Matches a single-residue token with an attached signed mass shift.
_RESIDUE_MOD_RE = re.compile(r"^([A-Z])([+-]\d+(?:\.\d+)?)$")


def _convert_seq(seq: str) -> str:
    """Convert a single MassIVE-KB peptide sequence string to ProForma.

    Handles N-terminal modifications (including multiple stacked shifts,
    which are summed into a single token), per-residue named modifications,
    and per-residue unnamed modifications (formatted as ``AA[+mass]``).

    Parameters
    ----------
    seq : str
        Peptide sequence in MassIVE-KB format, e.g.
        ``"+229.163+42.011AC+57.021GANHTLVLDSQK+229.163"``.

    Returns
    -------
    str
        Peptide sequence in ProForma format, e.g.
        ``"[+271.174]-AC[Carbamidomethyl]GANHTLVLDSQK[+229.163]"``.
    """
    nterm = ""

    # Extract and convert leading N-terminal modification(s).
    m = _LEADING_SHIFTS_RE.match(seq)
    if m:
        nterm_str = m.group(1)
        seq = seq[len(nterm_str) :]
        total = round(sum(float(s) for s in _SHIFT_RE.findall(nterm_str)), 3)
        # Canonicalize -0.0 → 0.0; a zero net shift needs no N-terminal token.
        total = total + 0.0
        if total == 0.0:
            pass  # net shift is zero — no N-terminal modification token needed
        elif total in _NTERM_NAMES:
            nterm = f"[{_NTERM_NAMES[total]}]-"
        elif total > 0:
            nterm = f"[+{total:.3f}]-"
        else:
            nterm = f"[{total:.3f}]-"

    # Split into per-residue tokens and convert each one.
    out_tokens = []
    for token in _RESIDUE_SPLIT_RE.split(seq):
        if token in _RESIDUE_NAMES:
            out_tokens.append(_RESIDUE_NAMES[token])
        else:
            rm = _RESIDUE_MOD_RE.match(token)
            if rm:
                aa, mass = rm.group(1), rm.group(2)
                out_tokens.append(f"{aa}[{mass}]")
            else:
                out_tokens.append(token)

    return nterm + "".join(out_tokens)


def convert(
    input_file: PathLike,
    output_file: PathLike,
    overwrite: bool = False,
) -> None:
    """Convert a MassIVE-KB annotated MGF file to ProForma sequence notation.

    Streams the input MGF file, rewrites the ``SEQ`` field of every spectrum
    from the MassIVE-KB modification format to ProForma, and writes the result
    to *output_file*.  All other spectrum fields are passed through unchanged.

    The following conversions are applied:

    *N-terminal modifications* — One or more leading signed mass shifts (e.g.
    ``+229.163+42.011``) are summed into a single ProForma N-terminal token.
    Common named modifications (Acetyl, Carbamyl, Ammonia-loss) are written
    using their ProForma names; all others use the numeric form
    ``[+mass]-``.

    *Per-residue modifications* — Common named modifications are written using
    their ProForma names (e.g. ``C+57.021`` → ``C[Carbamidomethyl]``,
    ``S+79.966`` → ``S[Phospho]``); all others use the numeric form
    ``AA[+mass]``.

    Parameters
    ----------
    input_file : PathLike
        Path to the input MassIVE-KB MGF file.
    output_file : PathLike
        Path for the converted output MGF file.  Must differ from
        *input_file*.
    overwrite : bool, default False
        If False, raise an error when *output_file* already exists.
    """
    input_file = pathlib.Path(input_file)
    output_file = pathlib.Path(output_file)

    if input_file.resolve() == output_file.resolve():
        raise ValueError(
            "input_file and output_file must be different paths; "
            "overwriting the input in-place is not supported."
        )

    if not overwrite and output_file.exists():
        raise FileExistsError(
            f"Output file already exists: {output_file}. "
            "Use --overwrite to overwrite."
        )

    configure_logging(output_file.with_suffix(".log"))
    logging.info("Converting %s -> %s", input_file, output_file)

    n_converted = 0

    def _convert_spectrum(spectrum: dict) -> dict:
        nonlocal n_converted
        params = dict(spectrum["params"])
        if "seq" in params:
            original = params["seq"]
            try:
                params["seq"] = _convert_seq(original)
                n_converted += 1
            except Exception as exc:
                title = params.get("title", "<no title>")
                raise ValueError(
                    f"Could not convert sequence {original!r} "
                    f"(spectrum title: {title!r}) in {input_file} "
                    f"to ProForma: {exc}"
                ) from exc
        return {**spectrum, "params": params}

    output_file = pathlib.Path(output_file)
    tmp_dir = output_file.parent
    with (
        pyteomics.mgf.read(
            str(input_file), use_index=False, use_header=False
        ) as reader,
        tempfile.NamedTemporaryFile(
            mode="w", dir=tmp_dir, suffix=".mgf", delete=False
        ) as tmp_fh,
    ):
        tmp_path = pathlib.Path(tmp_fh.name)
        header = reader.header
        spectra = tqdm.tqdm(reader, desc="Converting spectra", unit="spectrum")
        try:
            pyteomics.mgf.write(
                (_convert_spectrum(s) for s in spectra),
                output=tmp_fh,
                header=header,
            )
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    try:
        tmp_path.replace(output_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logging.info("Converted %d spectra", n_converted)


COMMANDS: Commands = convert


def main() -> None:
    """CLI entry point for mskb2proforma."""
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()

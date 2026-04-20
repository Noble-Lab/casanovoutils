"""
Utilities for reading and sampling spectra from mzML files.

Provides a single ``sample_spectra`` command that makes one streaming pass
through an mzML file, sampling a proportion ``k`` of spectra from each
read buffer.  Output can be written to MGF or mzML format based on the
output file extension.  A ``main`` entry point exposes commands as CLI
subcommands via ``fire``.
"""

import base64
import logging
import pathlib
import random
import zlib
from os import PathLike
from typing import Optional

import fire
import numpy as np
import pyteomics.mgf
import pyteomics.mzml
import tqdm
from lxml import etree

from . import configure_logging
from .types import Commands, PyteomicsSpectrum

_MZML_NS = "http://psi.hupo.org/ms/mzml"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_NSMAP = {None: _MZML_NS, "xsi": _XSI_NS}


def _encode_array(arr: np.ndarray) -> str:
    """Return zlib-compressed, base64-encoded float64 binary data."""
    return base64.b64encode(zlib.compress(arr.astype(np.float64).tobytes())).decode()


def _sub(parent: etree._Element, tag: str, **attrs: str) -> etree._Element:
    return etree.SubElement(parent, f"{{{_MZML_NS}}}{tag}", attrib=attrs)


def _cv(parent: etree._Element, accession: str, name: str, value: str = "") -> None:
    etree.SubElement(
        parent,
        f"{{{_MZML_NS}}}cvParam",
        attrib={"cvRef": "MS", "accession": accession, "name": name, "value": value},
    )


def _write_mzml(spectra: list[PyteomicsSpectrum], path: PathLike) -> None:
    """Write spectra to a minimal valid mzML 1.1.0 file."""
    root = etree.Element(
        f"{{{_MZML_NS}}}mzML",
        attrib={
            f"{{{_XSI_NS}}}schemaLocation": (
                "http://psi.hupo.org/ms/mzml "
                "http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd"
            )
        },
        nsmap=_NSMAP,
    )

    cv_list = _sub(root, "cvList", count="2")
    etree.SubElement(
        cv_list,
        f"{{{_MZML_NS}}}cv",
        attrib={
            "id": "MS",
            "fullName": "Proteomics Standards Initiative Mass Spectrometry Ontology",
            "version": "4.1.30",
            "URI": "https://psidev.info/sites/default/files/2018-11/psi-ms.obo",
        },
    )
    etree.SubElement(
        cv_list,
        f"{{{_MZML_NS}}}cv",
        attrib={
            "id": "UO",
            "fullName": "Unit Ontology",
            "version": "09:04:2014",
            "URI": "https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo",
        },
    )

    file_desc = _sub(root, "fileDescription")
    _sub(file_desc, "fileContent")

    sw_list = _sub(root, "softwareList", count="1")
    sw = _sub(sw_list, "software", id="casanovoutils", version="0")
    _cv(sw, "MS:1000799", "custom unreleased software tool", "casanovoutils")

    ic_list = _sub(root, "instrumentConfigurationList", count="1")
    ic = _sub(ic_list, "instrumentConfiguration", id="ic")
    _cv(ic, "MS:1000031", "instrument model")

    dp_list = _sub(root, "dataProcessingList", count="1")
    dp = _sub(dp_list, "dataProcessing", id="dp")
    pm = _sub(dp, "processingMethod", order="0", softwareRef="casanovoutils")
    _cv(pm, "MS:1000544", "Conversion to mzML")

    run = _sub(root, "run")
    spec_list = _sub(
        run, "spectrumList", count=str(len(spectra)), defaultDataProcessingRef="dp"
    )

    for idx, s in enumerate(spectra):
        mz = np.asarray(s["m/z array"])
        intensity = np.asarray(s["intensity array"])
        ms_level = str(s.get("ms level", 2))
        spec_id = s.get("id", f"scan={idx + 1}")

        spec_el = _sub(
            spec_list,
            "spectrum",
            index=str(idx),
            id=str(spec_id),
            defaultArrayLength=str(len(mz)),
        )
        _cv(spec_el, "MS:1000511", "ms level", ms_level)

        bda_list = _sub(spec_el, "binaryDataArrayList", count="2")
        for arr, accession, name in (
            (mz, "MS:1000514", "m/z array"),
            (intensity, "MS:1000515", "intensity array"),
        ):
            encoded = _encode_array(arr)
            bda = _sub(bda_list, "binaryDataArray", encodedLength=str(len(encoded)))
            _cv(bda, "MS:1000574", "zlib compression")
            _cv(bda, "MS:1000523", "64-bit float")
            _cv(bda, accession, name)
            binary_el = etree.SubElement(bda, f"{{{_MZML_NS}}}binary")
            binary_el.text = encoded

    etree.ElementTree(root).write(
        str(path), xml_declaration=True, encoding="utf-8", pretty_print=True
    )


def _write_spectra(spectra: list[PyteomicsSpectrum], path: PathLike) -> None:
    """Write spectra to MGF or mzML depending on the file extension of path."""
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".mgf":
        out_iter = tqdm.tqdm(spectra, desc=f"Writing {path}", unit="spectrum")
        pyteomics.mgf.write(out_iter, output=str(path))
    elif suffix == ".mzml":
        logging.info("Writing mzML to %s", path)
        _write_mzml(spectra, path)
    else:
        raise ValueError(
            f"Unsupported output extension {suffix!r}; expected '.mgf' or '.mzml'."
        )


def sample_spectra(
    input_file: PathLike,
    k: float,
    outfile: Optional[PathLike] = None,
    buffer_size: int = 1000,
    random_seed: int = 42,
) -> list[PyteomicsSpectrum]:
    """
    Sample a proportion of spectra from an mzML file in a single streaming pass.

    Reads the file in chunks of ``buffer_size`` spectra.  From each chunk,
    ``round(k * chunk_size)`` spectra are drawn without replacement using
    ``random.sample``.  No second pass or total-count is required.

    Note: the final sample count equals ``sum(round(k * b) for b in buffers)``
    which may differ slightly from ``round(k * total)`` due to per-buffer
    rounding.  Use a ``buffer_size`` that is large relative to ``1 / k`` to
    minimise this effect.

    Parameters
    ----------
    input_file : PathLike
        Path to the input mzML file.
    k : float
        Proportion of spectra to sample; must be in (0, 1).
    outfile : PathLike, optional
        If provided, write sampled spectra to this path.  The format is
        determined by the file extension: ``.mgf`` writes MGF,
        ``.mzml`` writes mzML 1.1.0.
    buffer_size : int, default=1000
        Number of spectra read per I/O chunk.
    random_seed : int, default=42
        Seed for reproducible sampling.

    Returns
    -------
    list[PyteomicsSpectrum]
        Sampled spectra in file order within each buffer.
    """
    configure_logging(pathlib.Path(outfile).with_suffix(".log") if outfile else None)

    if not isinstance(k, float) or not (0.0 < k < 1.0):
        raise ValueError(f"k must be a float proportion in (0, 1), got {k!r}.")

    rng = random.Random(random_seed)
    result: list[PyteomicsSpectrum] = []

    with pyteomics.mzml.MzML(str(input_file)) as reader:
        buf: list[PyteomicsSpectrum] = []
        for spectrum in tqdm.tqdm(reader, desc="Sampling spectra", unit="spectrum"):
            buf.append(spectrum)
            if len(buf) >= buffer_size:
                n_sample = round(k * len(buf))
                result.extend(rng.sample(buf, n_sample))
                buf.clear()
        if buf:
            n_sample = round(k * len(buf))
            result.extend(rng.sample(buf, n_sample))

    logging.info("Sampled %d spectra", len(result))

    if outfile is not None:
        _write_spectra(result, outfile)

    return result


COMMANDS: Commands = sample_spectra


def main() -> None:
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()

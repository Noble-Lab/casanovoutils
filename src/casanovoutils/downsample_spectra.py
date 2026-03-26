"""Downsample an MGF file to a target number or proportion of spectra.

Three strategies are provided:

* ``number``: sample exactly *k* spectra at random (loads all into memory).
* ``proportion``: sample exactly ``round(total × rate)`` spectra (loads all
  into memory so the exact target is always met).
* ``approx-proportion``: accept each spectrum independently with probability
  *rate* — a single streaming pass that needs no extra memory and never reads
  the file twice, at the cost of only approximate output size.
"""

import math
import random
import sys
from os import PathLike
from pathlib import Path

import fire
import pyteomics.mgf
import tqdm

_VALID_TYPES = frozenset({"number", "proportion", "approx-proportion"})


def downsample_spectra(
    input_file: PathLike,
    output_file: PathLike,
    downsample_type: str = "number",
    downsample_rate: float = 100,
    random_seed: int = 42,
) -> None:
    """Downsample an MGF file to a target number or proportion of spectra.

    Parameters
    ----------
    input_file : PathLike
        Path to the input MGF file.
    output_file : PathLike
        Path for the downsampled output MGF file.  Must differ from
        *input_file*; overwriting the input in-place is not supported.
    downsample_type : str, default ``"number"``
        Sampling strategy.  One of:

        * ``"number"`` – retain exactly *downsample_rate* spectra.
        * ``"proportion"`` – retain exactly
          ``round(total × downsample_rate)`` spectra.
        * ``"approx-proportion"`` – accept each spectrum independently
          with probability *downsample_rate* (streaming, no memory
          overhead, approximate output size).
    downsample_rate : float, default 100
        Target rate.  Must be a positive integer for ``"number"``; must
        be in ``(0, 1]`` for ``"proportion"`` and ``"approx-proportion"``.
    random_seed : int, default 42
        Seed for the random number generator (reproducibility).

    Returns
    -------
    None
        Writes the downsampled spectra to *output_file*.

    Raises
    ------
    ValueError
        If *input_file* and *output_file* resolve to the same path, if
        *downsample_type* is not recognised, or if *downsample_rate* is
        out of range for the chosen type.
    """
    if Path(input_file).resolve() == Path(output_file).resolve():
        raise ValueError(
            "input_file and output_file must be different paths; "
            "overwriting the input in-place is not supported."
        )

    if downsample_type not in _VALID_TYPES:
        raise ValueError(
            f"--downsample-type must be one of {sorted(_VALID_TYPES)}, "
            f"got {downsample_type!r}."
        )

    if downsample_type == "number":
        if (
            not math.isfinite(downsample_rate)
            or downsample_rate != int(downsample_rate)
            or int(downsample_rate) < 1
        ):
            raise ValueError(
                "--downsample-rate must be a positive integer when "
                f"--downsample-type is 'number', got {downsample_rate!r}."
            )
        k = int(downsample_rate)
    else:
        if not (0 < downsample_rate <= 1):
            raise ValueError(
                "--downsample-rate must be in (0, 1] when "
                f"--downsample-type is '{downsample_type}', "
                f"got {downsample_rate!r}."
            )

    rng = random.Random(random_seed)

    if downsample_type == "approx-proportion":
        _stream_sample(input_file, output_file, downsample_rate, rng)
        return

    # ---- exact sampling: load everything into memory -------------------------
    print(f"Reading {input_file} ...", file=sys.stderr)
    with pyteomics.mgf.read(str(input_file), use_index=False) as reader:
        spectra = list(
            tqdm.tqdm(reader, desc="Reading spectra", unit="spectrum")
        )

    total = len(spectra)
    if downsample_type == "proportion":
        k = round(total * downsample_rate)
    # Clamp k to the number of available spectra (relevant when downsample_rate
    # for "number" mode exceeds the total, or rounding pushes "proportion" > 1).
    k = min(k, total)

    # Sample by index and sort so output preserves the original input order.
    indices = sorted(rng.sample(range(total), k))
    sampled = [spectra[i] for i in indices]

    pct = k / total if total > 0 else 0.0
    print(
        f"Sampled {k:,} of {total:,} spectra ({pct:.1%}).",
        file=sys.stderr,
    )
    print(f"Writing {output_file} ...", file=sys.stderr)
    pyteomics.mgf.write(
        tqdm.tqdm(sampled, desc="Writing spectra", unit="spectrum"),
        output=str(output_file),
    )
    print("Done.", file=sys.stderr)


def _stream_sample(
    input_file: PathLike,
    output_file: PathLike,
    rate: float,
    rng: random.Random,
) -> None:
    """Stream-sample spectra, accepting each with probability *rate*.

    Parameters
    ----------
    input_file : PathLike
        Path to the input MGF file.
    output_file : PathLike
        Path for the output MGF file.
    rate : float
        Acceptance probability in ``(0, 1]``.
    rng : random.Random
        Local random number generator instance (caller-supplied for
        reproducibility without mutating global state).
    """
    counter = {"read": 0, "written": 0}

    def _filtered():
        with pyteomics.mgf.read(str(input_file), use_index=False) as reader:
            for spectrum in tqdm.tqdm(
                reader, desc="Streaming spectra", unit="spectrum"
            ):
                counter["read"] += 1
                if rng.random() < rate:
                    counter["written"] += 1
                    yield spectrum

    print(f"Streaming {input_file} ...", file=sys.stderr)
    pyteomics.mgf.write(_filtered(), output=str(output_file))

    n_read = counter["read"]
    n_written = counter["written"]
    actual_pct = n_written / n_read if n_read > 0 else 0.0
    print(
        f"Read {n_read:,} spectra, wrote {n_written:,} ({actual_pct:.1%}).",
        file=sys.stderr,
    )


def main() -> None:
    """CLI entry point for the ``downsample-spectra`` command."""
    fire.Fire(downsample_spectra)


if __name__ == "__main__":
    main()

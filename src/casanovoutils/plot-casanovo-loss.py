#!/usr/bin/env python3
"""Plot Casanovo training and validation loss.

This script reads one or more Casanovo training logs (either a log file or a
`metrics.csv` file) and produces a PNG plot of training and validation loss as
a function of step/iteration.

Output is written to `<root>.png`.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Sequence, TypeAlias

import matplotlib.pyplot as plt


LossSeries: TypeAlias = list[tuple[int, float]]


_FLOAT_RE_PART = r"-?(?:nan|inf|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


# Casanovo logs contain lines like:
#   ... model._log_history : 50000\tnan\t0.365210
# and a header line like:
#   ... model._log_history : Step\tTrain loss\tValid loss
_LOG_HISTORY_RE = re.compile(
    rf"model\._log_history\s*:\s*(?P<step>\d+)\s+"
    rf"(?P<train>{_FLOAT_RE_PART})\s+(?P<val>{_FLOAT_RE_PART})"
)


def read_from_logfile(input_path: Path) -> tuple[LossSeries, LossSeries]:
    """Read train/validation loss series from a Casanovo log file.

    Parameters
    ----------
    input_path
        Path to a text log file produced during Casanovo training.

    Returns
    -------
    (train_losses, val_losses)
        Two lists of ``(step, loss)`` tuples.
    """
    train_losses: LossSeries = []
    val_losses: LossSeries = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            match = _LOG_HISTORY_RE.search(line)
            if match is None:
                continue

            step = int(match.group("step"))

            train_loss = float(match.group("train"))
            if not math.isnan(train_loss):
                train_losses.append((step, train_loss))

            val_loss = float(match.group("val"))
            if not math.isnan(val_loss):
                val_losses.append((step, val_loss))

    if train_losses or val_losses:
        print(
            f"Read {len(train_losses)} train losses and {len(val_losses)} "
            "validation losses.",
            file=sys.stderr,
        )

    return train_losses, val_losses


def read_from_csvfile(input_path: Path) -> tuple[LossSeries, LossSeries]:
    """Read train/validation loss series from a Casanovo `metrics.csv` file.

    Parameters
    ----------
    input_path
        Path to a CSV file containing Casanovo metrics.

    Returns
    -------
    (train_losses, val_losses)
        Two lists of ``(step, loss)`` tuples.
    """
    train_losses: LossSeries = []
    val_losses: LossSeries = []

    required_fields = {"step", "train_CELoss", "valid_CELoss"}

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row.")

        missing = required_fields.difference(reader.fieldnames)
        if missing:
            found = ", ".join(reader.fieldnames)
            needed = ", ".join(sorted(required_fields))
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                "CSV file is missing required column(s): "
                f"{missing_str}. Required: {needed}. Found: {found}."
            )

        for row in reader:
            if not row.get("step"):
                continue
            step = int(row["step"])

            train_val = row.get("train_CELoss")
            if train_val:
                train_losses.append((step, float(train_val)))

            valid_val = row.get("valid_CELoss")
            if valid_val:
                val_losses.append((step, float(valid_val)))

    print(
        f"Read {len(train_losses)} train losses and {len(val_losses)} "
        "validation losses.",
        file=sys.stderr,
    )

    return train_losses, val_losses


def detect_input_format(input_path: Path) -> str:
    """Determine whether the input is a log file or a metrics CSV.

    The detection is based on filename conventions and a quick header sniff.

    Parameters
    ----------
    input_path
        Path to an input file.

    Returns
    -------
    format
        Either ``"csv"`` or ``"log"``.
    """
    if input_path.suffix.lower() == ".csv" or input_path.name == "metrics.csv":
        return "csv"

    # Sniff the first line: a metrics CSV should have a comma-delimited header
    # containing a "step" column.
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            first_line = input_file.readline()
    except OSError:
        # Let the caller surface a clearer error.
        return "log"

    if "," in first_line and "step" in first_line.split(","):
        return "csv"

    return "log"


def existing_file(path_str: str) -> Path:
    """Parse an existing file path for argparse.

    Parameters
    ----------
    path_str
        Path string provided by the user.

    Returns
    -------
    path
        A :class:`~pathlib.Path` that exists and is a file.
    """
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Not a file: {path}")
    return path


def read_from_file(input_path: Path) -> tuple[LossSeries, LossSeries]:
    """Read losses from a Casanovo log file or metrics CSV."""
    file_format = detect_input_format(input_path)
    if file_format == "csv":
        return read_from_csvfile(input_path)
    return read_from_logfile(input_path)


def plot_losses(
    root: str,
    train_loss_lists: Sequence[LossSeries],
    val_loss_lists: Sequence[LossSeries],
    max_y: float | None,
) -> None:
    """Create and save the loss plot.

    Parameters
    ----------
    root
        Output file root name. Output is written to ``{root}.png``.
    train_loss_lists
        A sequence of training loss series.
    val_loss_lists
        A sequence of validation loss series.
    max_y
        Optional y-axis maximum.
    """
    fig, ax = plt.subplots()

    for i, train_loss_list in enumerate(train_loss_lists):
        if not train_loss_list:
            continue
        label = "Training" if i == 0 else None
        steps, losses = zip(*train_loss_list)
        ax.plot(steps, losses, "-o", markersize=2, label=label)

    for i, val_loss_list in enumerate(val_loss_lists):
        if not val_loss_list:
            continue
        label = "Validation" if i == 0 else None
        steps, losses = zip(*val_loss_list)
        ax.plot(steps, losses, "-o", markersize=2, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(root)

    if max_y is not None:
        ax.set_ylim(0, max_y)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right")

    fig.set_figwidth(4)
    fig.set_figheight(3)
    fig.savefig(f"{root}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="plot-casanovo-loss.py",
        description=(
            "Read Casanovo log and/or metrics.csv files and plot training and "
            "validation loss."
        ),
    )
    parser.add_argument(
        "root",
        help="Output file root; plot will be written to <root>.png",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=existing_file,
        help="One or more input files (log files or metrics.csv files)",
    )
    parser.add_argument(
        "--max-y",
        type=float,
        default=None,
        help="Optional y-axis maximum",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the script."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    train_loss_lists: list[LossSeries] = []
    val_loss_lists: list[LossSeries] = []
    any_points = False

    for input_path in args.inputs:
        try:
            train_loss_list, val_loss_list = read_from_file(input_path)
        except (OSError, ValueError) as exc:
            print(f"Error reading {input_path}: {exc}", file=sys.stderr)
            return 2

        if not train_loss_list and not val_loss_list:
            print(
                f"Warning: no loss entries found in {input_path}.",
                file=sys.stderr,
            )
        else:
            any_points = True

        train_loss_lists.append(train_loss_list)
        val_loss_lists.append(val_loss_list)

    if not any_points:
        print(
            "Error: no loss entries found in any input file; nothing to plot.",
            file=sys.stderr,
        )
        return 2

    plot_losses(args.root, train_loss_lists, val_loss_lists, args.max_y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -----------------------------------------------------------------------------
# Embedded samples and pytest tests
#
# Run:
#   pytest plot-casanovo-loss.py
#
# These tests write the embedded samples to a temporary directory so the
# parsing code operates on real file paths.
# code is exercised exactly as it is in real usage (paths, encoding, etc.).

SAMPLE_CASA_BALANCED_LOG = """2026-01-17 08:50:26,334 INFO [casanovo/MainProcess] utils.log_system_info : ======= System Information =======
2026-01-17 08:50:26,334 INFO [casanovo/MainProcess] utils.log_system_info : Executed Command: /net/noble/vol1/home/noble/miniconda3/envs/casanovo_dev/bin/casanovo train --model ../2026-01-14training4-extend/casa_balanced.epoch=12-step=668876.ckpt --config ../2026-01-09training4/casanovo.yaml --output_dir . --output_root casa_balanced --validation_peak_path /tmp/6478606.1.noble-long.q/massivekb_82c0124b_val_rndDedupSEQCHG_proforma_UniSpec_processed.mgf /tmp/6478606.1.noble-long.q/massivekb_82c0124b_train_rndDedupSEQCHG_proforma_UniSpec_processed.mgf
2026-01-17 08:50:26,335 INFO [casanovo/MainProcess] utils.log_system_info : Host Machine: n002.grid.gs.washington.edu
2026-01-17 08:50:26,335 INFO [casanovo/MainProcess] utils.log_system_info : OS: Linux
2026-01-17 08:50:26,335 INFO [casanovo/MainProcess] utils.log_system_info : OS Version: #174-Ubuntu SMP Fri Nov 14 20:25:16 UTC 2025
2026-01-17 08:50:26,335 INFO [casanovo/MainProcess] utils.log_system_info : Python Version: 3.11.14
2026-01-17 08:50:26,335 INFO [casanovo/MainProcess] utils.log_system_info : Casanovo Version: 5.1.2.dev20+ge71136cf8
2026-01-17 08:50:26,335 INFO [casanovo/MainProcess] utils.log_system_info : Depthcharge Version: 0.4.8
2026-01-17 08:50:26,335 INFO [casanovo/MainProcess] utils.log_system_info : PyTorch Version: 2.9.1+cu128
2026-01-17 08:50:26,371 INFO [casanovo/MainProcess] utils.log_system_info : CUDA Version: 12.8
2026-01-17 08:50:26,403 INFO [casanovo/MainProcess] utils.log_system_info : cuDNN Version: 91002
2026-01-17 08:50:26,607 INFO [casanovo/MainProcess] casanovo.setup_model : Casanovo version 5.1.2.dev20+ge71136cf8
2026-01-17 08:50:26,607 DEBUG [casanovo/MainProcess] casanovo.setup_model : model = ../2026-01-14training4-extend/casa_balanced.epoch=12-step=668876.ckpt
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : config = ../2026-01-09training4/casanovo.yaml
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : output directory = /net/noble/vol1/home/noble/proj/2026_ukeich_ms-casa-balanced/results/bill/2026-01-17training4-extend2
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : output root name = casa_balanced
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : precursor_mass_tol = 50.0
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : isotope_error_range = (0, 1)
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : min_peptide_len = 6
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : max_peptide_len = 100
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : predict_batch_size = 1024
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : top_match = 1
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : accelerator = auto
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : devices = None
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : n_beams = 1
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : enzyme = trypsin
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : digestion = full
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : missed_cleavages = 0
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : max_mods = 1
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : allowed_fixed_mods = C:C[Carbamidomethyl]
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : allowed_var_mods = M:M[Oxidation],N:N[Deamidated],Q:Q[Deamidated],nterm:[Acetyl]-,nterm:[Carbamyl]-,nterm:[Ammonia-loss]-,nterm:[+25.980265]-
2026-01-17 08:50:26,608 DEBUG [casanovo/MainProcess] casanovo.setup_model : random_seed = 454
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : n_log = 1
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : tb_summarywriter = True
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : log_metrics = True
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : log_every_n_steps = 50
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : lance_dir = None
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : val_check_interval = 50000
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : min_peaks = 20
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : max_peaks = 150
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : min_mz = 50.0
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : max_mz = 2500.0
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : min_intensity = 0.01
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : remove_precursor_tol = 2.0
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : max_charge = 5
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : dim_model = 512
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : n_head = 8
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : dim_feedforward = 1024
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : n_layers = 9
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : dropout = 0.0
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : dim_intensity = None
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : warmup_iters = 100000
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : cosine_schedule_period_iters = 600000
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : learning_rate = 0.0005
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : weight_decay = 1e-05
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : train_label_smoothing = 0.01
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : train_batch_size = 32
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : max_epochs = 30
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : shuffle = True
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : shuffle_buffer_size = 10000
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : num_sanity_val_steps = 0
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : calculate_precision = False
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : accumulate_grad_batches = 1
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : gradient_clip_val = None
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : gradient_clip_algorithm = None
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : precision = 32-true
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : replace_isoleucine_with_leucine = False
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : massivekb_tokenizer = False
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : residues = {'G': 57.021464, 'A': 71.037114, 'S': 87.032028, 'P': 97.052764, 'V': 99.068414, 'T': 101.04767, 'C[Carbamidomethyl]': 160.030649, 'L': 113.084064, 'I': 113.084064, 'N': 114.042927, 'D': 115.026943, 'Q': 128.058578, 'K': 128.094963, 'E': 129.042593, 'M': 131.040485, 'H': 137.058912, 'F': 147.068414, 'R': 156.101111, 'Y': 163.063329, 'W': 186.079313, 'M[Oxidation]': 147.0354, '[Acetyl]-': 42.010565, '[Ammonia-loss]-': -17.026549}
2026-01-17 08:50:26,609 DEBUG [casanovo/MainProcess] casanovo.setup_model : n_workers = 0
2026-01-17 08:50:26,659 INFO [casanovo/MainProcess] casanovo.train : Training a model from:
2026-01-17 08:50:26,659 INFO [casanovo/MainProcess] casanovo.train :   /tmp/6478606.1.noble-long.q/massivekb_82c0124b_train_rndDedupSEQCHG_proforma_UniSpec_processed.mgf
2026-01-17 08:50:26,659 INFO [casanovo/MainProcess] casanovo.train : Using the following validation files:
2026-01-17 08:50:26,659 INFO [casanovo/MainProcess] casanovo.train :   /tmp/6478606.1.noble-long.q/massivekb_82c0124b_val_rndDedupSEQCHG_proforma_UniSpec_processed.mgf
2026-01-17 08:50:26,710 WARNING [casanovo/MainProcess] model_runner.initialize_tokenizer : Configured residue(s) not in model alphabet: C[Carbamidomethyl], M[Oxidation], [Acetyl]-, [Ammonia-loss]-
2026-01-17 08:59:29,083 WARNING [py.warnings/MainProcess] warnings._showwarnmsg : UserWarning: Skipped 194 spectra with invalid information.Last error was: 
 Insufficient number of peaks
2026-01-17 08:59:29,083 WARNING [py.warnings/MainProcess] warnings._showwarnmsg : UserWarning: Skipped 194 spectra with invalid information.Last error was: 
 Insufficient number of peaks
2026-01-17 08:59:47,693 WARNING [py.warnings/MainProcess] warnings._showwarnmsg : UserWarning: Skipped 6 spectra with invalid information.Last error was: 
 Insufficient number of peaks
2026-01-17 08:59:47,693 WARNING [py.warnings/MainProcess] warnings._showwarnmsg : UserWarning: Skipped 6 spectra with invalid information.Last error was: 
 Insufficient number of peaks
2026-01-17 08:59:49,451 WARNING [py.warnings/MainProcess] warnings._showwarnmsg : /net/noble/vol1/home/noble/miniconda3/envs/casanovo_dev/lib/python3.11/site-packages/lightning/pytorch/callbacks/model_checkpoint.py:881: Checkpoint directory /net/noble/vol1/home/noble/proj/2026_ukeich_ms-casa-balanced/results/bill/2026-01-17training4-extend2 exists and is not empty.

2026-01-17 08:59:49,451 WARNING [py.warnings/MainProcess] warnings._showwarnmsg : /net/noble/vol1/home/noble/miniconda3/envs/casanovo_dev/lib/python3.11/site-packages/lightning/pytorch/callbacks/model_checkpoint.py:881: Checkpoint directory /net/noble/vol1/home/noble/proj/2026_ukeich_ms-casa-balanced/results/bill/2026-01-17training4-extend2 exists and is not empty.

2026-01-17 09:54:24,590 INFO [casanovo/MainProcess] model._log_history : Step	Train loss	Valid loss	
2026-01-17 09:54:24,590 INFO [casanovo/MainProcess] model._log_history : 50000	nan	0.365210
2026-01-17 09:56:00,560 INFO [casanovo/MainProcess] model._log_history : 51452	0.382402	nan
2026-01-17 10:49:10,152 INFO [casanovo/MainProcess] model._log_history : 100000	nan	0.453382
2026-01-17 10:52:17,722 INFO [casanovo/MainProcess] model._log_history : 102904	0.490081	nan
2026-01-17 11:43:52,773 INFO [casanovo/MainProcess] model._log_history : 150000	nan	0.440560
2026-01-17 11:48:33,344 INFO [casanovo/MainProcess] model._log_history : 154356	0.523945	nan
2026-01-17 12:38:31,493 INFO [casanovo/MainProcess] model._log_history : 200000	nan	0.423084
2026-01-17 12:44:45,364 INFO [casanovo/MainProcess] model._log_history : 205808	0.504128	nan
2026-01-17 13:33:05,334 INFO [casanovo/MainProcess] model._log_history : 250000	nan	0.393148
2026-01-17 13:40:52,259 INFO [casanovo/MainProcess] model._log_history : 257260	0.479335	nan
2026-01-17 14:27:48,716 INFO [casanovo/MainProcess] model._log_history : 300000	nan	0.377105
2026-01-17 14:37:13,278 INFO [casanovo/MainProcess] model._log_history : 308712	0.452938	nan
2026-01-17 15:22:37,936 INFO [casanovo/MainProcess] model._log_history : 350000	nan	0.347271
2026-01-17 15:33:53,328 INFO [casanovo/MainProcess] model._log_history : 360164	0.425210	nan
2026-01-17 16:17:40,238 INFO [casanovo/MainProcess] model._log_history : 400000	nan	0.324995
2026-01-17 16:30:15,185 INFO [casanovo/MainProcess] model._log_history : 411616	0.397945	nan
2026-01-17 17:12:26,569 INFO [casanovo/MainProcess] model._log_history : 450000	nan	0.304107
2026-01-17 17:26:34,146 INFO [casanovo/MainProcess] model._log_history : 463068	0.372500	nan
2026-01-17 18:07:04,844 INFO [casanovo/MainProcess] model._log_history : 500000	nan	0.290138
2026-01-17 18:22:39,366 INFO [casanovo/MainProcess] model._log_history : 514520	0.350137	nan
2026-01-17 19:01:22,218 INFO [casanovo/MainProcess] model._log_history : 550000	nan	0.284223
2026-01-17 19:18:24,375 INFO [casanovo/MainProcess] model._log_history : 565972	0.333132	nan
2026-01-17 19:55:24,981 INFO [casanovo/MainProcess] model._log_history : 600000	nan	0.282990
2026-01-17 20:13:58,220 INFO [casanovo/MainProcess] model._log_history : 617424	0.324337	nan
2026-01-17 20:49:30,742 INFO [casanovo/MainProcess] model._log_history : 650000	nan	0.284116
2026-01-17 21:09:37,074 INFO [casanovo/MainProcess] model._log_history : 668876	0.325341	nan
2026-01-17 21:43:44,873 INFO [casanovo/MainProcess] model._log_history : 700000	nan	0.289146
2026-01-17 22:05:27,022 INFO [casanovo/MainProcess] model._log_history : 720328	0.334916	nan
2026-01-17 22:38:26,486 INFO [casanovo/MainProcess] model._log_history : 750000	nan	0.302169
2026-01-17 23:01:43,122 INFO [casanovo/MainProcess] model._log_history : 771780	0.352514	nan
2026-01-17 23:32:57,433 INFO [casanovo/MainProcess] model._log_history : 800000	nan	0.322934
2026-01-17 23:57:51,029 INFO [casanovo/MainProcess] model._log_history : 823232	0.377037	nan
2026-01-18 00:27:34,174 INFO [casanovo/MainProcess] model._log_history : 850000	nan	0.344516
2026-01-18 00:54:01,858 INFO [casanovo/MainProcess] model._log_history : 874684	0.404934	nan
2026-01-18 01:22:16,632 INFO [casanovo/MainProcess] model._log_history : 900000	nan	0.374117
2026-01-18 01:50:18,048 INFO [casanovo/MainProcess] model._log_history : 926136	0.433300	nan
2026-01-18 02:17:00,213 INFO [casanovo/MainProcess] model._log_history : 950000	nan	0.398734
2026-01-18 02:46:30,165 INFO [casanovo/MainProcess] model._log_history : 977588	0.461437	nan
2026-01-18 03:11:37,519 INFO [casanovo/MainProcess] model._log_history : 1000000	nan	0.429187
2026-01-18 03:42:41,730 INFO [casanovo/MainProcess] model._log_history : 1029040	0.486534	nan
2026-01-18 04:06:18,635 INFO [casanovo/MainProcess] model._log_history : 1050000	nan	0.456048
2026-01-18 04:38:49,882 INFO [casanovo/MainProcess] model._log_history : 1080492	0.507617	nan
2026-01-18 05:00:50,142 INFO [casanovo/MainProcess] model._log_history : 1100000	nan	0.471987
2026-01-18 05:34:42,962 INFO [casanovo/MainProcess] model._log_history : 1131944	0.521891	nan
2026-01-18 05:55:07,212 INFO [casanovo/MainProcess] model._log_history : 1150000	nan	0.472303
2026-01-18 06:30:39,164 INFO [casanovo/MainProcess] model._log_history : 1183396	0.529776	nan
2026-01-18 06:49:32,414 INFO [casanovo/MainProcess] model._log_history : 1200000	nan	0.477287
2026-01-18 07:26:28,859 INFO [casanovo/MainProcess] model._log_history : 1234848	0.530516	nan
2026-01-18 07:43:50,642 INFO [casanovo/MainProcess] model._log_history : 1250000	nan	0.457192
2026-01-18 08:22:18,011 INFO [casanovo/MainProcess] model._log_history : 1286300	0.526391	nan
2026-01-18 08:38:09,527 INFO [casanovo/MainProcess] model._log_history : 1300000	nan	0.461780
2026-01-18 09:18:09,676 INFO [casanovo/MainProcess] model._log_history : 1337752	0.515894	nan
2026-01-18 09:32:31,412 INFO [casanovo/MainProcess] model._log_history : 1350000	nan	0.433914
2026-01-18 10:13:58,207 INFO [casanovo/MainProcess] model._log_history : 1389204	0.498964	nan
2026-01-18 10:26:46,963 INFO [casanovo/MainProcess] model._log_history : 1400000	nan	0.415865
2026-01-18 11:09:54,748 INFO [casanovo/MainProcess] model._log_history : 1440656	0.477520	nan
2026-01-18 11:21:12,566 INFO [casanovo/MainProcess] model._log_history : 1450000	nan	0.394631
2026-01-18 12:05:48,411 INFO [casanovo/MainProcess] model._log_history : 1492108	0.453517	nan
2026-01-18 12:15:36,896 INFO [casanovo/MainProcess] model._log_history : 1500000	nan	0.368093
"""

SAMPLE_METRICS_CSV = """epoch,hp/optimizer_cosine_schedule_period_iters,hp/optimizer_warmup_iters,lr-Adam,lr-Adam-momentum,lr-Adam-weight_decay,step,train_CELoss,valid_CELoss
0,,,,,,49999,,0.3652101457118988
0,600000.0,100000.0,,,,51451,0.382401704788208,
1,,,,,,99999,,0.4533817172050476
1,,,,,,102903,0.490081250667572,
2,,,,,,149999,,0.44056037068367004
2,,,,,,154355,0.5239452123641968,
3,,,,,,199999,,0.42308369278907776
3,,,,,,205807,0.5041282773017883,
4,,,,,,249999,,0.39314785599708557
4,,,,,,257259,0.47933459281921387,
5,,,,,,299999,,0.3771052062511444
5,,,,,,308711,0.4529375433921814,
6,,,,,,349999,,0.34727099537849426
6,,,,,,360163,0.42520976066589355,
7,,,,,,399999,,0.3249945044517517
7,,,,,,411615,0.39794477820396423,
8,,,,,,449999,,0.3041072189807892
8,,,,,,463067,0.372500479221344,
9,,,,,,499999,,0.29013770818710327
9,,,,,,514519,0.3501371145248413,
10,,,,,,549999,,0.28422316908836365
10,,,,,,565971,0.3331315219402313,
11,,,,,,599999,,0.2829902768135071
11,,,,,,617423,0.3243366479873657,
12,,,,,,649999,,0.2841159701347351
12,,,,,,668875,0.32534098625183105,
13,,,,,,699999,,0.289145827293396
13,,,,,,720327,0.33491596579551697,
14,,,,,,749999,,0.30216944217681885
14,,,,,,771779,0.3525140881538391,
15,,,,,,799999,,0.32293400168418884
15,,,,,,823231,0.3770371079444885,
16,,,,,,849999,,0.3445155620574951
16,,,,,,874683,0.40493419766426086,
17,,,,,,899999,,0.3741174638271332
17,,,,,,926135,0.4332999587059021,
18,,,,,,949999,,0.39873382449150085
18,,,,,,977587,0.4614366888999939,
19,,,,,,999999,,0.42918747663497925
19,,,,,,1029039,0.48653364181518555,
20,,,,,,1049999,,0.4560483992099762
20,,,,,,1080491,0.5076171159744263,
21,,,,,,1099999,,0.471987247467041
21,,,,,,1131943,0.5218905210494995,
22,,,,,,1149999,,0.4723031222820282
22,,,,,,1183395,0.5297763347625732,
23,,,,,,1199999,,0.47728684544563293
23,,,,,,1234847,0.5305159687995911,
24,,,,,,1249999,,0.4571917653083801
24,,,,,,1286299,0.5263910293579102,
25,,,,,,1299999,,0.46178045868873596
25,,,,,,1337751,0.5158939361572266,
26,,,,,,1349999,,0.43391361832618713
26,,,,,,1389203,0.49896445870399475,
27,,,,,,1399999,,0.41586488485336304
27,,,,,,1440655,0.4775197207927704,
28,,,,,,1449999,,0.3946306109428406
28,,,,,,1492107,0.4535166621208191,
29,,,,,,1499999,,0.36809325218200684
"""


def _write_sample(tmp_path: Path, filename: str, content: str) -> Path:
    """Write an embedded sample to a temporary file and return its path."""
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_detect_input_format_csv(tmp_path) -> None:
    """CSV inputs should be detected as CSV."""
    metrics_path = _write_sample(tmp_path, "metrics.csv", SAMPLE_METRICS_CSV)
    assert detect_input_format(metrics_path) == "csv"


def test_detect_input_format_log(tmp_path) -> None:
    """Log inputs should be detected as logs."""
    log_path = _write_sample(
        tmp_path, "casa_balanced.log", SAMPLE_CASA_BALANCED_LOG
    )
    assert detect_input_format(log_path) == "log"


def test_read_from_logfile_parses_expected_points(tmp_path) -> None:
    """The logfile parser should extract train/val series from log_history."""
    import pytest  # type: ignore

    log_path = _write_sample(
        tmp_path, "casa_balanced.log", SAMPLE_CASA_BALANCED_LOG
    )
    train, val = read_from_logfile(log_path)

    assert len(train) == 29
    assert len(val) == 30

    assert train[0] == (51452, pytest.approx(0.382402, rel=1e-9))
    assert val[0] == (50000, pytest.approx(0.365210, rel=1e-9))

    assert train[-1] == (1492108, pytest.approx(0.453517, rel=1e-9))
    assert val[-1] == (1500000, pytest.approx(0.368093, rel=1e-9))


def test_read_from_csvfile_parses_expected_points(tmp_path) -> None:
    """The CSV parser should read the same series as the original metrics.csv."""
    import pytest  # type: ignore

    metrics_path = _write_sample(tmp_path, "metrics.csv", SAMPLE_METRICS_CSV)
    train, val = read_from_csvfile(metrics_path)

    assert len(train) == 29
    assert len(val) == 30

    assert train[0] == (51451, pytest.approx(0.382401704788208, rel=1e-12))
    assert val[0] == (49999, pytest.approx(0.3652101457118988, rel=1e-12))

    assert train[-1] == (1492107, pytest.approx(0.4535166621208191, rel=1e-12))
    assert val[-1] == (1499999, pytest.approx(0.36809325218200684, rel=1e-12))


def test_read_from_file_dispatches_by_format(tmp_path) -> None:
    """read_from_file should call the appropriate reader based on detection."""
    metrics_path = _write_sample(tmp_path, "metrics.csv", SAMPLE_METRICS_CSV)
    log_path = _write_sample(
        tmp_path, "casa_balanced.log", SAMPLE_CASA_BALANCED_LOG
    )

    train_csv, val_csv = read_from_file(metrics_path)
    train_log, val_log = read_from_file(log_path)

    assert len(train_csv) == 29
    assert len(val_csv) == 30
    assert len(train_log) == 29
    assert len(val_log) == 30

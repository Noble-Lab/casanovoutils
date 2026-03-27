#!/usr/bin/env python3
"""Plot Casanovo training and validation loss.

This script reads one or more Casanovo training logs (either a log file or a
`metrics.csv` file) and produces a PNG plot of training and validation loss as
a function of step/iteration.

Output is written to `<root>.png`.
"""

from __future__ import annotations

import csv
import logging
import math
import pathlib
import re
from pathlib import Path
from typing import Optional, TypeAlias

import fire
import matplotlib.pyplot as plt

from . import configure_logging
from .types import CommandDict

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
        logging.info(
            "Read %d train losses and %d validation losses.",
            len(train_losses),
            len(val_losses),
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

    logging.info(
        "Read %d train losses and %d validation losses.",
        len(train_losses),
        len(val_losses),
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


def read_from_file(input_path: Path) -> tuple[LossSeries, LossSeries]:
    """Read losses from a Casanovo log file or metrics CSV."""
    file_format = detect_input_format(input_path)
    if file_format == "csv":
        return read_from_csvfile(input_path)
    return read_from_logfile(input_path)


def plot_losses(
    root: str,
    train_loss_lists: list[LossSeries],
    val_loss_lists: list[LossSeries],
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


def plot(
    root: str,
    inputs: list[str],
    max_y: Optional[float] = None,
) -> None:
    """Read Casanovo log and/or metrics.csv files and plot training/validation loss.

    Parameters
    ----------
    root
        Output file root; plot will be written to ``<root>.png`` and log to
        ``<root>.log``.
    inputs
        One or more input files (Casanovo log files or ``metrics.csv`` files).
    max_y
        Optional y-axis maximum.
    """
    configure_logging(pathlib.Path(root).with_suffix(".log"))

    train_loss_lists: list[LossSeries] = []
    val_loss_lists: list[LossSeries] = []
    any_points = False

    for input_str in inputs:
        input_path = Path(input_str)
        try:
            train_loss_list, val_loss_list = read_from_file(input_path)
        except (OSError, ValueError) as exc:
            logging.error("Error reading %s: %s", input_path, exc)
            raise SystemExit(2) from exc

        if not train_loss_list and not val_loss_list:
            logging.warning("No loss entries found in %s.", input_path)
        else:
            any_points = True

        train_loss_lists.append(train_loss_list)
        val_loss_lists.append(val_loss_list)

    if not any_points:
        logging.error("No loss entries found in any input file; nothing to plot.")
        raise SystemExit(2)

    logging.info("Writing plot to %s.png", root)
    plot_losses(root, train_loss_lists, val_loss_lists, max_y)


COMMANDS: CommandDict = {
    "plot": plot,
}


def main() -> None:
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()

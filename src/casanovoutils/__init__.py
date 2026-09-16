"""
casanovoutils — utilities for working with de novo peptide sequencing data.

Provides MGF file processing, precision/coverage evaluation, sequence
alignment, and residue mass vocabulary management. Exposes
``configure_logging`` for consistent log setup across submodules.
"""

import logging
import os
import sys
from os import PathLike
from typing import Optional


def configure_logging(
    log_file: Optional[PathLike] = None,
    level: Optional[int] = None,
    file_mode: str = "a",
) -> Optional[logging.FileHandler]:
    """
    Configure logging to stdout, optionally also writing to a file.

    The stdout handler is only added once (subsequent calls are a no-op for
    the base configuration).  If *log_file* is provided, a file handler is
    always added, even if logging was already configured, so callers can
    direct output from a specific command to a dedicated log file.

    Parameters
    ----------
    log_file : PathLike, optional
        If provided, log output is written to this file in addition to stdout.
        If ``None`` (default), only stdout is used.
    level : int, optional
        Logging level. If ``None`` (default), reads from the ``LOG_LEVEL``
        environment variable. If that is not set, defaults to ``logging.INFO``.
    file_mode : str, default "a"
        File open mode passed to ``logging.FileHandler``.  Use ``"w"`` to
        truncate the file on each run, ``"a"`` (default) to append.

    Returns
    -------
    logging.FileHandler or None
        The file handler that was added, or ``None`` if *log_file* was not
        provided.  The caller is responsible for removing and closing the
        handler when it is no longer needed.
    """
    level = os.environ.get("LOG_LEVEL", logging.INFO) if level is None else level

    if not logging.root.handlers:
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=handlers,
        )

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logging.root.addHandler(file_handler)
        return file_handler

    return None

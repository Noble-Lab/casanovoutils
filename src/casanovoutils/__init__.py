"""
casanovoutils — utilities for working with de novo peptide sequencing data.

Provides MGF file processing, precision/coverage evaluation, sequence
alignment, and residue mass vocabulary management. Exposes
``configure_logging`` for consistent log setup across submodules.
"""

import logging
import sys
from os import PathLike
from typing import Optional


def configure_logging(log_file: Optional[PathLike] = None) -> None:
    """
    Configure logging to stdout, optionally also writing to a file.

    This function is a no-op if it's already been called.

    Parameters
    ----------
    log_file : PathLike, optional
        If provided, log output is written to this file in addition to stdout.
        The file is opened in append mode. If ``None`` (default), only stdout
        is used.
    """
    if logging.root.handlers:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )

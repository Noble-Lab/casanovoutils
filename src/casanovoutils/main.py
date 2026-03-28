"""
Top-level CLI entry point for casanovoutils.

Builds a nested command dict from each submodule's COMMANDS constant and
exposes them as a single ``casanovoutils`` CLI via ``fire``.
"""

import fire

from .datasets import COMMANDS as dataset_commands
from .denovoutils import COMMANDS as denovo_commands
from .graphloss import COMMANDS as loss_commands
from .mgfutils import COMMANDS as mgf_commands
from .residues import COMMANDS as residue_commands
from .types import CommandDict

COMMANDS: CommandDict = {
    "mgf": mgf_commands,
    "denovo": denovo_commands,
    "dump-residues": residue_commands,
    "graph-loss": loss_commands,
    "datasets": dataset_commands,
}


def main() -> None:
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()

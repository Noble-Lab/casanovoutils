from os import PathLike
from typing import Optional

import fire


def get_multi_enzyme(
    mskb_v2_path: PathLike,
    output_dir: Optional[PathLike] = None,
    peptide_allow_list: Optional[PathLike] = None,
    peptide_block_list: Optional[PathLike] = None,
) -> None:
    """
    Get the multi-enzymic splits used to train Casanovo.

    After respecting the allow and block list, PSMs from
    `mskb_v2_path` are assigned to train/val/test in an ~8:1:1 ratio.

    Parameters
    ----------
    mskb_v2_path : PathLike
        The path to the  MassIVE-KB v2.0.15 dataset, see
        https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp
    output_dir : Optional[PathLike], default = None
        The directory to route outputs to. If not set the current working
        directory will be used as the output directory
    peptide_allow_list : Optional[PathLike], default = None
        A JSON file specifying an allow list for the train, test, and validation
        splits.
    peptide_allow_list : Optional[PathLike], default = None
        A JSON file specifying a block list for the train, test, and validation
        splits. If an allow list is also specified, all peptides in the block
        list will be removed from the allow-list.

    Outputs
    -------
    multi-enzyme-simple.train.mgf
        An MGF file containing the multi-enzymic train splits
    multi-enzyme-simple.test.mgf
        An MGF file containing the multi-enzymic test splits
    multi-enzyme-simple.val.mgf
        An MGF file containing the multi-enzymic validation splits
    peptides.json
        A JSON file specifying the unique peptides in the train, test, and
        validation splits
    """
    raise NotImplementedError()


def get_mskb_final(
    mskb_v1_path: PathLike,
    output_dir: PathLike,
    peptide_allow_list: Optional[PathLike],
    peptide_block_list: Optional[PathLike],
) -> None:
    """
    Get the tryptic splits used to train Casanovo

    This function will create a train split containing 2,000,000 PSMs, a
    validation split containing 200,000 PSMs, and a test split containing
    200,000 PSMs from `mskb_v1_path`. All splits will be disjoint at the
    peptide level.

    Parameters
    ----------
    mskb_v1_path : PathLike
        The path to the  MassIVE-KB v1 dataset, see
        https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp
    output_dir : Optional[PathLike], default = None
        The directory to route outputs to. If not set the current working
        directory will be used as the output directory
    peptide_allow_list : Optional[PathLike], default = None
        A JSON file specifying an allow list for the train, test, and validation
        splits.
    peptide_allow_list : Optional[PathLike], default = None
        A JSON file specifying a block list for the train, test, and validation
        splits. If an allow list is also specified, all peptides in the block
        list will be removed from the allow-list.

    Outputs
    -------
    mskb_final.train.mgf
        An MGF file containing the tryptic train splits
    mskb_final.test.mgf
        An MGF file containing the tryptic test splits
    mskb_final.val.mgf
        An MGF file containing the tryptic validation splits
    peptides.json
        A JSON file specifying the unique peptides in the train, test, and
        validation splits
    """
    raise NotImplementedError()


def get_full_data(
    mskb_v1_path: PathLike,
    mskb_v2_path: PathLike,
    output_dir: PathLike,
    peptide_allow_list: Optional[PathLike],
    peptide_block_list: Optional[PathLike],
) -> None:
    """
    Get the tryptic and multi-enzymic splits for Casanovo

    This is a convenience utility that will

    1. Invoke get_mskb_final using the specified parameters, including the
    block list and allow list
    2. Construct a block list for creating the multi-enzyme splits, such that
    no peptide-level leakage occurs from the tryptic peptide list
    3. Invoke get_multi_enzyme with the block list created from the tryptic
    splits

    Parameters
    ----------
    mskb_v1_path : PathLike
        The path to the  MassIVE-KB v1 dataset, see
        https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp
    mskb_v2_path : PathLike
        The path to the  MassIVE-KB v2.0.15 dataset, see
        https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp
    output_dir : Optional[PathLike], default = None
        The directory to route outputs to. If not set the current working
        directory will be used as the output directory
    peptide_allow_list : Optional[PathLike], default = None
        A JSON file specifying an allow list for the train, test, and validation
        splits.
    peptide_allow_list : Optional[PathLike], default = None
        A JSON file specifying a block list for the train, test, and validation
        splits. If an allow list is also specified, all peptides in the block
        list will be removed from the allow-list.

    Outputs
    -------
    mskb_final/mskb_final.train.mgf
        An MGF file containing the tryptic train splits
    mskb_final/mskb_final.test.mgf
        An MGF file containing the tryptic test splits
    mskb_final/mskb_final.val.mgf
        An MGF file containing the tryptic validation splits
    mskb_final/peptides.json
        A JSON file specifying the unique peptides in the train, test, and
        validation splits
    multi_enzyme_simple/multi-enzyme-simple.train.mgf
        An MGF file containing the multi-enzymic train splits
    multi_enzyme_simple/multi-enzyme-simple.test.mgf
        An MGF file containing the multi-enzymic test splits
    multi_enzyme_simple/multi-enzyme-simple.val.mgf
        An MGF file containing the multi-enzymic validation splits
    multi_enzyme_simple/peptides.json
        A JSON file specifying the unique peptides in the train, test, and
        validation splits
    mskb_block_list.json
        The block list constructed from the MSKB splits
    """
    raise NotImplementedError()


def main() -> None:
    fire.Fire(
        {
            "multi_enzyme_simple": get_multi_enzyme,
            "mskb_final": get_mskb_final,
            "full_splits": get_full_data,
        }
    )


if __name__ == "__main__":
    main()

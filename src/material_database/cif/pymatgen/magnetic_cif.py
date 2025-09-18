from logging import WARNING, getLogger
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.operations import MagSymmOp
from pymatgen.symmetry.maggroups import MagneticSpaceGroup

from material_database.cif.pymatgen.block import CifBlock
from material_database.cif.pymatgen.utils import str2float

LOGGER = getLogger(__name__)
LOGGER.setLevel(WARNING)


def is_magcif(cif_block: CifBlock) -> bool:
    """Check if a file is a magCIF file (heuristic)."""
    # Doesn't seem to be a canonical way to test if file is magCIF or
    # not, so instead check for magnetic symmetry datanames
    prefixes = [
        "_space_group_magn",
        "_atom_site_moment",
        "_space_group_symop_magn",
    ]

    for key in cif_block.data:
        for prefix in prefixes:
            if prefix in key:
                return True

    return False


def is_magcif_incommensurate(cif_block: CifBlock) -> bool:
    """
    Check if a file contains an incommensurate magnetic
    structure (heuristic).
    """
    # Doesn't seem to be a canonical way to test if magCIF file
    # describes incommensurate structure or not, so instead check
    # for common datanames
    if not is_magcif(cif_block):
        return False

    prefixes = ["_cell_modulation_dimension", "_cell_wave_vector"]
    for key in cif_block:
        for prefix in prefixes:
            if prefix in key:
                return True
    return False


def parse_magnetic_moments(cif_block: CifBlock) -> dict[str, NDArray]:
    """Parse atomic magnetic moments from data."""
    try:
        magmoms = {
            cif_block["_atom_site_moment_label"][idx]: np.array(
                [
                    str2float(cif_block["_atom_site_moment_crystalaxis_x"][idx]),
                    str2float(cif_block["_atom_site_moment_crystalaxis_y"][idx]),
                    str2float(cif_block["_atom_site_moment_crystalaxis_z"][idx]),
                ]
            )
            for idx in range(len(cif_block["_atom_site_moment_label"]))
        }
    except (ValueError, KeyError):
        return {}
    return magmoms


def get_magnetic_symops(data: CifBlock) -> list[MagSymmOp]:
    """Equivalent to get_symops except for magnetic symmetry groups.
    Separate function since additional operation for time reversal symmetry
    (which changes magnetic moments on sites) needs to be returned.
    """
    # Get BNS label and number for magnetic space group
    bns_name = data.data.get("_space_group_magn.name_BNS", "")
    bns_num = data.data.get("_space_group_magn.number_BNS", "")

    mag_symm_ops = []
    # Check if magCIF file explicitly contains magnetic symmetry operations
    if xyzt := data.data.get("_space_group_symop_magn_operation.xyz"):
        if isinstance(xyzt, str):
            xyzt = [xyzt]
        mag_symm_ops = [MagSymmOp.from_xyzt_str(s) for s in xyzt]

        if data.data.get("_space_group_symop_magn_centering.xyz"):
            xyzt = data.data.get("_space_group_symop_magn_centering.xyz")
            if isinstance(xyzt, str):
                xyzt = [xyzt]
            centering_symops = [MagSymmOp.from_xyzt_str(s) for s in xyzt]

            all_ops = []
            for op in mag_symm_ops:
                for centering_op in centering_symops:
                    new_translation = [
                        i - np.floor(i)
                        for i in op.translation_vector + centering_op.translation_vector
                    ]
                    new_time_reversal = op.time_reversal * centering_op.time_reversal

                    all_ops.append(
                        MagSymmOp.from_rotation_and_translation_and_time_reversal(
                            rotation_matrix=op.rotation_matrix,
                            translation_vec=new_translation,
                            time_reversal=cast("Literal[-1, 1]", new_time_reversal),
                        )
                    )
            mag_symm_ops = all_ops

    # Else check if it specifies a magnetic space group
    elif bns_name or bns_num:
        label = bns_name or list(map(int, (bns_num.split("."))))

        if data.data.get("_space_group_magn.transform_BNS_Pp_abc") != "a,b,c;0,0,0":
            jonas_faithful = data.data.get("_space_group_magn.transform_BNS_Pp_abc")
            mag_sg = MagneticSpaceGroup(label, jonas_faithful)

        elif data.data.get("_space_group_magn.transform_BNS_Pp"):
            raise NotImplementedError("Incomplete specification to implement.")
        else:
            mag_sg = MagneticSpaceGroup(label)

        mag_symm_ops = mag_sg.symmetry_ops

    if not mag_symm_ops:
        msg = "No magnetic symmetry detected, using primitive symmetry."
        LOGGER.warning(msg)
        mag_symm_ops = [MagSymmOp.from_xyzt_str("x, y, z, 1")]

    return mag_symm_ops

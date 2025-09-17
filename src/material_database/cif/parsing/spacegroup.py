import re
from logging import WARNING, getLogger
from pathlib import Path

from monty.serialization import loadfn
from pymatgen.symmetry.analyzer import SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.site_symmetries import SymmOp

from material_database.cif.pymatgen.block import CifBlock
from material_database.cif.pymatgen.utils import str2float


def clean_up_spacegroup_string(s: str) -> str:
    return re.sub(r"[\s_]", "", s)


LOGGER = getLogger(__name__)
LOGGER.setLevel(WARNING)

SYMMETRY_OPS = loadfn(Path(__file__).parent / "symm_ops.json")
SPACEGROUPS = {
    clean_up_spacegroup_string(key): key for key in SYMM_DATA["space_group_encoding"]
}


def get_spacegroup_information(
    block: CifBlock,
) -> tuple[SpacegroupOperations, SpaceGroup] | None:
    spacegroup = _parse_spacegroup_number(block) or _parse_spacegroup_symbol(block)

    if spacegroup is None:
        LOGGER.warning(
            "Could not determine spacegroup from CIF block. Defaulting to P1."
        )
        return None

    symm_ops = _parse_symmetry_operations(block)

    if not symm_ops:
        symm_ops = spacegroup.symmetry_ops

    return (
        SpacegroupOperations(
            symmops=symm_ops,
            int_symbol=spacegroup.full_symbol,
            int_number=spacegroup.int_number,
        ),
        spacegroup,
    )


def _parse_spacegroup_number(block: CifBlock) -> SpaceGroup | None:
    KEYS_TO_TRY = [
        "_space_group_IT_number",
        "_space_group_IT_number_",
        "_symmetry_Int_Tables_number",
        "_symmetry_Int_Tables_number_",
    ]

    for key in KEYS_TO_TRY:
        spacegroup = block.get(key)
        if spacegroup is not None:
            try:
                return SpaceGroup.from_int_number(int(str2float(spacegroup)))
            except ValueError:
                continue

    return None


def _parse_spacegroup_symbol(block: CifBlock) -> SpaceGroup | None:
    KEYS_TO_TRY = [
        "_symmetry_space_group_name_H-M",
        "_symmetry_space_group_name_H_M",
        "_symmetry_space_group_name_H-M_",
        "_symmetry_space_group_name_H_M_",
        "_space_group_name_Hall",
        "_space_group_name_Hall_",
        "_space_group_name_H-M_alt",
        "_space_group_name_H-M_alt_",
        "_symmetry_space_group_name_hall",
        "_symmetry_space_group_name_hall_",
        "_symmetry_space_group_name_h-m",
        "_symmetry_space_group_name_h-m_",
    ]

    for key in KEYS_TO_TRY:
        spacegroup = block.get(key)

        if spacegroup:
            spacegroup = clean_up_spacegroup_string(spacegroup)
            try:
                if spg := SPACEGROUPS.get(spacegroup):
                    return SpaceGroup(spg)
            except ValueError:
                pass

            for spacegroup_dict in SYMMETRY_OPS:
                if spacegroup == re.sub(r"\s+", "", spacegroup_dict["hermann_mauguin"]):
                    return SpaceGroup.from_int_number(spacegroup_dict["number"])

    return None


def _parse_symmetry_operations(block: CifBlock) -> list[SymmOp]:
    KEYS_TO_TRY = [
        "_symmetry_equiv_pos_as_xyz",
        "_symmetry_equiv_pos_as_xyz_",
        "_space_group_symop_operation_xyz",
        "_space_group_symop_operation_xyz_",
    ]

    for key in KEYS_TO_TRY:
        xyz_strings = block.get(key, [])
        if xyz_strings:
            break

    if isinstance(xyz_strings, str):
        xyz_strings = [xyz_strings]

    return [SymmOp.from_xyz_str(s) for s in xyz_strings]

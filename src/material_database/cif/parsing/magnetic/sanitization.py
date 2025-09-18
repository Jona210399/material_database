from logging import WARNING, getLogger

from material_database.cif.parsing.block import CifBlock
from material_database.cif.parsing.sanitization import sanitize_cif_block

LOGGER = getLogger(__name__)
LOGGER.setLevel(WARNING)


def sanitize_magnetic_cif_block(cif_block: CifBlock) -> CifBlock:
    cif_block = sanitize_cif_block(cif_block)
    cif_block = _fix_magnetic_cif_tags(cif_block)
    return cif_block


def _fix_magnetic_cif_tags(cif_block: CifBlock) -> CifBlock:
    """This fixes inconsistencies in naming of several magCIF tags
    as a result of magCIF being in widespread use prior to
    specification being finalized (on advice of Branton Campbell).
    CIF-1 style has all underscores, interim standard
    had period before magn instead of before the final
    component (e.g. xyz)."""

    correct_keys = (
        "_space_group_symop_magn_operation.xyz",
        "_space_group_symop_magn_centering.xyz",
        "_space_group_magn.name_BNS",
        "_space_group_magn.number_BNS",
        "_atom_site_moment_crystalaxis_x",
        "_atom_site_moment_crystalaxis_y",
        "_atom_site_moment_crystalaxis_z",
        "_atom_site_moment_label",
    )

    # Cannot mutate dict during enumeration, so store changes
    changes_to_make = {}

    for original_key in cif_block.data:
        for correct_key in correct_keys:
            # convert to all underscore
            trial_key = "_".join(correct_key.split("."))
            test_key = "_".join(original_key.split("."))
            if trial_key == test_key:
                changes_to_make[correct_key] = original_key

    # Apply changes
    for correct_key, original_key in changes_to_make.items():
        cif_block.data[correct_key] = cif_block.data[original_key]

    # Map interim_keys to final_keys
    renamed_keys = {
        "_magnetic_space_group.transform_to_standard_Pp_abc": "_space_group_magn.transform_BNS_Pp_abc"
    }
    changes_to_make = {}

    for interim_key, final_key in renamed_keys.items():
        if cif_block.data.get(interim_key):
            changes_to_make[final_key] = interim_key

    if len(changes_to_make) > 0:
        LOGGER.warning("Keys changed to match new magCIF specification.")

    for final_key, interim_key in changes_to_make.items():
        cif_block.data[final_key] = cif_block.data[interim_key]

    return cif_block

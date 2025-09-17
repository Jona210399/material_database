import math
import re
from logging import WARNING, getLogger

from material_database.cif.pymatgen.block import CifBlock
from material_database.cif.pymatgen.utils import str2float

LOGGER = getLogger(__name__)
LOGGER.setLevel(WARNING)


def sanitize_cif_block(
    cif_block: CifBlock,
    is_magcif: bool,
    fraction_tolerance: float,
) -> CifBlock:
    """Some CIF files do not conform to spec. This method corrects
    known issues, particular in regards to Springer materials/
    Pauling files.

    Handles formats of data as found in CIF files extracted
    from the Springer Materials/Pauling File databases,
    and that are different from standard ICSD formats.

    This method is here so that CifParser can assume its
    input conforms to spec, simplifying its implementation.
    """

    _check_for_implicit_hydrogens(cif_block)
    cif_block = _check_atom_site_type_symbols(cif_block)

    if is_magcif:
        cif_block = _sanitize_magnetic_cif(cif_block)

    cif_block = _check_finite_precision_coordinates(cif_block, fraction_tolerance)
    return cif_block


def _sanitize_magnetic_cif(cif_block: CifBlock) -> CifBlock:
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
        print("Keys changed to match new magCIF specification.")

    for final_key, interim_key in changes_to_make.items():
        cif_block.data[final_key] = cif_block.data[interim_key]

    return cif_block


def _check_for_implicit_hydrogens(cif_block: CifBlock) -> None:
    if "_atom_site_attached_hydrogens" in cif_block.data:
        attached_hydrogens = [
            str2float(x)
            for x in cif_block.data["_atom_site_attached_hydrogens"]
            if str2float(x) != 0
        ]
        if len(attached_hydrogens) > 0:
            LOGGER.warning(
                "Structure has implicit hydrogens defined, parsed structure unlikely to be "
                "suitable for use in calculations unless hydrogens added."
            )


def _check_atom_site_type_symbols(cif_block: CifBlock) -> CifBlock:
    if "_atom_site_type_symbol" in cif_block.data:
        # Keep track of which data row needs to be removed.
        # Example of a row: Nb,Zr '0.8Nb + 0.2Zr' .2a .m-3m 0 0 0 1 14
        # 'rhombic dodecahedron, Nb<sub>14</sub>'
        # Without this, the above row in a structure would be parsed
        # as an ordered site with only Nb (since
        # CifParser would try to parse the first two characters of the
        # label "Nb,Zr") and occupancy=1.
        # However, this site is meant to be a disordered site with 0.8 of
        # Nb and 0.2 of Zr.
        idxs_to_remove: list[int] = []

        new_atom_site_label: list[str] = []
        new_atom_site_type_symbol: list[str] = []
        new_atom_site_occupancy: list[str] = []
        new_fract_x: list[str] = []
        new_fract_y: list[str] = []
        new_fract_z: list[str] = []

        for idx, el_row in enumerate(cif_block["_atom_site_label"]):
            # CIF files from the Springer Materials/Pauling File have
            # switched the label and symbol. Thus, in the
            # above shown example row, '0.8Nb + 0.2Zr' is the symbol.
            # Below, we split the strings on ' + ' to
            # check if the length (or number of elements) in the label and
            # symbol are equal.
            if len(cif_block["_atom_site_type_symbol"][idx].split(" + ")) > len(
                el_row.split(" + ")
            ):
                # Dictionary to hold elements and occupancies
                els_occu: dict[str, float] = {}

                # Parse symbol to get element names and occupancy
                symbol_str: str = cif_block["_atom_site_type_symbol"][idx]
                symbol_str_lst: list[str] = symbol_str.split(" + ")

                for _idx, symbol in enumerate(symbol_str_lst):
                    # Remove any bracketed items in the string
                    symbol_str_lst[_idx] = re.sub(r"\([0-9]*\)", "", symbol.strip())

                    # Extract element name and occupancy from the string
                    els_occu[
                        str(
                            re.findall(r"\D+", symbol_str_lst[_idx].strip())[1]
                        ).replace("<sup>", "")
                    ] = float(
                        "0" + re.findall(r"\.?\d+", symbol_str_lst[_idx].strip())[1]
                    )

                for et, occu in els_occu.items():
                    # New atom site labels have "fix" appended
                    new_atom_site_label.append(f"{et}_fix{len(new_atom_site_label)}")
                    new_atom_site_type_symbol.append(et)
                    new_atom_site_occupancy.append(str(occu))

                    new_fract_x.append(
                        str(str2float(cif_block["_atom_site_fract_x"][idx]))
                    )
                    new_fract_y.append(
                        str(str2float(cif_block["_atom_site_fract_y"][idx]))
                    )
                    new_fract_z.append(
                        str(str2float(cif_block["_atom_site_fract_z"][idx]))
                    )

                idxs_to_remove.append(idx)

        # Remove the original row by iterating over all keys in the CIF
        # data looking for lists, which indicates
        # multiple data items, one for each row, and remove items from the
        # list that corresponds to the removed row,
        # so that it's not processed by the rest of this function (which
        # would result in an error).
        for original_key in cif_block.data:
            if isinstance(cif_block.data[original_key], list):
                for idx in sorted(idxs_to_remove, reverse=True):
                    del cif_block.data[original_key][idx]

        if idxs_to_remove:
            LOGGER.warning("Pauling file corrections applied.")

            cif_block.data["_atom_site_label"] += new_atom_site_label
            cif_block.data["_atom_site_type_symbol"] += new_atom_site_type_symbol
            cif_block.data["_atom_site_occupancy"] += new_atom_site_occupancy
            cif_block.data["_atom_site_fract_x"] += new_fract_x
            cif_block.data["_atom_site_fract_y"] += new_fract_y
            cif_block.data["_atom_site_fract_z"] += new_fract_z

    return cif_block


def _check_finite_precision_coordinates(
    cif_block: CifBlock,
    fraction_tolerance: float,
) -> CifBlock:
    """Check for finite precision coordinates (e.g. 0.6667 instead of 0.6666666...),
    which can cause issues when applying symmetry operations."""

    important_fracs = (1 / 3, 2 / 3)
    fracs_to_change = {}
    for label in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"):
        if label in cif_block.data:
            for idx, frac in enumerate(cif_block.data[label]):
                try:
                    frac = str2float(frac)
                except Exception:
                    # Coordinate might not be defined, e.g. '?'
                    continue

                for comparison_frac in important_fracs:
                    if math.isclose(
                        frac / comparison_frac,
                        1,
                        abs_tol=fraction_tolerance,
                        rel_tol=0,
                    ):
                        fracs_to_change[label, idx] = str(comparison_frac)

    if fracs_to_change:
        LOGGER.warning(
            f"{len(fracs_to_change)} fractional coordinates rounded to ideal values to avoid issues with "
            "finite precision."
        )
        for (label, idx), val in fracs_to_change.items():
            cif_block.data[label][idx] = val

    return cif_block

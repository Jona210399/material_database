from inspect import getfullargspec

from pymatgen.core.lattice import Lattice

from material_database.cif.parsing.block import CifBlock
from material_database.cif.parsing.logger import LOGGER as PARSER_LOGGER
from material_database.cif.parsing.utils import str2float

LOGGER = PARSER_LOGGER.getChild("lattice")


def get_lattice(
    data: CifBlock,
    length_strings=("a", "b", "c"),
    angle_strings=("alpha", "beta", "gamma"),
    lattice_type=None,
) -> Lattice | None:
    """Generate the lattice from the provided lattice parameters.
    In the absence of all six lattice parameters, the crystal system
    and necessary parameters are parsed.
    """
    try:
        return get_lattice_no_exception(
            data=data,
            angle_strings=angle_strings,
            lattice_type=lattice_type,
            length_strings=length_strings,
        )

    except KeyError:
        # Missing Key search for cell setting
        for lattice_label in (
            "_symmetry_cell_setting",
            "_space_group_crystal_system",
        ):
            if data.data.get(lattice_label):
                lattice_type = data.data.get(lattice_label, "").lower()
                try:
                    required_args = getfullargspec(getattr(Lattice, lattice_type)).args

                    lengths = (
                        length for length in length_strings if length in required_args
                    )
                    angles = (a for a in angle_strings if a in required_args)
                    return get_lattice(
                        data,
                        lengths,
                        angles,
                        lattice_type=lattice_type,
                    )
                except AttributeError as exc:
                    LOGGER.warning(str(exc))

            else:
                return None
    return None


def get_lattice_no_exception(
    data: CifBlock,
    length_strings: tuple[str, str, str] = ("a", "b", "c"),
    angle_strings: tuple[str, str, str] = ("alpha", "beta", "gamma"),
    lattice_type: str | None = None,
) -> Lattice:
    """Convert a CifBlock to a pymatgen Lattice.

    Args:
        data: a dictionary of the CIF file
        length_strings: The strings that are used to identify the length parameters in the CIF file.
        angle_strings: The strings that are used to identify the angles in the CIF file.
        lattice_type (str): The type of lattice.

    Returns:
        Lattice object
    """
    lengths = [str2float(data[f"_cell_length_{i}"]) for i in length_strings]
    angles = [str2float(data[f"_cell_angle_{i}"]) for i in angle_strings]
    if not lattice_type:
        return Lattice.from_parameters(*lengths, *angles)
    return getattr(Lattice, lattice_type)(*(lengths + angles))


def passes_minimal_lattice_thickness_check(
    lattice: Lattice, minimal_thickness: float
) -> bool:
    thickness = [
        lattice.d_hkl((1, 0, 0)),
        lattice.d_hkl((0, 1, 0)),
        lattice.d_hkl((0, 0, 1)),
    ]

    return not any(t < minimal_thickness for t in thickness)

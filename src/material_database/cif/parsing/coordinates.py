import math

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Composition
from pymatgen.core.operations import SymmOp
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc

from material_database.cif.parsing.block import CifBlock
from material_database.cif.parsing.logger import LOGGER as PARSER_LOGGER
from material_database.cif.parsing.utils import str2float

LOGGER = PARSER_LOGGER.getChild("coordinates")


def parse_fractional_coordinates(
    block: CifBlock, num_sites: int
) -> list[tuple[float, float, float]] | None:
    x, y, z = [block.get(f"_atom_site_fract_{axis}", None) for axis in "xyz"]

    if None in (x, y, z):
        LOGGER.warning("Missing fractional coordinate data in CIF block.")
        return None

    if not (len(x) == len(y) == len(z) == num_sites):
        LOGGER.warning("Inconsistent number of fractional coordinates in CIF block.")
        return None

    fractional_coordinates = []
    for i in range(num_sites):
        try:
            coord = (str2float(x[i]), str2float(y[i]), str2float(z[i]))
            fractional_coordinates.append(coord)
        except ValueError as e:
            LOGGER.warning(
                f"Invalid fractional coordinate at index {i}: ({x[i]}, {y[i]}, {z[i]}): {e}"
            )
            return None

    return fractional_coordinates


def get_equivalent_coordinates(
    coordinate: tuple[float, float, float],
    symmetry_operations: list[SymmOp],
    site_tolerance: float,
) -> list[NDArray]:
    """Generate unique coordinates using coordinates and symmetry operations."""
    equivalent_coordinates: list[NDArray] = []

    for op in symmetry_operations:
        coord = op.operate(coordinate)
        coord = np.array([i - math.floor(i) for i in coord])
        if not in_coord_list_pbc(equivalent_coordinates, coord, atol=site_tolerance):
            equivalent_coordinates.append(coord)

    return equivalent_coordinates


def get_matching_coordinate(
    coordinates_to_composition: dict[tuple[float, float, float], Composition],
    coord: tuple[float, float, float],
    symmetry_operations: list[SymmOp],
    site_tolerance: float,
) -> tuple[float, float, float]:
    """Find site by coordinate."""
    coordinates: list[tuple[float, float, float]] = list(
        coordinates_to_composition.keys()
    )
    for op in symmetry_operations:
        frac_coord = op.operate(coord)
        indices: NDArray = find_in_coord_list_pbc(
            coordinates,
            frac_coord,
            atol=site_tolerance,
        )
        if len(indices) > 0:
            return coordinates[indices[0]]

    return coord

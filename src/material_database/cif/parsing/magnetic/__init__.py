from pymatgen.core.structure import Structure

from material_database.cif.parsing.block import CifBlock


def parse_magnetic_cif(cif_block: CifBlock, parsing_settings) -> list[Structure]:
    raise NotImplementedError("Magnetic CIF parsing not yet implemented.")

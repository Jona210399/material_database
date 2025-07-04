from io import StringIO

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifBlock, CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.site_symmetries import SymmOp
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.typing import PathLike


def spacegroup_from_cif(block: CifBlock) -> SpaceGroup | None:
    try:
        spacegroup = int(block["_symmetry_Int_Tables_number"])
        return SpaceGroup.from_int_number(spacegroup)

    except (KeyError, ValueError):
        pass

    try:
        spacegroup = block["_symmetry_space_group_name_H-M"]
        return SpaceGroup(spacegroup)
    except KeyError:
        pass

    return None


def get_spacegroup_operations(
    block: CifBlock,
    spacegroup: SpaceGroup,
) -> SpacegroupOperations:
    symm_ops = [
        SymmOp.from_xyz_str(xyz_str) for xyz_str in block["_symmetry_equiv_pos_as_xyz"]
    ]

    return SpacegroupOperations(
        symmops=symm_ops,
        int_symbol=spacegroup.full_symbol,
        int_number=spacegroup.int_number,
    )


def get_wyckoff_letters_as_in_cif(block: CifBlock) -> list[str] | None:
    if "_atom_site_Wyckoff_symbol" in block.data:
        return block["_atom_site_Wyckoff_symbol"]
    return None


def get_wyckoff_multiplicities(block: CifBlock) -> list[int] | None:
    if "_atom_site_symmetry_multiplicity" in block.data:
        return [int(mult) for mult in block["_atom_site_symmetry_multiplicity"]]
    return None


def get_wyckoff_symbols(multiplicities: list[int], letters: list[str]) -> list[str]:
    if len(multiplicities) != len(letters):
        raise ValueError("Multiplicity and letters lists must have the same length.")

    return [f"{mult}{letter}" for mult, letter in zip(multiplicities, letters)]


def get_wyckoff_letters(
    wyckoff_letters_as_in_cif: list[str], multiplicities: list[int]
) -> list[str]:
    if len(wyckoff_letters_as_in_cif) != len(multiplicities):
        raise ValueError("Length of wyckoff letters must match multiplicities")

    wyckoff_letters = []
    for letter, count in zip(wyckoff_letters_as_in_cif, multiplicities):
        wyckoff_letters.extend([letter] * count)
    return wyckoff_letters


def get_equivalent_positions(multiplicities: list[int]) -> list[int]:
    equivalent_positions = []
    for group_index, count in enumerate(multiplicities):
        equivalent_positions.extend([group_index] * count)
    return equivalent_positions


def symmetrized_structure_from_cif_block(
    structure: Structure,
    block: CifBlock,
) -> SymmetrizedStructure:
    spacegroup = spacegroup_from_cif(block)
    if spacegroup is None:
        raise ValueError("Could not determine space group from CIF block.")

    spacegroup_ops = get_spacegroup_operations(block, spacegroup)
    wyckoff_letters_as_in_cif = get_wyckoff_letters_as_in_cif(block)

    if wyckoff_letters_as_in_cif is None:
        raise ValueError("Could not determine Wyckoff letters from CIF block.")

    wyckoff_multiplicities = get_wyckoff_multiplicities(block)
    if wyckoff_multiplicities is None:
        raise ValueError("Could not determine Wyckoff multiplicities from CIF block.")

    equivalent_positions = get_equivalent_positions(wyckoff_multiplicities)
    wyckoff_letters = get_wyckoff_letters(
        wyckoff_letters_as_in_cif, wyckoff_multiplicities
    )

    return SymmetrizedStructure(
        structure=structure,
        spacegroup=spacegroup_ops,
        equivalent_positions=equivalent_positions,
        wyckoff_letters=wyckoff_letters,
    )


def symmetrized_structures_from_cif(
    cif_path: PathLike | StringIO,
) -> tuple[list[None | Structure], list[None | SymmetrizedStructure]]:
    parser = CifParser(cif_path)
    structures: list[None | Structure] = [None] * len(parser._cif.data)
    symmetrized_structures: list[None | SymmetrizedStructure] = [None] * len(
        parser._cif.data
    )

    for i, block in enumerate(parser._cif.data.values()):
        try:
            structure = parser._get_structure(
                block,
                primitive=False,
                symmetrized=False,
                check_occu=True,
            )
        except (KeyError, ValueError):
            structure = None

        if structure is None:
            continue

        structures[i] = structure

        try:
            symmetrized_structure = symmetrized_structure_from_cif_block(
                structure=structure,
                block=block,
            )
        except ValueError:
            analyzer = SpacegroupAnalyzer(structure)
            symmetrized_structure = analyzer.get_symmetrized_structure()

        symmetrized_structures[i] = symmetrized_structure

    return structures, symmetrized_structures


def main():
    cif_path = "tests/files/cif/Graphite.cif"
    structure = Structure.from_file(cif_path)
    analyzer = SpacegroupAnalyzer(structure)
    symmetrized_structure = analyzer.get_symmetrized_structure()

    structures, symmetrized_structures = symmetrized_structures_from_cif(cif_path)
    print(f"Number of structures: {len(structures)}")
    print(f"Number of symmetrized structures: {len(symmetrized_structures)}")
    for i, symm_struct in enumerate(symmetrized_structures):
        if symm_struct:
            symm_struct.wyckoff_symbols


if __name__ == "__main__":
    main()

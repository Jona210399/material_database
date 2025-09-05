from typing import Any, Sequence

from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.cif import CifBlock, CifFile
from pymatgen.symmetry.analyzer import (
    PeriodicSite,
    SpacegroupAnalyzer,
    SymmetrizedStructure,
    SymmOp,
)

SIGNIFICANT_FIGURES = 8
FORMAT_STR = f"{{:.{SIGNIFICANT_FIGURES}f}}"


def analyzer_to_cif(analyzer: SpacegroupAnalyzer):
    refined_structure = analyzer.get_refined_structure()
    symmetry_operations = analyzer.get_symmetry_operations()
    symmetrized_structure = analyzer.get_symmetrized_structure()
    spacegroup = (
        analyzer.get_space_group_symbol(),
        analyzer.get_space_group_number(),
    )

    return get_cif(
        refined_structure,
        symmetry_operations,
        symmetrized_structure,
        spacegroup,
    )


def structure_to_cif(structure: Structure):
    spacegroup_analyzer = SpacegroupAnalyzer(structure)
    return analyzer_to_cif(spacegroup_analyzer)


def get_cif(
    refined_structure: Structure,
    symmetry_operations: list[SymmOp],
    symmetrized_structure: SymmetrizedStructure,
    spacegroup: tuple[str, int],
) -> str:
    blocks: dict[str, Any] = {}
    loops: list[list[str]] = []
    get_symmetry_cif_string(spacegroup, symmetry_operations, blocks)
    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])
    get_cell_cif_string(refined_structure.lattice, blocks)
    get_formula_cif_string(refined_structure.composition.element_composition, blocks)
    blocks["_cell_formula_units_Z"] = str(
        int(
            refined_structure.composition.element_composition.get_reduced_composition_and_factor()[
                1
            ]
        )
    )
    get_atom_site_cif_string(
        symmetrized_structure.equivalent_sites,
        symmetrized_structure.wyckoff_letters,
        refined_structure.composition,
        blocks,
    )
    loops.append(["_atom_type_symbol", "_atom_type_oxidation_number"])
    loops.append(
        [
            "_atom_site_type_symbol",
            "_atom_site_label",
            "_atom_site_symmetry_multiplicity",
            "_atom_site_Wyckoff_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        ],
    )

    return str(
        CifFile(
            {
                refined_structure.composition.reduced_formula: CifBlock(
                    blocks, loops, refined_structure.composition.reduced_formula
                )
            }
        )
    )


def get_symmetry_cif_string(
    spacegroup: tuple[str, int],
    symmetry_operations: list[SymmOp],
    blocks: dict[str, Any],
) -> None:
    blocks["_symmetry_space_group_name_H-M"] = spacegroup[0]
    blocks["_symmetry_Int_Tables_number"] = spacegroup[1]

    ops: list[str] = [
        SymmOp.from_rotation_and_translation(
            op.rotation_matrix, op.translation_vector
        ).as_xyz_str()
        for op in symmetry_operations
    ]
    blocks["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    blocks["_symmetry_equiv_pos_as_xyz"] = ops


def get_cell_cif_string(lattice: Lattice, blocks: dict[str, Any]) -> None:
    for cell_attr in ("a", "b", "c"):
        blocks[f"_cell_length_{cell_attr}"] = FORMAT_STR.format(
            getattr(lattice, cell_attr)
        )
    for cell_attr in ("alpha", "beta", "gamma"):
        blocks[f"_cell_angle_{cell_attr}"] = FORMAT_STR.format(
            getattr(lattice, cell_attr)
        )

    blocks["_cell_volume"] = FORMAT_STR.format(lattice.volume)


def get_formula_cif_string(
    element_composition: Composition, blocks: dict[str, Any]
) -> None:
    blocks["_chemical_formula_structural"] = element_composition.reduced_formula
    blocks["_chemical_formula_sum"] = element_composition.formula


def get_atom_site_cif_string(
    equivalent_sites: list[list[PeriodicSite]],
    wyckoff_letters: Sequence[str],
    composition: Composition,
    blocks: dict[str, Any],
):
    try:
        symbol_to_oxi_num = {
            str(el): float(el.oxi_state or 0) for el in sorted(composition.elements)
        }

    except (TypeError, AttributeError):
        symbol_to_oxi_num = {el.symbol: 0 for el in sorted(composition.elements)}

    atom_site_type_symbol = []
    atom_site_symmetry_multiplicity = []
    atom_site_wyckoff_symbol = []
    atom_site_fract_x = []
    atom_site_fract_y = []
    atom_site_fract_z = []
    atom_site_label = []
    atom_site_occupancy = []

    count = 0
    unique_sites = [
        (
            min(sites, key=lambda site: tuple(abs(x) for x in site.frac_coords)),
            len(sites),
            wyck,
        )
        for sites, wyck in zip(equivalent_sites, wyckoff_letters)
    ]
    for site, mult, wyckoff in sorted(
        unique_sites,
        key=lambda t: (
            t[0].species.average_electroneg,
            -t[1],
            t[0].a,
            t[0].b,
            t[0].c,
        ),
    ):
        for sp, occu in site.species.items():
            atom_site_type_symbol.append(str(sp))
            atom_site_symmetry_multiplicity.append(f"{mult}")
            atom_site_fract_x.append(FORMAT_STR.format(site.a))
            atom_site_fract_y.append(FORMAT_STR.format(site.b))
            atom_site_fract_z.append(FORMAT_STR.format(site.c))
            site_label = (
                site.label
                if site.label != site.species_string
                else f"{sp.symbol}{count}"
            )
            atom_site_label.append(site_label)
            atom_site_occupancy.append(str(occu))
            atom_site_wyckoff_symbol.append(wyckoff)
            count += 1

    blocks["_atom_type_symbol"] = list(symbol_to_oxi_num)
    blocks["_atom_type_oxidation_number"] = list(symbol_to_oxi_num.values())
    blocks["_atom_site_type_symbol"] = atom_site_type_symbol
    blocks["_atom_site_label"] = atom_site_label
    blocks["_atom_site_symmetry_multiplicity"] = atom_site_symmetry_multiplicity
    blocks["_atom_site_Wyckoff_symbol"] = atom_site_wyckoff_symbol
    blocks["_atom_site_fract_x"] = atom_site_fract_x
    blocks["_atom_site_fract_y"] = atom_site_fract_y
    blocks["_atom_site_fract_z"] = atom_site_fract_z
    blocks["_atom_site_occupancy"] = atom_site_occupancy

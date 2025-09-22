from collections import defaultdict
from dataclasses import dataclass

from numpy.typing import NDArray
from pymatgen.core import (
    Composition,
    Structure,
)
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure

from material_database.cif.parsing.atomic_sites import (
    parse_compositions,
)
from material_database.cif.parsing.block import CifBlock
from material_database.cif.parsing.coordinates import (
    get_equivalent_coordinates,
    get_matching_coordinate,
    parse_fractional_coordinates,
)
from material_database.cif.parsing.file import CifFile
from material_database.cif.parsing.lattice import (
    get_lattice,
    passes_minimal_lattice_thickness_check,
)
from material_database.cif.parsing.logger import (
    LOGGER,
    add_file_handler,
    disable_console_logging,
    enable_console_logging,
    remove_file_handler,
)
from material_database.cif.parsing.magnetic import parse_magnetic_cif
from material_database.cif.parsing.magnetic_cif import (
    is_magcif,
    is_magcif_incommensurate,
)
from material_database.cif.parsing.sanitization import sanitize_cif_block
from material_database.cif.parsing.spacegroup import (
    get_spacegroup_information,
)
from material_database.cif.parsing.wyckoffs import (
    get_wyckoff_letters_as_in_cif,
    get_wyckoff_multiplicities,
)


@dataclass
class ParsingSettings:
    """Settings for parsing CIF files."""

    occupancy_tolerance: float = 1.0
    site_tolerance: float = 1e-4
    fraction_tolerance: float = 1e-4
    composition_tolerance: float = 0.01
    min_lattice_thickness: float = 0.01
    primitive: bool = False
    symmetrized: bool = True
    check_occupancies: bool = False
    check_cif: bool = True
    log_file: str | None = None
    log_to_console: bool = True


def parse_cif(
    cif_block: CifBlock,
    parsing_settings: ParsingSettings = ParsingSettings(),
) -> list[Structure]:
    if parsing_settings.log_to_console:
        enable_console_logging()
    else:
        disable_console_logging()

    if parsing_settings.log_file is not None:
        add_file_handler(parsing_settings.log_file)
    else:
        remove_file_handler()

    if is_magcif(cif_block):
        return parse_magnetic_cif(cif_block, parsing_settings)

    if is_magcif_incommensurate(cif_block):
        raise NotImplementedError("Incommensurate magnetic structures not supported.")

    return _parse_standard_cif(cif_block, parsing_settings)


def _parse_standard_cif(
    cif_block: CifBlock, parsing_settings: ParsingSettings
) -> Structure | SymmetrizedStructure:
    if parsing_settings.symmetrized and parsing_settings.primitive:
        raise ValueError("Cannot set both symmetrized and primitive to True.")

    try:
        parsing_result = _parse_raw_cif(cif_block, parsing_settings)
    except ValueError as exc:
        LOGGER.warning(f"Error parsing CIF block: {exc}")
        return None

    structure = _build_structure(
        parsing_result,
        site_tolerance=parsing_settings.site_tolerance,
        return_symmetrized=parsing_settings.symmetrized,
    )

    if parsing_settings.primitive:
        structure = structure.get_primitive_structure()
        structure = structure.get_reduced_structure()

    if parsing_settings.check_cif:
        cif_failure_reason = check(
            cif_block,
            structure,
            composition_tolerance=parsing_settings.composition_tolerance,
        )
        if cif_failure_reason is not None:
            LOGGER.warning(cif_failure_reason)
            print("failure reason:", cif_failure_reason)

    return structure


def _parse_raw_cif(
    cif_block: CifBlock, parsing_settings: ParsingSettings
) -> (
    tuple[
        dict[tuple[float, float, float], Composition],
        dict[tuple[float, float, float], str],
        dict[tuple[float, float, float], str],
        dict[tuple[float, float, float], int],
        Lattice,
        list[SymmOp],
        dict[str, list[int]],
    ]
    | None
):
    sanitize_cif_block(cif_block, parsing_settings.fraction_tolerance)

    lattice = get_lattice(cif_block)
    if lattice is None:
        raise ValueError("Could not parse lattice from CIF block.")

    if not passes_minimal_lattice_thickness_check(
        lattice, parsing_settings.min_lattice_thickness
    ):
        raise ValueError(
            f"Lattice thickness is below the minimum threshold of {parsing_settings.min_lattice_thickness}."
        )

    spacegroup_infos = get_spacegroup_information(cif_block)

    if spacegroup_infos is None:
        LOGGER.warning("No spacegroup information found, assuming P1.")
        spacegroup = SpaceGroup.from_int_number(1)
        spacegroup_operations = spacegroup.symmetry_ops

    else:
        spacegroup_operations, spacegroup = spacegroup_infos

    compositions_and_atom_sites = parse_compositions(cif_block)

    if compositions_and_atom_sites is None:
        raise ValueError("Could not parse atom sites into compositions.")

    compositions, atom_sites, site_properties = compositions_and_atom_sites

    wyckoff_letters = get_wyckoff_letters_as_in_cif(cif_block) or ["Not Parsed"] * len(
        compositions
    )
    wyckoff_multiplicities = get_wyckoff_multiplicities(cif_block) or [None] * len(
        compositions
    )

    fractional_coordinates = parse_fractional_coordinates(cif_block, len(compositions))

    if fractional_coordinates is None:
        raise ValueError("Could not parse fractional coordinates.")

    coord_to_composition: dict[tuple[float, float, float], Composition] = defaultdict(
        Composition
    )

    coord_to_site_label: dict[tuple[float, float, float], list[str]] = defaultdict(
        lambda: ""
    )
    coord_to_wyckoff: dict[tuple[float, float, float], list[str]] = defaultdict(list)
    coord_to_multiplicity: dict[tuple[float, float, float], list[int]] = defaultdict(
        list
    )

    for i in range(len(compositions)):
        coordinate = fractional_coordinates[i]
        composition = compositions[i]
        atom_site = atom_sites[i]
        wyckoff_letter = wyckoff_letters[i]
        wyckoff_multiplicity = wyckoff_multiplicities[i]

        match = get_matching_coordinate(
            coord_to_composition,
            coordinate,
            symmetry_operations=spacegroup_operations,
            site_tolerance=parsing_settings.site_tolerance,
        )
        coord_to_composition[match] += composition
        coord_to_site_label[match] += atom_site
        coord_to_wyckoff[match].append(wyckoff_letter)
        coord_to_multiplicity[match].append(wyckoff_multiplicity)

    coord_to_composition = _verify_composition_occupancies(
        coord_to_composition,
        parsing_settings.check_occupancies,
        parsing_settings.occupancy_tolerance,
    )

    for coord, wyckoff in coord_to_wyckoff.items():
        if len(set(wyckoff)) > 1:
            LOGGER.warning(
                f"Multiple Wyckoff letters {wyckoff} found for coordinate {coord}. Using the first one: {wyckoff[0]}."
            )
        coord_to_wyckoff[coord] = wyckoff[0]

    for coord, mult in coord_to_multiplicity.items():
        if len(set(mult)) > 1:
            LOGGER.warning(
                f"Multiple Wyckoff multiplicities {mult} found for coordinate {coord}. Using the first one: {mult[0]}."
            )
        coord_to_multiplicity[coord] = mult[0]

    return (
        dict(coord_to_composition),
        dict(coord_to_site_label),
        dict(coord_to_wyckoff),
        dict(coord_to_multiplicity),
        lattice,
        spacegroup_operations,
        site_properties,
    )


def _build_structure(
    parsing_result: tuple[
        dict[tuple[float, float, float], Composition],
        dict[tuple[float, float, float], str],
        dict[tuple[float, float, float], str],
        dict[tuple[float, float, float], int],
        Lattice,
        list[SymmOp],
        dict[str, list[int]],
    ],
    site_tolerance: float,
    return_symmetrized: bool,
) -> Structure | None:
    (
        coord_to_composition,
        coord_to_site_label,
        coord_to_wyckoff,
        coord_to_multiplicity,
        lattice,
        spacegroup_operations,
        site_properties,
    ) = parsing_result

    all_compositions: list[Composition] = []
    all_coordinates: list[tuple[float, float, float]] = []
    all_labels: list[str] = []
    all_wyckoff_letters: list[str] = []

    compositions_to_coords: dict[Composition, list[tuple[float, float, float]]] = (
        defaultdict(list)
    )
    for coord, composition in coord_to_composition.items():
        compositions_to_coords[composition].append(coord)

    coords_to_equivalent_coords: dict[tuple[float, float, float], list[NDArray]] = {
        coord: get_equivalent_coordinates(
            coordinate=coord,
            symmetry_operations=spacegroup_operations,
            site_tolerance=site_tolerance,
        )
        for coord in coord_to_composition.keys()
    }

    for coord, multiplicity in coord_to_multiplicity.items():
        if multiplicity is None:
            coord_to_multiplicity[coord] = len(coords_to_equivalent_coords[coord])
            continue

        if multiplicity != (mult := len(coords_to_equivalent_coords[coord])):
            LOGGER.warning(
                f"Wyckoff multiplicity {multiplicity} does not match the number of equivalent positions {mult} for coordinate {coord}. Using the number of equivalent positions."
            )
            multiplicity = mult

        coord_to_multiplicity[coord] = multiplicity

    for composition, coordinates in compositions_to_coords.items():
        equivalent_coords = [
            equiv_coord
            for coord in coordinates
            for equiv_coord in coords_to_equivalent_coords[coord]
        ]

        labels = [
            coord_to_site_label[coord]
            for coord in coordinates
            for _ in coords_to_equivalent_coords[coord]
        ]

        wyckoffs = [
            coord_to_wyckoff[coord]
            for coord in coordinates
            for _ in coords_to_equivalent_coords[coord]
        ]

        all_coordinates.extend(equivalent_coords)
        all_compositions.extend(len(equivalent_coords) * [composition])
        all_labels.extend(labels)
        all_wyckoff_letters.extend(wyckoffs)

    struct = Structure(
        lattice,
        all_compositions,
        all_coordinates,
        site_properties=site_properties,
        labels=all_labels,
    )

    if not return_symmetrized:
        return struct.get_sorted_structure()

    if "Not Parsed" in all_wyckoff_letters:
        LOGGER.warning(
            "Wyckoff letters not parsed, falling back to SpacegroupAnalyzer."
        )

        analyzer = SpacegroupAnalyzer(struct)
        struct = analyzer.get_symmetrized_structure()
        return struct

    equivalent_indices = [
        i for i, mult in enumerate(coord_to_multiplicity.values()) for _ in range(mult)
    ]

    return SymmetrizedStructure(
        struct,
        spacegroup_operations,
        equivalent_indices,
        all_wyckoff_letters,
    )


def _verify_composition_occupancies(
    coord_to_composition: dict[tuple[float, float, float], Composition],
    check_occupancies: bool,
    occupancy_tolerance: float,
) -> dict[tuple[float, float, float], Composition]:
    # Check occupancy

    if not check_occupancies:
        return coord_to_composition

    for key, composition in coord_to_composition.items():
        total_occupancy = sum(composition.values())
        if total_occupancy > occupancy_tolerance:
            LOGGER.warning(
                f"Occupancy of {composition} is {total_occupancy} and exceeds the set tolerance of {occupancy_tolerance}. "
                "Rescaling total composition occupancy to 1.0."
            )
        if total_occupancy > 1.0:
            coord_to_composition[key] = composition / total_occupancy

    return coord_to_composition


def check(
    cif_block: CifBlock,
    structure: Structure,
    composition_tolerance: float,
) -> str | None:
    """Check whether a Structure created from CIF passes sanity checks.

    Checks:
        - Composition from CIF is valid
        - CIF composition contains only valid elements
        - CIF and structure contain the same elements (often hydrogens
            are omitted from CIFs, as their positions cannot be determined from
            X-ray diffraction, needs more difficult neutron diffraction)
        -  CIF and structure have same relative stoichiometry. Thus
            if CIF reports stoichiometry LiFeO, and the structure has
            composition (LiFeO)4, this check passes.

    Args:
        structure (Structure) : Structure created from CIF.

    Returns:
        str | None: If any check fails, return a human-readable str for the
            reason (e.g., which elements are missing). None if all checks pass.
    """

    cif_formula = None
    for key in ("_chemical_formula_sum", "_chemical_formula_structural"):
        if cif_block.get(key):
            cif_formula = cif_block[key]
            break

    # In case of missing CIF formula keys, get non-stoichiometric formula from
    # unique sites and skip relative stoichiometry check (added in gh-3628)
    check_stoichiometry = True
    if cif_formula is None and cif_block.get("_atom_site_type_symbol"):
        check_stoichiometry = False
        cif_formula = " ".join(cif_block["_atom_site_type_symbol"])

    try:
        cif_composition = Composition(cif_formula)
    except ValueError as exc:
        return f"Cannot determine chemical composition from CIF! {exc}"

    try:
        orig_comp = cif_composition.remove_charges().as_dict()
        struct_comp = structure.composition.remove_charges().as_dict()
    except Exception as exc:
        return str(exc)

    orig_comp_elts = {str(elt) for elt in orig_comp}
    struct_comp_elts = {str(elt) for elt in struct_comp}
    failure_reason: str | None = None

    # Hard failure: missing elements
    if orig_comp_elts != struct_comp_elts:
        missing = set(orig_comp_elts).difference(set(struct_comp_elts))
        addendum = "from PMG structure composition"
        if not missing:
            addendum = "from CIF-reported composition"
            missing = set(struct_comp_elts).difference(set(orig_comp_elts))
        missing_str = ", ".join([str(x) for x in missing])
        failure_reason = f"Missing elements {missing_str} {addendum}"

    elif any(struct_comp[elt] - orig_comp[elt] != 0 for elt in orig_comp):
        # Check that CIF/PMG stoichiometry has same relative ratios of elements
        if check_stoichiometry:
            ratios = {elt: struct_comp[elt] / orig_comp[elt] for elt in orig_comp_elts}

            same_stoich = all(
                abs(ratios[elt_a] - ratios[elt_b]) < composition_tolerance
                for elt_a in orig_comp_elts
                for elt_b in orig_comp_elts
            )

            if not same_stoich:
                failure_reason = f"Incorrect stoichiometry:\n  CIF={orig_comp}\n  PMG={struct_comp}\n  {ratios=}"
        else:
            LOGGER.warning(
                "Skipping relative stoichiometry check because CIF does not contain formula keys."
            )

    return failure_reason


if __name__ == "__main__":
    from io import StringIO

    import polars as pl
    from pymatgen.io.cif import CifParser
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cifs = pl.read_parquet("data/icsd/cif/icsd_000.parquet").select("cif").to_series()

    cif = cifs[0]
    print(cif)

    cif_file = CifFile.from_str(cif)
    print("#" * 120)

    cif_block = list(cif_file.data.values())[0]

    parsed_byparser = CifParser(StringIO(cif)).parse_structures()[0]
    parsed_byparser = SpacegroupAnalyzer(parsed_byparser).get_symmetrized_structure()
    print("Parsed by parser:")
    print(parsed_byparser)
    print("Equivalent indices of final structure:", parsed_byparser.equivalent_indices)

    print("-" * 120)
    print("Parsed by function:")

    parsed_byfunc = parse_cif(cif_block)
    print(parsed_byfunc)

    print("Equivalent indices of final structure:", parsed_byfunc.equivalent_indices)

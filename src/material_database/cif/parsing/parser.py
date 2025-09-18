import math
import re
import warnings
from dataclasses import dataclass
from itertools import groupby
from logging import WARNING, getLogger
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import (
    Composition,
    Element,
    PeriodicSite,
    Species,
    Structure,
)
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc

from material_database.cif.parsing.atomic_sites import (
    get_species_from_atom_site,
    parse_atom_sites,
)
from material_database.cif.parsing.block import CifBlock
from material_database.cif.parsing.file import CifFile
from material_database.cif.parsing.lattice import (
    check_min_lattice_thickness,
    get_lattice,
)
from material_database.cif.parsing.magnetic_cif import (
    is_magcif,
    is_magcif_incommensurate,
)
from material_database.cif.parsing.sanitization import sanitize_cif_block
from material_database.cif.parsing.spacegroup import (
    get_spacegroup_information,
)
from material_database.cif.parsing.utils import str2float
from material_database.cif.parsing.wyckoffs import (
    get_wyckoff_letters_as_in_cif,
    get_wyckoff_multiplicities,
)

LOGGER = getLogger(__name__)
LOGGER.setLevel(WARNING)


@dataclass
class ParsingSettings:
    """Settings for parsing CIF files."""

    occupancy_tolerance: float = 1.0
    site_tolerance: float = 1e-4
    fraction_tolerance: float = 1e-4
    check_cif: bool = False
    composition_tolerance: float = 0.01
    min_lattice_thickness: float = 0.01
    primitive: bool = True
    symmetrized: bool = True
    check_occu: bool = False


def parse_magnetic_cif(
    cif_block: CifBlock, parsing_settings: ParsingSettings
) -> list[Structure]:
    pass


def parse_cif(
    cif_block: CifBlock, parsing_settings: ParsingSettings = ParsingSettings()
) -> list[Structure]:
    pass

    if is_magcif(cif_block):
        return parse_magnetic_cif(cif_block, parsing_settings)

    if is_magcif_incommensurate(cif_block):
        raise NotImplementedError("Incommensurate magnetic structures not supported.")

    return parse_standard_cif(cif_block, parsing_settings)


def parse_occupancies(block: CifBlock) -> list[float] | None:
    KEYS_TO_TRY = [
        "_atom_site_occupancy",
        "_atom_site_occupancies",
        "_atom_site_occupancy_ ",
        "_atom_site_occupancies_",
    ]

    for key in KEYS_TO_TRY:
        occupancies: list[str] = block.get(key, None)
        if occupancies is not None:
            break

    if occupancies is None:
        LOGGER.warning("No occupancy data found in CIF block.")
        return None

    occupancies_as_floats = []
    for occupancy in occupancies:
        try:
            occupancy = str2float(occupancy.strip())
            occupancies_as_floats.append(occupancy)
        except ValueError as e:
            LOGGER.warning(f"Invalid occupancy value '{occupancy}': {e}")
            return None

    return occupancies_as_floats


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


def parse_standard_cif(
    cif_block: CifBlock, parsing_settings: ParsingSettings
) -> list[Structure]:
    sanitize_cif_block(cif_block, parsing_settings.fraction_tolerance)

    lattice = get_lattice(cif_block)
    check_min_lattice_thickness(lattice, parsing_settings.min_lattice_thickness)

    spacegroup_infos = get_spacegroup_information(cif_block)
    if spacegroup_infos is None:
        LOGGER.warning("No spacegroup information found, assuming P1.")
        symmetry_operations = SpaceGroup.from_int_number(1).symmetry_ops

    spacegroup_operations, spacegroup = spacegroup_infos
    symmetry_operations = spacegroup_operations  # type:ignore[assignment]

    oxidation_states = _parse_oxidation_states(cif_block)

    atom_sites = parse_atom_sites(cif_block)
    if atom_sites is None:
        raise ValueError("Could not parse atom sites.")

    occupancies = parse_occupancies(cif_block)

    if occupancies is None:
        raise ValueError("Could not parse occupancies.")

    wyckoff_letters = get_wyckoff_letters_as_in_cif(cif_block) or ["Not Parsed"] * len(
        atom_sites
    )
    wyckoff_multiplicities = get_wyckoff_multiplicities(cif_block) or [1] * len(
        atom_sites
    )

    fractional_coordinates = parse_fractional_coordinates(cif_block, len(atom_sites))

    if fractional_coordinates is None:
        raise ValueError("Could not parse fractional coordinates.")

    coord_to_species: dict[tuple[float, float, float], Composition] = {}
    coord_to_site_label: dict[tuple[float, float, float], str] = {}
    coord_to_wyckoff: dict[tuple[float, float, float], str] = {}
    coord_to_multiplicity: dict[tuple[float, float, float], int] = {}

    for idx, atom_site in enumerate(atom_sites):
        species = get_species_from_atom_site(atom_site, oxidation_states)
        occupancy = occupancies[idx]
        coordinate = fractional_coordinates[idx]

        # Create Composition
        composition_dict: dict[Species | str, float] = {species: max(occupancy, 1e-8)}

        num_h = get_num_implicit_hydrogens(atom_site)
        if num_h > 0:
            composition_dict["H"] = num_h
            LOGGER.warning(
                "Structure has implicit hydrogens defined, parsed structure unlikely to be "
                "suitable for use in calculations unless hydrogens added."
            )

        composition = Composition(composition_dict)

        # Find matching site by coordinate
        match: tuple[float, float, float] | Literal[False] = get_matching_coordinate(
            coord_to_species,
            coordinate,
            symmetry_operations=symmetry_operations,
            site_tolerance=parsing_settings.site_tolerance,
        )
        if not match:
            coord_to_species[coordinate] = composition
            coord_to_site_label[coordinate] = atom_site
            coord_to_wyckoff[coordinate] = wyckoff_letters[idx]
            coord_to_multiplicity[coordinate] = wyckoff_multiplicities[idx]

        else:
            coord_to_species[match] += composition
            coord_to_site_label[match] += atom_site

            current_wyckoff = coord_to_wyckoff[match]
            if current_wyckoff != wyckoff_letters[idx]:
                raise ValueError("Mismatched Wyckoff letters for same site.")

            coord_to_wyckoff[match] = wyckoff_letters[idx]

            current_multiplicity = coord_to_multiplicity[match]
            if current_multiplicity != wyckoff_multiplicities[idx]:
                raise ValueError("Mismatched multiplicities for same site.")

            coord_to_multiplicity[match] = wyckoff_multiplicities[idx]

    # Check occupancy
    _sum_occupancies: list[float] = [
        sum(comp.values())
        for comp in coord_to_species.values()
        if set(comp.elements) != {Element("O"), Element("H")}
    ]

    if any(occu > 1.0 for occu in _sum_occupancies):
        LOGGER.warning(
            f"Some occupancies ({list(filter(lambda x: x > 1, _sum_occupancies))}) sum to > 1! If they are within "
            "the occupancy_tolerance, they will be rescaled. "
            f"The current occupancy_tolerance is set to: {parsing_settings.occupancy_tolerance}"
        )

    # Collect info for building Structure
    all_species: list[Composition] = []
    all_species_noedit: list[Composition] = []
    all_coords: list[tuple[float, float, float]] = []
    all_hydrogens: list[float] = []
    all_labels: list[str] = []
    all_wyckoff_letters: list[str] = []

    if coord_to_species.items():
        grouped = groupby(
            coord_to_species.items(),
            key=lambda x: x[1],
        )

        for composition, group in grouped:
            tmp_coords: list[tuple[float, float, float]] = [site[0] for site in group]

            coords, new_labels, new_wyckoff_letters = _unique_coords(
                tmp_coords,
                site_tolerance=parsing_settings.site_tolerance,
                symmetry_operations=symmetry_operations,
                labels=coord_to_site_label,
                wyckoff_letters=coord_to_wyckoff,
            )

            if set(composition.elements) == {Element("O"), Element("H")}:
                # O with implicit hydrogens
                im_h = composition["H"]
                species = Composition({"O": composition["O"]})
            else:
                im_h = 0
                species = composition

            all_hydrogens.extend(len(coords) * [im_h])
            all_coords.extend(coords)  # type:ignore[arg-type]
            all_species.extend(len(coords) * [species])
            all_labels.extend(new_labels)
            all_wyckoff_letters.extend(new_wyckoff_letters)

        # Scale occupancies if necessary
        all_species_noedit = (
            all_species.copy()
        )  # save copy before scaling in case of check_occu=False, used below
        for idx, species in enumerate(all_species):
            total_occu = sum(species.values())
            if (
                parsing_settings.check_occu
                and total_occu > parsing_settings.occupancy_tolerance
            ):
                raise ValueError(f"Occupancy {total_occu} exceeded tolerance.")

            if total_occu > 1:
                all_species[idx] = species / total_occu

    if all_species and len(all_species) == len(all_coords):
        site_properties: dict[str, list] = {}
        if any(all_hydrogens):
            if len(all_hydrogens) != len(all_coords):
                raise ValueError("lengths of all_hydrogens and all_coords mismatch")
            site_properties["implicit_hydrogens"] = all_hydrogens

        if not site_properties:
            site_properties = {}

        if any(all_labels):
            if len(all_labels) != len(all_species):
                raise ValueError("lengths of all_labels and all_species mismatch")
        else:
            all_labels = None  # type: ignore[assignment]

        struct: Structure = Structure(
            lattice,  # type:ignore[arg-type]
            all_species,
            all_coords,
            site_properties=site_properties,
            labels=all_labels,
        )

        if parsing_settings.symmetrized:
            try:
                equivalent_indices = [
                    i
                    for i, mult in enumerate(coord_to_multiplicity.values())
                    for _ in range(mult)
                ]
                struct = SymmetrizedStructure(
                    struct,
                    spacegroup_operations,
                    equivalent_indices,
                    all_wyckoff_letters,
                )

            except ValueError:
                analyzer = SpacegroupAnalyzer(struct)
                struct = analyzer.get_symmetrized_structure()

        if not parsing_settings.check_occu:
            if lattice is None:
                raise RuntimeError("Cannot generate Structure with lattice as None.")

            for idx in range(len(struct)):
                struct[idx] = PeriodicSite(
                    all_species_noedit[idx],
                    all_coords[idx],
                    lattice,
                    properties=site_properties,
                    label=all_labels[idx],
                    skip_checks=True,
                )

        if parsing_settings.symmetrized or not parsing_settings.check_occu:
            return struct

        struct = struct.get_sorted_structure()

        if parsing_settings.primitive:
            struct = struct.get_primitive_structure()
            struct = struct.get_reduced_structure()

        if parsing_settings.check_cif:
            cif_failure_reason = check(struct)
            if cif_failure_reason is not None:
                warnings.warn(cif_failure_reason, stacklevel=2)

        return struct
    return None


def _unique_coords(
    coords: list[tuple[float, float, float]],
    symmetry_operations: list[SymmOp],
    site_tolerance: float,
    labels: dict[tuple[float, float, float], str] | None = None,
    wyckoff_letters: dict[tuple[float, float, float], str] | None = None,
) -> tuple[
    list[NDArray],
    list[str],
    list[str],
]:
    """Generate unique coordinates using coordinates and symmetry positions."""
    coords_out: list[NDArray] = []
    labels_out: list[str] = []
    wyckoffs_out: list[str] = []

    labels = labels or {}
    wyckoff_letters = wyckoff_letters or {}

    for tmp_coord in coords:
        for op in symmetry_operations:
            coord = op.operate(tmp_coord)
            coord = np.array([i - math.floor(i) for i in coord])
            if not in_coord_list_pbc(coords_out, coord, atol=site_tolerance):
                coords_out.append(coord)
                labels_out.append(labels.get(tmp_coord, "no_label"))
                wyckoffs_out.append(wyckoff_letters.get(tmp_coord, "Not Parsed"))

    return coords_out, labels_out, wyckoffs_out


def _parse_oxidation_states(cif_block: CifBlock) -> dict[str, float] | None:
    try:
        oxi_states = {
            cif_block["_atom_type_symbol"][i]: str2float(
                cif_block["_atom_type_oxidation_number"][i]
            )
            for i in range(len(cif_block["_atom_type_symbol"]))
        }
        # Attempt to strip oxidation state from _atom_type_symbol
        # in case the label does not contain an oxidation state
        for idx, symbol in enumerate(cif_block["_atom_type_symbol"]):
            oxi_states[re.sub(r"\d?[\+,\-]?$", "", symbol)] = str2float(
                cif_block["_atom_type_oxidation_number"][idx]
            )

    except (ValueError, KeyError):
        oxi_states = None
    return oxi_states


def get_matching_coordinate(
    coord_to_species: dict[tuple[float, float, float], Composition],
    coord: tuple[float, float, float],
    symmetry_operations: list[SymmOp],
    site_tolerance: float,
) -> tuple[float, float, float] | Literal[False]:
    """Find site by coordinate."""
    coords: list[tuple[float, float, float]] = list(coord_to_species.keys())
    for op in symmetry_operations:
        frac_coord = op.operate(coord)
        indices: NDArray = find_in_coord_list_pbc(
            coords,
            frac_coord,
            atol=site_tolerance,
        )
        if len(indices) > 0:
            return coords[indices[0]]
    return False


def get_num_implicit_hydrogens(symbol: str) -> int:
    """Get number of implicit hydrogens."""
    num_h = {"Wat": 2, "wat": 2, "O-H": 1}
    return num_h.get(symbol[:3], 0)


def check(
    cif_file: CifFile,
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
    cif_as_dict = cif_file.as_dict()
    head_key = next(iter(cif_as_dict))

    cif_formula = None
    for key in ("_chemical_formula_sum", "_chemical_formula_structural"):
        if cif_as_dict[head_key].get(key):
            cif_formula = cif_as_dict[head_key][key]
            break

    # In case of missing CIF formula keys, get non-stoichiometric formula from
    # unique sites and skip relative stoichiometry check (added in gh-3628)
    check_stoichiometry = True
    if cif_formula is None and cif_as_dict[head_key].get("_atom_site_type_symbol"):
        check_stoichiometry = False
        cif_formula = " ".join(cif_as_dict[head_key]["_atom_site_type_symbol"])

    try:
        cif_composition = Composition(cif_formula)
    except Exception as exc:
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

    cif = cifs[9]
    print(cif)

    cif_file = CifFile.from_str(cif)
    print("#" * 120)

    cif_block = list(cif_file.data.values())[0]

    parsed_byparser = CifParser(StringIO(cif)).parse_structures()[0]
    parsed_byparser = SpacegroupAnalyzer(parsed_byparser).get_symmetrized_structure()
    print("Parsed by parser:")
    print(parsed_byparser)

    parsed_byfunc = parse_cif(cif_block)
    print("Parsed by function:")
    print(parsed_byfunc)

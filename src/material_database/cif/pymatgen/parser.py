import math
import os
import re
import warnings
from functools import partial
from io import StringIO
from itertools import groupby
from logging import WARNING, getLogger
from typing import Literal

import numpy as np
from monty.serialization import loadfn
from numpy.typing import NDArray
from pymatgen.core import (
    Composition,
    DummySpecies,
    Element,
    Lattice,
    PeriodicSite,
    Species,
    Structure,
    get_el_sp,
)
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.io.core import ParseError
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
from typing_extensions import Self

from material_database.cif.parsing.spacegroup import (
    get_spacegroup_information,
)
from material_database.cif.parsing.wyckoffs import parse_wyckoff_info
from material_database.cif.pymatgen.block import CifBlock
from material_database.cif.pymatgen.file import CifFile
from material_database.cif.pymatgen.magnetic_cif import (
    get_magnetic_symops,
    is_magcif,
    is_magcif_incommensurate,
    parse_magnetic_moments,
)
from material_database.cif.pymatgen.parse_lattice import (
    check_min_lattice_thickness,
    get_lattice,
)
from material_database.cif.pymatgen.sanitization import sanitize_cif_block
from material_database.cif.pymatgen.utils import safe_str2float, str2float

LOGGER = getLogger(__name__)
LOGGER.setLevel(WARNING)


class CifParser:
    """
    CIF file parser. Attempt to fix CIFs that are out-of-spec, but will issue warnings
    if corrections applied. These are also stored in the CifParser's warnings attribute.
    CIF file parser. Attempt to fix CIFs that are out-of-spec, but will issue warnings
    if corrections applied. These are also stored in the CifParser's errors attribute.
    """

    def __init__(
        self,
        cif_file: CifFile,
        occupancy_tolerance: float = 1.0,
        site_tolerance: float = 1e-4,
        frac_tolerance: float = 1e-4,
        check_cif: bool = True,
        comp_tol: float = 0.01,
    ) -> None:
        """
        Args:
            filename (PathLike): CIF file, gzipped or bzipped CIF files are fine too.
            occupancy_tolerance (float): If total occupancy of a site is between
                1 and occupancy_tolerance, it will be scaled down to 1.
            site_tolerance (float): This tolerance is used to determine if two sites are at the same position,
                in which case they will be combined to a single disordered site. Defaults to 1e-4.
            frac_tolerance (float): This tolerance is used to determine is a coordinate should be rounded to an ideal
                value. e.g. 0.6667 is rounded to 2/3. This is desired if symmetry operations are going to be applied.
                However, for very large CIF files, this may need to be set to 0.
            check_cif (bool): Whether to check that stoichiometry reported in CIF matches
                that of resulting Structure, and whether elements are missing. Defaults to True.
            comp_tol (float): Tolerance for how closely stoichiometries of CIF file and pymatgen should match.
                Defaults to 0.01. Context: Experimental CIF files often don't report hydrogens positions due to being
                hard-to-locate with X-rays. pymatgen warns if the stoichiometry of the CIF file and the Structure
                don't match to within comp_tol.
        """

        # Take tolerances
        self._occupancy_tolerance = occupancy_tolerance
        self._site_tolerance = site_tolerance
        self._frac_tolerance = frac_tolerance
        # Options related to checking CIFs for missing elements
        # or incorrect stoichiometries
        self.check_cif = check_cif
        self.comp_tol = comp_tol

        self._cif = cif_file

        self.is_magcif = is_magcif(self._cif)
        self.is_magcif_incommensurate = is_magcif_incommensurate(self._cif)

        # Store warnings during parsing
        self.warnings: list[str] = []

        # Pass individual CifBlocks to _sanitize_data
        for key in self._cif.data:
            self._cif.data[key] = sanitize_cif_block(
                self._cif.data[key],
                self.is_magcif,
                self._frac_tolerance,
            )

    @classmethod
    def from_str(cls, cif_string: str, **kwargs) -> Self:
        """Create a CifParser from a string.

        Args:
            cif_string (str): String representation of a CIF.

        Returns:
            CifParser
        """
        return cls(StringIO(cif_string), **kwargs)

    def _unique_coords(
        self,
        coords: list[tuple[float, float, float]],
        symmetry_operations: list[SymmOp],
        magmoms: list[Magmom] | None = None,
        lattice: Lattice | None = None,
        labels: dict[tuple[float, float, float], str] | None = None,
    ) -> tuple[list[NDArray], list[Magmom], list[str]]:
        """Generate unique coordinates using coordinates and symmetry
        positions, and their corresponding magnetic moments if supplied.
        """
        coords_out: list[NDArray] = []
        labels_out: list[str] = []
        labels = labels or {}

        if magmoms:
            if len(magmoms) != len(coords):
                raise ValueError("Length of magmoms and coords don't match.")

            magmoms_out: list[Magmom] = []
            for tmp_coord, tmp_magmom in zip(coords, magmoms, strict=True):
                for op in symmetry_operations:
                    coord = op.operate(tmp_coord)
                    coord = np.array([i - math.floor(i) for i in coord])
                    if isinstance(op, MagSymmOp):
                        # Up to this point, magmoms have been defined relative
                        # to crystal axis. Now convert to Cartesian and into
                        # a Magmom object.
                        if lattice is None:
                            raise ValueError("Lattice cannot be None.")
                        magmom = Magmom.from_moment_relative_to_crystal_axes(
                            op.operate_magmom(tmp_magmom), lattice=lattice
                        )
                    else:
                        magmom = Magmom(tmp_magmom)

                    if not in_coord_list_pbc(
                        coords_out, coord, atol=self._site_tolerance
                    ):
                        coords_out.append(coord)
                        magmoms_out.append(magmom)
                        labels_out.append(labels.get(tmp_coord, "no_label"))

            return coords_out, magmoms_out, labels_out

        for tmp_coord in coords:
            for op in symmetry_operations:
                coord = op.operate(tmp_coord)
                coord = np.array([i - math.floor(i) for i in coord])
                if not in_coord_list_pbc(coords_out, coord, atol=self._site_tolerance):
                    coords_out.append(coord)
                    labels_out.append(labels.get(tmp_coord, "no_label"))

        dummy_magmoms = [Magmom(0)] * len(coords_out)
        return coords_out, dummy_magmoms, labels_out

    def _get_structure(
        self,
        data: CifBlock,
        primitive: bool,
        symmetrized: bool,
        check_occu: bool = False,
        min_thickness: float = 0.01,
    ) -> Structure | None:
        """Generate structure from part of the CIF.

        Args:
            data (CifBlock): The data block to parse.
            primitive (bool): Whether to return primitive unit cells.
            symmetrized (bool): Whether to return SymmetrizedStructure.
            check_occu (bool): Whether to check site for unphysical occupancy > 1.
            min_thickness (float): Minimum thickness in Angstrom to consider structure as valid.
                This is added to guard against unphysical small/thin structure,
                which could result in infinite loop for searching near neighbours.

        Returns:
            Structure or None if not found.
        """

        lattice = get_lattice(data)
        check_min_lattice_thickness(lattice, min_thickness)

        if self.is_magcif_incommensurate:
            raise NotImplementedError(
                "Incommensurate structures not currently supported."
            )
        if self.is_magcif:
            if lattice is None:
                raise ValueError(
                    "Magmoms given in terms of crystal axes in magCIF spec."
                )
            symmetry_operations = get_magnetic_symops(data)
            magmoms = parse_magnetic_moments(data)

        else:
            spacegroup_infos = get_spacegroup_information(data)
            if spacegroup_infos is None:
                LOGGER.warning("No spacegroup information found, assuming P1.")
                symmetry_operations = SpaceGroup.from_int_number(1).symmetry_ops

            spacegroup_operations, spacegroup = spacegroup_infos
            symmetry_operations = spacegroup_operations  # type:ignore[assignment]

            magmoms = {}

        oxi_states = _parse_oxidation_states(data)

        coord_to_species: dict[tuple[float, float, float], Composition] = {}
        coord_to_magmoms: dict[tuple[float, float, float], NDArray] = {}
        coord_to_site_label: dict[tuple[float, float, float], str] = {}
        atom_site_labels = data.get("_atom_site_label", [])
        atom_site_type_symbols = data.get("_atom_site_type_symbol", None)
        occupancies = data.get("_atom_site_occupancy", [1.0] * len(atom_site_labels))
        occupancies = [safe_str2float(o) for o in occupancies]
        atom_sites = atom_site_type_symbols or atom_site_labels

        for idx, atom_site in enumerate(atom_sites):
            element_symbol = _parse_symbol(atom_site)
            if not element_symbol:
                continue

            # Get oxidation state
            if oxi_states is not None:
                o_s = oxi_states.get(element_symbol, 0)
                o_s = oxi_states.get(atom_site, o_s)

                try:
                    element = Species(element_symbol, o_s)
                except (ValueError, ParseError):
                    element = DummySpecies(element_symbol, o_s)
            else:
                element = get_el_sp(element_symbol)  # type: ignore[assignment]

            occupancy = occupancies[idx]

            # If don't check_occu or the occupancy is greater than 0, create comp_dict
            if not check_occu or occupancy > 0.0:
                # Create site coordinate
                coord = tuple(
                    (str2float(data[f"_atom_site_fract_{axis}"][idx])) for axis in "xyz"
                )

                # Create Composition
                comp_dict: dict[Species | str, float] = {element: max(occupancy, 1e-8)}

                num_h = get_num_implicit_hydrogens(atom_site)
                if num_h > 0:
                    comp_dict["H"] = num_h
                    LOGGER.warning(
                        "Structure has implicit hydrogens defined, parsed structure unlikely to be "
                        "suitable for use in calculations unless hydrogens added."
                    )

                comp = Composition(comp_dict)

                # Find matching site by coordinate
                match: tuple[float, float, float] | Literal[False] = (
                    get_matching_coordinate(
                        coord_to_species,
                        coord,
                        symmetry_operations=symmetry_operations,
                        site_tolerance=self._site_tolerance,
                    )
                )
                if not match:
                    coord_to_species[coord] = comp
                    coord_to_magmoms[coord] = magmoms.get(
                        atom_site, np.array([0, 0, 0])
                    )
                    coord_to_site_label[coord] = atom_site

                else:
                    coord_to_species[match] += comp
                    # Disordered magnetic currently not supported
                    coord_to_magmoms[match] = None  # type:ignore[assignment]
                    coord_to_site_label[match] = atom_site

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
                f"The current occupancy_tolerance is set to: {self._occupancy_tolerance}"
            )

        # Collect info for building Structure
        all_species: list[Composition] = []
        all_species_noedit: list[Composition] = []
        all_coords: list[tuple[float, float, float]] = []
        all_magmoms: list[Magmom] = []
        all_hydrogens: list[float] = []
        equivalent_indices: list[int] = []
        all_labels: list[str] = []

        # Check if a magCIF file is disordered
        if self.is_magcif:
            for val in coord_to_magmoms.values():
                if val is None:
                    # Proposed solution to this is to instead store magnetic
                    # moments as Species 'spin' property, instead of site
                    # property, but this introduces ambiguities for end user
                    # (such as unintended use of `spin` and Species will have
                    # fictitious oxidation state).
                    raise NotImplementedError(
                        "Disordered magnetic structures not currently supported."
                    )

        if coord_to_species.items():
            for idx, (comp, group) in enumerate(
                groupby(
                    sorted(coord_to_species.items(), key=lambda x: x[1]),
                    key=lambda x: x[1],
                )
            ):
                tmp_coords: list[tuple[float, float, float]] = [
                    site[0] for site in group
                ]
                tmp_magmom: list[Magmom] = [
                    coord_to_magmoms[tmp_coord] for tmp_coord in tmp_coords
                ]

                if self.is_magcif:
                    coords, _magmoms, new_labels = self._unique_coords(
                        tmp_coords,
                        symmetry_operations=symmetry_operations,
                        magmoms=tmp_magmom,
                        labels=coord_to_site_label,
                        lattice=lattice,
                    )
                else:
                    coords, _magmoms, new_labels = self._unique_coords(
                        tmp_coords,
                        symmetry_operations=symmetry_operations,
                        labels=coord_to_site_label,
                    )

                if set(comp.elements) == {Element("O"), Element("H")}:
                    # O with implicit hydrogens
                    im_h = comp["H"]
                    species = Composition({"O": comp["O"]})
                else:
                    im_h = 0
                    species = comp

                # The following might be a more natural representation of equivalent indices,
                # but is not in the format expect by SymmetrizedStructure:
                #   equivalent_indices.append(list(range(len(all_coords), len(coords)+len(all_coords))))
                # The above gives a list like:
                #   [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]] where the
                # integers are site indices, whereas the version used below will give a version like:
                #   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
                # which is a list in the same order as the sites, but where if a site has the same integer
                # it is equivalent.
                equivalent_indices += len(coords) * [idx]

                all_hydrogens.extend(len(coords) * [im_h])
                all_coords.extend(coords)  # type:ignore[arg-type]
                all_species.extend(len(coords) * [species])
                all_magmoms.extend(_magmoms)
                all_labels.extend(new_labels)

            # Scale occupancies if necessary
            all_species_noedit = (
                all_species.copy()
            )  # save copy before scaling in case of check_occu=False, used below
            for idx, species in enumerate(all_species):
                total_occu = sum(species.values())
                if check_occu and total_occu > self._occupancy_tolerance:
                    raise ValueError(f"Occupancy {total_occu} exceeded tolerance.")

                if total_occu > 1:
                    all_species[idx] = species / total_occu

        if (
            all_species
            and len(all_species) == len(all_coords)
            and len(all_species) == len(all_magmoms)
        ):
            site_properties: dict[str, list] = {}
            if any(all_hydrogens):
                if len(all_hydrogens) != len(all_coords):
                    raise ValueError("lengths of all_hydrogens and all_coords mismatch")
                site_properties["implicit_hydrogens"] = all_hydrogens

            if self.is_magcif:
                site_properties["magmom"] = all_magmoms

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

            if symmetrized:
                try:
                    wyckoffs = parse_wyckoff_info(data)
                    struct = SymmetrizedStructure(
                        struct,
                        spacegroup_operations,
                        equivalent_indices,
                        wyckoffs,
                    )
                except ValueError:
                    analyzer = SpacegroupAnalyzer(struct)
                    struct = analyzer.get_symmetrized_structure()

            if not check_occu:
                if lattice is None:
                    raise RuntimeError(
                        "Cannot generate Structure with lattice as None."
                    )

                for idx in range(len(struct)):
                    struct[idx] = PeriodicSite(
                        all_species_noedit[idx],
                        all_coords[idx],
                        lattice,
                        properties=site_properties,
                        label=all_labels[idx],
                        skip_checks=True,
                    )

            if symmetrized or not check_occu:
                return struct

            struct = struct.get_sorted_structure()

            if primitive and self.is_magcif:
                struct = struct.get_primitive_structure(use_site_props=True)
            elif primitive:
                struct = struct.get_primitive_structure()
                struct = struct.get_reduced_structure()

            if self.check_cif:
                cif_failure_reason = self.check(struct)
                if cif_failure_reason is not None:
                    warnings.warn(cif_failure_reason, stacklevel=2)

            return struct
        return None

    def parse_structures(
        self,
        primitive: bool | None = None,
        symmetrized: bool = False,
        check_occu: bool = True,
        on_error: Literal["ignore", "warn", "raise"] = "warn",
    ) -> list[Structure]:
        """Return list of structures in CIF file.

        Args:
            primitive (bool): Whether to return primitive unit cells.
                Defaults to False. With magnetic CIF files, will return primitive
                magnetic cell which may be larger than nuclear primitive cell.
            symmetrized (bool): Whether to return a SymmetrizedStructure which will
                include the equivalent indices and symmetry operations used to
                create the Structure as provided by the CIF (if explicit symmetry
                operations are included in the CIF) or generated from information
                in the CIF (if only space group labels are provided). Note that
                currently Wyckoff labels and space group labels or numbers are
                not included in the generated SymmetrizedStructure, these will be
                notated as "Not Parsed" or -1 respectively.
            check_occu (bool): Whether to check site for unphysical occupancy > 1.
                Useful for experimental results in which occupancy was allowed to
                refine to unphysical values. Warning: unphysical occupancies are
                incompatible with many pymatgen features. Defaults to True.
            on_error ("ignore" | "warn" | "raise"): What to do in case of KeyError
                or ValueError while parsing CIF file. Defaults to "warn".

        Returns:
            list[Structure]: All structures in CIF file.
        """
        if primitive is None:
            primitive = False
            warnings.warn(
                "The default value of primitive was changed from True to False in "
                "https://github.com/materialsproject/pymatgen/pull/3419. CifParser now returns the cell "
                "in the CIF file as is. If you want the primitive cell, please set primitive=True explicitly.",
                stacklevel=2,
            )

        if primitive and symmetrized:
            raise ValueError(
                "Using both 'primitive' and 'symmetrized' arguments is not currently supported "
                "since unexpected behavior might result."
            )

        structures: list[Structure] = []
        for idx, data in enumerate(self._cif.data.values()):
            try:
                if struct := self._get_structure(
                    data, primitive, symmetrized, check_occu=check_occu
                ):
                    structures.append(struct)

            except (KeyError, ValueError) as exc:
                msg = f"No structure parsed for section {idx + 1} in CIF.\n{exc}"
                if on_error == "raise":
                    raise ValueError(msg) from exc
                if on_error == "warn":
                    warnings.warn(msg, stacklevel=2)
                self.warnings.append(msg)

        if self.warnings and on_error == "warn":
            warnings.warn(
                "Issues encountered while parsing CIF: " + "\n".join(self.warnings),
                stacklevel=2,
            )

        if not structures:
            raise ValueError("Invalid CIF file with no structures!")
        return structures

    def as_dict(self) -> dict:
        """MSONable dict."""
        dct: dict = {}
        for key, val in self._cif.data.items():
            dct[key] = {}
            for sub_key, sub_val in val.data.items():
                dct[key][sub_key] = sub_val
        return dct

    @property
    def has_errors(self) -> bool:
        """Whether there are errors/warnings detected in CIF parsing."""
        return len(self.warnings) > 0

    def check(self, structure: Structure) -> str | None:
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
        cif_as_dict = self.as_dict()
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
                ratios = {
                    elt: struct_comp[elt] / orig_comp[elt] for elt in orig_comp_elts
                }

                same_stoich = all(
                    abs(ratios[elt_a] - ratios[elt_b]) < self.comp_tol
                    for elt_a in orig_comp_elts
                    for elt_b in orig_comp_elts
                )

                if not same_stoich:
                    failure_reason = f"Incorrect stoichiometry:\n  CIF={orig_comp}\n  PMG={struct_comp}\n  {ratios=}"
            else:
                self.warnings += [
                    "Skipping relative stoichiometry check because CIF does not contain formula keys."
                ]

        return failure_reason


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


def _parse_symbol(symbol: str) -> str | None:
    """Parse a string with a symbol to extract a string representing an element.

    Args:
        sym (str): A symbol to be parsed.

    Returns:
        A string for the parsed symbol. None if no parsing was possible.
    """
    # Common representations for elements/water in CIF files
    # TODO: fix inconsistent handling of water
    special_syms = {
        "Hw": "H",
        "Ow": "O",
        "Wat": "O",
        "wat": "O",
        "OH": "",
        "OH2": "",
        "NO3": "N",
    }

    parsed_sym = None
    # Try with special symbols, otherwise check the first two letters,
    # then the first letter alone. If everything fails try extracting the
    # first letter.
    m_sp = re.match("|".join(special_syms), symbol)
    if m_sp:
        parsed_sym = special_syms[m_sp.group()]
    elif Element.is_valid_symbol(symbol[:2].title()):
        parsed_sym = symbol[:2].title()
    elif Element.is_valid_symbol(symbol[0].upper()):
        parsed_sym = symbol[0].upper()
    elif match := re.match(r"w?[A-Z][a-z]*", symbol):
        parsed_sym = match.group()

    if parsed_sym is not None and (m_sp or not re.match(rf"{parsed_sym}\d*", symbol)):
        msg = f"{symbol} parsed as {parsed_sym}"
        LOGGER.warning(msg)

    return parsed_sym


def get_symops(data: CifBlock) -> list[SymmOp]:
    """
    Get the symmetry operations, in order to generate symmetry
    equivalent positions. If no symops are present, the space
    group symbol is parsed, and symops are generated.
    """
    sym_ops = []
    for symmetry_label in (
        "_symmetry_equiv_pos_as_xyz",
        "_symmetry_equiv_pos_as_xyz_",
        "_space_group_symop_operation_xyz",
        "_space_group_symop_operation_xyz_",
    ):
        if data.data.get(symmetry_label):
            xyz = data.data.get(symmetry_label)
            if xyz is None:
                raise RuntimeError("Cannot get symmetry_label.")

            if isinstance(xyz, str):
                LOGGER.warning("A 1-line symmetry op P1 CIF is detected!")
                xyz = [xyz]
            try:
                sym_ops = [SymmOp.from_xyz_str(s) for s in xyz]
                break
            except ValueError:
                continue

    sub_space_group = partial(re.sub, r"[\s_]", "")

    space_groups = {
        sub_space_group(key): key for key in SYMM_DATA["space_group_encoding"]
    }

    if not sym_ops:
        # Try to parse symbol
        for symmetry_label in (
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
        ):
            sg = data.data.get(symmetry_label)

            if sg:
                sg = sub_space_group(sg)
                try:
                    if spg := space_groups.get(sg):
                        sym_ops = list(SpaceGroup(spg).symmetry_ops)
                        LOGGER.warning(
                            f"No _symmetry_equiv_pos_as_xyz type key found. Spacegroup from {symmetry_label} used."
                        )
                        break
                except ValueError:
                    pass

                try:
                    cod_data = loadfn(
                        os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            "symmetry",
                            "symm_ops.json",
                        )
                    )
                    for _data in cod_data:
                        if sg == re.sub(r"\s+", "", _data["hermann_mauguin"]):
                            xyz = _data["symops"]
                            sym_ops = [SymmOp.from_xyz_str(s) for s in xyz]
                            LOGGER.warning(
                                f"No _symmetry_equiv_pos_as_xyz type key found. Spacegroup from {symmetry_label} used."
                            )
                            break
                except Exception:
                    continue

                if sym_ops:
                    break

    if not sym_ops:
        # Try to parse International number
        for symmetry_label in (
            "_space_group_IT_number",
            "_space_group_IT_number_",
            "_symmetry_Int_Tables_number",
            "_symmetry_Int_Tables_number_",
        ):
            if data.data.get(symmetry_label):
                try:
                    integer = int(str2float(data.data.get(symmetry_label, "")))
                    sym_ops = list(SpaceGroup.from_int_number(integer).symmetry_ops)
                    break
                except ValueError:
                    continue

    if not sym_ops:
        LOGGER.warning(
            "No _symmetry_equiv_pos_as_xyz type key found. Defaulting to P1."
        )
        sym_ops = [SymmOp.from_xyz_str("x, y, z")]

    return sym_ops


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


if __name__ == "__main__":
    from io import StringIO

    import polars as pl
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cifs = pl.read_parquet("data/icsd/cif/icsd_000.parquet").select("cif").to_series()

    cif = cifs[9]
    print(cif)

    cif_file = CifFile.from_str(cif)

    parser = CifParser(cif_file)
    structs = parser.parse_structures(on_error="raise", symmetrized=True)
    for s in structs:
        print(s)
        print("SYMMETRIZED:")
        a = SpacegroupAnalyzer(s)
        s = a.get_symmetrized_structure()
        print(s)

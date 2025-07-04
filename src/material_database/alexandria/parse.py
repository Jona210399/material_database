from monty.json import MontyDecoder
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import (
    ComputedStructureEntry,
)

from material_database.alexandria.sanitize import (
    sanitize_alexandria_entry,
)


def fix_mismatching_composition(entry: dict) -> dict | None:
    composition = Composition((entry).get("composition", {})).get_el_amt_dict()
    structure: Structure = MontyDecoder().process_decoded(entry.get("structure", {}))
    structure_composition = structure.composition.get_el_amt_dict()
    if sorted((composition).items()) == sorted(structure_composition.items()):
        entry["composition"] = structure_composition
        return entry
    return None


def alexandria_entry_to_pymatgen(entry: dict) -> ComputedStructureEntry | None:
    entry = sanitize_alexandria_entry(entry)
    try:
        structure_entry = ComputedStructureEntry.from_dict(entry)
        return structure_entry
    except ValueError as e:
        if e.args[0] == "Mismatching composition provided.":
            fixed_entry = fix_mismatching_composition(entry)
            if fixed_entry is None:
                return None
            return alexandria_entry_to_pymatgen(fixed_entry)

        return None

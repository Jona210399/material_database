from material_database.serialization import remove_empty_fields


def remove_none_entries_from_entry_composition(alexandria_entry: dict) -> dict:
    composition: dict = alexandria_entry.get("composition", dict())
    for k, v in list(composition.items()):
        if v is None:
            composition.pop(k)
    alexandria_entry["composition"] = composition
    return alexandria_entry


def sanitize_alexandria_entry(entry: dict) -> dict:
    entry = remove_none_entries_from_entry_composition(entry)
    entry = remove_empty_fields(entry)
    return entry

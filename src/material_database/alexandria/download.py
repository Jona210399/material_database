from pathlib import Path

from material_database.utils.url_download import (
    download_file,
    get_file_name_from_file_url,
)


def is_valid_alexandria_file(file_name: str) -> bool:
    return file_name.startswith("alexandria_") and file_name.endswith(".json.bz2")


def download_alexandria_file(file_url: str, destination_folder: Path) -> None:
    filename = get_file_name_from_file_url(file_url)
    if not is_valid_alexandria_file(filename):
        return

    destination_file = destination_folder / filename
    download_file(file_url, destination_file)

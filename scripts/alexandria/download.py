from pathlib import Path

from tqdm import tqdm

from material_database.utils.url_download import (
    download_file,
    get_file_name_from_file_url,
    get_file_urls_from_url,
    write_time_stamp,
)


def is_valid_alexandria_file(file_name: str) -> bool:
    return file_name.startswith("alexandria_") and file_name.endswith(".json.bz2")


def download_alexandria_files(url: str, destination_folder: Path) -> None:
    destination_folder.mkdir(parents=True, exist_ok=True)
    file_urls = get_file_urls_from_url(url)
    for file_url in tqdm(file_urls, desc="Downloading Alexandria files", unit="file"):
        filename = get_file_name_from_file_url(file_url)
        if not is_valid_alexandria_file(filename):
            continue

        destinantion_file = destination_folder / filename
        download_file(file_url, destinantion_file)


def main():
    ALEXANDRIA_URL = "https://alexandria.icams.rub.de/data/pbe/"
    DESTINATION_FOLDER = Path.cwd() / "data" / "alexandria" / "raw"
    download_alexandria_files(
        url=ALEXANDRIA_URL,
        destination_folder=DESTINATION_FOLDER,
    )
    write_time_stamp(destination_folder=DESTINATION_FOLDER)


if __name__ == "__main__":
    main()

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

from tqdm import tqdm

from material_database.alexandria.download import download_alexandria_file
from material_database.utils.url_download import (
    get_file_urls_from_url,
    write_time_stamp,
)


def download_alexandria_files(url: str, destination_folder: Path) -> None:
    destination_folder.mkdir(parents=True, exist_ok=True)
    file_urls = get_file_urls_from_url(url)
    for file_url in tqdm(file_urls, desc="Downloading Alexandria files", unit="file"):
        download_alexandria_file(file_url, destination_folder)


def download_alexandria_files_concurrently(
    url: str, destination_folder: Path, max_workers: int = 1
) -> None:
    destination_folder.mkdir(parents=True, exist_ok=True)
    file_urls = get_file_urls_from_url(url)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    download_alexandria_file, file_urls, repeat(destination_folder)
                ),
                total=len(file_urls),
                desc="Downloading Alexandria files",
                unit="file",
            )
        )


def main():
    ALEXANDRIA_URL = "https://alexandria.icams.rub.de/data/pbe/"
    DESTINATION_FOLDER = Path.cwd() / "data" / "alexandria" / "raw" / "json_bz2"
    DESTINATION_FOLDER.mkdir(parents=True, exist_ok=True)
    MAX_WORKERS = 10
    download_alexandria_files_concurrently(
        url=ALEXANDRIA_URL,
        destination_folder=DESTINATION_FOLDER,
        max_workers=MAX_WORKERS,
    )
    write_time_stamp(destination_folder=DESTINATION_FOLDER)


if __name__ == "__main__":
    main()

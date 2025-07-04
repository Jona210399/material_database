from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

TIMESTAMP_FILE_NAME = "timestamp.txt"


def get_file_urls_from_url(url: str) -> list[str]:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    file_urls = []
    for link in soup.find_all("a"):
        href = str(link.get("href"))  # type: ignore
        if href and is_valid_href(href):
            file_urls.append(urljoin(url, href))
    return file_urls


def is_valid_href(href: str) -> bool:
    return not href.startswith("?") and not href.startswith("/")


def get_file_name_from_file_url(file_url: str) -> str:
    url = urlparse(file_url)
    file_name = PurePosixPath(url.path).name
    return file_name


def download_file(file_url: str, destination_file: Path) -> None:
    with requests.get(file_url, stream=True) as r:
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            print(f"Failed to download {file_url}: {e}")
            return

        with destination_file.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def write_time_stamp(destination_folder: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamp_file = destination_folder / TIMESTAMP_FILE_NAME
    with timestamp_file.open("w") as f:
        f.write(f"Downloaded on: {timestamp}\n")

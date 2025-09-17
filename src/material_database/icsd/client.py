import datetime
import platform
import re
from logging import INFO, FileHandler, Formatter, getLogger
from urllib.parse import ParseResult, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import dotenv_values

from material_database.icsd.constants import (
    SEARCH_DICT,
    AvailableProperties,
    ContentType,
)


def get_credentials() -> tuple[str, str]:
    config = dotenv_values(".env")
    login_id = config.get("ICSD_LOGIN_ID")
    password = config.get("ICSD_PASSWORD")

    if login_id is None or password is None:
        raise ValueError(
            "No login_id or password found in .env file. Please add ICSD_LOGIN_ID and ICSD_PASSWORD to your .env file."
        )

    return login_id, password


class ICSDClient:
    BASE_URL: ParseResult = urlparse("https://icsd.fiz-karlsruhe.de/ws/")
    MAX_BATCH_SIZE = 500

    def __init__(
        self,
        login_id: str,
        password: str,
        timeout: float = 15.0,
        verbose: bool = True,
    ):
        self.auth_token = None
        self.session_history = []
        self.windows_client = platform.system() == "Windows"
        self.timeout = timeout
        self.verbose = verbose

        self.login_id = login_id
        self.password = password

        self.session = requests.Session()
        self.session.headers.update({"accept": "application/xml"})

        self.response_logger = self.build_response_logger(log_file="icsd_client.log")
        self.logger = self.build_logger()

    def build_logger(self):
        logger = getLogger("ICSDClient")
        logger.setLevel(INFO)
        logger.disabled = not self.verbose
        return logger

    def build_response_logger(self, log_file: str):
        logger = getLogger("ICSDClientResponses")
        if not self.verbose:
            logger.disabled = True
            return logger
        logger.setLevel(INFO)
        handler = FileHandler(filename=log_file)
        handler.setFormatter(
            Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        return logger

    def __enter__(self):
        self.authorize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logout()
        self.auth_token = None

    def authorize(self):
        try:
            response = self.session.post(
                url=urljoin(self.BASE_URL.geturl(), "auth/login"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"loginid": self.login_id, "password": self.password},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Authorization failed: {e}")
            raise ConnectionError(
                "Authorization failed. Check credentials and logs."
            ) from e

        self.auth_token = response.headers.get("ICSD-Auth-Token")
        if not self.auth_token:
            raise ConnectionError("Authorization failed: No auth token received.")

        # Add token to session headers
        self.session.headers.update({"ICSD-Auth-Token": self.auth_token})
        self.logger.info("Authentication succeeded. Token expires in 1 hour.")
        self.response_logger.info(f"Authorization response: {response.status_code}")

    def logout(self):
        if self.auth_token is None:
            return
        try:
            response = self.session.get(
                url=urljoin(self.BASE_URL.geturl(), "auth/logout"),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.warning(f"Logout failed: {e}")
        finally:
            self.auth_token = None

    def search(
        self,
        search_dict: dict[str, str],
        search_type: str = "or",
        properties: list[AvailableProperties] = ["CollectionCode", "StructuredFormula"],
        content_type: ContentType = "EXPERIMENTAL_INORGANIC",
    ):
        # Remove invalid or None search terms
        search_dict = {
            k: v for k, v in search_dict.items() if k in SEARCH_DICT and v is not None
        }
        if not search_dict:
            raise ValueError(
                f"No valid search terms provided. Available: {list(SEARCH_DICT.keys())}"
            )

        search_string = f" {search_type} ".join(
            f"{k} : {v}" for k, v in search_dict.items()
        )

        try:
            response = self.session.get(
                url=urljoin(self.BASE_URL.geturl(), "search/expert"),
                params={"query": search_string, "content type": content_type},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Search request failed: {e}")
            return []

        self.response_logger.info(f"Search request: {search_string}")
        self.response_logger.info(f"Search response: {response.text[:500]}...")

        soup = BeautifulSoup(response.content, "html.parser")
        if "<idnums></idnums>" in str(soup):
            return []

        search_results = soup.idnums.contents[0].split(" ")
        properties_data = self.fetch_data(search_results, properties=properties)
        return list(zip(search_results, properties_data))

    def fetch_data(
        self,
        ids: list[str],
        properties: list[AvailableProperties] = ["CollectionCode", "StructuredFormula"],
    ):
        if not ids:
            return []

        all_data = []
        chunks = np.array_split(ids, np.ceil(len(ids) / self.MAX_BATCH_SIZE))

        for i, chunk in enumerate(chunks):
            try:
                response = self.session.get(
                    url=urljoin(self.BASE_URL.geturl(), "csv"),
                    params={
                        "idnum": list(chunk),
                        "windowsclient": self.windows_client,
                        "listSelection": properties,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch data for chunk {i}: {e}")
                continue

            data = str(response.content).split("\\t\\n")[1:-1]
            if not data and len(chunk) != 0:
                data = str(response.content).split("\\t\\r\\n")[1:-1]

            if len(properties) > 1:
                data = [row.split("\\t") for row in data]

            all_data.extend(data)
            self.response_logger.info(
                f"Fetched {len(data)} records for chunk {i}/{len(chunks)}"
            )

            # Re-authenticate periodically
            if i % 2 == 0 and len(chunks) > 1:
                self.logout()
                self.authorize()

        return all_data

    def fetch_cif(self, id: str):
        if self.auth_token is None:
            raise RuntimeError("Not authenticated. Call client.authorize() first.")

        try:
            response = self.session.get(
                url=urljoin(self.BASE_URL.geturl(), f"cif/{id}"),
                params={
                    "celltype": "experimental",
                    "windowsclient": self.windows_client,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch CIF {id}: {e}")
            return None

        self.response_logger.info(f"Fetched CIF for ID: {id}")
        return response.content.decode("UTF-8").strip()

    def fetch_cifs(self, ids: list[str]):
        if self.auth_token is None:
            raise RuntimeError("Not authenticated. Call client.authorize() first.")
        if not ids:
            return []

        if isinstance(ids[0], tuple):
            ids = [x[0] for x in ids]

        all_cifs = []
        chunks = np.array_split(ids, np.ceil(len(ids) / self.MAX_BATCH_SIZE))

        for i, chunk in enumerate(chunks):
            try:
                response = self.session.get(
                    url=urljoin(self.BASE_URL.geturl(), "cif/multiple"),
                    params={
                        "idnum": list(chunk),
                        "celltype": "experimental",
                        "windowsclient": self.windows_client,
                        "filetype": "cif",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch CIFs for chunk {i}: {e}")
                continue

            cifs = re.split(r"\(C\) \d{4} by FIZ Karlsruhe", response.text)[1:]
            cifs = [
                f"(C) {datetime.datetime.now().year} by FIZ Karlsruhe{x}" for x in cifs
            ]
            all_cifs.extend(cifs)

            # Re-authenticate periodically
            if i % 2 == 0 and len(chunks) > 1:
                self.logout()
                self.authorize()

        return all_cifs

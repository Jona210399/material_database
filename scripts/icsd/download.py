import pickle
from pathlib import Path
from typing import get_args

import polars as pl

from material_database.constants import ColumnNames
from material_database.icsd.client import ICSDClient, get_credentials
from material_database.icsd.constants import AvailableProperties, ContentType
from material_database.utils.url_download import write_time_stamp

PROPERTIES = [
    "CollectionCode",
    "StructureType",
    "SumFormula",
    "MineralName",
    "StructuredFormula",
    "ChemicalName",
    "MineralGroup",
]


def fetch_all_cifs(
    icsd_credentials: tuple[str, str],
    properties: list[AvailableProperties],
):
    login_id, password = icsd_credentials

    search_results: list[tuple] = []
    all_cifs: list[str] = []
    content_types: list[str] = []

    for content_type in get_args(ContentType):
        print(f"Fetching {content_type}")
        for start in range(0, 1000000, ICSDClient.MAX_BATCH_SIZE):
            with ICSDClient(login_id, password) as client:
                end = start + ICSDClient.MAX_BATCH_SIZE - 1
                print(f"{start}-{end}")
                search_res = client.search(
                    {"collectioncode": f"{start}-{end}"},
                    content_type=content_type,
                    properties=properties,
                )

                if not search_res:
                    continue

                cifs = client.fetch_cifs([x[0] for x in search_res])
                search_results.append(search_res)
                all_cifs.append(cifs)
                content_types.append([content_type] * len(cifs))

    return search_results, all_cifs, content_types


def combine_search(
    search_results: list[tuple],
    all_cifs: list[str],
    content_types: list[str],
):
    data = []

    assert len(search_results) == len(all_cifs) == len(content_types), (
        "Lengths of search_results, all_cifs and content_types must be equal."
    )

    for i, item in enumerate(search_results):
        assert len(item) == len(all_cifs[i]) == len(content_types[i]), (
            "Lengths of item, all_cifs sublist and content_types sublist must be equal."
        )

        for j in range(len(item)):
            db_code, properties = item[j]
            collection_code, *rest_properties = properties
            cif_string = all_cifs[i][j]
            content_type = content_types[i][j]

            data.append(
                [
                    int(db_code),
                    cif_string,
                    content_type,
                    int(collection_code),
                    *rest_properties,
                ]
            )

    column_labels = [
        "DB_ID",
        ColumnNames.CIF,
        "ContentType",
    ] + PROPERTIES

    data = pl.DataFrame(
        data,
        schema=column_labels,
    )
    data = data.select(
        [
            pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
            if data[c].dtype == pl.Utf8
            else pl.col(c)
            for c in data.columns
        ]
    )

    return data


def main():
    out_dir = Path.cwd() / "data" / "icsd" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    icsd_credentials = get_credentials()

    search_results, all_cifs, content_types = itermediate = fetch_all_cifs(
        icsd_credentials,
        PROPERTIES,
    )

    with open(out_dir / "intermediate.pkl", "wb") as f:
        pickle.dump(itermediate, f)

    icsd = combine_search(search_results, all_cifs, content_types)
    print(icsd)

    icsd.write_parquet(out_dir / "icsd.parquet")
    print(f"ICSD data saved to {out_dir / 'icsd.parquet'}")

    (out_dir / "intermediate.pkl").unlink(missing_ok=True)
    write_time_stamp(out_dir)


if __name__ == "__main__":
    main()

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from material_database.constants import ColumnNames
from material_database.deserialization import structure_from_serialized
from material_database.iterate import DatabaseFileIterator
from material_database.matbind_graphs.core import (
    GaussianDistanceCalculator,
    create_graph_data,
)
from material_database.types_ import (
    SerializedMatBindGraph,
    SerializedSymmetrizedStructure,
)

NUM_CPUS = 32  # None uses all available CPUs, use carfully since it can lead to high memory usage

RADIUS = 8.0
MAX_NUM_NBR = 12
ATOM_NUM_UPPER = NUM_NODE_FEATURES = 96

CALCULATOR = GaussianDistanceCalculator(dmin=0, dmax=RADIUS, step=0.2)
NUM_EDGE_FEATURES = CALCULATOR.num_edge_features()


def process_entry(entry: SerializedSymmetrizedStructure) -> SerializedMatBindGraph:
    structure = structure_from_serialized(entry)

    node_features, edge_index, edge_attr = create_graph_data(
        data_input=structure,
        max_num_nbr=MAX_NUM_NBR,
        radius=RADIUS,
        calculators=[CALCULATOR],
        atom_num_upper=ATOM_NUM_UPPER,
    )
    return SerializedMatBindGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def main():
    source = Path(
        "/p/project1/solai/oestreicher1/repos/material_database/data/alexandria"
    )

    destination = source / "matbind_graphs"
    destination.mkdir(exist_ok=True, parents=True)

    for file, df in DatabaseFileIterator(source=source / "pymatgen"):
        if (destination / file.name).exists():
            print(f"Skipping {file.name}, since it was already processed.")
            continue

        print(f"Processing file: {file.name}")
        df = df.collect()

        structures = df[ColumnNames.SYMMETRIZED_STRUCTURE]
        ids = df[ColumnNames.ID]

        with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
            graph_data: list[SerializedMatBindGraph] = list(
                tqdm(
                    executor.map(process_entry, structures),
                    total=len(structures),
                    desc=f"Processing structures in {file.name}",
                )
            )

        graph_df = (
            pl.DataFrame({ColumnNames.ID: ids})
            .with_columns(
                pl.DataFrame(
                    graph_data,
                    schema=SerializedMatBindGraph.get_schema(
                        num_node_features=NUM_NODE_FEATURES,
                        num_edge_features=NUM_EDGE_FEATURES,
                    ),
                )
            )
            .drop_nulls()
        )

        graph_df.write_parquet(destination / file.name)


if __name__ == "__main__":
    main()

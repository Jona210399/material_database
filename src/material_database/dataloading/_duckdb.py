from collections import Counter
from pathlib import Path
from typing import Optional, TypedDict

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from material_database.constants import ColumnNames


class SplitIDs(TypedDict):
    train: NDArray
    val: NDArray
    test: NDArray


class DistributedLoader:
    def __init__(
        self,
        base_dir: str | Path,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        id_column: str = ColumnNames.ID,
        columns_to_query: list[str] | None = None,
        db_path: Optional[str] = None,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        """
        Args:
            base_dir: Path to alexandria folder with subfolders of Parquet files.
            split_ratios: Tuple for train/val/test split fractions (must sum to 1.0).
            seed: Random seed for reproducibility.
            db_path: Optional path for DuckDB database on disk.
            num_replicas: Total number of distributed processes (GPUs).
            rank: Rank of this process (0-based).
        """
        self.base_dir = Path(base_dir)
        self.seed = seed
        self.conn = duckdb.connect(db_path or ":memory:")

        self.id_column = id_column
        self.columns_to_query = columns_to_query or ColumnNames.all_columns()
        self.num_replicas = num_replicas
        self.rank = rank
        self.split_ratios = split_ratios

        self.tables = self.register_subfolders()
        self.verify_tables()
        self.unique_ids = self.compute_unique_ids()
        self.split_ids = self.create_splits()
        self.split_ids = self.partition_splits()

    def register_subfolders(self) -> dict[str, list[str]]:
        tables = {}
        for subfolder in self.base_dir.iterdir():
            if not subfolder.is_dir():
                continue

            parquet_files = sorted(subfolder.glob("alexandria_000.parquet"))
            if not parquet_files:
                continue

            table_name = subfolder.name
            parquet_paths = [str(f) for f in parquet_files]
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet({parquet_paths})"
            )
            tables[table_name] = (
                self.conn.execute(f"PRAGMA table_info('{table_name}')")
                .fetchdf()["name"]
                .tolist()
            )

        return tables

    def verify_tables(self):
        all_columns = []
        for available_columns in self.tables.values():
            all_columns.extend(available_columns)

        counter = Counter(all_columns)
        print(counter)
        duplicates = [
            col for col, count in counter.items() if col != self.id_column and count > 1
        ]
        print("Duplicate columns:", duplicates)
        if duplicates:
            raise ValueError(
                f"Duplicate columns found across tables: {duplicates}. Please ensure column names are unique across tables."
            )

    def compute_unique_ids(self) -> NDArray:
        ids_sets = []
        for table in self.tables:
            ids = self.conn.execute(f"SELECT {self.id_column} FROM {table}").fetchall()
            ids_sets.append(set(id_[0] for id_ in ids))
        all_unique_ids = sorted(set.union(*ids_sets))
        all_unique_ids = np.array(all_unique_ids)

        return all_unique_ids

    def create_splits(self) -> SplitIDs:
        np.random.seed(self.seed)
        shuffled_ids = np.random.permutation(self.unique_ids)

        n = len(shuffled_ids)
        train_end = int(n * self.split_ratios[0])
        val_end = train_end + int(n * self.split_ratios[1])
        split_ids = {
            "train": shuffled_ids[:train_end],
            "val": shuffled_ids[train_end:val_end],
            "test": shuffled_ids[val_end:],
        }

        return split_ids

    def partition_splits(self) -> SplitIDs:
        return {
            split: ids[self.rank :: self.num_replicas]
            for split, ids in self.split_ids.items()
        }

    def build_batch_query(
        self,
        tables: dict[str, list[str]],
        columns_to_query: list[str],
        batch_temp_table: str,
        id_column: str = "id",
    ) -> str:
        """
        Build a SQL query that uses the batch temp table as the driving table.
        Ensures that even if some tables don't contain certain IDs, those IDs still appear.
        """
        select_clauses = [f"q.{id_column} AS {id_column}"]
        join_clauses = [f"{batch_temp_table} AS q"]
        used_aliases = set()

        def make_unique_alias(table_name: str) -> str:
            base = table_name[0].lower()
            alias = base
            i = 1
            while alias in used_aliases:
                alias = f"{base}{i}"
                i += 1
            used_aliases.add(alias)
            return alias

        for table_name, available_cols in tables.items():
            alias = make_unique_alias(table_name)
            cols = [
                c for c in columns_to_query if c in available_cols and c != id_column
            ]
            select_clauses.extend([f"{alias}.{c} AS {c}" for c in cols])

            join_clauses.append(
                f"LEFT JOIN {table_name} AS {alias} ON {alias}.{id_column} = q.{id_column}"
            )

        query = f"SELECT {', '.join(select_clauses)} FROM {' '.join(join_clauses)}"
        return query

    def query_ids(self, ids: list[str]) -> pd.DataFrame:
        query_ids = pd.DataFrame(ids, columns=[self.id_column])
        self.conn.register("query_ids", query_ids)
        query = self.build_batch_query(
            tables=self.tables,
            columns_to_query=self.columns_to_query,
            batch_temp_table="query_ids",
            id_column=self.id_column,
        )
        df = self.conn.execute(query).fetchdf()
        self.conn.unregister("query_ids")
        return df

    def batch_iterator(self, split: str, batch_size: int = 1024):
        """
        Yield batches as a single pandas.DataFrame combining all tables.
        Adds a column 'table_name' to indicate the source table.
        """
        ids = self.split_ids[split]
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            yield self.query_ids(batch_ids)


if __name__ == "__main__":
    loader = DistributedLoader(
        base_dir="/p/project1/solai/oestreicher1/repos/material_database/data/alexandria",
        split_ratios=(0.7, 0.15, 0.15),
        seed=42,
        db_path=":memory:",
    )

    print("Unique IDs:", len(loader.unique_ids))
    print("Train IDs:", len(loader.split_ids["train"]))
    print("Val IDs:", len(loader.split_ids["val"]))
    print("Test IDs:", len(loader.split_ids["test"]))

    print("Columns to query:", loader.columns_to_query)
    print("Tables:", loader.tables)

    for batch in loader.batch_iterator(split="train", batch_size=3):
        print(batch.columns)
        break

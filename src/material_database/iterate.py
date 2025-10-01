from pathlib import Path

import polars as pl


class DatabaseFileIterator:
    def __init__(self, source: Path):
        self.source = source
        self.files = sorted(source.glob("*.parquet"))
        if not self.files:
            raise ValueError(f"No parquet files found in {source}")
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.files):
            raise StopIteration
        file = self.files[self.index]
        df = pl.scan_parquet(file)
        self.index += 1
        return file, df

    def __len__(self):
        return len(self.files)

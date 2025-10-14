import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


def get_rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class StreamingParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_files: list[Path],
        columns: list[str] | None = None,
        buffer_size: int = 10000,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
    ):
        self.files = parquet_files

        self.buffer_size = buffer_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.columns = columns
        self.shuffle = shuffle

    def _yield_rows(self, file):
        pq_file = pq.ParquetFile(file)
        for batch in pq_file.iter_batches(
            batch_size=self.buffer_size * 2,
            columns=self.columns,
        ):
            df: pd.DataFrame = batch.to_pandas()
            for _, row in df.iterrows():
                yield row.to_dict()

    def __iter__(self):
        files = self.files[self.rank :: self.world_size]

        if self.shuffle:
            rng = random.Random(self.seed + self.rank)
            rng.shuffle(files)

        worker_info = get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
            print(
                f"Rank {self.rank}, Worker {worker_info.id}: {len(files)} files assigned: {[file.name for file in files]}"
            )

        buffer = []
        for file in files:
            for row in self._yield_rows(file):
                buffer.append(row)
                if len(buffer) >= self.buffer_size:
                    if self.shuffle:
                        rng.shuffle(buffer)
                    while buffer:
                        yield buffer.pop()

        if self.shuffle:
            rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()

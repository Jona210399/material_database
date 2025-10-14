import random
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule, Trainer
from torch.utils.data import DataLoader

from material_database.dataloading._torch.streaming_parquet_dataset import (
    StreamingParquetDataset,
)


class ParquetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path | str,
        columns: list[str] | None = None,
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
        batch_size: int = 2048,
        drop_last: bool = True,
        buffer_size: int = 10000,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()

        data_dir = Path(data_dir)
        self.columns = columns
        self.seed = seed

        files = sorted(data_dir.glob("*.parquet"))
        (
            self.train_files,
            self.validation_files,
            self.test_files,
        ) = self._split_files(
            files,
            frac_train,
            frac_val,
            frac_test,
            seed,
        )

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.num_workers = num_workers

    def _split_files(
        self,
        files: list[Path],
        frac_train: float,
        frac_val: float,
        frac_test: float,
        seed: int,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        random.seed(seed)
        random.shuffle(files)
        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
            "Fractions of training, validation, and test sets must sum to 1."
        )

        n = len(files)
        train_files = files[: int(frac_train * n)]
        validation_files = files[int(frac_train * n) : int((frac_train + frac_val) * n)]
        test_files = files[int((frac_train + frac_val) * n) :]
        return train_files, validation_files, test_files

    def get_rank_and_world_size(self) -> tuple[int, int]:
        if self.trainer is None:
            raise ValueError(
                "Trainer is not set. Please set the trainer before calling build_dataloader."
            )

        rank_and_num_replicas = getattr(
            self.trainer.strategy,
            "distributed_sampler_kwargs",
            {"rank": 0, "num_replicas": 1},
        )
        return rank_and_num_replicas["rank"], rank_and_num_replicas["num_replicas"]

    def _make_loader(
        self,
        files: list[Path],
        shuffle: bool = False,
    ):
        rank, world_size = self.get_rank_and_world_size()

        dataset = StreamingParquetDataset(
            parquet_files=files,
            columns=self.columns,
            seed=self.seed,
            buffer_size=self.buffer_size,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            prefetch_factor=2 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_files, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.validation_files, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_files, shuffle=False)

    def predict_dataloader(self):
        return self._make_loader(self.test_files, shuffle=False)


if __name__ == "__main__":
    import psutil
    from lightning.pytorch import LightningModule, Trainer

    class DummyModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)

        def training_step(self, batch, batch_idx):
            # no-op: just print shape
            print(f"Rank {self.global_rank}: batch {batch_idx} -> {batch}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    trainer = Trainer(accelerator="gpu", devices=4, max_epochs=1, limit_train_batches=2)

    print(psutil.virtual_memory())

    print("Creating data module...")
    dm = ParquetDataModule(
        data_dir=Path.cwd() / "data" / "alexandria" / "combined",
        columns=["crystallite_size", "intensities"],
        batch_size=2,
        num_workers=4,
    )

    trainer.fit(DummyModel(), datamodule=dm)

    exit()
    dm.trainer = trainer

    print("Iterating through training dataloader...")
    for i, batch in enumerate(dm.train_dataloader()):
        print(f"Batch {i}: {batch}")
        print(psutil.virtual_memory())
        if i >= 5:
            break

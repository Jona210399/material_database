from typing import Callable

import pandas as pd
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Batch, Data

Transform = Callable[[pd.Series], pd.Series | None]


class TransformPipeline:
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, row: pd.Series) -> pd.Series:
        for t in self.transforms:
            row = t(row)
        return row


class TransformDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for row in self.dataset:
            row = self.transform(row)
            yield row.to_dict()


def graph_from_columns(
    row: pd.Series,
    graph_columns: list[str] | None,
    new_column: str = "crystal_structure",
) -> pd.Series:
    graph_columns = graph_columns or ["edge_index", "edge_attr", "num_nodes"]

    graph = row[graph_columns].to_dict()
    row = row.drop(labels=graph_columns)

    graph = {k: torch.tensor(v) for k, v in graph.items()}

    row[new_column] = Data(**graph)
    return row


def align_to_central_modality(
    row: pd.Series,
    central_modality: str,
    modalities: list[str] | None = None,
) -> pd.Series:
    """
    Aligns all other modalities in a row to the given central modality.

    Each non-central modality becomes a dictionary containing:
      {
        modality_name: tensor(value_of_modality),
        central_modality: tensor(value_of_central_modality)
      }

    Missing values (None) are skipped.

    Returns:
        pd.Series where each key is a modality name (except the central one),
        and each value is a dict of aligned tensors.
    """

    modalities = modalities or [m for m in row.index if m != central_modality]
    result = {}

    central_value = row[central_modality]
    if central_value is None:
        return pd.Series(dtype=object)

    if not isinstance(central_value, Data):
        central_value = torch.atleast_1d(torch.tensor(central_value))

    for modality in modalities:
        value = row[modality]
        if value is None:
            continue

        if not isinstance(value, Data):
            value = torch.atleast_1d(torch.tensor(value))

        result[modality] = {
            central_modality: central_value,
            modality: value,
        }

    return pd.Series(result, dtype=object)


def collate_stack_per_modality(
    batch: list[dict[str, torch.Tensor | Data]],
) -> dict[str, dict[str, torch.Tensor | Data]]:
    """
    Collate a batch of dicts (from `align_to_central_modality`) into:
    { modality_name: { modality_name: Tensor, central_modality: Tensor }, ... }

    - Skips missing modalities (None or empty dicts)
    - Stacks tensors across the batch (assumes shapes are consistent)
    """
    collated: dict[str, dict[str, torch.Tensor]] = {}

    batch = [b for b in batch if b is not None]

    # Collect all modality keys that appear in any sample
    all_modalities = set().union(*(b.keys() for b in batch))

    for modality in all_modalities:
        # Gather valid entries for this modality
        valid = [
            b[modality]
            for b in batch
            if isinstance(b, dict) and modality in b and b[modality] is not None
        ]
        if not valid:
            continue

        collated_modality = {}
        for k in valid[0].keys():
            values = [v[k] for v in valid]
            if all(isinstance(v, Data) for v in values):
                collated_modality[k] = Batch.from_data_list(values)
            else:
                collated_modality[k] = torch.stack([v[k] for v in valid])

        collated[modality] = collated_modality

    return collated


if __name__ == "__main__":
    from functools import partial

    from torch.utils.data import DataLoader, Dataset

    df = pd.DataFrame(
        [
            {
                "edge_index": [[0, 1], [1, 2]],  # list of edges
                "edge_attr": [[1.0], [2.0]],  # attributes for each edge
                "num_nodes": 3,  # number of nodes
                "intensities": [1.0, 2.0],  # another modality for testing
                "temperature": 20.1,
            },
            {
                "edge_index": [[0, 1], [1, 0]],
                "edge_attr": [[0.5], [1.5]],
                "num_nodes": 2,
                "intensities": [3.0, 4.0],
                "temperature": 22.5,
            },
        ]
    )

    class PandasDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)

        def __getitem__(self, idx):
            item = self.df.iloc[idx]
            return item

        def __len__(self):
            return len(self.df)

    dataset = TransformDataset(
        dataset=PandasDataset(df),
        transform=TransformPipeline(
            transforms=[
                partial(
                    graph_from_columns,
                    graph_columns=["edge_index", "edge_attr", "num_nodes"],
                    new_column="crystal_structure",
                ),
                partial(
                    align_to_central_modality, central_modality="crystal_structure"
                ),
            ]
        ),
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_stack_per_modality,
    )

    for i, batch in enumerate(loader):
        print(f"Batch {i}: {batch}")

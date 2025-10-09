import pickle
from pathlib import Path
from typing import Optional
import multiprocessing

import lightning as L
from torch_geometric.loader import DataLoader
from gotennet.data.components.utils import CountNodesPerGraph, ConformationDataset


class RDB7DataModule(L.LightningDataModule):
    def __init__(self, data_file: str, split_path: str, split_file: str, batch_size: int = 64,
                 num_workers: int = 0, pin_memory: bool = False, multiprocessing_context: Optional[str] = None):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        # DataLoader runtime options (configurable via hydra overrides)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.multiprocessing_context = multiprocessing_context
        self.transforms = CountNodesPerGraph()
        
        # Load train, val, test split indices from the pickle file
        with open(Path(split_path) / split_file, "rb") as f:
            self.split_dict = pickle.load(f)
            
        self.train_indices = self.split_dict["train"]
        self.val_indices = self.split_dict["val"]
        self.test_indices = self.split_dict["test"]

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        train_set = ConformationDataset(
            self.data_file, 
            data_indices=self.train_indices, 
            transform=self.transforms
        )
        kwargs = dict(batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.multiprocessing_context:
            # Use a safe multiprocessing context if requested (e.g., 'forkserver' or 'spawn')
            ctx = multiprocessing.get_context(self.multiprocessing_context)
            kwargs['multiprocessing_context'] = ctx
        return DataLoader(train_set, **kwargs)

    def val_dataloader(self) -> DataLoader:
        val_set = ConformationDataset(
            self.data_file, 
            data_indices=self.val_indices, 
            transform=self.transforms
        )
        kwargs = dict(batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.multiprocessing_context:
            ctx = multiprocessing.get_context(self.multiprocessing_context)
            kwargs['multiprocessing_context'] = ctx
        return DataLoader(val_set, **kwargs)

    def test_dataloader(self) -> DataLoader:
        test_set = ConformationDataset(
            self.data_file, 
            data_indices=self.test_indices, 
            transform=self.transforms
        )
        kwargs = dict(batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        if self.multiprocessing_context:
            ctx = multiprocessing.get_context(self.multiprocessing_context)
            kwargs['multiprocessing_context'] = ctx
        return DataLoader(test_set, **kwargs)
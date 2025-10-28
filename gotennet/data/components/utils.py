import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
import pickle
from torch_geometric.data import Dataset
import os


class ConformationDataset(Dataset):
    def __init__(self, data_file, data_indices, transform=None):
        """
        Select data according to rxn_index attribute.
        
        Parameters:
        - data_file: Path to the pickle file containing all data samples
        - data_indices: Indices matching the rxn_index attribute in data objects
        - transform: Optional transforms to apply to each data sample
        """
        super().__init__()
        
        with open(data_file, "rb") as f:
            all_data = pickle.load(f)
        
        # Create a mapping from rxn_index to the data object
        rxn_index_to_data = {}
        for i, data_obj in enumerate(all_data):
            # Try common attribute names that might store the reaction index
            key = None
            for attr in ("rxn_idx", "rxn_index", "idx"):
                if hasattr(data_obj, attr):
                    try:
                        val = getattr(data_obj, attr)
                        # If it's a tensor, convert to python int
                        if hasattr(val, "item"):
                            key = int(val.item())
                        else:
                            key = int(val)
                    except Exception:
                        # If conversion failed, ignore and keep trying
                        key = None
                    break
            if key is None:
                # Fallback: use the list index
                key = i
                rank_zero_warn(
                    f"ConformationDataset: data object at position {i} has no 'rxn_idx'/'rxn_index'/'idx' attribute — using list index {i} as key."
                )
            if key in rxn_index_to_data:
                rank_zero_warn(f"ConformationDataset: duplicate rxn index {key} found; overwriting previous entry.")
            rxn_index_to_data[key] = data_obj

        # Validate requested indices are present
        missing = [idx for idx in data_indices if idx not in rxn_index_to_data]
        if len(missing) > 0:
            raise MissingLabelException(
                f"Requested rxn_indices not found in data file: {missing}. "
                f"Available keys (sample): {list(rxn_index_to_data.keys())[:20]}"
            )
        
        # Get data by the provided rxn_indices
        self.data = [rxn_index_to_data[idx] for idx in data_indices]
        self.transform = transform
        
        # Cache atom and edge types
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):
        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.data)

    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for i, graph in enumerate(self.data):
            if hasattr(graph, 'atom_type_N'):
                atom_types.update(graph.atom_type_N.tolist())
            elif hasattr(graph, 'atom_type'):
                atom_types.update(graph.atom_type.tolist())
            else:
                print(f"Graph {i} has no 'atom_type_N' or 'atom_type' attribute")
                if i > 2:  # Only print first few for debugging
                    break
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for i, graph in enumerate(self.data):
            if hasattr(graph, 'edge_type'):
                edge_types.update(graph.edge_type.tolist())
            else:
                if i == 0:  # Only print once for debugging
                    print(f"Graph {i} has no 'edge_type' attribute")
            if i > 2:  # Only print first few for debugging
                break
        return sorted(edge_types)
    
    
class CountNodesPerGraph(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        if not hasattr(data, "__num_nodes__"):
            # Handle different position data formats
            if hasattr(data, 'pos_N_3_I'):
                # T1X format: [atoms, 3D_coords, NEB_images]
                data.num_nodes = data.pos_N_3_I.shape[0]  # First dimension is number of atoms
                # Create pos attribute from first NEB image for compatibility
                data.pos = data.pos_N_3_I[:, :, 0]  # Shape: [atoms, 3]
            elif hasattr(data, 'pos') and data.pos is not None:
                # Standard format
                data.num_nodes = len(data.pos)
            else:
                raise ValueError("Data has no position information (pos or pos_N_3_I)")
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data


class MissingLabelException(Exception):
    pass

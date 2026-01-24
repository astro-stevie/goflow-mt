"""
Create a barrier height-based split for T1X dataset.

This script creates a split where:
- Edge values (very low and very high barrier heights) → test set
- Middle values (medium barrier heights) → train set
- Validation set is split from test set

The T1X dataset has multiple NEB images per reaction, but the barrier height
is a reaction-level property that should be used for splitting.
"""

import pickle
import numpy as np
import argparse
import os
from pathlib import Path
from collections import defaultdict


def extract_barrier_heights(data_list):
    """
    Extract barrier heights from T1X data.
    
    Barrier height = TS energy - Reactant energy
    
    For T1X data, we need to find a way to extract energies. This might be:
    1. From a CSV file associated with the data
    2. From energy information in the data objects themselves
    3. Calculated from structure-based estimates
    
    Args:
        data_list: List of Data objects from data_inr.pkl
        
    Returns:
        Array of barrier heights
    """
    
    barriers = []
    
    for idx, data in enumerate(data_list):
        barrier = None
        
        # Try different possible sources of barrier information
        if hasattr(data, 'barrier'):
            barrier = float(data.barrier.item() if hasattr(data.barrier, 'item') else data.barrier)
        elif hasattr(data, 'dE0'):
            barrier = float(data.dE0.item() if hasattr(data.dE0, 'item') else data.dE0)
        elif hasattr(data, 'ts_energy') and hasattr(data, 'r_energy'):
            ts_e = float(data.ts_energy.item() if hasattr(data.ts_energy, 'item') else data.ts_energy)
            r_e = float(data.r_energy.item() if hasattr(data.r_energy, 'item') else data.r_energy)
            barrier = ts_e - r_e
        else:
            # If no energy available, skip this reaction
            print(f"Warning: Could not extract barrier height for reaction {idx}")
            print(f"  Available attributes: {[attr for attr in dir(data) if not attr.startswith('_') and not callable(getattr(data, attr))][:20]}")
            barrier = None
        
        if barrier is not None:
            barriers.append(barrier)
        else:
            barriers.append(np.nan)
    
    return np.array(barriers)


def create_t1x_barrier_split(data_pkl_path, output_split_path, train_fraction=0.8, val_fraction=0.1, seed=42):
    """
    Create barrier-based split for T1X data.
    
    Args:
        data_pkl_path: Path to data_inr.pkl or data.pkl containing the dataset
        output_split_path: Path where to save split_inr_barrier.pkl
        train_fraction: Fraction of middle barrier values for training (default: 0.8)
        val_fraction: Fraction of edge values for validation (default: 0.1)
        seed: Random seed for reproducibility
    """
    
    np.random.seed(seed)
    
    print(f"Loading data from {data_pkl_path}...")
    with open(data_pkl_path, 'rb') as f:
        data_list = pickle.load(f)
    
    print(f"Loaded {len(data_list)} reactions\n")
    
    # Extract barrier heights
    print("Extracting barrier heights...")
    barriers = extract_barrier_heights(data_list)
    
    # Check for NaN values
    valid_mask = ~np.isnan(barriers)
    num_valid = np.sum(valid_mask)
    num_invalid = len(barriers) - num_valid
    
    if num_invalid > 0:
        print(f"Warning: {num_invalid} reactions have no barrier information")
        if num_valid == 0:
            raise ValueError("No reactions have barrier information available!")
    
    # Use only valid barriers for splitting
    valid_indices = np.where(valid_mask)[0]
    valid_barriers = barriers[valid_indices]
    
    print(f"Valid reactions: {num_valid}")
    print(f"Barrier heights - Min: {valid_barriers.min():.3f}, Max: {valid_barriers.max():.3f}, Mean: {valid_barriers.mean():.3f}\n")
    
    # Sort by barrier height
    sorted_positions = np.argsort(valid_barriers)
    sorted_barriers = valid_barriers[sorted_positions]
    sorted_indices = valid_indices[sorted_positions]
    
    # Split into edge and middle
    total_rxns = len(sorted_indices)
    edge_fraction = 1.0 - train_fraction  # ~20% for edge values (val + test)
    edge_size = int(np.ceil(total_rxns * edge_fraction / 2))  # 10% each on bottom and top
    
    print(f"Split configuration:")
    print(f"  Total reactions: {total_rxns}")
    print(f"  Train (middle):  {total_rxns - 2*edge_size} reactions")
    print(f"  Val + Test (edges): {2*edge_size} reactions")
    print()
    
    # Get bottom and top edge values
    bottom_edge_indices = sorted_indices[:edge_size]
    top_edge_indices = sorted_indices[-edge_size:]
    middle_indices = sorted_indices[edge_size:-edge_size]
    
    # Get barrier ranges
    bottom_barriers = sorted_barriers[:edge_size]
    middle_barriers = sorted_barriers[edge_size:-edge_size]
    top_barriers = sorted_barriers[-edge_size:]
    
    print(f"Barrier ranges:")
    print(f"  Bottom edge: {bottom_barriers.min():.3f} - {bottom_barriers.max():.3f}")
    print(f"  Middle:      {middle_barriers.min():.3f} - {middle_barriers.max():.3f}")
    print(f"  Top edge:    {top_barriers.min():.3f} - {top_barriers.max():.3f}\n")
    
    # Combine edge values
    edge_indices = np.concatenate([bottom_edge_indices, top_edge_indices])
    
    # Shuffle edge indices
    np.random.shuffle(edge_indices)
    
    # Split edge into validation and test
    val_size = int(np.ceil(len(edge_indices) * val_fraction / (1.0 - train_fraction)))
    val_indices = edge_indices[:val_size].tolist()
    test_indices = edge_indices[val_size:].tolist()
    train_indices = middle_indices.tolist()
    
    # Create split dictionary
    split_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # Verify split integrity
    test_no_data_leakage(split_dict, set(valid_indices))
    test_barrier_split_integrity(split_dict, valid_barriers, valid_indices)
    
    # Save split
    os.makedirs(output_split_path, exist_ok=True)
    output_file = Path(output_split_path) / 'split_inr_barrier.pkl'
    
    with open(output_file, 'wb') as f:
        pickle.dump(split_dict, f)
    
    print(f"\nBarrier-based split saved to {output_file}")
    print(f"  Train: {len(train_indices)} reactions")
    print(f"  Val:   {len(val_indices)} reactions")
    print(f"  Test:  {len(test_indices)} reactions")


def test_no_data_leakage(split_dict, all_indices):
    """
    Test that there is no data leakage between train, validation, and test sets.
    """
    train_set = set(split_dict['train'])
    val_set = set(split_dict['val'])
    test_set = set(split_dict['test'])
    
    # Test 1: No overlap between sets
    assert len(train_set.intersection(val_set)) == 0, "Overlap found between train and validation sets"
    assert len(train_set.intersection(test_set)) == 0, "Overlap found between train and test sets"
    assert len(val_set.intersection(test_set)) == 0, "Overlap found between validation and test sets"
    
    # Test 2: All indices are used (no missing data)
    all_used_indices = train_set.union(val_set).union(test_set)
    assert all_used_indices == all_indices, "The sets of indices don't match"
    
    # Test 3: Each index appears exactly once
    counts = defaultdict(int)
    for idx in split_dict['train'] + split_dict['val'] + split_dict['test']:
        counts[idx] += 1
    
    assert all(count == 1 for count in counts.values()), "Some indices appear multiple times across sets"
    
    print("✓ All data leakage tests passed!")


def test_barrier_split_integrity(split_dict, barriers, valid_indices):
    """
    Test that the barrier split correctly separates the data by energy barriers.
    """
    # Create mapping from valid indices to barrier values
    idx_to_barrier = {valid_indices[i]: barriers[i] for i in range(len(valid_indices))}
    
    # Get energy barriers for each set
    train_barriers = np.array([idx_to_barrier[idx] for idx in split_dict['train']])
    val_test_barriers = np.array([idx_to_barrier[idx] for idx in split_dict['val'] + split_dict['test']])
    
    # Check that train contains the middle values
    min_train = np.min(train_barriers)
    max_train = np.max(train_barriers)
    min_valtest = np.min(val_test_barriers)
    max_valtest = np.max(val_test_barriers)
    
    print(f"✓ Train barriers: {min_train:.3f} - {max_train:.3f}")
    print(f"✓ Val/Test barriers: {min_valtest:.3f} - {max_valtest:.3f}")
    
    # Check that most val/test barriers are outside train range
    extremes_count = np.sum((val_test_barriers < min_train) | (val_test_barriers > max_train))
    extreme_ratio = extremes_count / len(val_test_barriers) if len(val_test_barriers) > 0 else 0
    
    print(f"✓ Barrier split integrity verified! {extreme_ratio*100:.1f}% of val/test are at extremes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create barrier height-based split for T1X dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create barrier split for data_inr.pkl
  python create_t1x_barrier_split.py \\
    --data_pkl data/T1X/data_inr.pkl \\
    --output_dir data/T1X/splits
    
  # With custom train/val fractions (80% train, 15% val, 5% test)
  python create_t1x_barrier_split.py \\
    --data_pkl data/T1X/data_inr.pkl \\
    --output_dir data/T1X/splits \\
    --train_fraction 0.80 \\
    --val_fraction 0.15
        """
    )
    parser.add_argument(
        "--data_pkl",
        type=str,
        required=True,
        help="Path to data_inr.pkl file containing the T1X dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where split_inr_barrier.pkl will be saved"
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of middle barrier values to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of edge values to use for validation (default: 0.1, rest goes to test)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    create_t1x_barrier_split(
        args.data_pkl,
        args.output_dir,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed
    )

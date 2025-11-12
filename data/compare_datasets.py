import pickle
from hydra import main
import torch
from ase import Atoms
from ase.gui.gui import GUI
import numpy as np

with open('T1X/data_inr.pkl', 'rb') as f:
    t1x_data = pickle.load(f)

with open('RDB7/processed_data/data.pkl', 'rb') as f:
    rdb7_data = pickle.load(f)


# Display types and basic info
type(t1x_data), type(rdb7_data), len(t1x_data) if hasattr(t1x_data, '__len__') else None, len(rdb7_data) if hasattr(rdb7_data, '__len__') else None


# ============================================================================
# COMPARISON: T1X vs RDB7 Datasets
# ============================================================================

print("=" * 80)
print("DATASET COMPARISON: T1X vs RDB7")
print("=" * 80)

# Dataset sizes
print(f"\nDataset Sizes:")
print(f"  T1X:  {len(t1x_data)} samples")
print(f"  RDB7: {len(rdb7_data)} samples")

# Sample a record from each
t1x_sample = t1x_data[1]
rdb7_sample = rdb7_data[1]

print(f"\n{'-' * 80}")
print("KEY STRUCTURAL DIFFERENCES:")
print(f"{'-' * 80}")

# Check for critical attributes
t1x_keys = set(t1x_sample.keys()) if hasattr(t1x_sample, 'keys') else set()
rdb7_keys = set(rdb7_sample.keys()) if hasattr(rdb7_sample, 'keys') else set()

print(f"\nT1X attributes:  {sorted(t1x_keys)}")
print(f"RDB7 attributes: {sorted(rdb7_keys)}")

# Unique to each
only_t1x = t1x_keys - rdb7_keys
only_rdb7 = rdb7_keys - t1x_keys

if only_t1x:
    print(f"\n✓ UNIQUE TO T1X: {sorted(only_t1x)}")
if only_rdb7:
    print(f"\n✓ UNIQUE TO RDB7: {sorted(only_rdb7)}")

common = t1x_keys & rdb7_keys
if common:
    print(f"\n✓ COMMON ATTRIBUTES: {sorted(common)}")

print(f"\n{'-' * 80}")
print("DETAILED FIELD COMPARISON:")
print(f"{'-' * 80}")

# Critical difference 1: Transition State (TS) handling
print(f"\n1. TRANSITION STATE (TS) HANDLING:")
print(f"   T1X ts_idx: {t1x_sample.ts_idx} (scalar per-sample TS index)")
if hasattr(rdb7_sample, 'ts_idx'):
    print(f"   RDB7 ts_idx: {rdb7_sample.ts_idx}")
else:
    print(f"   RDB7 ts_idx: NOT PRESENT (no per-sample TS tracking)")

# Critical difference 2: Coordinates/Positions
print(f"\n2. COORDINATES/POSITIONS:")
if hasattr(t1x_sample, 'pos_N_3_I'):
    pos_shape = t1x_sample.pos_N_3_I.shape
    print(f"   T1X pos_N_3_I shape: {pos_shape}")
    print(f"      → (atoms={pos_shape[0]}, xyz={pos_shape[1]}, images={pos_shape[2]})")
    print(f"      → Contains {pos_shape[2]} images of the reaction path (NEB-like)")
else:
    print(f"   T1X pos_N_3_I: NOT PRESENT")

if hasattr(rdb7_sample, 'pos'):
    pos_shape = rdb7_sample.pos.shape
    print(f"   RDB7 pos shape: {pos_shape}")
    print(f"      → (atoms={pos_shape[0]}, xyz={pos_shape[1]}, images={pos_shape[2] if len(pos_shape) > 2 else 'N/A'})")
else:
    print(f"   RDB7 pos: NOT PRESENT")

# Critical difference 3: Feature dimensions
print(f"\n3. FEATURE DIMENSIONS:")
if hasattr(t1x_sample, 'r_feat_N_Z'):
    print(f"   T1X r_feat_N_Z shape: {t1x_sample.r_feat_N_Z.shape}")
else:
    print(f"   T1X r_feat_N_Z: NOT PRESENT")

if hasattr(rdb7_sample, 'r_feat'):
    print(f"   RDB7 r_feat shape: {rdb7_sample.r_feat.shape}")
else:
    print(f"   RDB7 r_feat: NOT PRESENT")

# Critical difference 4: Indices and mappings
print(f"\n4. INDICES & MAPPINGS:")
print(f"   T1X has:")
if hasattr(t1x_sample, 'r_index_to_mapnum'):
    print(f"      - r_index_to_mapnum (reactant atom → mapping number)")
if hasattr(t1x_sample, 'mapnum_to_r_index'):
    print(f"      - mapnum_to_r_index (mapping number → reactant atom)")
if hasattr(t1x_sample, 'mapnum_to_p_index'):
    print(f"      - mapnum_to_p_index (mapping number → product atom)")

print(f"\n   RDB7 has:")
if hasattr(rdb7_sample, 'rxn_index'):
    print(f"      - rxn_index: {rdb7_sample.rxn_index}")
if hasattr(rdb7_sample, 'augmented'):
    print(f"      - augmented: {rdb7_sample.augmented} (data augmentation flag)")


import pandas as pd

# ============================================================================
# SUMMARY TABLE
# ============================================================================

comparison_data = {
    'Feature': [
        'Transition State Index',
        'Coordinates Field',
        'Coordinate Shape',
        'Reaction Path',
        'Feature Field (Reactant)',
        'Feature Field (Product)',
        'Augmentation Flag',
        'Reaction ID',
        'Atom Type Field',
    ],
    'T1X': [
        'ts_idx (scalar per-sample)',
        'pos_N_3_I',
        '(N_atoms, 3, N_images)',
        'Multi-image NEB path (typically 10 images)',
        'r_feat_N_Z',
        'p_feat_N_Z',
        'Not present',
        'rxn_idx (string ID)',
        'atom_type_N',
    ],
    'RDB7': [
        'Not present',
        'pos',
        '(N_atoms, 3, N_states) — typically 3 states (R, TS, P)',
        'Usually 3 images: reactant, transition state, product',
        'r_feat',
        'p_feat',
        'augmented (bool)',
        'rxn_index (integer)',
        'atom_type',
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + "=" * 120)
print("SIDE-BY-SIDE COMPARISON TABLE:")
print("=" * 120)
print(df_comparison.to_string(index=False))
print("=" * 120)

print("\n" + "▶ KEY TAKEAWAYS:" + "\n")
print("  1. T1X → Per-sample TS index (ts_idx) selects which image is the transition state")
print("       RDB7 → No explicit per-sample TS index; the structure is typically (R, TS, P)")
print()
print("  2. T1X → Multi-image NEB pathway (e.g., 10 images)")
print("       RDB7 → Sparse 3-image representation (R, TS, P)")
print()
print("  3. T1X → Features: r_feat_N_Z, p_feat_N_Z (per-atom feature matrices, shape [N, Z])")
print("       RDB7 → Features: r_feat, p_feat (different dimensions, likely shape [N, 27])")
print()
print("  4. T1X → Reaction identified by rxn_idx (string, e.g., 'rxn2092')")
print("       RDB7 → Reaction identified by rxn_index (integer) + augmented flag")
print()
print("  5. T1X → Coordinates stored as (N, 3, I) — atoms × xyz × images")
print("       RDB7 → Coordinates stored as (N, 3, I) — same shape, but typically I=3")


# ============================================================================
# STEP 1: Build a lookup table of SMILES → data for quick matching
# ============================================================================

from collections import defaultdict

# Create SMILES lookup for both datasets
t1x_by_smiles = {}
for idx, sample in enumerate(t1x_data):
    smiles = sample.smiles
    if smiles not in t1x_by_smiles:
        t1x_by_smiles[smiles] = []
    t1x_by_smiles[smiles].append(idx)

rdb7_by_smiles = {}
for idx, sample in enumerate(rdb7_data):
    smiles = sample.smiles
    if smiles not in rdb7_by_smiles:
        rdb7_by_smiles[smiles] = []
    rdb7_by_smiles[smiles].append(idx)

# Find common SMILES
common_smiles = set(t1x_by_smiles.keys()) & set(rdb7_by_smiles.keys())

print(f"Total unique SMILES in T1X:  {len(t1x_by_smiles)}")
print(f"Total unique SMILES in RDB7: {len(rdb7_by_smiles)}")
print(f"Common reactions (by SMILES): {len(common_smiles)}")

# Show a few examples
print(f"\nFirst 5 common reactions:")
for i, smiles in enumerate(list(common_smiles)[:5]):
    print(f"  {i+1}. T1X indices: {t1x_by_smiles[smiles]}, RDB7 indices: {rdb7_by_smiles[smiles]}")

# ============================================================================
# STEP 2: Helper functions to extract and visualize transition states
# ============================================================================

def get_ts_from_t1x(data_sample):
    """
    Extract transition state coordinates from T1X sample.
    T1X has shape (N_atoms, 3, N_images) and ts_idx tells us which image is the TS.
    """
    ts_idx = data_sample.ts_idx
    # Convert to int if tensor
    try:
        if hasattr(ts_idx, 'item'):  # torch tensor
            ts_idx = int(ts_idx.item())
        else:
            ts_idx = int(ts_idx)
    except:
        ts_idx = int(ts_idx)
    
    pos_N_3_I = data_sample.pos_N_3_I  # shape: (N_atoms, 3, N_images)
    ts_coords = pos_N_3_I[:, :, ts_idx]  # shape: (N_atoms, 3)
    
    return ts_coords, ts_idx

def get_ts_from_rdb7(data_sample):
    """
    Extract transition state coordinates from RDB7 sample.
    RDB7 typically has shape (N_atoms, 3, 3) where index 1 is the TS.
    """
    pos = data_sample.pos  # shape: (N_atoms, 3, 3) or (N_atoms, 3)
    ts_idx = 1
    ts_coords = pos[:, ts_idx, :]
    
    return ts_coords, ts_idx

def atoms_from_coords(coords_N_3, atom_types_N, name="molecule"):
    """
    Create an ASE Atoms object from coordinates and atom types.
    
    Args:
        coords_N_3: (N, 3) numpy array of coordinates
        atom_types_N: (N,) array of atomic numbers or symbol strings
        name: name for the atoms object
    
    Returns:
        ASE Atoms object
    """
    import numpy as np
    
    # Convert coords to numpy if it's a tensor
    if hasattr(coords_N_3, 'cpu'):  # torch tensor
        coords_N_3 = coords_N_3.cpu().numpy()
    else:
        coords_N_3 = np.array(coords_N_3)
    
    # Convert atom_types to numpy if needed
    if hasattr(atom_types_N, 'cpu'):  # torch tensor
        atom_types_N = atom_types_N.cpu().numpy()
    else:
        atom_types_N = np.array(atom_types_N)
    
    # Create Atoms object
    atoms = Atoms(numbers=atom_types_N, positions=coords_N_3)
    return atoms

def kabsch_align(P, Q):
    """
    Align Q onto P using the Kabsch algorithm.
    Both P and Q are (N,3) numpy arrays. Returns rotated Q (aligned to P).
    """
    import numpy as _np

    P = _np.asarray(P)
    Q = _np.asarray(Q)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape for Kabsch alignment")

    # Center
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # Covariance
    C = Q_centered.T @ P_centered
    V, S, Wt = _np.linalg.svd(C)
    d = _np.sign(_np.linalg.det(V @ Wt))
    D = _np.diag([1.0, 1.0, d])
    U = V @ D @ Wt

    Q_rot = (Q_centered @ U)
    # Translate back to P centroid
    Q_aligned = Q_rot + P_centroid
    return Q_aligned

def visualize_two_structures(atoms1, atoms2, title1="Structure 1", title2="Structure 2", backend='matplotlib'):
    """
    Visualize two structures using plotly, nglview, matplotlib, or ase.
    
    Parameters:
    -----------
    atoms1, atoms2 : ASE Atoms objects
    title1, title2 : str
    backend : str
        'plotly': Interactive 3D scatter (side-by-side in HTML)
        'nglview': Molecular viewer with proper bonds (interactive in Jupyter)
        'matplotlib': Static 3D plots side-by-side
        'ase': ASE GUI (if available)
    
    Returns: None, None
    """
    
    # ===== Plotly backend =====
    if backend == 'plotly':
        try:
            import plotly.graph_objects as go
            import numpy as _np
            
            coords1 = atoms1.get_positions()
            coords2 = atoms2.get_positions()
            
            fig = go.Figure()
            
            # Add T1X
            fig.add_trace(go.Scatter3d(
                x=coords1[:, 0], y=coords1[:, 1], z=coords1[:, 2],
                mode='markers',
                name=title1,
                marker=dict(size=5, color='red'),
                text=[f"Atom {i}" for i in range(len(coords1))],
                hoverinfo='text'
            ))
            
            # Add RDB7
            fig.add_trace(go.Scatter3d(
                x=coords2[:, 0], y=coords2[:, 1], z=coords2[:, 2],
                mode='markers',
                name=title2,
                marker=dict(size=5, color='blue'),
                text=[f"Atom {i}" for i in range(len(coords2))],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title=f"{title1} vs {title2}",
                scene=dict(aspectmode='data'),
                hovermode='closest'
            )
            fig.show()
            return None, None
        except Exception as e:
            print("Plotly failed:", e)
            backend = 'matplotlib'  # Fall back

    # ===== nglview backend =====
    if backend == 'nglview':
        try:
            import nglview as nv
            
            # nglview works best with structure files; convert atoms to view
            view1 = nv.show_ase(atoms1)
            view1.add_representation('ball+stick')
            print(f"Showing {title1} in nglview...")
            
            view2 = nv.show_ase(atoms2)
            view2.add_representation('ball+stick')
            print(f"Showing {title2} in nglview...")
            
            return view1, view2
        except Exception as e:
            print("nglview failed:", e)
            backend = 'matplotlib'  # Fall back

    # ===== ASE backend =====
    if backend == 'ase':
        try:
            from ase.visualize import view
            print(f"Opening viewer for: {title1}")
            view(atoms1)
            print(f"Opening viewer for: {title2}")
            view(atoms2)
            return None, None
        except Exception as e:
            print("ASE viewer failed:", e)
            backend = 'matplotlib'  # Fall back

    # ===== Matplotlib fallback =====
    if backend == 'matplotlib':
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            import numpy as _np

            coords1 = atoms1.get_positions()
            coords2 = atoms2.get_positions()

            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

            ax1.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2], s=60)
            ax2.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], s=60)

            ax1.set_title(title1)
            ax2.set_title(title2)

            # Label a few atoms
            for i in range(min(6, len(coords1))):
                ax1.text(coords1[i, 0], coords1[i, 1], coords1[i, 2], str(i), fontsize=8)
            for i in range(min(6, len(coords2))):
                ax2.text(coords2[i, 0], coords2[i, 1], coords2[i, 2], str(i), fontsize=8)

            # Match axes limits for easier comparison
            all_coords = _np.vstack([coords1, coords2])
            mins = all_coords.min(axis=0)
            maxs = all_coords.max(axis=0)
            for ax in (ax1, ax2):
                ax.set_xlim(mins[0], maxs[0])
                ax.set_ylim(mins[1], maxs[1])
                ax.set_zlim(mins[2], maxs[2])

            plt.tight_layout()
            plt.show()
            return None, None
        except Exception as e:
            print("Matplotlib fallback failed:", e)
            return None, None

print("✓ Helper functions loaded successfully!")
print("  - get_ts_from_t1x(): Extract T1X transition state")
print("  - get_ts_from_rdb7(): Extract RDB7 transition state")
print("  - atoms_from_coords(): Create ASE Atoms object")
print("  - visualize_two_structures(): Open side-by-side comparison (ase or matplotlib)")



# ============================================================================
# STEP 3: Pick a common reaction and compare transition states
# ============================================================================

# Pick the first common reaction as an example
example_smiles = list(common_smiles)[0]

# Get indices
t1x_idx = t1x_by_smiles[example_smiles][0]
rdb7_idx = rdb7_by_smiles[example_smiles][0]

# Load the samples
t1x_sample = t1x_data[t1x_idx]
rdb7_sample = rdb7_data[rdb7_idx]

print(f"=" * 100)
print(f"EXAMPLE REACTION COMPARISON")
print(f"=" * 100)
print(f"\nReaction SMILES:\n  {example_smiles}")
print(f"\nT1X sample index:  {t1x_idx}")
print(f"RDB7 sample index: {rdb7_idx}")

# Extract transition states
t1x_ts_coords, t1x_ts_idx = get_ts_from_t1x(t1x_sample)
rdb7_ts_coords, rdb7_ts_idx = get_ts_from_rdb7(rdb7_sample)

print(f"\nT1X TS Information:")
print(f"  - Image index (ts_idx): {t1x_ts_idx}")
print(f"  - Coordinates shape: {t1x_ts_coords.shape}")
print(f"  - Atom types shape: {t1x_sample.atom_type_N.shape}")

print(f"\nRDB7 TS Information:")
print(f"  - Image index: {rdb7_ts_idx} (always middle image)")
print(f"  - Coordinates shape: {rdb7_ts_coords.shape}")
print(f"  - Atom types shape: {rdb7_sample.atom_type.shape}")

# Create ASE Atoms objects
t1x_atoms = atoms_from_coords(t1x_ts_coords, t1x_sample.atom_type_N, f"T1X TS (img {t1x_ts_idx})")
rdb7_atoms = atoms_from_coords(rdb7_ts_coords, rdb7_sample.atom_type, f"RDB7 TS (img {rdb7_ts_idx})")

print(f"\n✓ ASE Atoms objects created for comparison")



# ============================================================================
# STEP 4: Visualize the transition states with ASE GUI
# ============================================================================

# Uncomment the line below to open ASE GUI windows for 3D visualization
# Note: This opens interactive windows that allow rotation, zooming, etc.

# gui1, gui2 = visualize_two_structures(
#     t1x_atoms, 
#     rdb7_atoms,
#     title1=f"T1X TS (image {t1x_ts_idx})",
#     title2=f"RDB7 TS (image {rdb7_ts_idx})"
# )

print("\n" + "=" * 100)
print("TO VISUALIZE IN 3D:")
print("=" * 100)
print("\nUncomment the code cell below and run it to open interactive ASE GUI windows.")

# Calculate some basic geometry comparison
import numpy as np

def coords_to_np(coords):
    """Convert coordinates to numpy array."""
    if hasattr(coords, 'cpu'):
        return coords.cpu().numpy()
    return np.array(coords)

t1x_coords_np = coords_to_np(t1x_ts_coords)
rdb7_coords_np = coords_to_np(rdb7_ts_coords)

# Compute centroid
t1x_centroid = t1x_coords_np.mean(axis=0)
rdb7_centroid = rdb7_coords_np.mean(axis=0)

print(f"\nCentroid (center of mass):")
print(f"  T1X:  {t1x_centroid}")
print(f"  RDB7: {rdb7_centroid}")
print(f"  Difference: {np.linalg.norm(t1x_centroid - rdb7_centroid):.4f} Å")

# Center both structures
t1x_centered = t1x_coords_np - t1x_centroid
rdb7_centered = rdb7_coords_np - rdb7_centroid

# Compute pairwise distances (first 5 atoms)
print(f"\nPairwise distances (inter-atomic) - first 5 atoms:")
print(f"  T1X distances from atom 0:")
for i in range(1, min(6, len(t1x_centered))):
    dist = np.linalg.norm(t1x_centered[i] - t1x_centered[0])
    print(f"    Atom 0-{i}: {dist:.4f} Å")

print(f"\n  RDB7 distances from atom 0:")
for i in range(1, min(6, len(rdb7_centered))):
    dist = np.linalg.norm(rdb7_centered[i] - rdb7_centered[0])
    print(f"    Atom 0-{i}: {dist:.4f} Å")

print(f"\n✓ Ready to visualize! See next cell for ASE GUI launch code.")


# ============================================================================
# STEP 5: Reusable function to compare ANY reaction by SMILES
# ============================================================================

def compare_reaction_by_smiles(smiles_string, visualize=False, visualize_backend='matplotlib', align=True):
    """
    Compare transition states for a given reaction SMILES string.
    
    Parameters:
    -----------
    smiles_string : str
        The reaction SMILES string to search for
    visualize : bool
        If True, open viewers for 3D visualization
    visualize_backend : str
        'plotly': Interactive 3D scatter plots in HTML (recommended!)
        'nglview': Molecular viewer with bonds (best in Jupyter)
        'matplotlib': Static 3D plots side-by-side
        'ase': ASE GUI (requires Tcl/Tk)
    align : bool
        If True, align the RDB7 coordinates to the T1X coordinates using Kabsch
    
    Returns:
    --------
    dict with comparison data
    """
    
    # Find the reaction in both datasets
    if smiles_string not in common_smiles:
        print(f"❌ Reaction not found in both datasets")
        print(f"   Check that the SMILES matches exactly")
        return None
    
    t1x_idx = t1x_by_smiles[smiles_string][0]
    rdb7_idx = rdb7_by_smiles[smiles_string][0]
    
    t1x_sample = t1x_data[t1x_idx]
    rdb7_sample = rdb7_data[rdb7_idx]
    
    # Extract TS
    t1x_ts_coords, t1x_ts_idx = get_ts_from_t1x(t1x_sample)
    rdb7_ts_coords, rdb7_ts_idx = get_ts_from_rdb7(rdb7_sample)
    
    # Convert to numpy for alignment/visualization
    import numpy as _np
    if hasattr(t1x_ts_coords, 'cpu'):
        t1x_np = t1x_ts_coords.cpu().numpy()
    else:
        t1x_np = _np.array(t1x_ts_coords)
    if hasattr(rdb7_ts_coords, 'cpu'):
        rdb7_np = rdb7_ts_coords.cpu().numpy()
    else:
        rdb7_np = _np.array(rdb7_ts_coords)

    if align:
        try:
            rdb7_np_aligned = kabsch_align(t1x_np, rdb7_np)
        except Exception as e:
            print("Kabsch alignment failed:", e)
            rdb7_np_aligned = rdb7_np
    else:
        rdb7_np_aligned = rdb7_np

    # Create Atoms objects using aligned coords for RDB7
    t1x_atoms = atoms_from_coords(t1x_np, t1x_sample.atom_type_N)
    rdb7_atoms = atoms_from_coords(rdb7_np_aligned, rdb7_sample.atom_type)
    
    print(f"\n{'='*100}")
    print(f"REACTION: {smiles_string[:80]}...")
    print(f"{'='*100}")
    print(f"\nT1X index: {t1x_idx}, TS image: {t1x_ts_idx}")
    print(f"RDB7 index: {rdb7_idx}, TS image: {rdb7_ts_idx}")
    print(f"\nT1X TS atoms: {len(t1x_np)}, RDB7 TS atoms: {len(rdb7_np)}")
    
    if visualize:
        print(f"\nOpening 3D visualization (backend={visualize_backend})...")
        visualize_two_structures(t1x_atoms, rdb7_atoms, title1=f"T1X TS (img {t1x_ts_idx})", title2=f"RDB7 TS", backend=visualize_backend)
        return {
            't1x_atoms': t1x_atoms,
            'rdb7_atoms': rdb7_atoms,
            't1x_gui': None,
            'rdb7_gui': None
        }
    else:
        return {
            't1x_atoms': t1x_atoms,
            'rdb7_atoms': rdb7_atoms,
            't1x_gui': None,
            'rdb7_gui': None
        }

# Example: How to use it
print("\n" + "=" * 100)
print("HOW TO USE:")
print("=" * 100)
print("""
# To compare any reaction by SMILES:
result = compare_reaction_by_smiles(
    smiles_string="YOUR_REACTION_SMILES_HERE",
    visualize=True,               # open viewer
    visualize_backend='matplotlib',# 'ase' or 'matplotlib'
    align=True                    # align RDB7 to T1X
)

# Or get a list of available reactions:
print(list(common_smiles)[:10])  # First 10 common reactions

# Pick one and compare:
example_smiles = list(common_smiles)[5]  # Or any index
result = compare_reaction_by_smiles(example_smiles, visualize=True, visualize_backend='matplotlib', align=True)
""")

print("\n✓ Comparison function ready!")



if __name__ == "__main__":
    # Pick one and compare:
    example_smiles = list(common_smiles)[9000]  # Or any index
    # Use 'plotly' for interactive HTML, 'matplotlib' for static plots, or 'nglview' in Jupyter
    result = compare_reaction_by_smiles(example_smiles, visualize=True, visualize_backend='plotly', align=True)
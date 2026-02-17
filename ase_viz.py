#!/usr/bin/env python3
"""
Load GoFlow predictions and visualize NEB pathway using ASE
Run on cluster: python visualize_ase.py --pkl_path logs/run_name/test_samples/samples_all.pkl --rxn_idx 0
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
import sys

# Try importing ASE - install if needed on cluster
try:
    import ase
    from ase import Atoms
    from ase.io import write, read
    from ase.visualize import view
    from ase.visualize.plot import plot_atoms
except ImportError:
    print("ASE not found. Install with: pip install ase")
    sys.exit(1)

# Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

# Atomic number mapping
ATOMIC_NUM = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53
}

SYMBOL_MAP = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 9: 'F', 17: 'Cl', 35: 'Br', 53: 'I'}

def load_predictions(pkl_path):
    """Load pickle file with predictions"""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results

def get_element_symbols(data):
    """Extract element symbols from atom_type indices"""
    # Try atom_type_N first (ground truth data)
    if hasattr(data, 'atom_type_N'):
        atom_types = data.atom_type_N.numpy() if hasattr(data.atom_type_N, 'numpy') else data.atom_type_N
        symbols = [SYMBOL_MAP.get(int(at), 'X') for at in atom_types]
    # Fall back to atom_type
    elif hasattr(data, 'atom_type'):
        atom_types = data.atom_type.numpy() if hasattr(data.atom_type, 'numpy') else data.atom_type
        symbols = [SYMBOL_MAP.get(int(at), 'X') for at in atom_types]
    else:
        # Fallback: guess from SMILES (basic implementation)
        symbols = ['C'] * data.pos_gen_neb_I_T_Nm_3.shape[2]
    return symbols

def create_ase_trajectory(data, neb_img_idx=None, sample_idx=None):
    """
    Create ASE Atoms objects for a reaction pathway
    
    Args:
        data: PyG Data object from predictions
        neb_img_idx: Which NEB image to visualize (None = all)
        sample_idx: Which sample to use (None = best/median)
    
    Returns:
        List of ASE Atoms objects (one per ODE step)
    """
    symbols = get_element_symbols(data)
    
    # Select geometry
    if hasattr(data, 'pos_gen_neb_I_T_Nm_3'):
        # pos_gen_neb_I_T_Nm_3: [I, T, N, 3]
        pos_I_T_N_3 = data.pos_gen_neb_I_T_Nm_3.numpy()
        
        if neb_img_idx is not None:
            # Single NEB image trajectory
            pos_T_N_3 = pos_I_T_N_3[neb_img_idx]
        else:
            # Take first image (TS in most cases)
            pos_T_N_3 = pos_I_T_N_3[0]
    else:
        # Fallback
        pos_T_N_3 = data.pos_gen
    
    # Create ASE trajectory
    trajectory = []
    for t_step in range(pos_T_N_3.shape[0]):
        coords = pos_T_N_3[t_step]
        atoms = Atoms(symbols=symbols, positions=coords)
        trajectory.append(atoms)
    
    return trajectory

def save_trajectory_xyz(trajectory, output_file):
    """Save trajectory as XYZ file readable by ASE and external viewers"""
    with open(output_file, 'w') as f:
        for frame_idx, atoms in enumerate(trajectory):
            f.write(f"{len(atoms)}\n")
            f.write(f"Frame {frame_idx}\n")
            for atom in atoms:
                f.write(f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n")

def visualize_on_cluster(trajectory, output_base, save_only_final=False):
    """
    Create visualization files for cluster environments (no GUI)
    
    Args:
        trajectory: List of ASE Atoms objects
        output_base: Base filename for output
        save_only_final: If True, only save the final frame as XYZ
    """
    # Save as trajectory file (full trajectory)
    traj_file = f"{output_base}.traj"
    write(traj_file, trajectory)
    print(f"✓ Saved ASE trajectory: {traj_file}")
    
    if save_only_final:
        # Save only final frame as XYZ
        xyz_file = f"{output_base}.xyz"
        write(xyz_file, trajectory[-1])
        print(f"✓ Saved final frame: {xyz_file}")
    else:
        # Save full trajectory as XYZ
        xyz_file = f"{output_base}.xyz"
        save_trajectory_xyz(trajectory, xyz_file)
        print(f"✓ Saved XYZ trajectory: {xyz_file}")
        
        # Save per-frame XYZ files
        for i, atoms in enumerate(trajectory):
            write(f"{output_base}_frame_{i:04d}.xyz", atoms)
        print(f"✓ Saved {len(trajectory)} individual frames: {output_base}_frame_*.xyz")


def visualize_all_neb_images(data, output_dir, rxn_idx, save_only_final=True):
    """
    Visualize all NEB images for a single reaction
    
    Args:
        data: PyG Data object
        output_dir: Output directory
        rxn_idx: Reaction index
        save_only_final: If True, save only final frame per image as XYZ
    """
    if not hasattr(data, 'pos_gen_neb_I_T_Nm_3'):
        print("Error: No NEB image data available")
        return
    
    pos_I_T_N_3 = data.pos_gen_neb_I_T_Nm_3.numpy()
    num_images = pos_I_T_N_3.shape[0]
    
    print(f"Extracting all {num_images} NEB images for reaction {rxn_idx}...")
    
    for neb_idx in range(num_images):
        # Create trajectory for this NEB image
        trajectory = []
        pos_T_N_3 = pos_I_T_N_3[neb_idx]
        symbols = get_element_symbols(data)
        
        for t_step in range(pos_T_N_3.shape[0]):
            coords = pos_T_N_3[t_step]
            atoms = Atoms(symbols=symbols, positions=coords)
            trajectory.append(atoms)
        
        # Save this NEB image (with rxn_idx in filename)
        output_base = f"{output_dir}/rxn{rxn_idx}_neb{neb_idx:02d}"
        visualize_on_cluster(trajectory, output_base, save_only_final=save_only_final)
        print(f"  [{neb_idx+1}/{num_images}] NEB image {neb_idx}")
    
    print(f"✓ All {num_images} NEB images saved!")

def plot_neb_pathway_2d(output_dir, rxn_idx, num_images, figsize=(20, 4)):
    """
    Create a 2D side-by-side visualization of all NEB images (final ODE step)
    """
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]
    
    print(f"Creating 2D pathway visualization...")
    
    for neb_idx in range(num_images):
        xyz_file = f"{output_dir}/rxn{rxn_idx}_neb{neb_idx:02d}.xyz"
        
        if not Path(xyz_file).exists():
            print(f"Warning: {xyz_file} not found, skipping")
            continue
        
        # Read last frame from trajectory
        try:
            atoms = read(xyz_file, index=-1)
            ax = axes[neb_idx]
            
            # Use ASE's plot function for nice 2D projection
            plot_atoms(atoms, ax, radii=0.3, rotation=('10x,10y,0z'))
            ax.set_title(f'Image {neb_idx}', fontsize=10)
            ax.axis('off')
        except Exception as e:
            print(f"Error plotting NEB image {neb_idx}: {e}")
    
    plt.tight_layout()
    
    # Save as SVG and PDF
    svg_path = f"{output_dir}/neb_pathway_2d_rxn{rxn_idx}.svg"
    pdf_path = f"{output_dir}/neb_pathway_2d_rxn{rxn_idx}.pdf"
    
    fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"✓ Saved 2D pathway: {svg_path}")
    print(f"✓ Saved 2D pathway: {pdf_path}")

def plot_neb_pathway_3d(output_dir, rxn_idx, num_images, figsize=(20, 4)):
    """
    Create a 3D side-by-side visualization of all NEB images
    """
    fig = plt.figure(figsize=figsize)
    
    print(f"Creating 3D pathway visualization...")
    
    for neb_idx in range(num_images):
        xyz_file = f"{output_dir}/rxn{rxn_idx}_neb{neb_idx:02d}.xyz"
        
        if not Path(xyz_file).exists():
            continue
        
        try:
            atoms = read(xyz_file, index=-1)
            
            ax = fig.add_subplot(1, num_images, neb_idx + 1, projection='3d')
            
            # Get positions and atom types
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            
            # Color map for elements
            color_map = {'H': 'white', 'C': 'gray', 'N': 'blue', 
                        'O': 'red', 'S': 'yellow', 'F': 'green',
                        'Cl': 'green', 'Br': 'brown', 'I': 'purple'}
            colors = [color_map.get(s, 'pink') for s in symbols]
            
            # Size map for elements
            size_map = {'H': 50, 'C': 100, 'N': 100, 'O': 100, 
                       'S': 120, 'F': 80, 'Cl': 120, 'Br': 140, 'I': 160}
            sizes = [size_map.get(s, 100) for s in symbols]
            
            # Plot atoms
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Draw bonds (simple: connect nearby atoms)
            for i in range(len(atoms)):
                for j in range(i+1, len(atoms)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < 1.8:  # Bond threshold in Angstroms
                        ax.plot([positions[i, 0], positions[j, 0]],
                               [positions[i, 1], positions[j, 1]],
                               [positions[i, 2], positions[j, 2]],
                               'k-', alpha=0.3, linewidth=1)
            
            ax.set_title(f'Image {neb_idx}', fontsize=10)
            ax.set_xlabel('X (Å)', fontsize=8)
            ax.set_ylabel('Y (Å)', fontsize=8)
            ax.set_zlabel('Z (Å)', fontsize=8)
            ax.tick_params(labelsize=6)
            
            # Equal aspect ratio
            max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                                 positions[:, 1].max()-positions[:, 1].min(),
                                 positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
            mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
            mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
            mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
        except Exception as e:
            print(f"Error plotting 3D NEB image {neb_idx}: {e}")
    
    plt.tight_layout()
    
    # Save as SVG and PDF
    svg_path = f"{output_dir}/neb_pathway_3d_rxn{rxn_idx}.svg"
    pdf_path = f"{output_dir}/neb_pathway_3d_rxn{rxn_idx}.pdf"
    
    fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"✓ Saved 3D pathway: {svg_path}")
    print(f"✓ Saved 3D pathway: {pdf_path}")

def plot_trajectory_animation(output_dir, rxn_idx, neb_idx):
    """
    Create a visualization showing ODE trajectory for a single NEB image
    """
    xyz_file = f"{output_dir}/rxn{rxn_idx}_neb{neb_idx:02d}.xyz"
    
    if not Path(xyz_file).exists():
        print(f"Warning: {xyz_file} not found")
        return
    
    try:
        # Read all frames
        frames = read(xyz_file, index=':')
        num_frames = len(frames)
        
        # Create figure with subplots for selected frames
        selected_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, -1]
        fig, axes = plt.subplots(1, len(selected_indices), figsize=(20, 4))
        
        for i, idx in enumerate(selected_indices):
            atoms = frames[idx]
            plot_atoms(atoms, axes[i], radii=0.3, rotation=('10x,10y,0z'))
            axes[i].set_title(f'Step {idx if idx >= 0 else num_frames + idx}', fontsize=10)
            axes[i].axis('off')
        
        plt.suptitle(f'ODE Integration Trajectory - NEB Image {neb_idx}', fontsize=14)
        plt.tight_layout()
        
        # Save
        svg_path = f"{output_dir}/trajectory_neb{neb_idx:02d}_rxn{rxn_idx}.svg"
        pdf_path = f"{output_dir}/trajectory_neb{neb_idx:02d}_rxn{rxn_idx}.pdf"
        
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"✓ Saved ODE trajectory: {svg_path}")
        
    except Exception as e:
        print(f"Error creating trajectory animation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize GoFlow predictions with ASE")
    parser.add_argument("--pkl_path", required=True, help="Path to samples_all.pkl")
    parser.add_argument("--list", action="store_true", help="List all reactions in the pickle file with indices")
    parser.add_argument("--rxn_idx", type=int, default=0, help="Reaction index to visualize")
    parser.add_argument("--neb_img", type=int, default=None, help="NEB image index (None=TS only)")
    parser.add_argument("--all_neb", action="store_true", help="Extract all NEB images for reaction")
    parser.add_argument("--save_only_final", action="store_true", help="Save only final frame per NEB image (not full trajectory)")
    parser.add_argument("--plot", action="store_true", help="Create publication-quality plots (SVG/PDF)")
    parser.add_argument("--plot_2d", action="store_true", help="Create 2D pathway visualization")
    parser.add_argument("--plot_3d", action="store_true", help="Create 3D pathway visualization")
    parser.add_argument("--plot_trajectory", type=int, default=None, help="Plot ODE trajectory for specific NEB image")
    parser.add_argument("--output_dir", default="ase_visualizations", help="Output directory")
    parser.add_argument("--view", action="store_true", help="Open in ASE GUI if available (local only)")
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.pkl_path}...")
    results = load_predictions(args.pkl_path)
    
    # List all reactions if requested
    if args.list:
        print(f"\nFound {len(results)} reactions in {args.pkl_path}\n")
        print(f"{'Idx':<5} {'Atoms':<7} {'NEB Imgs':<10} {'SMILES':<80}")
        print("=" * 105)
        for idx, data in enumerate(results):
            num_atoms = data.pos_gen_neb_I_T_Nm_3.shape[2] if hasattr(data, 'pos_gen_neb_I_T_Nm_3') else 0
            num_neb = data.pos_gen_neb_I_T_Nm_3.shape[0] if hasattr(data, 'pos_gen_neb_I_T_Nm_3') else 0
            smiles = data.smiles if hasattr(data, 'smiles') else 'N/A'
            print(f"{idx:<5} {num_atoms:<7} {num_neb:<10} {smiles:<80}")
        print("\nUse --rxn_idx <idx> to visualize a specific reaction")
        return
    # Validate rxn_idx
    if args.rxn_idx >= len(results):
        print(f"Error: Reaction index {args.rxn_idx} out of range (max: {len(results)-1})")
        print(f"Use --list to see all available reactions")
        sys.exit(1)
    
    data = results[args.rxn_idx]
    
    print(f"Reaction SMILES: {data.smiles}")
    print(f"Number of atoms: {data.pos_gen_neb_I_T_Nm_3.shape[2]}")
    num_neb_images = data.pos_gen_neb_I_T_Nm_3.shape[0]
    print(f"NEB images: {num_neb_images}")
    print(f"ODE steps: {data.pos_gen_neb_I_T_Nm_3.shape[1]}")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Handle all NEB images or single image
    if args.all_neb:
        visualize_all_neb_images(data, args.output_dir, args.rxn_idx, save_only_final=args.save_only_final)
    else:
        # Create trajectory for single NEB image
        trajectory = create_ase_trajectory(data, neb_img_idx=args.neb_img)
        print(f"Created trajectory with {len(trajectory)} frames")
        
        # Save output (with rxn_idx in filename)
        img_label = f"_neb{args.neb_img:02d}" if args.neb_img is not None else "_ts"
        output_base = f"{args.output_dir}/rxn{args.rxn_idx}{img_label}"
        visualize_on_cluster(trajectory, output_base, save_only_final=args.save_only_final)
        # View if requested
        if args.view:
            try:
                view(trajectory)
            except Exception as e:
                print(f"Could not open GUI: {e}")
                print("Files saved successfully for offline viewing")
    
    # Create publication plots if requested
    if args.plot or args.plot_2d:
        plot_neb_pathway_2d(args.output_dir, args.rxn_idx, num_neb_images)
    
    if args.plot or args.plot_3d:
        plot_neb_pathway_3d(args.output_dir, args.rxn_idx, num_neb_images)
    
    if args.plot_trajectory is not None:
        plot_trajectory_animation(args.output_dir, args.rxn_idx, args.plot_trajectory)

if __name__ == "__main__":
    main()
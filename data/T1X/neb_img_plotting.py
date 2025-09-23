import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
import re
import os
import argparse

# Command line argument parsing
parser = argparse.ArgumentParser(description="Choose between individual reaction and reaction with largest distance moved.")
parser.add_argument("--rxn_num", help="Choose the individual reaction number", default=None, type=int)
args = parser.parse_args()

SAVE_PATH = f"figures/{args.rxn_num if args.rxn_num is not None else 'interesting_reaction'}/"

def find_interesting_reaction(data, min_displacement=0.5):
    """
    Find reactions with significant atomic movement for better visualization
    """
    interesting_reactions = []
    
    for idx, reaction in enumerate(data[:100]):  # Check first 100
        if hasattr(reaction, 'pos_N_3_I'):
            pos = reaction.pos_N_3_I.numpy()
            num_atoms, _, num_images = pos.shape
            
            # Calculate total displacement for each atom
            total_displacement = 0
            for atom_idx in range(num_atoms):
                start_pos = pos[atom_idx, :, 0]
                end_pos = pos[atom_idx, :, -1]
                displacement = np.linalg.norm(end_pos - start_pos)
                total_displacement += displacement
            
            avg_displacement = total_displacement / num_atoms
            
            if avg_displacement > min_displacement:
                interesting_reactions.append({
                    'idx': idx,
                    'smiles': reaction.smiles,
                    'avg_displacement': avg_displacement,
                    'num_atoms': num_atoms,
                    'num_images': num_images
                })
    
    # Sort by displacement
    interesting_reactions.sort(key=lambda x: x['avg_displacement'], reverse=True)
    
    print("Most interesting reactions for visualization:")
    for i, rxn in enumerate(interesting_reactions[:5]):
        print(f"{i+1}. Index {rxn['idx']}: {rxn['smiles']}")
        print(f"   Avg displacement: {rxn['avg_displacement']:.3f} Å")
        print(f"   Atoms: {rxn['num_atoms']}, Images: {rxn['num_images']}")
        print()
    
    return interesting_reactions


def plot_neb_trajectory_3d(neb_images, atom_types=None, save_path=None):
    """
    Plot all NEB images in a 3D trajectory
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    num_images, num_atoms, _ = neb_images.shape
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different NEB images
    colors = plt.cm.viridis(np.linspace(0, 1, num_images))
    
    for img_idx in range(num_images):
        positions = neb_images[img_idx]  # [atoms, 3]
        
        # Plot atoms as points
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c=[colors[img_idx]], s=100, alpha=0.7, 
                  label=f'Image {img_idx}')
        
        # Connect atoms with bonds (simple - connect consecutive atoms)
        for atom_idx in range(num_atoms-1):
            ax.plot([positions[atom_idx, 0], positions[atom_idx+1, 0]],
                   [positions[atom_idx, 1], positions[atom_idx+1, 1]], 
                   [positions[atom_idx, 2], positions[atom_idx+1, 2]],
                   c=colors[img_idx], alpha=0.3, linewidth=1)
    
    # Draw trajectory lines for each atom
    for atom_idx in range(num_atoms):
        atom_traj = neb_images[:, atom_idx, :]  # [images, 3]
        ax.plot(atom_traj[:, 0], atom_traj[:, 1], atom_traj[:, 2], 
               'k--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'NEB Reaction Pathway - {num_images} Images')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=num_images-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('NEB Image Index')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.close()
    return fig, ax

def save_individual_neb_images(neb_images, reaction_data, output_dir=None):
    """
    Save each NEB image as individual XYZ file for external visualization
    """
    if output_dir is None:
        output_dir = os.path.join(SAVE_PATH, 'neb_snapshots')
    os.makedirs(output_dir, exist_ok=True)
    
    num_images, num_atoms, _ = neb_images.shape
    
    # Get atom types if available
    if hasattr(reaction_data, 'atom_type'):
        atom_types = reaction_data.atom_type.numpy()
        # Convert atom type indices to element symbols (you might need to adjust this)
        element_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}  # Add more as needed
        elements = [element_map.get(at, 'X') for at in atom_types]
    else:
        elements = ['C'] * num_atoms  # Default to carbon
    
    for img_idx in range(num_images):
        xyz_file = os.path.join(output_dir, f'neb_image_{img_idx:02d}.xyz')
        
        with open(xyz_file, 'w') as f:
            f.write(f"{num_atoms}\n")
            f.write(f"NEB Image {img_idx} - {reaction_data.smiles}\n")
            
            for atom_idx in range(num_atoms):
                pos = neb_images[img_idx, atom_idx]
                f.write(f"{elements[atom_idx]} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    
    print(f"Saved {num_images} XYZ files to {output_dir}/")

def create_neb_animation(neb_images, save_path=None):
    """
    Create an animated GIF showing the reaction progression
    """
    from matplotlib.animation import FuncAnimation
    
    num_images, num_atoms, _ = neb_images.shape
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot limits
    all_coords = neb_images.reshape(-1, 3)
    margin = 1.0
    ax.set_xlim([all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin])
    ax.set_ylim([all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin])
    ax.set_zlim([all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin])
    
    def animate(frame):
        ax.clear()
        ax.set_xlim([all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin])
        ax.set_ylim([all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin])
        ax.set_zlim([all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin])
        
        positions = neb_images[frame]
        
        # Plot current frame
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=100, alpha=0.8)
        
        # Connect atoms with bonds
        for atom_idx in range(num_atoms-1):
            ax.plot([positions[atom_idx, 0], positions[atom_idx+1, 0]],
                   [positions[atom_idx, 1], positions[atom_idx+1, 1]], 
                   [positions[atom_idx, 2], positions[atom_idx+1, 2]],
                   'b-', alpha=0.6, linewidth=2)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'NEB Image {frame}/{num_images-1}')
    
    anim = FuncAnimation(fig, animate, frames=num_images, interval=500, repeat=True)
    anim.save(save_path, writer='pillow', fps=2)
    plt.close()
    
    return anim

def strip_atom_mapping(smiles):
    # Remove atom mapping numbers, e.g. [O:1] -> [O]
    return re.sub(r'\:([0-9]+)', '', smiles)

def get_elements_from_smiles(smiles, neb_images):
    # If reaction SMILES, take first reactant
    if '>>' in smiles:
        reactants = smiles.split('>>')[0]
        # If multiple reactants, take the first one
        reactant = reactants.split('.')[
            0] if '.' in reactants else reactants
    else:
        reactant = smiles
    reactant = strip_atom_mapping(reactant)
    mol = Chem.MolFromSmiles(reactant)
    if mol is None:
        raise ValueError("Invalid SMILES string after stripping atom mapping")
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # add hydrogens
    num_atoms = neb_images.shape[1]
    num_hydrogens = num_atoms - len(elements)
    elements_with_h = elements + ['H'] * num_hydrogens
    return elements_with_h

def plot_molecular_structure(neb_images, elements, frame_idx=0, 
                           atomic_colors=None, save_path=None):
    """
    Plot single molecular structure with proper atomic colors
    """
    if atomic_colors is None:
        # Standard CPK colors
        atomic_colors = {
            'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red', 
            'S': 'yellow', 'P': 'orange', 'F': 'lightgreen',
            'Cl': 'green', 'Br': 'darkred', 'I': 'purple'
        }
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = neb_images[frame_idx]
    
    # Plot atoms
    for i, (pos, element) in enumerate(zip(positions, elements)):
        color = atomic_colors.get(element, 'gray')
        size = 200 if element != 'H' else 100  # Smaller for hydrogen
        ax.scatter(pos[0], pos[1], pos[2], 
                  c=color, s=size, alpha=0.8, 
                  edgecolors='black', linewidth=1)
        
        # Add atom labels
        # Offset label to the side (e.g., along +x direction)
        offset = 0.2  # adjust as needed for clarity
        ax.text(pos[0] + offset, pos[1], pos[2], f'{element}{i}', 
            fontsize=8, ha='left', va='center')
    
    # Draw bonds (simple distance-based)
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            # Typical bond lengths (adjust as needed)
            max_bond_length = 2.0  # Angstroms
            if dist < max_bond_length:
                ax.plot([positions[i][0], positions[j][0]],
                       [positions[i][1], positions[j][1]],
                       [positions[i][2], positions[j][2]], 
                       'k-', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Molecular Structure - Frame {frame_idx}')
    
    # Equal aspect ratio
    max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                         positions[:,1].max()-positions[:,1].min(),
                         positions[:,2].max()-positions[:,2].min()]).max() / 2.0
    mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
    mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
    mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig, ax

def run(rxn_idx=None):
    if rxn_idx is not None:
        print(f"Using reaction {rxn_idx} for visualization")
        print(f"SMILES: {data[rxn_idx].smiles}")
        reaction_data = data[rxn_idx]
        neb_images = reaction_data.pos_N_3_I.numpy().transpose(2, 0, 1)

        # Create all visualizations
        fig, ax = plot_neb_trajectory_3d(neb_images, save_path=SAVE_PATH+f"neb_trajectory_3d_{rxn_idx}.pdf")
        save_individual_neb_images(neb_images, reaction_data, output_dir=SAVE_PATH+f'neb_snapshots_{rxn_idx}')
        anim = create_neb_animation(neb_images, save_path=SAVE_PATH+f'neb_animation_{rxn_idx}.gif')

        # Extract elements from SMILES
        elements = get_elements_from_smiles(reaction_data.smiles, neb_images)

        # Usage
        for i in range(len(neb_images)):
            plot_molecular_structure(neb_images, elements, frame_idx=i, save_path=f'{SAVE_PATH}molecule_frame_{i:02d}_{rxn_idx}.pdf')

    else:
        # Find good reactions for visualization
        interesting = find_interesting_reaction(data)

        # Use the most interesting one
        if interesting:
            best_idx = interesting[0]['idx']
            print(f"Using reaction {best_idx} for visualization")
            print(f"SMILES: {interesting[0]['smiles']}")
            reaction_data = data[best_idx]
            neb_images = reaction_data.pos_N_3_I.numpy().transpose(2, 0, 1)

            # Create all visualizations
            fig, ax = plot_neb_trajectory_3d(neb_images, save_path=SAVE_PATH+f"neb_trajectory_3d_{best_idx}.pdf")
            save_individual_neb_images(neb_images, reaction_data, output_dir=SAVE_PATH+f'neb_snapshots_{best_idx}')
            anim = create_neb_animation(neb_images, save_path=SAVE_PATH+f'neb_animation_{best_idx}.gif')

            # Extract elements from SMILES
            elements = get_elements_from_smiles(reaction_data.smiles, neb_images)

            # Usage
            for i in range(len(neb_images)):
                plot_molecular_structure(neb_images, elements, frame_idx=i, save_path=SAVE_PATH+f'molecule_frame_{i:02d}_{best_idx}.pdf')
        else:
            print("No interesting reactions found.")


if __name__ == "__main__":

    # Load data.pkl
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Load feat_dict.pkl
    with open('feat_dict.pkl', 'rb') as f:
        feat_dict = pickle.load(f)

    run(rxn_idx=args.rxn_num)
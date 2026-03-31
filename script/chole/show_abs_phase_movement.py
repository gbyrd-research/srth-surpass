import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import os

def calculate_absolute_movement(values):
    """Calculate the absolute movement in 1D or 3D space."""
    diffs = np.diff(values, axis=0)
    
    # Check if diffs is 1D or 2D
    if diffs.ndim == 1:
        distances = np.abs(diffs)  # For 1D data, just take the absolute value of the differences
    else:
        distances = np.linalg.norm(diffs, axis=1)  # For 2D or 3D data, calculate the norm along axis 1
    
    distances = np.insert(distances, 0, 0)  # Insert zero at the beginning
    return distances

def plot_absolute_movements(base_path, output_path, phase_idx, num_demos_to_plot=3, scale_by_max=False, first_last_index_range=None, show_flag=True):
    if scale_by_max:
        print("Scaling the absolute movements by the maximum value.")
        output_path = output_path / "scaled_by_max"
    
    num_demos_plotted = 0
    while num_demos_plotted < num_demos_to_plot:
        # List all folders starting with "tissue"
        tissue_folders = [folder for folder in Path(base_path).glob('tissue_*') if folder.is_dir()]
        
        if not tissue_folders:
            raise ValueError(f"No tissue folders found in the specified dataset path: {base_path}")

        # Randomly select one tissue folder
        selected_tissue_folder_path = random.choice(tissue_folders)
        selected_tissue_folder_name = selected_tissue_folder_path.name
        print(f"Selected tissue folder: {selected_tissue_folder_name}")

        # Construct the phase folder path
        phase_folder_start = f"{phase_idx}_*"
        phase_folder_paths = list(selected_tissue_folder_path.glob(phase_folder_start))
        if not phase_folder_paths:
            print(f"No phase folders found for phase index {phase_idx} in tissue {selected_tissue_folder_path}\n")
            continue
        elif len(phase_folder_paths) == 1:
            phase_folder_path = phase_folder_paths[0]
        else:
            phase_folder_path = random.choice(phase_folder_paths) 
        phase_folder_name = phase_folder_path.name
        
        # Get all demo folders for that specific tissue and phase
        date_folders = list(phase_folder_path.glob('*-*'))
        if not date_folders:
            print(f"No demo folders found for tissue {selected_tissue_folder_path} and phase {phase_idx}\n")
            continue
        
        # Select a random demo folder
        selected_date_folder_path = random.choice(date_folders)
        selected_date_folder_name = selected_date_folder_path.name
        print(f"Selected demo folder: {selected_date_folder_name}\n")
        
        # Load the ee_csv.csv file
        ee_csv_path = selected_date_folder_path / 'ee_csv.csv'
        if not ee_csv_path.exists():
            print(f"No kinematics csv file found at {ee_csv_path}\n")
            continue
        
        ee_df = pd.read_csv(ee_csv_path)
        
        # Extract the absolute movement (position) for psm1 and psm2
        psm1_positions = ee_df[['psm1_pose.position.x', 'psm1_pose.position.y', 'psm1_pose.position.z']].values
        psm2_positions = ee_df[['psm2_pose.position.x', 'psm2_pose.position.y', 'psm2_pose.position.z']].values
        
        # Scale the absolute movement by the maximum value
        if scale_by_max:
            psm1_positions /= np.max(psm1_positions, axis=0)
            psm2_positions /= np.max(psm2_positions, axis=0)
        
        # Extract the jaw positions for psm1 and psm2
        psm1_jaw = ee_df['psm1_jaw'].values
        psm2_jaw = ee_df['psm2_jaw'].values
        
        # Determine indices to plot
        indices_to_plot = [(0, len(psm1_positions)), (0, first_last_index_range), (len(psm1_positions)-first_last_index_range, len(psm1_positions))] if first_last_index_range else [(0, len(psm1_positions))]
        plot_labels = ["all", "first", "last"] if first_last_index_range else ["all"]
        for (start_idx, end_idx), plot_label in zip(indices_to_plot, plot_labels):
            # Slice the data based on the specified index range
            psm1_movement = calculate_absolute_movement(psm1_positions[start_idx:end_idx])
            psm2_movement = calculate_absolute_movement(psm2_positions[start_idx:end_idx])
            psm1_jaw_values = psm1_jaw[start_idx:end_idx]
            psm2_jaw_values = psm2_jaw[start_idx:end_idx]
            psm1_jaw_movement = calculate_absolute_movement(psm1_jaw_values)
            psm2_jaw_movement = calculate_absolute_movement(psm2_jaw_values)

            # Plotting the absolute movements and jaw movements
            fig, axs = plt.subplots(3, 2, figsize=(16, 8))

            x_axes_values = np.arange(start_idx, end_idx)

            plot_label_add = f"{plot_label} {first_last_index_range}" if first_last_index_range else plot_label

            # PSM2 Absolute Movement
            axs[0, 0].plot(x_axes_values, psm2_movement, label='3D Movement')
            axs[0, 0].set_title(f'PSM2 Absolute Movement in 3D Space ({plot_label_add})')
            axs[0, 0].legend()

            # PSM1 Absolute Movement
            axs[0, 1].plot(x_axes_values, psm1_movement, label='3D Movement')
            axs[0, 1].set_title(f'PSM1 Absolute Movement in 3D Space ({plot_label_add})')
            axs[0, 1].legend()

            # PSM2 Jaw Movement
            axs[1, 0].plot(x_axes_values, psm2_jaw_values, label='Jaw Value')
            axs[1, 0].set_title(f'PSM2 Jaw Values ({plot_label_add})')
            axs[1, 0].legend()

            # PSM1 Jaw Movement
            axs[1, 1].plot(x_axes_values, psm1_jaw_values, label='Jaw Value')
            axs[1, 1].set_title(f'PSM1 Jaw Values ({plot_label_add})')
            axs[1, 1].legend()
            
            # PSM2 Jaw Absolute Movement
            axs[2, 0].plot(x_axes_values, psm2_jaw_movement, label='Jaw Movement')
            axs[2, 0].set_title(f'PSM2 Jaw Absolute Movement ({plot_label_add})')
            axs[2, 0].legend()
            
            # PSM1 Jaw Absolute Movement
            axs[2, 1].plot(x_axes_values, psm1_jaw_movement, label='Jaw Movement')
            axs[2, 1].set_title(f'PSM1 Jaw Absolute Movement ({plot_label_add})')
            axs[2, 1].legend()

            # Set figure title
            fig.suptitle(f'Absolute movements and Jaw values for PSM1 and PSM2\n--> Tissue: {selected_tissue_folder_name}, phase: {phase_folder_name}), demo: {selected_date_folder_name} ({plot_label} {first_last_index_range})') 

            plt.tight_layout()
            output_path_subfolder = Path(output_path) / "recovery" if "recovery" in phase_folder_name else Path(output_path) / "normal"
            if not output_path_subfolder.exists():
                output_path_subfolder.mkdir(parents=True, exist_ok=True)
            if plot_label == "all":
                save_path = Path(output_path_subfolder) / f"absolute_movements_tissue_{selected_tissue_folder_name}_phase_{phase_folder_name}_demo_{selected_date_folder_name}_{plot_label}.png"
            else:
                save_path = Path(output_path_subfolder) / f"absolute_movements_tissue_{selected_tissue_folder_name}_phase_{phase_folder_name}_demo_{selected_date_folder_name}_{plot_label}_{first_last_index_range}.png"
            fig.savefig(save_path)
            if show_flag:
                plt.show()
            plt.close(fig)
            
        num_demos_plotted += 1

if __name__ == "__main__":

    # Fixed phase index for testing
    phase_indices = [1,2] # range(1,18)
    scale_by_max = False
    first_last_index_range = 40
    base_path =  os.path.join(os.getenv("PATH_TO_DATASET"), "experiments") # "base_chole_clipping_cutting"
    nun_demos_to_plot = 2
    chole_scripts_path = Path(__file__).parent
    for phase_idx in phase_indices:
        print(f"\nPlotting absolute movements for phase {phase_idx}...")
        output_path = Path(os.path.join(chole_scripts_path, "demo_abs_movements"), f"phase_{phase_idx}")
        plot_absolute_movements(base_path, output_path, phase_idx, nun_demos_to_plot, scale_by_max, first_last_index_range, show_flag=False)

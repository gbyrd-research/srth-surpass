import json
from pathlib import Path
import os
import sys

import cv2

# Import the necessary modules from this package
PATH_TO_YAY_ROBOT = os.getenv('PATH_TO_YAY_ROBOT')
if PATH_TO_YAY_ROBOT:
    sys.path.append(os.path.join(PATH_TO_YAY_ROBOT, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
from instructor.dataset_daVinci import get_valid_demo_start_end_indices, SequenceDataset

def save_tissue_phase_demo_structure_all(base_path, output_json_path, tissue_prefix="tissue"):
    # Create a dictionary to store the tissue, phase, and corresponding demo folders
    tissue_phase_demo_dict = {}

    # Get all tissue folders in the base path
    tissue_folders = [folder for folder in Path(base_path).glob(f"{tissue_prefix}_*") if folder.is_dir()]
    tissue_folders = sorted(tissue_folders, key=lambda p: int(p.stem.split('_')[-1]))

    # Iterate over each tissue folder
    for tissue_folder_path in tissue_folders:
        tissue_idx = tissue_folder_path.stem.split('_')[-1]

        # Get and sort phase folders based on the number before the first "_"
        phase_folders = [phase_folder for phase_folder in tissue_folder_path.glob('*_*') if phase_folder.name.split('_')[0].isdigit()]
        phase_folders_sorted = sorted(phase_folders, key=lambda p: int(p.name.split('_')[0]))

        # Iterate through each of the sorted phase folders
        for phase_folder_path in phase_folders_sorted:
            phase_name = phase_folder_path.stem
            # Check if the phase name contains "back" or "recovery"
            if ("grabbing" in phase_name.lower() or "clipping" in phase_name.lower() or "cutting" in phase_name.lower()) and "recovery" in phase_name.lower():
                # Get all demo folders for that specific tissue and phase
                date_folders = list(phase_folder_path.glob('*-*'))
                if not date_folders:
                    print(f"No demo folders found for {phase_folder_path}")
                    continue

                # Sort the demo folders by name
                date_folder_sorted = sorted(date_folders, key=lambda p: p.name)
                demo_folder_names = [demo_folder.stem for demo_folder in date_folder_sorted]

                # Store the tissue, phase, and demo folder names in the dictionary
                if tissue_idx not in tissue_phase_demo_dict:
                    tissue_phase_demo_dict[tissue_idx] = {}

                tissue_phase_demo_dict[tissue_idx][phase_name] = demo_folder_names

        print(f"Processed {tissue_prefix}_{tissue_idx}")

    # Save the dictionary to a JSON file at the first level of the dataset folder
    with open(output_json_path, 'w') as json_file:
        json.dump(tissue_phase_demo_dict, json_file, indent=4)

    print(f"Saved tissue-phase-demo structure to {output_json_path}")


def save_first_frames_all_demos(base_path, output_base_path, tissue_prefix="tissue"):
    # Get all tissue folders in the base path
    tissue_folders = [folder for folder in Path(base_path).glob(f"{tissue_prefix}_*") if folder.is_dir()]

    # Iterate over each tissue folder
    for tissue_folder_path in tissue_folders:
        tissue_idx = tissue_folder_path.stem.split('_')[-1]

        # Get and sort phase folders based on the number before the first "_"
        phase_folders = [phase_folder for phase_folder in tissue_folder_path.glob('*_*') if phase_folder.name.split('_')[0].isdigit()]
        phase_folders_sorted = sorted(phase_folders, key=lambda p: int(p.name.split('_')[0]))

        # Iterate through each of the sorted phase folders
        for phase_folder_path in phase_folders_sorted:
            phase_name = phase_folder_path.stem
            # Check if the phase name contains "clipping", "cutting", and "recovery"
            if ("grabbing" in phase_name.lower() or "clipping" in phase_name.lower() or "cutting" in phase_name.lower()) and "recovery" in phase_name.lower():
                # Get all demo folders for that specific tissue and phase
                date_folders = list(phase_folder_path.glob('*-*'))
                if not date_folders:
                    print(f"No demo folders found for {phase_folder_path}")
                    continue

                # Sort the demo folders by name
                date_folder_sorted = sorted(date_folders, key=lambda p: p.name)

                # Create the output directory for the phase
                output_image_dir = Path(output_base_path) / f"{tissue_prefix}_{tissue_idx}" / phase_folder_path.stem
                output_image_dir.mkdir(parents=True, exist_ok=True)

                # Iterate through each demo folder and save the first frame
                for selected_date_folder_path in date_folder_sorted:
                    left_img_dir_path = selected_date_folder_path / "left_img_dir"
                    if not left_img_dir_path.exists():
                        print(f"No left image directory found for {selected_date_folder_path}")
                        continue
                    
                    # Try to get the start index; if not available, use 0
                    start, _, _ = get_valid_demo_start_end_indices(selected_date_folder_path, 0, 0)
                    frame_idx = start

                    img_path = left_img_dir_path / f"frame{str(frame_idx).zfill(6)}_left.jpg"

                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            # Save the image directly in the phase folder with a unique name
                            output_image_path = output_image_dir / f"{selected_date_folder_path.stem}_frame{str(frame_idx).zfill(6)}_left.jpg"
                            cv2.imwrite(str(output_image_path), img)
                            print(f"Saved {output_image_path}")
                        else:
                            raise ValueError(f"Image corrupt for {img_path}")
                    else:
                        print(f"Image not found for {img_path}")

        print(f"Processed {tissue_prefix}_{tissue_idx}")

def add_grabbing_cutting_clipping_demos(tissue_folder_path, output_json_path):
    # Load the existing JSON structure if the file exists
    if Path(output_json_path).exists():
        with open(output_json_path, 'r') as json_file:
            tissue_phase_demo_dict = json.load(json_file)
    else:
        tissue_phase_demo_dict = {}

    # Get the specific tissue folder
    tissue_folder_path = Path(tissue_folder_path)
    if not tissue_folder_path.exists() or not tissue_folder_path.is_dir():
        print(f"Tissue folder {tissue_folder_path} does not exist or is not a directory.")
        return

    tissue_idx = tissue_folder_path.stem.split('_')[-1]

    # Get and sort phase folders based on the number before the first "_"
    phase_folders = [phase_folder for phase_folder in tissue_folder_path.glob('*_*') if phase_folder.name.split('_')[0].isdigit()]
    phase_folders_sorted = sorted(phase_folders, key=lambda p: int(p.name.split('_')[0]))

    # Iterate through each of the sorted phase folders
    for phase_folder_path in phase_folders_sorted:
        phase_name = phase_folder_path.stem

        # Check for "grabbing", "cutting", or "clipping" in the phase name
        if any(keyword in phase_name.lower() for keyword in ["grabbing", "cutting", "clipping"]) and "recovery" in phase_name.lower():
            # Get all demo folders for that specific tissue and phase
            date_folders = list(phase_folder_path.glob('*-*'))
            if not date_folders:
                print(f"No demo folders found for {phase_folder_path}")
                continue

            # Sort the demo folders by name
            date_folder_sorted = sorted(date_folders, key=lambda p: p.name)
            demo_folder_names = [demo_folder.stem for demo_folder in date_folder_sorted]

            # Ensure the tissue index is in the dictionary
            if tissue_idx not in tissue_phase_demo_dict:
                tissue_phase_demo_dict[tissue_idx] = {}

            # Add or update only the specific phase in the dictionary
            tissue_phase_demo_dict[tissue_idx][phase_name] = demo_folder_names

        print(f"Processed {tissue_folder_path} - {phase_name}")

    # Save the updated dictionary back to the JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(tissue_phase_demo_dict, json_file, indent=4)

    print(f"Updated tissue-phase-demo structure with grabbing, cutting, and clipping phases and saved to {output_json_path}")


if __name__ == "__main__":
    tissue_prefix = "tissue"  # "tissue" or "phantom"
    dataset_name = "base_chole_clipping_cutting"  # Default data folder name

    # Set the base path and output JSON path
    base_path = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    output_json_path = os.path.join(base_path, "corrections.json")

    # Generate and save the tissue-phase-demo structure for all tissues
    # save_tissue_phase_demo_structure_all(base_path, output_json_path, tissue_prefix)

    # ------------ Add new tissue ------------

    tissue_idx = 80 # 72
    tissue_name = f"{tissue_prefix}_{str(tissue_idx)}"
    tissue_folder_path = os.path.join(base_path, tissue_name)
    add_grabbing_cutting_clipping_demos(tissue_folder_path, output_json_path)

    # ------------ Create first frames for all demos ------------

    # tissue_prefix = "tissue"  # "tissue" or "phantom"
    # dataset_name = "base_chole_clipping_cutting"  # Default data folder name

    # # Set the base path and output paths
    # base_path = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    # output_image_base_path = os.path.join(base_path, "FirstFramesRecoveryClippingCutting")

    # # Save the first frames of all demos for relevant phases
    # save_first_frames_all_demos(base_path, output_image_base_path, tissue_prefix)

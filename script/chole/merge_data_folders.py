import os
import shutil
from pathlib import Path

def move_folders_src_target(source_folder, target_folder):
    # Iterate over each tissue folder in the source directory
    for tissue_folder in os.listdir(source_folder):
        source_tissue_path = os.path.join(source_folder, tissue_folder)
        target_tissue_path = os.path.join(target_folder, tissue_folder)

        if not os.path.isdir(source_tissue_path):
            continue  # Skip non-directory files

        # Ensure the tissue folder exists in the target directory
        if not os.path.exists(target_tissue_path):
            os.makedirs(target_tissue_path, exist_ok=True)

        # Iterate over each phase folder in the tissue folder
        for phase_folder in os.listdir(source_tissue_path):
            source_phase_path = os.path.join(source_tissue_path, phase_folder)
            target_phase_path = os.path.join(target_tissue_path, phase_folder)

            if not os.path.isdir(source_phase_path):
                continue  # Skip non-directory files

            # Ensure the phase folder exists in the target directory
            if not os.path.exists(target_phase_path):
                os.makedirs(target_phase_path, exist_ok=True)

            # Iterate over each demo folder in the phase folder
            for demo_folder in os.listdir(source_phase_path):
                source_demo_path = os.path.join(source_phase_path, demo_folder)
                target_demo_path = os.path.join(target_phase_path, demo_folder)

                if not os.path.isdir(source_demo_path):
                    continue  # Skip non-directory files

                # Move each camera/segmask/etc. folder from the source to the target
                for sub_folder in os.listdir(source_demo_path):
                    source_sub_path = os.path.join(source_demo_path, sub_folder)
                    target_sub_path = os.path.join(target_demo_path, sub_folder)

                    if os.path.exists(target_sub_path):
                        print(f"Skipped: {target_sub_path} already exists.")
                    else:
                        shutil.move(source_sub_path, target_sub_path)
                        print(f"Moved: {source_sub_path} to {target_sub_path}")

        print(f"Finished processing tissue folder: {tissue_folder}\n")

def merge_and_cleanup_tissue_folders(source_folder):
    # Iterate over each tissue folder in the source directory
    for tissue_folder in os.listdir(source_folder):
        outer_tissue_path = os.path.join(source_folder, tissue_folder)

        if not os.path.isdir(outer_tissue_path):
            continue  # Skip non-directory files

        # Check if the tissue folder contains another folder with the same name
        inner_tissue_path = os.path.join(outer_tissue_path, tissue_folder)
        if not os.path.exists(inner_tissue_path):
            continue  # Skip if the inner tissue folder doesn't exist

        # Iterate over each phase folder in the inner tissue folder
        for phase_folder in os.listdir(inner_tissue_path):
            source_phase_path = os.path.join(inner_tissue_path, phase_folder)
            target_phase_path = os.path.join(outer_tissue_path, phase_folder)

            if not os.path.isdir(source_phase_path):
                continue  # Skip non-directory files

            # Ensure the phase folder exists in the outer tissue folder
            if not os.path.exists(target_phase_path):
                os.makedirs(target_phase_path, exist_ok=True)

            # Move each demo folder from the inner tissue folder to the outer tissue folder
            for demo_folder in os.listdir(source_phase_path):
                source_demo_path = os.path.join(source_phase_path, demo_folder)
                target_demo_path = os.path.join(target_phase_path, demo_folder)

                if not os.path.isdir(source_demo_path):
                    continue  # Skip non-directory files

                # Move each camera/segmask/etc. folder from the inner tissue folder to the outer tissue folder
                for sub_folder in os.listdir(source_demo_path):
                    source_sub_path = os.path.join(source_demo_path, sub_folder)
                    target_sub_path = os.path.join(target_demo_path, sub_folder)

                    if os.path.exists(target_sub_path):
                        print(f"Skipped: {target_sub_path} already exists.")
                    else:
                        shutil.move(source_sub_path, target_sub_path)
                        print(f"Moved: {source_sub_path} to {target_sub_path}")

        # After merging, remove the now-empty inner tissue folder
        shutil.rmtree(inner_tissue_path)
        print(f"Removed empty folder: {inner_tissue_path}\n")

        print(f"Finished processing tissue folder: {tissue_folder}\n")

if __name__ == "__main__":
    # source_folder = os.path.join(os.getenv("PATH_TO_DATASET"), "seg_masks")
    # target_folder = os.path.join(os.getenv("PATH_TO_DATASET"), "base_chole_clipping_cutting")

    # move_folders_src_target(source_folder, target_folder)
    
    # ----------
    
    source_folder = os.path.join(os.getenv("PATH_TO_DATASET"), "base_chole_clipping_cutting")
    merge_and_cleanup_tissue_folders(source_folder)    


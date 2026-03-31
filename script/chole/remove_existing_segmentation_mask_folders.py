import os
import shutil

def remove_target_folders(input_folder, target_folders, max_index):
    # Iterate through the directory and its subdirectories
    for root, dirs, _ in os.walk(input_folder):        
        # Only remove certain tissue folders
        root_split = root.split("/")
        tissue_root_elements = [element for element in root_split if "tissue" in element]
        if len(tissue_root_elements) > 0:
            tissue = tissue_root_elements[0]
            tissue_index = int(tissue.split("_")[1])
            if max_index is not None and int(tissue_index) > max_index:
                continue
        
        for dir in dirs:
            if dir in target_folders:
                # Path to the folder that should be removed
                target_folder_path = os.path.join(root, dir)
                
                # Remove the folder and its contents
                shutil.rmtree(target_folder_path)
                print(f"Removed: {target_folder_path}")

if __name__ == "__main__":
    dataset_name = "base_chole_clipping_cutting"  # "base_chole_clipping_cutting" "phantom_chole" 
    dataset_folder = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    target_folders = ["seg_masks"]  #  "contours_on_image" - List of folder names to match
    max_index = None  # Only search within folders with an index <= max_index

    # Remove target folders within the dataset folder
    remove_target_folders(dataset_folder, target_folders, max_index)

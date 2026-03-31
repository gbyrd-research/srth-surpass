import os

def remove_target_files(input_folder, target_files=["indices_curated.json"], max_index=None):
    # Iterate through the directory and its subdirectories
    for root, _, files in os.walk(input_folder):
        # Only process certain tissue folders
        root_split = root.split(os.sep)
        tissue_root_elements = [element for element in root_split if "tissue" in element]
        if len(tissue_root_elements) > 0:
            tissue = tissue_root_elements[0]
            tissue_index = int(tissue.split("_")[1])
            if max_index is not None and tissue_index > max_index:
                continue

        for file in files:
            if file in target_files:
                # Path to the file that should be removed
                target_file_path = os.path.join(root, file)
                
                # Remove the file
                os.remove(target_file_path)
                print(f"Removed: {target_file_path}")

if __name__ == "__main__":
    dataset_name = "base_chole_clipping_cutting"  # "base_chole_clipping_cutting" "phantom_chole" 
    dataset_folder = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    target_files = ["indices_curated.json"]  # List of file names to remove
    max_index = None  # Only search within folders with an index <= max_index

    # Remove target files within the dataset folder
    remove_target_files(dataset_folder, target_files, max_index)

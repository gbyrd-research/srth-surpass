import os
import shutil

def check_and_copy_missing_or_corrupt_files(src_folder, dst_folder, target_folders):
    missing_or_corrupt_files = []

    for root, dirs, files in os.walk(src_folder):
        # Only check certain target folders
        root_split = root.split("/")
        relevant_folders = [element for element in root_split if element in target_folders]

        if len(relevant_folders) > 0:
            rel_path = os.path.relpath(root, src_folder)
            corresponding_dst_dir = os.path.join(dst_folder, rel_path)

            for file_name in files:
                src_file_path = os.path.join(root, file_name)
                dst_file_path = os.path.join(corresponding_dst_dir, file_name)

                if not os.path.exists(dst_file_path):
                    print(f"Missing: {dst_file_path}")
                    missing_or_corrupt_files.append((src_file_path, dst_file_path))
                else:
                    src_size = os.path.getsize(src_file_path)
                    dst_size = os.path.getsize(dst_file_path)
                    if src_size != dst_size:
                        print(f"Corrupt: {dst_file_path} (Size mismatch: source={src_size}, dest={dst_size})")
                        missing_or_corrupt_files.append((src_file_path, dst_file_path))
    
    print("\n--------------------------------------------------------\n")
    
    # Copy the missing or corrupt files
    for src_file, dst_file in missing_or_corrupt_files:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        
        try:
            shutil.copy(src_file, dst_file)
            print(f"Copied: {src_file} to {dst_file}")
        except Exception as e:
            print(f"Failed to copy {src_file} to {dst_file}: {e}")
    
    return missing_or_corrupt_files

if __name__ == "__main__":
    dataset_name = "base_chole_clipping_cutting"  # Example dataset name
    src_folder = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    dst_folder = os.path.join(os.getenv("PATH_TO_DATASET"), "seg_masks")
    target_folders = ["labelled_img_depth_relative", "contours_on_image", "ee_labels.csv"]  # Example list of target folders/files

    missing_or_corrupt_files = check_and_copy_missing_or_corrupt_files(src_folder, dst_folder, target_folders)

    print("\n--------------------------------------------------------\n")

    if missing_or_corrupt_files:
        print("\nSummary of files that were missing or corrupt and copied:")
        print("------------------------------------------")
        for src_file, dst_file in missing_or_corrupt_files:
            print(f"Copied: {src_file} to {dst_file}")
        print("------------------------------------------")
    else:
        print("\nAll files were already copied successfully.")

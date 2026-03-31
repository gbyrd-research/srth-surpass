import os
import shutil
import time

def copy_folders_with_parent_structure(input_folder, target_folders, output_folder, max_index, min_index, sleep_time_ms=None):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List to track failed paths
    failed_copies = []
    
    # Iterate through the directory and its subdirectories
    for root, dirs, files in os.walk(input_folder):       
        # Only copy over certain tissue folders
        root_split = root.split("/")
        tissue_root_elements = [element for element in root_split if "tissue" in element]
        if len(tissue_root_elements) > 0:
            tissue = tissue_root_elements[0]
            tissue_index = int(tissue.split("_")[1])
            if max_index is not None and tissue_index > max_index or min_index is not None and tissue_index < min_index:
                continue
        
        for name in dirs + files:
            if name in target_folders:
                # Calculate relative path to maintain directory structure
                rel_dir = os.path.relpath(root, input_folder)
                dest_dir = os.path.join(output_folder, rel_dir)
                
                # Ensure the destination directory exists
                os.makedirs(dest_dir, exist_ok=True)
                
                src_path = os.path.join(root, name)
                dst_path = os.path.join(dest_dir, name)
                
                # Check if the destination file or directory already exists
                if os.path.exists(dst_path):
                    print(f"Skipped: {dst_path} already exists.")
                    continue
                
                for attempt in range(4):  # Attempt up to four times
                    try:
                        # Copy the file or directory
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                            print(f"Copied directory: {src_path} to {dst_path}")
                        else:
                            shutil.copy(src_path, dst_path)
                            print(f"Copied file: {src_path} to {dst_path}")
                        break  # Exit the retry loop if successful
                    except Exception as e:
                        print(f"Error copying {src_path} to {dst_path}: {e}")
                        if attempt < 3:  # Less than 4 attempts, prepare for retry
                            print("Retrying after a short delay...")
                            time.sleep(1)  # Wait for 1 second before retrying
                            # Clean up the partially copied directory or file
                            if os.path.isdir(dst_path):
                                shutil.rmtree(dst_path)
                            elif os.path.exists(dst_path):
                                os.remove(dst_path)
                        else:  # Fourth attempt failed, log the failed path
                            print(f"Final retry failed for {src_path} to {dst_path}: {e}")
                            failed_copies.append(src_path)
                            continue  # Skip this file or directory after a failed retry
            
                if sleep_time_ms:
                    time.sleep(sleep_time_ms / 1000)

    print("\nCopy process completed.")

    # Output the list of failed copies
    if failed_copies:
        print("\nThe following files or directories failed to copy after 4 attempts:")
        for failed in failed_copies:
            print(failed)

if __name__ == "__main__":
    dataset_name = "base_chole_clipping_cutting"  # "base_chole_clipping_cutting" "phantom_chole" 
    dataset_folder = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name)
    target_folders = ["labelled_img_depth_relative", "ee_labels.csv"]  #  "contours_on_image" - List of folder names to match
    output_folder = os.path.join(os.getenv("PATH_TO_DATASET"), "depth_seg_training")
    max_index = 80
    min_index = 78

    # Copy folders with parent directory structure
    sleep_time_ms = 15
    copy_folders_with_parent_structure(dataset_folder, target_folders, output_folder, max_index, min_index, sleep_time_ms)

import os
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

from natsort import natsorted
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories

def compute_relative_actions_in_SE3(qpos, action):
    """
    Note: this is the proper implementation
    qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
    action: set point on the dvrk (action_horizon x 8)
    
    returns: relative position and rotation w.r.t qpos
    """
    
    diff = np.zeros((action.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm

    # convert current pose to SE(3)
    qpos_wxyz = rotations.quaternion_wxyz_from_xyzw(qpos[3:7])
    qpos_py3d = np.concatenate((qpos[0:3], qpos_wxyz))
    g_qpos = transformations.transform_from_pq(qpos_py3d) # no jaw angle!

    # convert actions to SE(3)
    action_wxyz = batch_rotations.batch_quaternion_wxyz_from_xyzw(action[:, 3:7]) 
    action_py3d = np.concatenate((action[:, 0:3], action_wxyz), axis = 1)
    g_action = trajectories.transforms_from_pqs(action_py3d)

    # invert current pose
    g_qpos_inv = transformations.invert_transform(g_qpos)
    diff_SE3 = trajectories.concat_one_to_many(g_qpos_inv, g_action)

    # construct 6d rot
    diff_6d = diff_SE3[:,0:3,:2]
    diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
    
    # fill in translation elements
    diff[:, 0:3] = diff_SE3[:, 0:3, 3] # replace the translations with the last column first three rows of SE3
    # fill in 6d rot
    diff[:, 3:9] = diff_6d
    # fill in jaw angle (note: jaw angle is absolute, not relative)
    diff[:, 9] = action[:, 7]
    return diff

def compute_quat_diff(quat1, quat2):
    r1 = R.from_quat(quat1) # single element
    r2 = R.from_quat(quat2) # many rows of elements
    diff = r1.inv()*r2
    diff = diff.as_quat()
    return diff

def computer_diff_actions(qpos, action):

    """
    qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
    action: set point on the dvrk (action_horizon x 8)
    
    returns: relative position and rotation w.r.t qpos
    """

    # find diff first and then fill-in the quaternion differences properly
    diff = action - qpos

    quat_init = qpos[3:7]
    quat_actions = action[:, 3:7]

    # convert quaternions to rotation matrices
    r_init = R.from_quat(quat_init)
    r_actions = R.from_quat(quat_actions)
    # find their diff
    diff_rs = r_init.inv()*r_actions 
    # extract their first two columns
    diff_6d = diff_rs.as_matrix()[:,:,:2]
    diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
    
    diff_exp = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
    diff_exp[:diff.shape[0], 0:diff.shape[1]] = diff
    diff = diff_exp

    diff[:, 3:9] = diff_6d
    diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
    return diff

def compute_diff_actions_wrt_camera(qpos, action):
        """
        qpos: current position [9]
        action: actions commanded by the user [n_actions x 9]
        returns: relative actions w.r.t qpos
        """
        # find diff first and then fill-in the quaternion differences properly
        diff = action - qpos
        quat_actions = action[:, 3:7]

        # convert quaternions to rotation matrices
        r_actions = R.from_quat(quat_actions)

        # extract their first two columns
        diff_6d = r_actions.as_matrix()[:,:,:2]
        diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
        diff_exp = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
        diff_exp[:diff.shape[0], 0:diff.shape[1]] = diff 
        diff = diff_exp

        diff[:, 3:9] = diff_6d
        diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
        return diff    

def compute_diffs(ids, data_dir, chunk_size=100, phantoms=False, use_multiprocessing=True):
    cp_psm1 = [ "psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
            "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
            "psm1_jaw"]

    sp_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
            "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
            "psm1_jaw_sp"]

    cp_psm2 = [ "psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
            "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
            "psm2_jaw"]

    sp_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
            "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
            "psm2_jaw_sp"]

    # Collect all CSV file paths first
    csv_paths = []
    
    for id in ids:
        if phantoms:
            root = os.path.join(data_dir, f"phantom_{id}")
        else:
            root = os.path.join(data_dir, f"tissue_{id}")
        
        if not os.path.exists(root):
            print(f"Warning: Directory {root} does not exist, skipping...")
            continue
            
        print(f"Scanning {root}")
        dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
        dirlist = natsorted(dirlist)

        for phase in dirlist:
            phase_path = os.path.join(root, phase)
            items = [item for item in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, item))]
            
            for item in items:
                if item == "Corrections":
                    sample_dir = os.path.join(phase_path, item)
                    new_samples = os.listdir(sample_dir)
                    if new_samples:
                        sample_dir = os.path.join(sample_dir, new_samples[0])
                        pth = os.path.join(sample_dir, "ee_csv.csv")
                else:
                    pth = os.path.join(phase_path, item, "ee_csv.csv")
                
                if os.path.isfile(pth):
                    csv_paths.append(pth)
    
    print(f"Total CSV files found: {len(csv_paths)}")
    
    # Process CSV files
    if use_multiprocessing and len(csv_paths) > 1:
        print(f"Using multiprocessing with {min(cpu_count(), len(csv_paths))} workers")
        process_func = partial(process_single_csv, 
                              cp_psm1=cp_psm1, sp_psm1=sp_psm1,
                              cp_psm2=cp_psm2, sp_psm2=sp_psm2,
                              chunk_size=chunk_size)
        
        with Pool(processes=min(cpu_count(), len(csv_paths))) as pool:
            results = pool.map(process_func, csv_paths)
        
        # Filter out None results and concatenate
        diffs = [r for r in results if r is not None and len(r) > 0]
    else:
        print("Processing sequentially...")
        diffs = []
        for pth in csv_paths:
            result = process_single_csv(pth, cp_psm1, sp_psm1, cp_psm2, sp_psm2, chunk_size)
            if result is not None and len(result) > 0:
                diffs.append(result)
    
    if not diffs:
        raise ValueError("No valid data found!")
    
    num_episodes = len(csv_paths)  # Number of CSV files = number of episodes
    print(f"Total episodes (CSV files): {num_episodes}")
    print(f"Total diff arrays: {len(diffs)}")
    diffs_np = np.concatenate(diffs, axis=0)
    print(f"Total samples: {diffs_np.shape[0]}")
    
    mean = diffs_np.mean(axis=0)
    std = diffs_np.std(axis=0).clip(1e-2, 10)
    min_vals = diffs_np.min(axis=0)
    max_vals = diffs_np.max(axis=0)

    return mean, std, min_vals, max_vals, num_episodes


def process_single_csv(pth, cp_psm1, sp_psm1, cp_psm2, sp_psm2, chunk_size):
    """Process a single CSV file and return the diffs."""
    try:
        csv = pd.read_csv(pth)
        
        # Pre-extract all data at once (much faster than row-by-row)
        cp_psm1_data = csv[cp_psm1].to_numpy()
        sp_psm1_data = csv[sp_psm1].to_numpy()
        cp_psm2_data = csv[cp_psm2].to_numpy()
        sp_psm2_data = csv[sp_psm2].to_numpy()
        
        csv_len = len(csv)
        diffs_list = []
        
        # Process in batches for better efficiency
        for jj in range(csv_len):
            end_idx = min(jj + chunk_size, csv_len)
            
            first_el_psm1 = cp_psm1_data[jj]
            chunk_el_psm1 = sp_psm1_data[jj:end_idx]
            diff_psm1 = compute_relative_actions_in_SE3(first_el_psm1, chunk_el_psm1)

            first_el_psm2 = cp_psm2_data[jj]
            chunk_el_psm2 = sp_psm2_data[jj:end_idx]
            diff_psm2 = compute_relative_actions_in_SE3(first_el_psm2, chunk_el_psm2)

            diff_stacked = np.column_stack((diff_psm1, diff_psm2))
            diffs_list.append(diff_stacked)
        
        if diffs_list:
            return np.concatenate(diffs_list, axis=0)
        return None
        
    except Exception as e:
        print(f"Warning: Failed to process {pth}: {e}")
        return None

# Define the main function to generate the task configuration file
def generate_task_config():
    import time
    
    #ids = [5]
    ids = [1, 2, 3, 4, 10]
    #ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    data_dir = "/home/iulian/chole_ws/data/cnh_exvivo_chole"
    # data_dir = "/home/iulian/chole_ws/data/Xinhao"
    # data_dir = "/home/iulian/chole_ws/data/Jesse/"

    print(f"Starting computation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    mean, std, min_vals, max_vals, num_episodes = compute_diffs(ids, data_dir, use_multiprocessing=True)
    
    elapsed_time = time.time() - start_time
    print(f"Computation completed in {elapsed_time:.2f} seconds")

    # Prepare results dictionary
    results = {
        "metadata": {
            "tissue_ids": ids,
            "data_directory": data_dir,
            "num_episodes": num_episodes,
            "computation_time_seconds": elapsed_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        },
        "statistics": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": min_vals.tolist(),
            "max": max_vals.tolist()
        }
    }

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Number of Episodes: {num_episodes}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Min: {min_vals}")
    print(f"Max: {max_vals}")
    print("="*80 + "\n")

    # Write results to JSON file
    output_filename = "./std_mean_invivo.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to {output_filename}")
    
    # Also write a compact text summary for quick reference
    summary_filename = "./std_mean_invivo_summary.txt"
    with open(summary_filename, "w") as f:
        f.write(f"Tissue IDs: {ids}\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Computation Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mean: {', '.join(map(str, mean))}\n")
        f.write(f"Std: {', '.join(map(str, std))}\n")
        f.write(f"Min: {', '.join(map(str, min_vals))}\n")
        f.write(f"Max: {', '.join(map(str, max_vals))}\n")
    
    print(f"Summary written to {summary_filename}")
    
    return results


# Run the main function to generate the task configuration
if __name__ == "__main__":
    generate_task_config()

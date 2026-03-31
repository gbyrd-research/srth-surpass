# other imports...
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
ROBOT_STATION = "4"
CHUNK_SIZE = 32

# define constants for the axes indices
X_INDEX = 0
Y_INDEX = 1
Z_INDEX = 2

# define the positive and negative directions
LEFT = "move to the left"
RIGHT = "move to the right"
FORWARD = "move away from me"
BACKWARD = "move towards me"
UPWARD = "go higher"
DOWNWARD = "go lower"

# define gripper movements
CLOSE_GRIPPER = "close gripper"
OPEN_GRIPPER = "open gripper"

# define rotation directions
CLOCKWISE = "rotate clockwise"
COUNTERCLOCKWISE = "rotate counterclockwise"



def compute_trend(array, threshold):
    # check if the array is long enough to compute the gradient
    if len(array) < 2:
        return False, False, None
    gradient = np.diff(array, axis=0)  # compute the gradient
    avg_gradient = np.mean(gradient)  # calculate the average gradient
    absolute_change = abs(array[-1] - array[0])  # calculate the absolute change

    increase = avg_gradient > 0 and absolute_change > threshold
    decrease = avg_gradient < 0 and absolute_change > threshold

    # calculate the percentage change of absolute change compared to threshold
    percentage_change = absolute_change / threshold

    return increase, decrease, percentage_change



def get_dominant_direction(percentages, labels):
    assert len(percentages) == len(labels)

    if len(percentages) == 0:
        return None
    max_index = np.argmax(percentages)
    assert percentages[max_index] > 1
    return labels[max_index]


def get_auto_label(ee_csv, start_ts, chunk_size = 5, jaw_threshold=0.3):

    header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                    "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                    "psm1_jaw"]
            
    header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                    "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                    "psm2_jaw"]
    
    header_name_actions_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                                "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                                "psm1_jaw_sp"]

    header_name_actions_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                                "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                                "psm2_jaw_sp"]

    # Compute trends for end-effector positions
    left_percentages = []
    right_percentages = []
    left_labels = []
    right_labels = []

    action_psm1 = ee_csv[header_name_qpos_psm1].iloc[start_ts : start_ts + chunk_size].to_numpy() # note 400 added here
    action_psm2 = ee_csv[header_name_qpos_psm2].iloc[start_ts : start_ts + chunk_size].to_numpy() # note 400 added here

    ee_l_positions = action_psm2[:, :3]
    ee_r_positions = action_psm1[:, :3]

    gripper_positions = np.stack([action_psm2[:, -1], action_psm1[:, -1]], axis=1)
    
    x_threshold_l = 0.0005
    y_threshold_l = 0.0005
    z_threshold_l = 0.0008
    x_threshold_r = 0.001
    y_threshold_r = 0.0008
    z_threshold_r = 0.0015

    for i, threshold in zip(range(3), [x_threshold_l, y_threshold_l, z_threshold_l]):
        increase, decrease, percentage = compute_trend(ee_l_positions[:, i], threshold)
        if increase:
            left_labels.append(f'move left arm {["to the left", "higher", "away from me"][i]}')
            left_percentages.append(percentage)
        elif decrease:
            left_labels.append(f'move left arm {["to the right", "lower", "towards me"][i]}')
            left_percentages.append(percentage)

    # print("left_labels:", left_labels)
    # print("left_percentages:", left_percentages)

    for i, threshold in zip(range(3), [x_threshold_r, y_threshold_r, z_threshold_r]):
        increase, decrease, percentage = compute_trend(ee_r_positions[:, i], threshold)
        # print("increase:", increase, "decrease:", decrease, "percentage:", percentage)
        if increase:
            right_labels.append(f'move right arm {["to the left", "higher", "away from me"][i]}')
            right_percentages.append(percentage)
        elif decrease:
            right_labels.append(f'move right arm {["to the right", "lower", "towards me"][i]}')
            right_percentages.append(percentage)

    # print("right_labels:", right_labels)
    # print("right_percentages:", right_percentages)


    # Determine dominant direction
    left_dominant_label = get_dominant_direction(left_percentages, left_labels)
    right_dominant_label = get_dominant_direction(right_percentages, right_labels)
    if left_dominant_label and right_dominant_label:
        if max(left_percentages) > max(right_percentages):
            return left_dominant_label
        elif max(right_percentages) >= max(left_percentages):
            return right_dominant_label
    else:
        if left_dominant_label:
            return left_dominant_label
        elif right_dominant_label:
            return right_dominant_label


    # Compute trends for gripper positions
    left_open, left_close, _ = compute_trend(gripper_positions[:, 0], threshold=jaw_threshold)
    right_open, right_close, _ = compute_trend(gripper_positions[:, 1], threshold=jaw_threshold)

    if left_close and right_close:
        return "close both grippers"
    elif left_close:
        return "close left gripper"
    elif right_close:
        return "close right gripper"
    elif left_open and right_open:
        return "open both grippers"
    elif left_open:
        return "open left gripper"
    elif right_open:
        return "open right gripper"
    
    return "do not move"


if __name__ == "__main__":
    view = ...
    get_auto_label(view)

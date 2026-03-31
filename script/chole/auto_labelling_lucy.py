# other imports...
from scipy.spatial.transform import Rotation as R


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


def compute_rotation(quaternions, rotation_threshold=20):
    rotations = R.from_quat(quaternions)
    euler_angles = rotations.as_euler("xyz", degrees=True)

    yaw_diff = np.diff(euler_angles[:, 2])  # yaw corresponds to rotation around the z-axis
    total_yaw_change = np.sum(yaw_diff)

    counterclockwise = total_yaw_change > rotation_threshold
    clockwise = total_yaw_change < -rotation_threshold

    return counterclockwise, clockwise


def get_dominant_direction(percentages, labels):
    assert len(percentages) == len(labels)

    if len(percentages) == 0:
        return None
    max_index = np.argmax(percentages)
    assert percentages[max_index] > 1
    return labels[max_index]


def get_label_list(is_biarm=False):
    return get_biarm_label_list() if is_biarm else get_single_arm_label_list()


def get_single_arm_label_list():
    return [CLOSE_GRIPPER, OPEN_GRIPPER, LEFT, RIGHT, FORWARD, BACKWARD, UPWARD, DOWNWARD, CLOCKWISE, COUNTERCLOCKWISE]


def get_biarm_label_list():
    return [
        "close left gripper",
        "close right gripper",
        "open left gripper",
        "open right gripper",
        "move left arm to the left",
        "move left arm to the right",
        "move left arm towards me",
        "move left arm away from me",
        "move left arm higher",
        "move left arm lower",
        "move right arm to the left",
        "move right arm to the right",
        "move right arm towards me",
        "move right arm away from me",
        "move right arm higher",
        "move right arm lower",
        "close both grippers",
        "open both grippers",
        "move both arms to the left",
        "move both arms to the right",
        "move both arms towards me",
        "move both arms away from me",
        "move both arms higher",
        "move both arms lower",
    ]


def get_labels_for_chunk(chunk, robot_station, threshold=0.2):
    if chunk["ee_position"] is not None:
        ee_pos = chunk["ee_position"]
        if isinstance(robot_station, int):
            robot_station = str(robot_station)
        if robot_station not in ["4", "5", "6", "5/6"]:
            # print(f"robot station is {robot_station}")
            return None
        # axes_mapping: (axis, positive direction, negative direction)
        if robot_station == "4":
            axes_mapping = [
                (X_INDEX, LEFT, RIGHT),
                (Y_INDEX, BACKWARD, FORWARD),
                (Z_INDEX, UPWARD, DOWNWARD),
            ]
        else:
            # station 5/6
            axes_mapping = [
                (X_INDEX, FORWARD, BACKWARD),
                (Y_INDEX, LEFT, RIGHT),
                (Z_INDEX, UPWARD, DOWNWARD),
            ]

        ee_label_candidates = []
        ee_percentages = []
        for dimension, text_positive, text_negative in axes_mapping:
            increase, decrease, percentage = compute_trend(ee_pos[:, dimension], threshold)
            if increase:
                ee_label_candidates.append(text_positive)
                ee_percentages.append(percentage)
            if decrease:
                ee_label_candidates.append(text_negative)
                ee_percentages.append(percentage)
        dominant_direction = get_dominant_direction(ee_percentages, ee_label_candidates)
        if dominant_direction:
            return dominant_direction

    if chunk["ee_quaternion_xyzw"] is not None:
        quaternions = chunk["ee_quaternion_xyzw"]
        counterclockwise, clockwise = compute_rotation(quaternions)
        if counterclockwise:
            return COUNTERCLOCKWISE
        if clockwise:
            return CLOCKWISE

    if chunk["gripper_position"] is not None:
        gripper_pos = chunk["gripper_position"]
        increase, decrease, _ = compute_trend(gripper_pos, 0.4)
        if increase:
            return CLOSE_GRIPPER
        if decrease:
            return OPEN_GRIPPER

    return None  # no valid label


def get_arx_label(gripper_positions, ee_positions):
    # Compute trends for end-effector positions
    left_percentages = []
    right_percentages = []
    left_labels = []
    right_labels = []

    for i, threshold in zip(range(3), [0.1, 0.1, 0.06]):
        increase, decrease, percentage = compute_trend(ee_positions[:, i], threshold)
        if increase:
            left_labels.append(f'move left arm {["to the right", "away from me", "higher"][i]}')
            left_percentages.append(percentage)
        elif decrease:
            left_labels.append(f'move left arm {["to the left", "towards me", "lower"][i]}')
            left_percentages.append(percentage)

    for i, threshold in zip(range(3, 6), [0.1, 0.1, 0.06]):
        increase, decrease, percentage = compute_trend(ee_positions[:, i], threshold)
        if increase:
            right_labels.append(f'move right arm {["to the left", "towards me", "higher"][i-3]}')
            right_percentages.append(percentage)
        elif decrease:
            right_labels.append(f'move right arm {["to the right", "away from me", "lower"][i-3]}')
            right_percentages.append(percentage)

    # Determine dominant direction
    left_dominant_label = get_dominant_direction(left_percentages, left_labels)
    right_dominant_label = get_dominant_direction(right_percentages, right_labels)

    # if both left_dominant_label and right_dominant_label, replace left arm with both arms
    # if only one of them, return that one
    if left_dominant_label and right_dominant_label:
        return left_dominant_label.replace("left arm", "both arms")
    elif left_dominant_label:
        return left_dominant_label
    elif right_dominant_label:
        return right_dominant_label

    # Compute trends for gripper positions
    left_open, left_close, _ = compute_trend(gripper_positions[:, 0], threshold=0.1)
    right_open, right_close, _ = compute_trend(gripper_positions[:, 1], threshold=0.1)

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

    return None


def get_auto_label(self, view):
    ee_position = view.data["observation/ee_position"][view.step : view.step + self.action_chunk_size]
    gripper_position = view.data["observation/gripper_position"][view.step : view.step + self.action_chunk_size]
    if not self.is_biarm:
        ee_rotation = view.data["observation/ee_quaternion_xyzw"][view.step : view.step + self.action_chunk_size]
        chunk = {
            "ee_position": ee_position,
            "gripper_position": gripper_position,
            "ee_quaternion_xyzw": ee_rotation,
        }
        robot_station = view.episode_data.metadata["robot_platform_increment"]
        return auto_label.get_labels_for_chunk(chunk, robot_station)
    else:
        return auto_label.get_arx_label(gripper_position, ee_position)


if __name__ == "__main__":
    view = ...
    get_auto_label(view)

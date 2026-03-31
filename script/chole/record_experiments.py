import argparse
import cv2
import os
from datetime import datetime
import time
import signal
import sys
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import pandas as pd

# --------------------------------- Global variables ---------------------------------

is_recording = False
exit_flag = False

usb_image_left = endo_cam_psm1 = endo_cam_psm2 = None
direction = psm = None

psm1_pose = None
psm1_sp = None
psm1_rcm_pose = None
psm1_jaw = None
psm1_jaw_sp = None

psm2_pose = None
psm2_sp = None
psm2_rcm_pose = None
psm2_jaw = None
psm2_jaw_sp = None

ecm_pose = None
ecm_rcm_pose = None

kinematics_timestamp = None

# SUJ measured_cp and js
suj1_pose = suj1_jp = None # SUJ/PSM1/measured_cp, measured_js
suj2_pose = suj2_jp = None # SUJ/PSM2/measured_cp, measured_js
suj_ecm_pose = suj_ecm_jp = None # SUJ/ECM/measured_cp, measured_js

# psm / ecm measured_js and setpoint_js
psm1_js = psm1_set_js = None  #PSM1/measured_js, setpoint_js
psm2_js = psm2_set_js = None  #PSM2/measured_js, setpoint_js
ecm_js = ecm_set_js = None # ECM/measured_js, setpoint_js


# --------------------------------- Helper classes ---------------------------------

class RecordingManager:
    def __init__(self, output_folder_path, fps, wrist_image_sav_res, endo_image_save_res):
        self.vid_left = None
        self.vid_psm1_endo = None
        self.vid_psm2_endo = None
        
        self.output_folder_path = output_folder_path            
        self.fps = fps
        self.wrist_image_sav_res = wrist_image_sav_res
        self.endo_image_save_res = endo_image_save_res
        
    def start_new_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vid_left = cv2.VideoWriter(os.path.join(self.output_folder_path, 'left_img_dir.mp4'), fourcc, self.fps, self.endo_image_save_res)
        self.vid_psm1_endo = cv2.VideoWriter(os.path.join(self.output_folder_path, 'endo_psm1.mp4'), fourcc, self.fps, self.wrist_image_sav_res)
        self.vid_psm2_endo = cv2.VideoWriter(os.path.join(self.output_folder_path, 'endo_psm2.mp4'), fourcc, self.fps, self.wrist_image_sav_res)

    def stop_current_recording(self):
        if self.vid_left.isOpened():
            self.vid_left.release()
        if self.vid_psm1_endo.isOpened():
            self.vid_psm1_endo.release()
        if self.vid_psm2_endo.isOpened():
            self.vid_psm2_endo.release()

class ros_topics:
    def __init__(self, output_folder_path):
        
        # Initialize recording state
        self.current_instruction = None
        self.start_time = None
        
        # Init the labels file
        self.label_file_path = os.path.join(output_folder_path, f'labels_w_metadata_infos.txt')

        # --------------------------
        
        # Initialize ROS nodes        
        self.bridge = CvBridge()
        
        self.usb_camera_sub_left = rospy.Subscriber("/jhu_daVinci/left/image_raw", Image, self.get_camera_image_left, queue_size=1)
        self.endo_cam_psm1_sub = rospy.Subscriber("/PSM1/endoscope_img", Image, self.get_endo_cam_psm1, queue_size=1)
        self.endo_cam_psm2_sub = rospy.Subscriber("/PSM2/endoscope_img", Image, self.get_endo_cam_psm2, queue_size=1)
        self.instructor_sub = rospy.Subscriber("/instructor_prediction", String, self.get_instruction, queue_size=1)
        self.pause_robot_sub = rospy.Subscriber("/pause_robot", Bool, self.pause_robot_callback, queue_size=1)
        self.user_correction_instruction_sub = rospy.Subscriber("/hl_policy_correction_phase_instruction", String, self.get_instruction_correction, queue_size=1)
        self.user_direction_instruction_sub = rospy.Subscriber("/direction_instruction_user", String, self.get_direction_correction, queue_size=1)
        self.direction_sub = rospy.Subscriber("/robot_direction", String, self.get_direction, queue_size=1)
        self.psm_sub = rospy.Subscriber("/psm", String, self.get_psm, queue_size=1)
        
        #psm1
        self.psm1_sub = rospy.Subscriber("/PSM1/measured_cp", PoseStamped, self.get_psm1_pose, queue_size=1)
        self.psm1_sp_sub = rospy.Subscriber("/PSM1/setpoint_cp", PoseStamped, self.get_psm1_setpoint, queue_size=1)
        self.psm1_rcm_sub = rospy.Subscriber("PSM1/local/measured_cp", PoseStamped, self.get_psm1_rcm_pose, queue_size=1)
        self.psm1_jaw_sub = rospy.Subscriber("PSM1/jaw/measured_js", JointState, self.get_psm1_jaw, queue_size=1)
        self.psm1_jaw_sp_sub = rospy.Subscriber("PSM1/jaw/setpoint_js", JointState, self.get_psm1_jaw_sp, queue_size=1)

        #psm2
        self.psm2_sub = rospy.Subscriber("/PSM2/measured_cp", PoseStamped, self.get_psm2_pose, queue_size=1)    
        self.psm2_sp_sub = rospy.Subscriber("/PSM2/setpoint_cp", PoseStamped, self.get_psm2_setpoint, queue_size=1)
        self.psm2_rcm_sub = rospy.Subscriber("PSM2/local/measured_cp", PoseStamped, self.get_psm2_rcm_pose, queue_size=1)
        self.psm2_jaw_sub = rospy.Subscriber("PSM2/jaw/measured_js", JointState, self.get_psm2_jaw, queue_size=1)
        self.psm2_jaw_sp_sub = rospy.Subscriber("PSM2/jaw/setpoint_js", JointState, self.get_psm2_jaw_sp, queue_size=1)

        # ecm
        self.ecm_sub = rospy.Subscriber("/ECM/measured_cp", PoseStamped, self.get_ecm_pose, queue_size=1)
        self.ecm_rcm_sub = rospy.Subscriber("ECM/local/measured_cp", PoseStamped, self.get_ecm_rcm_pose, queue_size=1)
        
        # sujs
        self.sub1 = rospy.Subscriber("/SUJ/PSM1/measured_cp", PoseStamped, self.c1, queue_size=1)
        self.sub2 = rospy.Subscriber("/SUJ/PSM1/measured_js", JointState, self.c2, queue_size=1)
        self.sub3 = rospy.Subscriber("/SUJ/PSM2/measured_cp", PoseStamped, self.c3, queue_size=1)
        self.sub4 = rospy.Subscriber("/SUJ/PSM2/measured_js", JointState, self.c4, queue_size=1)
        self.sub7 = rospy.Subscriber("/SUJ/ECM/measured_cp", PoseStamped, self.c7, queue_size=1)
        self.sub8 = rospy.Subscriber("/SUJ/ECM/measured_js", JointState, self.c8, queue_size=1)
        
        # psms js
        self.sub9 = rospy.Subscriber("/PSM1/measured_js", JointState, self.c9, queue_size=1)
        self.sub10 = rospy.Subscriber("/PSM1/setpoint_js", JointState, self.c10, queue_size=1)
        self.sub11 = rospy.Subscriber("/PSM2/measured_js", JointState, self.c11, queue_size=1)
        self.sub12 = rospy.Subscriber("/PSM2/setpoint_js", JointState, self.c12, queue_size=1)
        self.sub15 = rospy.Subscriber("/ECM/measured_js", JointState, self.c15, queue_size=1)
        self.sub16 = rospy.Subscriber("/ECM/setpoint_js", JointState, self.c16, queue_size=1)

    # -------------------------------- Callbacks --------------------------------

    def get_camera_image_left(self, data):
        global usb_image_left
        usb_image_left = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # RGB

    def get_endo_cam_psm1(self, data):
        global endo_cam_psm1
        endo_cam_psm1 = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # BGR

    def get_endo_cam_psm2(self, data):
        global endo_cam_psm2
        endo_cam_psm2 = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # BGR
        
    def get_instruction(self, data):
        global is_recording
        new_instruction = data.data
        if not record_wo_labels_flag:
            if self.current_instruction is None:
                self.current_instruction = new_instruction
            elif is_recording and new_instruction != self.current_instruction:
                current_time = datetime.now() - self.start_time
                current_time_str = f"{current_time.seconds // 3600}:{(current_time.seconds % 3600) // 60}:{current_time.seconds % 60}:{current_time.microseconds // 1000}"
                self.current_instruction = new_instruction
                label_line = f"{current_time_str}, {self.current_instruction}"
                with open(self.label_file_path, 'a') as f:
                    f.write(f"{label_line}\n")
                print(label_line)

    def get_instruction_correction(self, data):
        global is_recording
        global record_wo_labels_flag
        if is_recording and not record_wo_labels_flag:
            correction = data.data
            if correction != self.current_instruction:
                current_time = datetime.now() - self.start_time
                current_time_str = f"{current_time.seconds // 3600}:{(current_time.seconds % 3600) // 60}:{current_time.seconds % 60}:{current_time.microseconds // 1000}"
                self.current_instruction = correction
                label_line = f"{current_time_str}, {self.current_instruction}, correction"
                with open(self.label_file_path, 'a') as f:
                    f.write(f"{label_line}\n")
                print(label_line)

    def get_direction_correction(self, data): # TODO: Check if this works
        global is_recording
        global record_wo_labels_flag
        if is_recording and not record_wo_labels_flag:
            current_time = datetime.now() - self.start_time
            current_time_str = f"{current_time.seconds // 3600}:{(current_time.seconds % 3600) // 60}:{current_time.seconds % 60}:{current_time.microseconds // 1000}"
            direction_correction = data.data
            label_line = f"{current_time_str}, {self.current_instruction} recovery, direction correction: {direction_correction}"
            with open(self.label_file_path, 'a') as f:
                f.write(f"{label_line}\n")
            print(label_line)

    def get_direction(self, data):
        global is_recording
        global psm      
        if is_recording and not record_wo_labels_flag:
            current_time = datetime.now() - self.start_time
            current_time_str = f"{current_time.seconds // 3600}:{(current_time.seconds % 3600) // 60}:{current_time.seconds % 60}:{current_time.microseconds // 1000}"
            direction_correction = data.data
            if psm:
                language_correction = get_language_correction(direction_correction, psm) # TODO: Check if this works
            else:
                language_correction = direction_correction
                print("Couldn't not map it to instrument as psm not published since started the recording script")
            label_line = f"{current_time_str}, {self.current_instruction} recovery, direction correction: {language_correction}" 
            with open(self.label_file_path, 'a') as f:
                f.write(f"{label_line}\n")
            print(label_line)
            
    def get_psm(self, data):
        global psm
        psm = data.data
       
    def get_language_correction(direction_correction, psm):
        # Mapping of the directions and psm to the language instruction
        directions_mapping = {
            "left": "to the left",
            "right": "to the right",
            "up": "higher",
            "down": "lower",
            "forward": "away from me",
            "backward": "towards me",
            "open": "open",
            "close": "close"
        }

        arms_mapping = {
            "psm1": "right arm",
            "psm2": "left arm"
        }

        if direction in ["open", "close"]:
            if psm == "psm1":
                return f"{directions_mapping[direction]} right gripper"
            elif psm == "psm2":
                return f"{directions_mapping[direction]} left gripper"
        else:
            return f"move {arms_mapping[psm]} {directions_mapping[direction]}"
            
    def pause_robot_callback(self, data):
        global is_recording
        if is_recording and not record_wo_labels_flag:
            robot_paused = data.data
            current_time = datetime.now() - self.start_time
            current_time_str = f"{current_time.seconds // 3600}:{(current_time.seconds % 3600) // 60}:{current_time.seconds % 60}:{current_time.microseconds // 1000}"
            if robot_paused:
                label_line = f"{current_time_str}, robot_paused"
                with open(self.label_file_path, 'a') as f:
                    f.write(f"{label_line}\n")
            else:
                label_line = f"{current_time_str}, robot_resumed"
                with open(self.label_file_path, 'a') as f:
                    f.write(f"{label_line}\n")
            print(label_line)

    def c1(self, data):
        global suj1_pose
        suj1_pose = data.pose

    def c2(self, data):
        global suj1_jp
        suj1_jp = data.position

    def c3(self, data):
        global suj2_pose
        suj2_pose = data.pose

    def c4(self, data):
        global suj2_jp
        suj2_jp = data.position

    def c7(self, data):
        global suj_ecm_pose
        suj_ecm_pose = data.pose

    def c8(self, data):
        global suj_ecm_jp
        suj_ecm_jp = data.position

    def c9(self, data):
        global psm1_js
        psm1_js = data.position

    def c10(self, data):
        global psm1_set_js
        psm1_set_js = data.position

    def c11(self, data):
        global psm2_js
        psm2_js = data.position

    def c12(self, data):
        global psm2_set_js
        psm2_set_js = data.position

    def c15(self, data):
        global ecm_js
        ecm_js = data.position

    def c16(self, data):
        global ecm_set_js
        ecm_set_js = data.position

    def get_ecm_pose(self, data):
        global ecm_pose
        ecm_pose = data.pose
        global kinematics_timestamp
        kinematics_timestamp = data.header.stamp

    def get_ecm_rcm_pose(self, data):
        global ecm_rcm_pose
        ecm_rcm_pose = data.pose

    def get_psm1_pose(self, data):
        global psm1_pose
        psm1_pose = data.pose

    def get_psm1_setpoint(self, data):
        global psm1_sp
        psm1_sp = data.pose

    def get_psm1_rcm_pose(self, data):
        global psm1_rcm_pose
        psm1_rcm_pose = data.pose

    def get_psm2_pose(self, data):
        global psm2_pose
        psm2_pose = data.pose

    def get_psm2_setpoint(self, data):
        global psm2_sp
        psm2_sp = data.pose

    def get_psm2_rcm_pose(self, data):
        global psm2_rcm_pose
        psm2_rcm_pose = data.pose

    def get_psm1_jaw(self, data):
        global psm1_jaw
        psm1_jaw = data.position[0]

    def get_psm1_jaw_sp(self, data):
        global psm1_jaw_sp
        psm1_jaw_sp = data.position[0]

    def get_psm2_jaw(self, data):
        global psm2_jaw
        psm2_jaw = data.position[0]

    def get_psm2_jaw_sp(self, data):
        global psm2_jaw_sp
        psm2_jaw_sp = data.position[0]


# --------------------------------- Helper functions ---------------------------------

def save_kinematics_batch(output_folder_path, ee_points, first_batch = True):
    header =  [
        "timestamp",
        
        "psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z", # PSM1
        "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
        
        "psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
        "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
        
        "psm1_jaw", "psm1_jaw_sp",

        "psm1_rcm_pose.position.x", "psm1_rcm_pose.position.y", "psm1_rcm_pose.position.z", 
        "psm1_rcm_pose.orientation.x", "psm1_rcm_pose.orientation.y", "psm1_rcm_pose.orientation.z", "psm1_rcm_pose.orientation.w",
        
        "psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z", # PSM 2
        "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
        
        "psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
        "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",

        "psm2_jaw", "psm2_jaw_sp",

        "psm2_rcm_pose.position.x", "psm2_rcm_pose.position.y", "psm2_rcm_pose.position.z",
        "psm2_rcm_pose.orientation.x", "psm2_rcm_pose.orientation.y", "psm2_rcm_pose.orientation.z", "psm2_rcm_pose.orientation.w",

        "ecm_pose.position.x", "ecm_pose.position.y", "ecm_pose.position.z", # ECM
        "ecm_pose.orientation.x", "ecm_pose.orientation.y", "ecm_pose.orientation.z", "ecm_pose.orientation.w",

        "ecm_rcm_pose.position.x", "ecm_rcm_pose.position.y", "ecm_rcm_pose.position.z",
        "ecm_rcm_pose.orientation.x", "ecm_rcm_pose.orientation.y", "ecm_rcm_pose.orientation.z", "ecm_rcm_pose.orientation.w",

        "suj1_pose.position.x", "suj1_pose.position.y", "suj1_pose.position.z",
        "suj1_pose.orientation.x", "suj1_pose.orientation.y", "suj1_pose.orientation.z", "suj1_pose.orientation.w",
        "suj1_jp[0]", "suj1_jp[1]", "suj1_jp[2]", "suj1_jp[3]",

        "suj2_pose.position.x", "suj2_pose.position.y", "suj2_pose.position.z",
        "suj2_pose.orientation.x", "suj2_pose.orientation.y", "suj2_pose.orientation.z", "suj2_pose.orientation.w",
        "suj2_jp[0]", "suj2_jp[1]", "suj2_jp[2]", "suj2_jp[3]",

        "suj_ecm_pose.position.x", "suj_ecm_pose.position.y", "suj_ecm_pose.position.z",
        "suj_ecm_pose.orientation.x", "suj_ecm_pose.orientation.y", "suj_ecm_pose.orientation.z", "suj_ecm_pose.orientation.w",
        "suj_ecm_jp[0]", "suj_ecm_jp[1]", "suj_ecm_jp[2]", "suj_ecm_jp[3]",

        "psm1_js[0]", "psm1_js[1]", "psm1_js[2]", "psm1_js[3]", "psm1_js[4]", "psm1_js[5]",
        "psm1_set_js[0]", "psm1_set_js[1]", "psm1_set_js[2]", "psm1_set_js[3]", "psm1_set_js[4]", "psm1_set_js[5]",

        "psm2_js[0]", "psm2_js[1]", "psm2_js[2]", "psm2_js[3]", "psm2_js[4]", "psm2_js[5]",
        "psm2_set_js[0]", "psm2_set_js[1]", "psm2_set_js[2]", "psm2_set_js[3]", "psm2_set_js[4]", "psm2_set_js[5]",

        "ecm_js[0]", "ecm_js[1]", "ecm_js[2]", "ecm_js[3]",
        "ecm_set_js[0]", "ecm_set_js[1]", "ecm_set_js[2]", "ecm_set_js[3]",
    ]
    ee_save_path = os.path.join(output_folder_path, f"ee_csv.csv")
    with open(ee_save_path, 'a', newline='') as f:
        csv_data = pd.DataFrame(ee_points, columns=header)
        csv_data.to_csv(f, index=False, header=first_batch)

def get_kinematic_data():    
    current_kinematic_data = [
        kinematics_timestamp,
        #PSM1
        psm1_pose.position.x, psm1_pose.position.y, psm1_pose.position.z, # PSM1
        psm1_pose.orientation.x, psm1_pose.orientation.y, psm1_pose.orientation.z, psm1_pose.orientation.w,
        psm1_sp.position.x, psm1_sp.position.y, psm1_sp.position.z,
        psm1_sp.orientation.x, psm1_sp.orientation.y, psm1_sp.orientation.z, psm1_sp.orientation.w,
        psm1_jaw, psm1_jaw_sp,
        psm1_rcm_pose.position.x, psm1_rcm_pose.position.y, psm1_rcm_pose.position.z,
        psm1_rcm_pose.orientation.x, psm1_rcm_pose.orientation.y, psm1_rcm_pose.orientation.z, psm1_rcm_pose.orientation.w,
        
        # PSM2
        psm2_pose.position.x, psm2_pose.position.y, psm2_pose.position.z, # PSM 2
        psm2_pose.orientation.x, psm2_pose.orientation.y, psm2_pose.orientation.z, psm2_pose.orientation.w,
        psm2_sp.position.x, psm2_sp.position.y, psm2_sp.position.z,
        psm2_sp.orientation.x, psm2_sp.orientation.y, psm2_sp.orientation.z, psm2_sp.orientation.w,
        psm2_jaw, psm2_jaw_sp,
        psm2_rcm_pose.position.x, psm2_rcm_pose.position.y, psm2_rcm_pose.position.z,
        psm2_rcm_pose.orientation.x, psm2_rcm_pose.orientation.y, psm2_rcm_pose.orientation.z, psm2_rcm_pose.orientation.w,
        # ECM
        ecm_pose.position.x, ecm_pose.position.y, ecm_pose.position.z, # ECM
        ecm_pose.orientation.x, ecm_pose.orientation.y, ecm_pose.orientation.z, ecm_pose.orientation.w,
        # ECM RCM
        ecm_rcm_pose.position.x, ecm_rcm_pose.position.y, ecm_rcm_pose.position.z,
        ecm_rcm_pose.orientation.x, ecm_rcm_pose.orientation.y, ecm_rcm_pose.orientation.z, ecm_rcm_pose.orientation.w,
        # suj poses
        suj1_pose.position.x, suj1_pose.position.y, suj1_pose.position.z,
        suj1_pose.orientation.x, suj1_pose.orientation.y, suj1_pose.orientation.z, suj1_pose.orientation.w,
        suj1_jp[0], suj1_jp[1], suj1_jp[2], suj1_jp[3],

        suj2_pose.position.x, suj2_pose.position.y, suj2_pose.position.z,
        suj2_pose.orientation.x, suj2_pose.orientation.y, suj2_pose.orientation.z, suj2_pose.orientation.w,
        suj2_jp[0], suj2_jp[1], suj2_jp[2], suj2_jp[3],

        suj_ecm_pose.position.x, suj_ecm_pose.position.y, suj_ecm_pose.position.z,
        suj_ecm_pose.orientation.x, suj_ecm_pose.orientation.y, suj_ecm_pose.orientation.z, suj_ecm_pose.orientation.w,
        suj_ecm_jp[0], suj_ecm_jp[1], suj_ecm_jp[2], suj_ecm_jp[3],

        # joints
        psm1_js[0], psm1_js[1], psm1_js[2], psm1_js[3], psm1_js[4], psm1_js[5],
        psm1_set_js[0], psm1_set_js[1], psm1_set_js[2], psm1_set_js[3], psm1_set_js[4], psm1_set_js[5],

        psm2_js[0], psm2_js[1], psm2_js[2], psm2_js[3], psm2_js[4], psm2_js[5],
        psm2_set_js[0], psm2_set_js[1], psm2_set_js[2], psm2_set_js[3], psm2_set_js[4], psm2_set_js[5],

        ecm_js[0], ecm_js[1], ecm_js[2], ecm_js[3],
        ecm_set_js[0], ecm_set_js[1], ecm_set_js[2], ecm_set_js[3],
    ]
    
    return current_kinematic_data
    
# Signal handler for stopping the recording
def stop_recording_signal_handler(sig, frame):
    print("Ctrl+C detected, stopping the recording...")  
    global exit_flag  
    exit_flag = True

# ------------------------------- Main code -------------------------------

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Parameters
    global record_wo_labels_flag
    record_wo_labels_flag = args.record_wo_labels_flag
    wrist_image_sav_res = tuple(args.wrist_image_sav_res)
    endo_image_save_res = tuple(args.endo_image_save_res)
    batch_interval = args.batch_interval
    output_folder_path = args.output_folder_path
    
    # Create output folder if it doesn't exist yet
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    time.sleep(2)
    
    # Create ROS node and start recording manager
    rospy.init_node('rostopic_recorder', anonymous=True)
    ros_fps = 30  # 30hz
    rm = RecordingManager(output_folder_path, ros_fps, wrist_image_sav_res, endo_image_save_res)
    rt = ros_topics(output_folder_path)
    time.sleep(2)
    rate = rospy.Rate(ros_fps)

    # Set up exit condition
    rm.start_new_recording()
    signal.signal(signal.SIGINT, stop_recording_signal_handler)

    # Init batch variables
    ee_points_batch = []
    first_batch = True
    
    # Main loop - labels saved in the background via ros topic class callbacks
    started_recording = False
    while not rospy.is_shutdown() and not exit_flag:
        global is_recording
        # Check if images are received
        if usb_image_left is None or endo_cam_psm1 is None or endo_cam_psm2 is None:
            print("No images received, cancelling the recording")
            break
        elif not is_recording and not record_wo_labels_flag:
            # Wait for the first instruction
            while rt.current_instruction is None:
                if exit_flag:
                    break
                time.sleep(0.1)
            if exit_flag:
                break
            
            # Once the first instruction is received, start the recording
            rt.start_time = datetime.now()
            label_line = f"0:0:0:0, {rt.current_instruction}"
            with open(rt.label_file_path, 'a') as f: # Save the first instruction in labels file
                f.write(f"{label_line}\n")
            started_recording = True
            print("Starting the recording now. Stop recording by pressing Ctrl+C\n")
            print(label_line)
            is_recording = True
            last_batch_time = time.time()
        elif not is_recording:
            print("Starting the recording now. Stop recording by pressing Ctrl+C\n")
            is_recording = True
            last_batch_time = time.time()

        # Collect kinematic data
        current_kinematic_data = get_kinematic_data()
        ee_points_batch.append(current_kinematic_data)

        # Save batches every batch_interval seconds
        if time.time() - last_batch_time > batch_interval:
            save_kinematics_batch(output_folder_path, ee_points_batch, first_batch)
            if first_batch:
                first_batch = False
            ee_points_batch = [] # Clear the batch
            last_batch_time = time.time()

        # Save frames to video files
        rm.vid_left.write(cv2.cvtColor(cv2.resize(usb_image_left, endo_image_save_res), cv2.COLOR_RGB2BGR))
        rm.vid_psm1_endo.write(cv2.resize(endo_cam_psm1, wrist_image_sav_res))
        rm.vid_psm2_endo.write(cv2.resize(endo_cam_psm2, wrist_image_sav_res))

        rate.sleep()

    # Save remaining batch elements
    if ee_points_batch:
        save_kinematics_batch(output_folder_path, ee_points_batch, first_batch)

    # Stop the recording + ROS
    rm.stop_current_recording()
    rospy.signal_shutdown("Application closed")
    print(f"Recording stopped and saved to {output_folder_path}")
    
    # Save a copy to correct of the labels (this should be postprocessed and renamed into labels.txt - for generating the experiment data folder structure)
    if not record_wo_labels_flag and started_recording: # TODO: Check if this works preventing error when no labels has been saved
        with open(rt.label_file_path, 'r') as f:
            lines = f.readlines()
        with open(os.path.join(output_folder_path, 'labels_to_correct.txt'), 'w') as f:
            for line in lines:
                f.write(line)


def parse_args():
    parser = argparse.ArgumentParser(description="ROS topic recorder with video and kinematics logging.")
    
    # Wrist and endoscope image resolutions
    parser.add_argument('--wrist_image_sav_res', type=int, nargs=2, default=[640, 480],
                        help="Resolution for wrist camera images (default: [640, 480])")
    parser.add_argument('--endo_image_save_res', type=int, nargs=2, default=[960, 540],
                        help="Resolution for endoscope camera images (default: [960, 540])")
    
    # Record without labels flag
    parser.add_argument('--record_wo_labels_flag', action='store_true', help="Flag to record without labels (default: False)")
    
    # Save batch every n seconds
    parser.add_argument('--batch_interval', type=int, default=10, help="Save batch every n seconds (default: 10)")
    
    # Output folder path
    PATH_TO_YAY_ROBOT = os.getenv('PATH_TO_YAY_ROBOT')
    default_experiment_folder_name = f"experiment_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    default_output_folder_path = os.path.join(PATH_TO_YAY_ROBOT, "experiment_recordings", default_experiment_folder_name) # Alternatively in home directory: os.path.join(os.path.expanduser("~"), "_experiments", experiments_folder_name)
    parser.add_argument('--output_folder_path', type=str, default=default_output_folder_path, help="Base output folder path")
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
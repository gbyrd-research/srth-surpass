import rospy
import crtk
from std_msgs.msg import String, Bool
from dvrk_scripts.dvrk_control import example_application
from rostopics import ros_topics
import time
import PyKDL
import numpy as np
class RobotDirectionListener:
    def __init__(self):
        # ------------------ Initialize ROS node------------------ #
        # rospy.init_node('robot_direction_listener', anonymous=True)

        # ------------------ Initialize CRTK and example applications ------------------ #
        self.rt = ros_topics()
        self.ral = crtk.ral('dvrk_arm_test')
        self.psm1_app = example_application(self.ral, "PSM1", 1)
        self.psm2_app = example_application(self.ral, "PSM2", 1)
        self.user_correction_start_t = time.time()
        self.psm_app = self.psm1_app
        self.current_psm = "psm1"
        self.robot_pose_psm1 = None
        self.jaw_angle_psm1 = None
        self.robot_pose_psm2 = None
        self.jaw_angle_psm2 = None
        self.use_preprogrammed_correction = False
        self.is_correction = False

        # ------------------ ROS Subscribers ------------------ #
        rospy.Subscriber("/robot_direction", String, self.robot_direction_callback)
        rospy.Subscriber("/psm", String, self.psm_callback)
        rospy.Subscriber("/use_preprogrammed_correction", Bool, self.use_preprogrammed_correction_callback)
        rospy.Subscriber("/direction_instruction", String, self.direction_instruction_callback, queue_size=1)
        rospy.Subscriber('/is_correction', Bool, self.is_correction_callback, queue_size=1)
        rospy.Subscriber('/direction_instruction_user', String, self.user_correction_callback, queue_size=1)

    ## ------------------------ Callbacks ------------------------ ##

    def psm_callback(self, msg):
        self.current_psm = msg.data
        print(f"PSM received: {self.current_psm}")
        if self.current_psm == "psm1":
            self.psm_app = self.psm1_app
        else:
            self.psm_app = self.psm2_app

    def remember_robot_pose(self):
        if self.current_psm == "psm1":
            self.robot_pose_psm1 = self.psm_app.arm.setpoint_cp()
            self.jaw_angle_psm1 = self.psm_app.arm.jaw.setpoint_jp()
            
            print(f"PSM1 pose remembered: {self.robot_pose_psm1}")
        elif self.current_psm == "psm2":
            self.robot_pose_psm2 = self.psm_app.arm.setpoint_cp()
            self.jaw_angle_psm2 = self.psm_app.arm.jaw.setpoint_jp()
            
            print(f"PSM2 pose remembered: {self.robot_pose_psm2}")

    def move_robot_to_pose(self):
        robot_pose = []
        if self.current_psm == "psm1":
            print(self.robot_pose_psm1.p)
            print(self.robot_pose_psm1.M.GetQuaternion())
            print(self.jaw_angle_psm1)
            robot_pose = np.array((self.robot_pose_psm1.p.x(),
                                   self.robot_pose_psm1.p.y(),
                                   self.robot_pose_psm1.p.z(),
                                   self.robot_pose_psm1.M.GetQuaternion()[0],
                                   self.robot_pose_psm1.M.GetQuaternion()[1],
                                   self.robot_pose_psm1.M.GetQuaternion()[2],
                                   self.robot_pose_psm1.M.GetQuaternion()[3],
                                   self.jaw_angle_psm1[0]))

        
        elif self.current_psm == "psm2":
            robot_pose = np.array((self.robot_pose_psm2.p.x(),
                                   self.robot_pose_psm2.p.y(),
                                   self.robot_pose_psm2.p.z(),
                                   self.robot_pose_psm2.M.GetQuaternion()[0],
                                   self.robot_pose_psm2.M.GetQuaternion()[1],
                                   self.robot_pose_psm2.M.GetQuaternion()[2],
                                   self.robot_pose_psm2.M.GetQuaternion()[3],
                                   self.jaw_angle_psm2[0]))
            
            
        print(f"Robot pose: {robot_pose}")
        
        done = self.ral.spin_and_execute(self.psm_app.run_full_pose_goal, robot_pose)
        print(f"Robot moved to pose: {robot_pose}")

    def robot_direction_callback(self, msg):
        if type(msg) == str:
            robot_dir = msg
        else:
            robot_dir = msg.data
            
        if robot_dir == "store_pose":
            self.remember_robot_pose()
        elif robot_dir == "move_to_pose":
            self.move_robot_to_pose()
        else:
            robot_move = True
            if robot_move:
                # Execute robot movement based on the direction received
                done = self.ral.spin_and_execute(self.psm_app.move_robot_in_direction, robot_dir)
                print(f"Robot moved in direction {robot_dir}")
                time.sleep(0.5)
                robot_move = False
                

    def direction_instruction_callback(self, msg):
        message = msg.data
        if self.is_correction:
            self.correction(message)

    def correction(self, message):
        if message.startswith("move left arm"):
            self.current_psm = "psm2"
            self.psm_app = self.psm2_app
            ## remove the first three words from the message
            message = message.split(' ', 3)[3]
            self.robot_direction_callback(message)

        elif message.startswith("move right arm"):
            self.current_psm = "psm1"
            self.psm_app = self.psm1_app
            ## remove the first three words from the message
            message = message.split(' ', 3)[3]
            self.robot_direction_callback(message)

        elif message.startswith("open") or message.startswith("close"):
            if message.split(' ', 2)[1] == "left":
                self.current_psm = "psm2"
                self.psm_app = self.psm2_app
                self.robot_direction_callback(message.split(' ', 2)[0])
            elif message.split(' ', 2)[1] == "right":
                self.current_psm = "psm1"
                self.psm_app = self.psm1_app
                self.robot_direction_callback(message.split(' ', 2)[0])
            elif message.split(' ', 2)[1] == "both":
                self.current_psm = "psm1"
                self.psm_app = self.psm1_app
                self.robot_direction_callback(message.split(' ', 2)[0])
                self.current_psm = "psm2"
                self.psm_app = self.psm2_app
                self.robot_direction_callback(message.split(' ', 2)[0])

    # def is_correction_callback(self, msg):
    #     if self.use_preprogrammed_correction:
    #         self.is_correction = msg.data
    #         print(f"Correction: {self.is_correction}")

    def use_preprogrammed_correction_callback(self, msg):
        self.use_preprogrammed_correction = msg.data
        print(f"Use preprogrammed correction: {self.use_preprogrammed_correction}")

    def user_correction_callback(self, msg):
        if self.use_preprogrammed_correction:
            self.user_correction_start_t = time.time()
            
            self.is_correction = True  # Set the correction flag immediately when user issues a correction
            self.correction(msg.data)

    def is_correction_callback(self, msg):
        if self.use_preprogrammed_correction:
        
            if self.user_correction is not None and time.time() - self.user_correction_start_t < 3:
                self.is_correction = True
            else:
                self.is_correction = msg.data
                if not self.is_correction:
                    self.user_correction = None  # Clear user_correction if it's not active

            print("Is correction active: ", self.is_correction)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    listener = RobotDirectionListener()
    listener.run()

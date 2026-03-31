import rospy
import crtk
from std_msgs.msg import String, Bool
from dvrk_scripts.dvrk_control import example_application
from rostopics import ros_topics
import time
import PyKDL
import numpy as np

class RobotPause:

    def __init__(self):

        # ------------------ Initialize ROS node------------------ #

        rospy.init_node('robot_pause_listener', anonymous=True)

        # ------------------ Initialize variables ------------------ #

        self.idx = 1
        self.current_task = None
        self.previous_task = None
        self.intervention = False


        self.pause_trigger_predictions = ["clipping second clip left tube",
                                          "clipping third clip left tube",
                                          "go to the cutting position left tube", 
                                          "clipping first clip right tube",
                                          "clipping second clip right tube",
                                          "clipping third clip right tube",
                                          "go to the cutting position right tube"]

        self.corresponding_previous_tasks = ["going back first clip left tube", 
                                             "going back second clip left tube",
                                             "going back third clip left tube",
                                             "go back from the cut left tube",
                                             "going back first clip right tube",
                                             "going back second clip right tube",
                                             "going back third clip right tube"]

        self.print_statements = ["Please load the second clip !",
                                 "Please load the third clip !",
                                 "Please switch psm1 to scissors !",
                                 "Please switch psm1 to clip applier and load the first clip !",
                                 "Please load the second clip !",
                                 "Please load the third clip !",
                                 "Please switch psm1 to scissors !"]

        # ------------------ ROS Publishers ------------------ #

        self.pause_publisher = rospy.Publisher('/pause_robot', Bool, queue_size=10)

        self.pause_hl_publisher = rospy.Publisher('/pause_hl', Bool, queue_size=10)

        self.robot_dir_pub = rospy.Publisher("/robot_direction", String, queue_size=10)
        
        # ------------------ ROS Subscribers ------------------ #
        rospy.Subscriber("/clip_loading_tool_switching_required", Bool, self.intervention_callback, queue_size=10)
        rospy.Subscriber("/instructor_prediction", String, self.language_instruction_callback, queue_size=10)


    ## ------------------------ Callbacks ------------------------ ##
    def intervention_callback(self, msg):
        self.intervention = msg.data


    def language_instruction_callback(self, msg):
        self.current_task = msg.data
        # if self.currect_task != self.previous_task:
        #     print(f"Current task: {self.current_task}\n")
        if self.idx == len(self.pause_trigger_predictions):
            if self.current_task == "go back from the cut right tube":
                time.sleep(6)
                print("task completed")
                rospy.signal_shutdown("task completed")
                exit()
        else:

            if (self.previous_task == self.corresponding_previous_tasks[self.idx] and 
                self.current_task == self.pause_trigger_predictions[self.idx]):

                # Publish True to pause the robot and highlight
                self.pause_publisher.publish(True)
                self.pause_hl_publisher.publish(True)
                # self.robot_dir_pub.publish("store_pose")

                if not self.intervention:
                    print(f"Pausing robot due to prediction: {self.current_task}\n")
                else:
                    print(f"Pausing robot due to high level flag\n")
                print("\n--------------------------------------------------\n",self.print_statements[self.idx],"\n--------------------------------------------------\n")

                # Wait for user input to continue
                input("Press Enter to resume...")
                
                # Publish False to resume the robot and highlight
                self.pause_publisher.publish(False)
                self.pause_hl_publisher.publish(False)

                self.idx += 1
                print("Resumed robot operation\n")
                print("idx", self.idx)    

        # update previous task
        self.previous_task = self.current_task

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    listener = RobotPause()
    listener.run()

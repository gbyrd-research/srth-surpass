from visualization_msgs.msg import Marker
import rospy


# Publish the mesh
rospy.init_node('mesh_publisher', anonymous=True)
pub = rospy.Publisher('mesh', Marker, queue_size=10)
rate = rospy.Rate(10)  # 10 Hz
# Initialize the marker message
msg = Marker()
msg.header.frame_id = "baselink"  # Set to an appropriate frame ID (e.g., "world" or your specific frame)
msg.header.stamp = rospy.Time.now()
msg.type = Marker.MESH_RESOURCE
msg.action = Marker.ADD
msg.mesh_resource = "file:///home/iulian/Downloads/animal_OR_12_04_06/textured_output.obj"
msg.mesh_use_embedded_materials = True  # Enable textures for the mesh
msg.scale.x = 1.0  # Adjust scale if necessary
msg.scale.y = 1.0
msg.scale.z = 1.0

# Set a default position and orientation
msg.pose.position.x = 0.0
msg.pose.position.y = 0.0
msg.pose.position.z = 0.0
msg.pose.orientation.x = 0.0
msg.pose.orientation.y = 1.0
msg.pose.orientation.z = 0.0
msg.pose.orientation.w = 0.0

while not rospy.is_shutdown():
    msg.header.stamp = rospy.Time.now()  # Update timestamp
    pub.publish(msg)
    rate.sleep()


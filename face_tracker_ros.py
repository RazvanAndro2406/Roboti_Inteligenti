#!/usr/bin/env python3
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class FaceTrackerNode:
    def __init__(self):
        rospy.init_node('face_tracker_node', anonymous=True)
        
        # Initialize CvBridge and Haar Cascade
        self.bridge = CvBridge()
        self.haar_cascade = cv.CascadeClassifier('haar_face.xml')
        
        if self.haar_cascade.empty():
            rospy.logerr("Failed to load haar_face.xml! Check your file path.")
        
        # ROS Subscribers and Publishers
        # Change '/camera/image_raw' to match your robot's camera topic
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Control parameters (tune these for your specific robot)
        self.angular_kp = 0.002       # How aggressively the robot turns
        self.forward_speed = 0.15     # Forward speed in m/s
        self.stop_face_width = 180    # Stop moving forward when face is this wide (in pixels)
        self.deadband_pixels = 25     # Ignore small horizontal errors to prevent oscillation

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image message to OpenCV BGR image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        height, width, _ = frame.shape
        frame_center_x = width / 2.0

        # 2. Enhance image for Haar Cascade (Grayscale + Histogram Equalization)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        enhanced_gray = cv.equalizeHist(gray)

        # 3. Detect faces
        faces = self.haar_cascade.detectMultiScale(
            enhanced_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        twist = Twist()

        if len(faces) > 0:
            # Track the largest face (closest person) by area (w * h)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            # Calculate horizontal error from the center of the camera frame
            face_center_x = x + (w / 2.0)
            error_x = frame_center_x - face_center_x

            # Angular control: positive error means face is on the left -> turn left (+z)
            if abs(error_x) > self.deadband_pixels:
                twist.angular.z = self.angular_kp * error_x
            else:
                twist.angular.z = 0.0

            # Linear control: move forward if the face is smaller than the stopping threshold
            if w < self.stop_face_width:
                twist.linear.x = self.forward_speed
            else:
                twist.linear.x = 0.0

            # Optional visual debugging window
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.line(frame, (int(frame_center_x), 0), (int(frame_center_x), height), (255, 0, 0), 1)
        else:
            # No face detected -> stop the robot safely
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # 4. Publish velocity command
        self.cmd_vel_pub.publish(twist)

        # Display debug window (comment out if running headless over SSH)
        cv.imshow("Face Tracker", frame)
        cv.waitKey(1)

if __name__ == '__main__':
    try:
        node = FaceTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv.destroyAllWindows()
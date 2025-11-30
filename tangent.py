#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import SetMode, CommandBool
from ml_detector.msg import DetectionArray
from pymavlink import mavutil


class NFZAvoidance:
    def __init__(self):
        rospy.init_node('nfz_avoidance_node')

        # NFZ Parameters
        self.nfz_center = np.array([0.0, 25.0, 0.0])
        self.nfz_radius = 5.0
        self.extension_distance = 25.0
        self.tolerance = 0.5

        # State variables
        self.current_pose = None
        self.first_detection_pose = None
        self.second_detection_pose = None
        self.detection_active = False
        self.waypoint_calculated = False
        self.tangent_point = None
        self.straight_target = None
        self.following_tangent = False
        self.state = State()

        # Publishers
        self.setpoint_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)

        # Subscribers
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/yolo_detections', DetectionArray, self.yolo_callback)
        rospy.Subscriber('/mavros/state', State, self.state_callback)

        # Services
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd/arming')
        self.set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arming_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)

        rospy.loginfo("NFZ Avoidance Node Initialized")

    def state_callback(self, msg):
        self.state = msg
        # Enforce GUIDED mode
        if self.state.mode != "GUIDED":
            self.set_mode_srv(custom_mode="GUIDED")

    def odom_callback(self, msg):
        """Update current drone position"""
        self.current_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Send setpoints continuously in GUIDED mode
        if self.state.mode == "GUIDED":
            if self.following_tangent and self.tangent_point is not None:
                self.publish_setpoint(self.tangent_point)
                if np.linalg.norm(self.current_pose - self.tangent_point) < self.tolerance:
                    rospy.loginfo("Reached tangent point, switching to straight-line target")
                    self.following_tangent = False
            elif self.waypoint_calculated and self.straight_target is not None:
                self.publish_setpoint(self.straight_target)

    def yolo_callback(self, msg):
        if self.waypoint_calculated or self.current_pose is None:
            return
        if len(msg.detections) == 0:
            return
        if not self.detection_active:
            self.first_detection_pose = self.current_pose.copy()
            self.detection_active = True
            rospy.loginfo(f"First detection at: {self.first_detection_pose}")
            rospy.Timer(rospy.Duration(2.0), self.get_second_position, oneshot=True)

    def get_second_position(self, event):
        if self.current_pose is not None:
            self.second_detection_pose = self.current_pose.copy()
            rospy.loginfo(f"Second detection at: {self.second_detection_pose}")
            self.calculate_avoidance_waypoint()

    def calculate_avoidance_waypoint(self):
        p1 = self.first_detection_pose
        p2 = self.second_detection_pose
        direction = p2 - p1
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            rospy.logwarn("Drone moved too little to calculate trajectory")
            return

        direction /= distance
        rospy.loginfo(f"Trajectory direction: {direction}")

        intersections = self.ray_circle_intersection(p2, direction, self.nfz_center, self.nfz_radius)

        if intersections:
            rospy.logwarn("Trajectory intersects NFZ!")
            exit_point = intersections[1] if len(intersections) > 1 else intersections[0]
            self.straight_target = exit_point + direction * self.extension_distance
            rospy.loginfo(f"Straight-line target beyond NFZ: {self.straight_target}")

            self.tangent_point = self.compute_tangent_point(p2, direction)
            if self.tangent_point is not None:
                rospy.loginfo(f"Tangent point on NFZ: {self.tangent_point}")
                self.following_tangent = True

            self.waypoint_calculated = True
        else:
            rospy.loginfo("Trajectory does NOT intersect NFZ. Moving straight")
            self.straight_target = p2 + direction * self.extension_distance
            self.waypoint_calculated = True

    def ray_circle_intersection(self, ray_origin, ray_dir, circle_center, circle_radius):
        origin_2d = ray_origin[:2]
        dir_2d = ray_dir[:2]
        center_2d = circle_center[:2]
        oc = origin_2d - center_2d

        a = np.dot(dir_2d, dir_2d)
        b = 2 * np.dot(oc, dir_2d)
        c = np.dot(oc, oc) - circle_radius**2
        disc = b**2 - 4*a*c

        if disc < 0:
            return []

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        points = []
        for t in [t1, t2]:
            if t > 0:
                pt_2d = origin_2d + t * dir_2d
                pt_3d = np.array([pt_2d[0], pt_2d[1], ray_origin[2]])
                points.append(pt_3d)

        return points

    def compute_tangent_point(self, external_point, trajectory_dir):
        C = self.nfz_center[:2]
        P = external_point[:2]
        r = self.nfz_radius

        CP = P - C
        d_sq = np.dot(CP, CP)
        d = np.sqrt(d_sq)

        if d < r:
            rospy.logwarn("External point inside NFZ, cannot compute tangent")
            return None

        l = r**2 / d_sq
        m = r * np.sqrt(d_sq - r**2) / d_sq

        R = np.array([[0, -1], [1, 0]])
        T1_2d = C + l * CP + m * R.dot(CP)
        T2_2d = C + l * CP - m * R.dot(CP)

        T1 = np.array([T1_2d[0], T1_2d[1], external_point[2]])
        T2 = np.array([T2_2d[0], T2_2d[1], external_point[2]])

        dir1 = T1 - external_point
        dir2 = T2 - external_point
        dir1 /= np.linalg.norm(dir1)
        dir2 /= np.linalg.norm(dir2)

        return T1 if np.dot(dir1, trajectory_dir) > np.dot(dir2, trajectory_dir) else T2

    def publish_setpoint(self, target_point):
        msg = PositionTarget()
        msg.coordinate_frame = mavutil.mavlink.MAV_FRAME_LOCAL_NED
        msg.type_mask = (PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                         PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                         PositionTarget.IGNORE_YAW | PositionTarget.IGNORE_YAW_RATE)
        msg.position.x = target_point[0]
        msg.position.y = target_point[1]
        msg.position.z = target_point[2]
        msg.header.stamp = rospy.Time.now()
        self.setpoint_pub.publish(msg)


if __name__ == '__main__':
    try:
        nfz = NFZAvoidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


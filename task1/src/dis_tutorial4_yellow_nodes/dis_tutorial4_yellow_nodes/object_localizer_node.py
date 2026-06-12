#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from dis_tutorial4_yellow_interfaces.msg import DetectionArray
from rclpy.time import Time

import tf2_ros
import tf2_geometry_msgs
import numpy as np


class ObjectLocalizerNode(Node):

    def __init__(self):
        super().__init__('object_localizer_node')

        # The buffer stores all the transforms,
        # and the listener fills it automatically by listening to the /tf topic
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # store the most recent detections here
        self.latest_detections = None

        # velocity from odometry — detections are ignored while the robot moves
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.LINEAR_THRESHOLD = 0.05   # m/s
        self.ANGULAR_THRESHOLD = 0.05  # rad/s

        # store all the unique objects the robot has found so far
        self.localized_faces = []
        self.localized_rings = []
        self.dedup_distance = 0.5

        self.marker_id = 0

        # accumulated markers — the full list is republished periodically so
        # RViz shows all objects even after reconnecting
        self.all_markers: list = []

        # subscribers
        self.detection_sub = self.create_subscription(
            DetectionArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/oakd/rgb/preview/depth/points',
            self.pointcloud_callback,
            qos_profile_sensor_data
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/localized_objects',
            10
        )

        # one-shot: clear any markers left in RViz from a previous run
        self._clear_markers_once()
        # republish every 2 s so RViz shows current state after reconnect
        self.create_timer(2.0, self._republish_all_markers)

        self.get_logger().info("object_localizer_node started.")

    def _clear_markers_once(self):
        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.header.frame_id = "map"
        clear.header.stamp = self.get_clock().now().to_msg()
        self.all_markers.clear()
        self.localized_faces.clear()
        self.localized_rings.clear()
        self.marker_pub.publish(MarkerArray(markers=[clear]))
        self.all_markers.clear()

    def _republish_all_markers(self):
        if self.all_markers:
            self.marker_pub.publish(MarkerArray(markers=self.all_markers))

    def detection_callback(self, msg):
        self.latest_detections = msg

    def map_callback(self, msg):
        pass  # stored for the future

    def odom_callback(self, msg):
        self.linear_x = msg.twist.twist.linear.x
        self.angular_z = msg.twist.twist.angular.z

    def pointcloud_callback(self, data):
        if self.latest_detections is None:
            return

        if abs(self.linear_x) > self.LINEAR_THRESHOLD or abs(self.angular_z) > self.ANGULAR_THRESHOLD:
            self.latest_detections = None
            return

        height = data.height
        width = data.width

        detections = self.latest_detections
        self.latest_detections = None  # consume detections

        new_markers = []

        # get 3D point from pointcloud
        a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        a = a.reshape((height, width, 3))

        for det in detections.detections:
            cx = det.cx
            cy = det.cy

            # bounds check
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                continue

            d = a[cy, cx, :]

            # for unrecognisable values, skip them
            if np.any(np.isnan(d)) or np.any(np.isinf(d)):
                continue
            if d[0] == 0.0 and d[1] == 0.0 and d[2] == 0.0:
                continue

            # ignore objects that are >3m
            # dist from cam = sqrt(x^2 + y^2 + z^2)
            distance = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            if distance > 3.0:
                continue

            # transform point from camera frame to map frame
            point_in_camera = PointStamped()
            point_in_camera.header.frame_id = data.header.frame_id
            point_in_camera.header.stamp = Time().to_msg()
            point_in_camera.point.x = float(d[0])
            point_in_camera.point.y = float(d[1])
            point_in_camera.point.z = float(d[2])

            try:
                point_in_map = self.tf_buffer.transform(point_in_camera, "map")
                mx = point_in_map.point.x
                my = point_in_map.point.y
                mz = point_in_map.point.z

                if np.isnan(mx) or np.isnan(my) or np.isnan(mz):
                    continue

                # deduplication
                if det.label == "face":
                    obj_list = self.localized_faces
                else:
                    obj_list = self.localized_rings

                is_duplicate = False
                for saved in obj_list:
                    dist = np.sqrt((mx - saved['x'])**2 + (my - saved['y'])**2)
                    if dist < self.dedup_distance:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    obj_entry = {
                        'x': mx,
                        'y': my,
                        'z': mz,
                        'label': det.label,
                        'color': det.color
                    }
                    obj_list.append(obj_entry)
                    self.get_logger().info(
                        f"New {det.label} localized at ({mx:.2f}, {my:.2f}, {mz:.2f})"
                        + (f" color={det.color}" if det.color else "")
                        + f" | Total faces: {len(self.localized_faces)}, rings: {len(self.localized_rings)}"
                    )

                    # publish sphere marker
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = data.header.stamp
                    marker.ns = det.label
                    marker.id = self.marker_id
                    self.marker_id += 1
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.scale.x = 0.3
                    marker.scale.y = 0.3
                    marker.scale.z = 0.3
                    marker.pose.position.x = mx
                    marker.pose.position.y = my
                    marker.pose.position.z = mz

                    if det.label == "face":
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 1.0
                        marker.color.a = 1.0
                    elif det.color == "red":
                        marker.color.r = 1.0
                        marker.color.g = 0.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0
                    elif det.color == "green":
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0
                    elif det.color == "blue":
                        marker.color.r = 0.0
                        marker.color.g = 0.0
                        marker.color.b = 1.0
                        marker.color.a = 1.0
                    elif det.color == "black":
                        marker.color.r = 0.1
                        marker.color.g = 0.1
                        marker.color.b = 0.1
                        marker.color.a = 1.0
                    elif det.color == "yellow":
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker.color.a = 1.0
                    else:
                        marker.color.r = 0.5
                        marker.color.g = 0.5
                        marker.color.b = 0.5
                        marker.color.a = 1.0

                    new_markers.append(marker)

                    # text marker above the object
                    text_marker = Marker()
                    text_marker.header.frame_id = "map"
                    text_marker.header.stamp = data.header.stamp
                    text_marker.ns = det.label + "_text"
                    text_marker.id = self.marker_id
                    self.marker_id += 1
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.scale.z = 0.2
                    text_marker.pose.position.x = mx
                    text_marker.pose.position.y = my
                    text_marker.pose.position.z = mz + 0.4
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.color.a = 1.0

                    if det.label == "face":
                        text_marker.text = "Face"
                    else:
                        text_marker.text = f"Ring: {det.color}"

                    new_markers.append(text_marker)

            except Exception as e:
                self.get_logger().warn(f"Error: {e}")

        if new_markers:
            self.all_markers.extend(new_markers)
            self.marker_pub.publish(MarkerArray(markers=self.all_markers))


def main(args=None):
    rclpy.init(args=args)
    node = ObjectLocalizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

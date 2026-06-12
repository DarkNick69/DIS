#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
import time
import math
import subprocess
import numpy as np
from collections import deque


from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import MarkerArray

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


GREET_DISTANCE = 0.3


waypoints_1 = [
    [1.1141, -0.0508],
    [1.3020, -2.5941],
    [-0.1384, -2.5628],
    [-1.3909, -0.8177],
    [-2.4164, 0.6770],
    [-2.3695, 2.5082],
    [0.6679, 2.5082],
    [0.7540, 1.2483],
]


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.current_waypoint = 0
        self.faces = []
        self.faces_greeted = 0
        self.rings = []
        self.rings_greeted = 0

        # ROS2 subscribers
        self.create_subscription(DockStatus, 'dock_status', self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        #self.chat_pub = self.create_publisher(String, '/chat', 10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')
        
        # markers
        self.create_subscription(
            MarkerArray,
            '/localized_objects',
            self.markers_callback,
            10
        )

        self.get_logger().info(f"Robot commander has been initialized!")

    def publish_message(self, text: str):
        # msg = String()
        # msg.data = text
        # self.chat_pub.publish(msg)
        # self.get_logger().info(f'Published to /chat: "{text}"')
        # time.sleep(1)
        try:
            subprocess.run(
                ['espeak-ng', '-s', '150', text],
                timeout=10
            )
        except FileNotFoundError:
            self.get_logger().warn("espeak is not installed. Falling back to terminal.")
            self.get_logger().info(f"ANNOUNCEMENT: {text}")
        except Exception as e:
            self.get_logger().error(f"Announcement failed: {e}")

        
    def task1(self, waypoints):
        self.info("Starting task1.")
        for waypoint in waypoints:
            self.current_waypoint += 1
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()

            goal_pose.pose.position.x = waypoint[0]
            goal_pose.pose.position.y = waypoint[1]
            goal_pose.pose.orientation = self.YawToQuaternion(0.0)

            self.info("Moving to next waypoint")
            self.goToPose(goal_pose)
            while not self.isTaskComplete():
                time.sleep(0.5)
            self.info("Scanning surroundings")
            full_circle = 2 * math.pi
            for i in range(8):
                self.spin(full_circle/8)
                time.sleep(0.25)
            #self.info("Done scanning")
            
            self.greet_rings()
            self.greet_faces()

    def _goto_greet_pose(self, target_x, target_y):
        pose = self.getCurrentPose()
        if pose is not None:
            robot_x, robot_y, _ = pose
        else:
            robot_x, robot_y = target_x, target_y

        dx = target_x - robot_x
        dy = target_y - robot_y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            nx, ny = dx / dist, dy / dist
        else:
            nx, ny = 1.0, 0.0

        greeting_x = target_x - nx * GREET_DISTANCE
        greeting_y = target_y - ny * GREET_DISTANCE
        yaw = math.atan2(dy, dx)

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = greeting_x
        goal_pose.pose.position.y = greeting_y
        goal_pose.pose.orientation = self.YawToQuaternion(yaw)

        self.goToPose(goal_pose)
        while not self.isTaskComplete():
            time.sleep(0.5)

    def greet_rings(self):
        for ring in self.rings[self.rings_greeted:]:
            ring_x = ring[0]
            ring_y = ring[1]
            ring_color = ring[2]

            self.info(f"Greeting {ring_color} ring at {ring_x}, {ring_y}")
            self._goto_greet_pose(ring_x, ring_y)
            self.publish_message(f'Hello {ring_color} ring!')
            self.rings_greeted += 1

    def greet_faces(self):
        for face in self.faces[self.faces_greeted:]:
            face_x = face[0]
            face_y = face[1]

            self.info(f"Greeting face at {face_x}, {face_y}")
            self._goto_greet_pose(face_x, face_y)
            self.publish_message(f'Hello human face!')
            self.faces_greeted += 1
                
    def getCurrentPose(self):
        """Returns (x, y, yaw) of the robot in map frame."""
        if not self.initial_pose_received:
            self.warn('No pose received yet!')
            return None
        
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        
        # Extract yaw from quaternion
        q = self.current_pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        
        return x, y, yaw
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")

        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")

        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        #self.info(f'Spinning to angle {goal_msg.target_yaw}')

        send_goal_future = self.spin_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedbackCallback
        )

        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self.result_future)

        result = self.result_future.result().result
        #self.info("Spin completed")

        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def _vect_cmp(self, v1, v2, allowance):
        for i in range(len(v1)):
            if abs(v1[i] - v2[i]) > allowance:
                return False
        return True
        
    def markers_callback(self, msg):
        for marker in msg.markers:
            x = marker.pose.position.x
            y = marker.pose.position.y
            label = marker.ns
            
            if label == 'face':
                unique = True
                for face in self.faces:
                    if face[0] == x and face[1] == y:
                        unique = False
                        break
                if unique:
                    self.faces.append([x, y])
            elif label == 'ring':
                color = None
                rgb = [marker.color.r, marker.color.g, marker.color.b]
                if self._vect_cmp(rgb, [1.0, 0.0, 0.0], 0.05):
                    color = 'red'
                elif self._vect_cmp(rgb, [0.0, 1.0, 0.0], 0.05):
                    color = 'green'
                elif self._vect_cmp(rgb, [0.0, 0.0, 1.0], 0.05):
                    color = 'blue'
                elif self._vect_cmp(rgb, [1.0, 1.0, 0.0], 0.05):
                    color = 'yellow'
                elif self._vect_cmp(rgb, [0.1, 0.1, 0.1], 0.05):
                    color = 'black'
                else:
                    self.get_logger().info(f'Failed to handle ring color R={marker.color.r} G={marker.color.g} B={marker.color.b}')
                if color is not None:
                    unique = True
                    for ring in self.rings:
                        if (ring[0] == x and ring[1] == y):
                            unique = False
                            break
                    if unique:
                        self.rings.append([x, y, color])
                        self.get_logger().info(f'{label} at ({x:.2f}, {y:.2f})')

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()
        
    rc.task1(waypoints_1)
    time.sleep(2)

    rc.destroyNode()

if __name__=="__main__":
    main()

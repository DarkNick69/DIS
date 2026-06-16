#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from tf2_ros import TransformException
from visualization_msgs.msg import Marker


class Edge:
    def __init__(self, eid, v1, target_yaw=None):
        self.id = eid
        self.v1 = v1                  # Vertex this edge departs from
        self.explored = False
        self.target_yaw = target_yaw  # World yaw (rad) the robot must face to follow this branch

    def __repr__(self):
        return f"E{self.id}(explored={self.explored})"


class Vertex:
    def __init__(self, vid, x, y):
        self.id = vid
        self.x = x
        self.y = y
        self.edges = []          # Edge list, left-to-right as first detected
        self.parent_edge = None  # Edge we arrived on (None = root)

    def next_unexplored(self):
        return next((e for e in self.edges if not e.explored), None)

    def all_explored(self):
        return all(e.explored for e in self.edges)

    def dist(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def __repr__(self):
        return f"V{self.id}({self.x:.2f},{self.y:.2f}) edges={self.edges}"


class BlueLineFollower(Node):

    CENTERING          = 0
    FOLLOWING          = 1
    TURNING_AROUND     = 2
    RETURNING          = 3
    DONE               = 4
    SPINNING_LEFT      = 5
    VERTEX_PAUSE       = 6
    HOMING             = 7
    APPROACHING_PERSON = 8
    WAITING            = 9

    STATE_NAMES = {
        0: 'CENTERING',
        1: 'FOLLOWING',
        2: 'TURNING_AROUND',
        3: 'RETURNING',
        4: 'DONE',
        5: 'SPINNING_LEFT',
        6: 'VERTEX_PAUSE',
        7: 'HOMING',
        8: 'APPROACHING_PERSON',
        9: 'WAITING',
    }

    # Fraction of image height used for fork detection (above the junction)
    UPPER_FRACTION = 0.45

    # Camera horizontal field of view (radians). Used to convert image-x offset
    # to world angle when storing branch directions at a fork vertex.
    # Tune this to match the actual camera — typical preview cam is ~60-90°.
    CAMERA_HFOV = 1.25

    def __init__(self):
        super().__init__('blue_line')

        # --- Tunable parameters ---
        self.linear_speed  = 0.15
        self.angular_gain  = 1.2
        self.turn_speed    = 0.8
        self.min_blob_area = 300
        self.fork_radius   = 0.35  # m — proximity threshold for vertex detection

        self.blue_lower = np.array([80,  80,  50])
        self.blue_upper = np.array([130, 255, 255])

        # --- State ---
        self.state   = self.FOLLOWING
        self.enabled = False

        # --- Graph ---
        self._vertices       = []
        self._vertex_counter = 0
        self._edge_counter   = 0
        self._current_vertex = None  # Vertex we are exploring from
        self._current_edge   = None  # Edge we are currently traversing
        self._return_target  = None  # Vertex to drive back to in RETURNING

        self.bridge      = CvBridge()
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_image       = None
        self.latest_blobs       = []   # full-image blobs (dead-end detection)
        self.latest_upper_blobs = []   # top UPPER_FRACTION  — fork detection
        self.latest_lower_blobs = []   # bot (1-UPPER_FRACTION) — steering
        self.arm_set            = False

        # Vertex-pause bookkeeping (no delay, kept for pending-blob fallback)
        self._pending_fork_blobs = []
        self._pending_fork_rx    = None
        self._pending_fork_ry    = None

        self._centering_ok_count = 0
        self._centering_start    = None   # timestamp for recovery spin when no blobs
        self._turn_start_yaw     = None
        self._left_fork          = False  # True once robot has moved away from current vertex

        self._person_detected    = False
        self._last_marker_time   = None  # for debouncing people_marker
        self._person_world_x     = None  # world position of detected person
        self._person_world_y     = None
        self._home_x             = None  # initial position stored on first enabled tick
        self._home_y             = None
        self._fork_exit_time     = None  # set when SPINNING_LEFT → FOLLOWING

        self._log_tick = 0

        self.create_subscription(
            Image, '/top_camera/rgb/preview/image_raw',
            self.top_camera_callback, qos_profile_sensor_data)
        self.create_subscription(
            Marker, '/people_marker',
            self._people_marker_callback, 10)
        self.create_subscription(
            Bool, '/blue_line/enable',
            self.enable_callback, 10)

        self.cmd_vel_pub     = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.arm_pub         = self.create_publisher(String, '/arm_command', 10)
        self.debug_image_pub = self.create_publisher(Image, '/blue_line/debug_image', 10)
        self.debug_mask_pub  = self.create_publisher(Image, '/blue_line/debug_mask', 10)

        self.arm_timer = self.create_timer(1.0, self.set_arm_once)
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("=" * 50)
        self.get_logger().info("Blue Line Follower  (graph-based DFS)  initialized.")
        self.get_logger().info("Waiting for /blue_line/enable=true to start.")
        self.get_logger().info("=" * 50)

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def set_arm_once(self):
        if not self.arm_set:
            msg = String()
            msg.data = 'line_follow'
            self.arm_pub.publish(msg)
            self.arm_set = True
            self.arm_timer.cancel()
            self.get_logger().info("[ARM] Set to line_follow.")

    def enable_callback(self, msg: Bool):
        if msg.data and not self.enabled:
            self.enabled = True
            self._reset_to_centering()
            self._vertices.clear()
            self._vertex_counter = 0
            self._edge_counter   = 0
            self._current_vertex     = None
            self._current_edge       = None
            self._return_target      = None
            self._pending_fork_blobs = []
            self._centering_start    = None
            self.get_logger().info("=" * 50)
            self.get_logger().info("[ENABLE] Blue line follower STARTED.")
            self.get_logger().info("=" * 50)
        elif not msg.data:
            self.enabled = False
            self.stop_robot()
            self.get_logger().info("[ENABLE] Blue line follower STOPPED.")

    def _reset_to_centering(self):
        self.state               = self.CENTERING
        self._centering_ok_count = 0
        self._turn_start_yaw     = None
        self._left_fork          = True

    def _set_state(self, new_state):
        if new_state != self.state:
            self.get_logger().info(
                f"[STATE] {self.STATE_NAMES[self.state]} → {self.STATE_NAMES[new_state]}")
            self.state = new_state

    # ------------------------------------------------------------------ #
    # TF helpers                                                           #
    # ------------------------------------------------------------------ #

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None, None

    def get_robot_yaw(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny, cosy)
        except TransformException:
            return None

    @staticmethod
    def _angle_diff(a, b):
        """Signed shortest-path difference a − b, in (−π, π]."""
        d = a - b
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    def _near_vertex(self, rx, ry, v: Vertex) -> bool:
        if rx is None:
            return False
        return v.dist(rx, ry) <= self.fork_radius

    # ------------------------------------------------------------------ #
    # Camera callback                                                      #
    # ------------------------------------------------------------------ #

    def top_camera_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"[CAMERA] CV bridge error: {e}")
            return

        self.latest_image = cv_image

        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        blobs = []
        for i in range(1, len(stats)):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_blob_area:
                blobs.append((int(centroids[i][0]),
                              int(centroids[i][1]),
                              int(stats[i, cv2.CC_STAT_AREA])))
        blobs.sort(key=lambda b: b[0])
        self.latest_blobs = blobs

        # Upper-portion blob detection — used to spot fork branches before
        # they merge into one blob at the junction center.
        upper_h = int(cv_image.shape[0] * self.UPPER_FRACTION)
        _, _, stats_u, centroids_u = cv2.connectedComponentsWithStats(mask[:upper_h, :])
        upper_blobs = []
        for i in range(1, len(stats_u)):
            if stats_u[i, cv2.CC_STAT_AREA] >= self.min_blob_area // 2:
                upper_blobs.append((int(centroids_u[i][0]),
                                    int(centroids_u[i][1]),
                                    int(stats_u[i, cv2.CC_STAT_AREA])))
        upper_blobs.sort(key=lambda b: b[0])
        self.latest_upper_blobs = upper_blobs

        # Lower-portion blob detection — steering reference.
        # Restricting to below the fork-detection boundary keeps the centroid
        # on the current stem even when fork branches are visible above.
        _, _, stats_l, centroids_l = cv2.connectedComponentsWithStats(mask[upper_h:, :])
        lower_blobs = []
        for i in range(1, len(stats_l)):
            if stats_l[i, cv2.CC_STAT_AREA] >= self.min_blob_area // 2:
                lower_blobs.append((int(centroids_l[i][0]),
                                    int(centroids_l[i][1]) + upper_h,  # shift back to full coords
                                    int(stats_l[i, cv2.CC_STAT_AREA])))
        lower_blobs.sort(key=lambda b: b[0])
        self.latest_lower_blobs = lower_blobs

        debug_img = cv_image.copy()
        # Full-image blobs in green (dead-end detection)
        for bx, by, area in blobs:
            cv2.circle(debug_img, (bx, by), 8, (0, 255, 0), -1)
            cv2.putText(debug_img, f"{area}px", (bx + 10, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # Upper-portion blobs in red (fork branch detection)
        for bx, by, area in upper_blobs:
            cv2.circle(debug_img, (bx, by), 6, (0, 0, 255), 2)
        img_cx = cv_image.shape[1] // 2

        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))
            self.debug_mask_pub.publish(
                self.bridge.cv2_to_imgmsg(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 'bgr8'))
        except CvBridgeError as e:
            self.get_logger().error(f"[DEBUG] Bridge error: {e}")

    # ------------------------------------------------------------------ #
    # Control loop                                                         #
    # ------------------------------------------------------------------ #

    def control_loop(self):
        self._log_tick += 1

        if not self.enabled or self.state == self.DONE:
            return
        if self.latest_image is None:
            if self._log_tick % 20 == 0:
                self.get_logger().warn("[CAMERA] No image yet from top camera.")
            return

        # Store initial position on the first tick (TF may not be ready at enable time)
        if self._home_x is None:
            rx, ry = self.get_robot_pose()
            if rx is not None:
                self._home_x, self._home_y = rx, ry
                self.get_logger().info(f"[HOME] Initial position stored: ({rx:.2f}, {ry:.2f})")

        blobs  = self.latest_blobs
        img_cx = self.latest_image.shape[1] // 2

        if self._log_tick % 10 == 0:
            blob_summary = ', '.join(f"cx={b[0]} area={b[2]}" for b in blobs)
            self.get_logger().info(
                f"[TICK] state={self.STATE_NAMES[self.state]}  "
                f"blobs={len(blobs)}  [{blob_summary}]")

        if   self.state == self.CENTERING:
            self._handle_centering(blobs, img_cx)
        elif self.state == self.FOLLOWING:
            self._handle_following(blobs, img_cx)
        elif self.state == self.TURNING_AROUND:
            self._handle_turning_around()
        elif self.state == self.RETURNING:
            self._handle_returning(blobs, img_cx)
        elif self.state == self.SPINNING_LEFT:
            self._handle_spinning_left(blobs, img_cx)
        elif self.state == self.VERTEX_PAUSE:
            self._handle_vertex_pause(blobs, img_cx)
        elif self.state == self.HOMING:
            self._handle_homing()
        elif self.state == self.APPROACHING_PERSON:
            self._handle_approaching_person()
        elif self.state == self.WAITING:
            self.stop_robot()

    # ------------------------------------------------------------------ #
    # State handlers                                                       #
    # ------------------------------------------------------------------ #

    def _handle_centering(self, blobs, img_cx):
        if not blobs:
            if self._centering_start is None:
                self._centering_start = self.get_clock().now()
            elapsed = (self.get_clock().now() - self._centering_start).nanoseconds / 1e9
            if self._log_tick % 10 == 0:
                self.get_logger().warn(f"[CENTER] No line visible — {elapsed:.0f}s.")
            self._centering_ok_count = 0
            if elapsed > 5.0:
                # Slow CCW spin to scan for the line rather than waiting forever
                twist = TwistStamped()
                twist.header.stamp    = self.get_clock().now().to_msg()
                twist.header.frame_id = 'base_link'
                twist.twist.angular.z = self.turn_speed * 0.5
                self.cmd_vel_pub.publish(twist)
            else:
                self.stop_robot()
            return

        self._centering_start = None  # reset timer when line is found
        cx    = max(blobs, key=lambda b: b[2])[0]
        error = (img_cx - cx) / img_cx

        if abs(error) < 0.05:
            self._centering_ok_count += 1
            self.stop_robot()
            if self._centering_ok_count >= 5:
                self.get_logger().info("[CENTER] Centered — FOLLOWING.")
                self._left_fork          = False
                self._centering_ok_count = 0
                self._set_state(self.FOLLOWING)
        else:
            self._centering_ok_count = 0
            twist = TwistStamped()
            twist.header.stamp    = self.get_clock().now().to_msg()
            twist.header.frame_id = 'base_link'
            twist.twist.angular.z = self.angular_gain * error
            self.cmd_vel_pub.publish(twist)

    def _handle_following(self, blobs, img_cx):
        rx, ry = self.get_robot_pose()

        # Once the robot drives away from its current vertex, set the flag.
        if (self._current_vertex and not self._left_fork
                and not self._near_vertex(rx, ry, self._current_vertex)):
            self._left_fork = True

        # ── Dead end ──────────────────────────────────────────────────
        if not blobs:
            self.get_logger().info("[FOLLOW] Dead end.")
            if self._current_edge:
                self._current_edge.explored = True
                self.get_logger().info(f"  Marked E{self._current_edge.id} explored.")
            self.stop_robot()
            self._turn_start_yaw = None
            self._return_target  = self._current_vertex
            self._set_state(self.TURNING_AROUND)
            return

        # ── Fork detection via upper-portion blobs ────────────────────
        # At a Y-junction the full image sees ONE merged blob; the upper
        # portion (above the junction point) still sees the two branches
        # as separate blobs.
        upper = self.latest_upper_blobs
        fork_detected = (len(upper) >= 2 and upper[-1][0] - upper[0][0] >= 40)

        # ── No fork — follow the line using lower-portion blob ───────
        if not fork_detected:
            # Lower blobs ignore fork branches ahead and track only the
            # current stem, keeping the steering centroid on the line center.
            steer_pool = self.latest_lower_blobs if self.latest_lower_blobs else blobs
            best = max(steer_pool, key=lambda b: b[2])
            if self._log_tick % 5 == 0:
                self.get_logger().info(
                    f"[FOLLOW] cx={best[0]} error={(img_cx - best[0]) / img_cx:.3f}")
            self.follow_centroid(best[0], img_cx)
            return

        # ── Fork: guard against duplicates and TF outages ────────────
        # Two conditions both block new-vertex creation:
        #   1. Grace period (3 s after SPINNING_LEFT exits): the robot is still
        #      clearing the junction it was just aligned to — blob ordering is
        #      camera-orientation-dependent and cannot be trusted here.
        #   2. Near a known vertex: same junction seen again.
        in_grace   = (
            self._fork_exit_time is not None and
            (self.get_clock().now() - self._fork_exit_time).nanoseconds / 1e9 < 3.0
        )
        near_known = rx is not None and any(self._near_vertex(rx, ry, v) for v in self._vertices)

        if in_grace or near_known:
            # Steer by the stored world angle for the current edge — immune to
            # blob-ordering changes as the robot rotates.
            if self._current_edge and self._current_edge.target_yaw is not None:
                yaw = self.get_robot_yaw()
                if yaw is not None:
                    err = self._angle_diff(self._current_edge.target_yaw, yaw)
                    twist = TwistStamped()
                    twist.header.stamp    = self.get_clock().now().to_msg()
                    twist.header.frame_id = 'base_link'
                    twist.twist.linear.x  = self.linear_speed
                    twist.twist.angular.z = max(-self.turn_speed,
                                                min(self.turn_speed, self.angular_gain * err))
                    self.cmd_vel_pub.publish(twist)
                    return
            steer_pool = self.latest_lower_blobs if self.latest_lower_blobs else blobs
            self.follow_centroid(max(steer_pool, key=lambda b: b[2])[0], img_cx)
            return

        # ── Fork: new vertex ──────────────────────────────────────────
        if rx is None:
            # Position unknown — skip vertex creation, steer on lower blob
            steer_pool = self.latest_lower_blobs if self.latest_lower_blobs else blobs
            self.follow_centroid(max(steer_pool, key=lambda b: b[2])[0], img_cx)
            return

        self._pending_fork_blobs = upper
        self._pending_fork_rx    = rx
        self._pending_fork_ry    = ry
        self.stop_robot()
        self.get_logger().info(
            f"[FORK] Spotted at ({rx:.2f},{ry:.2f})  upper_blobs={len(upper)}")
        self._set_state(self.VERTEX_PAUSE)

    def _handle_vertex_pause(self, blobs, img_cx):
        """Create vertex at the fork immediately (no delay)."""
        self.stop_robot()

        # Use freshest upper blobs; fall back to what was seen when fork was first spotted
        upper = self.latest_upper_blobs
        if len(upper) < 2 or upper[-1][0] - upper[0][0] < 40:
            upper = self._pending_fork_blobs

        rx  = self._pending_fork_rx
        ry  = self._pending_fork_ry
        yaw = self.get_robot_yaw()

        self._vertex_counter += 1
        v = Vertex(self._vertex_counter, rx, ry)
        for blob in upper:
            self._edge_counter += 1
            bx, _, _ = blob
            ty = yaw + (img_cx - bx) / img_cx * (self.CAMERA_HFOV / 2.0) if yaw is not None else None
            v.edges.append(Edge(self._edge_counter, v, target_yaw=ty))
        v.parent_edge    = self._current_edge
        self._vertices.append(v)

        self._current_vertex = v
        self._current_edge   = v.edges[0]
        self._left_fork      = False

        yaw_strs = [
            f"{math.degrees(e.target_yaw):.1f}°" if e.target_yaw is not None else "?"
            for e in v.edges
        ]
        self.get_logger().info("─" * 50)
        self.get_logger().info(
            f"[V{v.id}] Created at ({rx:.2f},{ry:.2f})  "
            f"branches={len(upper)}  "
            f"edges=[{', '.join(f'E{e.id}' for e in v.edges)}]  "
            f"target_yaws={yaw_strs}")
        self.get_logger().info(
            f"[V{v.id}] Following E{v.edges[0].id} (leftmost cx={upper[0][0]})")
        self.get_logger().info("─" * 50)

        self._set_state(self.FOLLOWING)

    def _handle_turning_around(self):
        yaw = self.get_robot_yaw()

        if self._turn_start_yaw is None:
            if yaw is None:
                self.stop_robot()
                return
            self._turn_start_yaw = yaw
            self.get_logger().info(f"[TURN] 180° from yaw={math.degrees(yaw):.1f}°")

        if yaw is not None:
            turned = abs(self._angle_diff(yaw, self._turn_start_yaw))
            if turned >= math.pi * 0.95:
                self.stop_robot()
                self._turn_start_yaw = None
                self._left_fork      = True
                self.get_logger().info("[TURN] Done — RETURNING.")
                self._set_state(self.RETURNING)
                return
            if self._log_tick % 5 == 0:
                self.get_logger().info(f"[TURN] {math.degrees(turned):.1f}° / 180°")

        twist = TwistStamped()
        twist.header.stamp    = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        twist.twist.angular.z = self.turn_speed
        self.cmd_vel_pub.publish(twist)

    def _handle_returning(self, blobs, img_cx):
        if self._return_target is None:
            if self._person_detected:
                self.get_logger().info("[HOMING] No vertices — heading straight to start.")
                self._set_state(self.HOMING)
            else:
                self.get_logger().info("[WAIT] No vertices — waiting for person detection.")
                self._set_state(self.WAITING)
            self.stop_robot()
            return

        rx, ry = self.get_robot_pose()
        v      = self._return_target

        if self._log_tick % 5 == 0:
            dist_s = f"{v.dist(rx, ry):.2f} m" if rx is not None else "?"
            self.get_logger().info(
                f"[RETURN] → V{v.id}  dist={dist_s}  blobs={len(blobs)}")

        # ── Arrived at the target vertex ──────────────────────────────
        if self._left_fork and self._near_vertex(rx, ry, v):
            self.stop_robot()
            self._current_vertex = v
            next_e = v.next_unexplored()

            if next_e is None:
                # All edges of v explored — backtrack to parent vertex
                self.get_logger().info(f"[V{v.id}] All edges explored — backtracking.")
                if v.parent_edge is None:
                    if self._person_detected:
                        self.get_logger().info("[HOMING] Backtrack complete — heading to start.")
                        self._set_state(self.HOMING)
                    else:
                        self.get_logger().info("[WAIT] Root fully explored — waiting for person detection.")
                        self.stop_robot()
                        self._set_state(self.WAITING)
                else:
                    v.parent_edge.explored = True
                    parent_v = v.parent_edge.v1
                    self.get_logger().info(
                        f"  Marked E{v.parent_edge.id} explored. "
                        f"Returning to V{parent_v.id}.")
                    self._return_target  = parent_v
                    self._turn_start_yaw = None
                    self._set_state(self.TURNING_AROUND)
                return

            # Next unexplored edge — spin left to find it
            self._current_edge   = next_e
            self._turn_start_yaw = None
            self.get_logger().info(f"[V{v.id}] Next unexplored: E{next_e.id} — SPINNING_LEFT.")
            self._set_state(self.SPINNING_LEFT)
            return

        # ── Still en route ────────────────────────────────────────────
        steer_pool = self.latest_lower_blobs if self.latest_lower_blobs else blobs
        if steer_pool:
            self.follow_centroid(max(steer_pool, key=lambda b: b[2])[0], img_cx)
        elif blobs:
            self.follow_centroid(max(blobs, key=lambda b: b[2])[0], img_cx)
        else:
            twist = TwistStamped()
            twist.header.stamp    = self.get_clock().now().to_msg()
            twist.header.frame_id = 'base_link'
            twist.twist.linear.x  = self.linear_speed * 0.5
            self.cmd_vel_pub.publish(twist)

    def _handle_spinning_left(self, blobs, img_cx):
        yaw = self.get_robot_yaw()
        if yaw is None:
            self.stop_robot()
            return

        # ── Primary: turn to stored world angle for this edge ─────────
        # This avoids ambiguity when multiple blobs are visible at the fork.
        if self._current_edge and self._current_edge.target_yaw is not None:
            target = self._current_edge.target_yaw
            diff   = self._angle_diff(target, yaw)

            if self._log_tick % 5 == 0:
                self.get_logger().info(
                    f"[SPIN_L] yaw={math.degrees(yaw):.1f}°  "
                    f"target={math.degrees(target):.1f}°  "
                    f"diff={math.degrees(diff):.1f}°")

            if abs(diff) < math.radians(8):
                self.stop_robot()
                self._turn_start_yaw = None
                self._left_fork      = False
                self._fork_exit_time = self.get_clock().now()
                self.get_logger().info(
                    f"[SPIN_L] Aligned to E{self._current_edge.id} "
                    f"yaw={math.degrees(yaw):.1f}° — FOLLOWING.")
                self._set_state(self.FOLLOWING)
                return

            twist = TwistStamped()
            twist.header.stamp    = self.get_clock().now().to_msg()
            twist.header.frame_id = 'base_link'
            twist.twist.angular.z = math.copysign(self.turn_speed, diff)
            self.cmd_vel_pub.publish(twist)
            return

        # ── Fallback: visual blob centering (no stored target_yaw) ────
        if self._turn_start_yaw is None:
            self._turn_start_yaw = yaw
            self.get_logger().info(f"[SPIN_L] Start yaw={math.degrees(yaw):.1f}° (visual fallback)")

        turned = abs(self._angle_diff(yaw, self._turn_start_yaw))

        if self._log_tick % 5 == 0:
            self.get_logger().info(f"[SPIN_L] Turned {math.degrees(turned):.1f}°")

        if turned >= math.radians(60) and blobs:
            best = min(blobs, key=lambda b: abs(b[0] - img_cx))
            if abs(best[0] - img_cx) / img_cx < 0.10:
                self.stop_robot()
                self._turn_start_yaw = None
                self._left_fork      = False
                self._fork_exit_time = self.get_clock().now()
                self.get_logger().info(
                    f"[SPIN_L] Locked cx={best[0]} → E{self._current_edge.id}  FOLLOWING.")
                self._set_state(self.FOLLOWING)
                return

        if turned >= 2 * math.pi:
            self.stop_robot()
            self._turn_start_yaw     = None
            self._centering_ok_count = 0
            self.get_logger().warn("[SPIN_L] 360° no lock — CENTERING.")
            self._set_state(self.CENTERING)
            return

        twist = TwistStamped()
        twist.header.stamp    = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        twist.twist.angular.z = self.turn_speed  # CCW = left
        self.cmd_vel_pub.publish(twist)

    # ------------------------------------------------------------------ #
    # OAK-D person detection                                              #
    # ------------------------------------------------------------------ #

    def _people_marker_callback(self, msg: Marker):
        """Triggered by detect_people.py whenever a person is spotted.
        Requires two markers within 2 s to guard against single-frame false positives.
        On confirmation: compute person world position and enter APPROACHING_PERSON."""
        if self._person_detected or self.state in (
                self.APPROACHING_PERSON, self.DONE) or not self.enabled:
            return

        now = self.get_clock().now()
        if self._last_marker_time is not None:
            elapsed = (now - self._last_marker_time).nanoseconds / 1e9
            if elapsed < 2.0:
                # Person position in base_link frame (from detect_people.py pointcloud)
                px = msg.pose.position.x
                py = msg.pose.position.y
                if math.isnan(px) or math.isnan(py):
                    return
                # Convert to world frame using current robot pose
                rx, ry = self.get_robot_pose()
                yaw    = self.get_robot_yaw()
                if rx is None or yaw is None:
                    return
                self._person_world_x = rx + px * math.cos(yaw) - py * math.sin(yaw)
                self._person_world_y = ry + px * math.sin(yaw) + py * math.cos(yaw)
                self._person_detected = True
                print("Person detected!")
                self.get_logger().info(
                    f"Person at world ({self._person_world_x:.2f}, {self._person_world_y:.2f})"
                    f" — approaching.")
                self._set_state(self.APPROACHING_PERSON)
                return

        self._last_marker_time = now

    def _handle_approaching_person(self):
        """Follow the blue line toward the person; stop 1.0 m away."""
        rx, ry = self.get_robot_pose()
        if rx is None or self._person_world_x is None:
            self.stop_robot()
            return

        dist = math.sqrt((rx - self._person_world_x) ** 2 + (ry - self._person_world_y) ** 2)

        if self._log_tick % 10 == 0:
            self.get_logger().info(f"[APPROACH] dist={dist:.2f} m to person")

        if dist <= 1.0:
            self.stop_robot()
            print("In position. Done.")
            self.get_logger().info("[APPROACH] Reached person. Terminating.")
            self._set_state(self.DONE)
            return

        # Stay on the blue line — steer toward the largest lower blob
        blobs = self.latest_lower_blobs or self.latest_blobs
        if not blobs:
            self.stop_robot()
            return
        img_cx = self.latest_image.shape[1] // 2
        bx = max(blobs, key=lambda b: b[2])[0]
        self.follow_centroid(bx, img_cx)

    def _handle_homing(self):
        """Drive toward the stored initial position (unused in current flow)."""
        rx, ry = self.get_robot_pose()
        if rx is None or self._home_x is None:
            self.stop_robot()
            return
        dist = math.sqrt((rx - self._home_x) ** 2 + (ry - self._home_y) ** 2)
        if dist < 0.5:
            self.stop_robot()
            self._set_state(self.DONE)
            return
        yaw = self.get_robot_yaw()
        if yaw is None:
            self.stop_robot()
            return
        bearing = math.atan2(self._home_y - ry, self._home_x - rx)
        err     = self._angle_diff(bearing, yaw)
        twist = TwistStamped()
        twist.header.stamp    = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        twist.twist.angular.z = self.angular_gain * err
        if abs(err) < math.radians(20):
            twist.twist.linear.x = self.linear_speed
        self.cmd_vel_pub.publish(twist)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def follow_centroid(self, cx: int, img_cx: int):
        twist = TwistStamped()
        twist.header.stamp    = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        error = (img_cx - cx) / img_cx
        twist.twist.linear.x  = self.linear_speed
        twist.twist.angular.z = self.angular_gain * error
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        twist = TwistStamped()
        twist.header.stamp    = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(twist)


def main():
    print("Version 11")
    rclpy.init(args=None)
    node = BlueLineFollower()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.state == node.DONE:
                node.get_logger().info("Terminating — mission complete.")
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

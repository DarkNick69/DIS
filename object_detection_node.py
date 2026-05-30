#!/usr/bin/env python3
"""Ring + face detection node — v13h.

Follows the dis_tutorial5 algorithm exactly:
  1. BGR → grayscale
  2. Adaptive threshold → binary image
  3. Extract contours
  4. Fit ellipses to contours ≥ MIN_CNT_PTS points; filter by eccentricity
  5. For every pair of ellipses whose centres are within CENTER_THR pixels,
     and where one axis of one ellipse is strictly larger on both axes,
     declare a ring candidate.

3D confirmation via depth hole check:
  A real 3D ring hanging in the air has a hole — the depth seen through the
  hole is the background (farther away) vs the ring band (closer).
  2D painted rings have no depth difference between hole and band.
  Candidates where hole_depth - band_depth < HOLE_DIFF_MM are rejected.
  If there is no valid depth data the check is skipped (candidate accepted).
"""

import os
try:
    from ament_index_python.packages import get_package_share_directory as _get_share
    _PKG_SHARE = _get_share('dis_tutorial3')
except Exception:
    _PKG_SHARE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', '..', 'share', 'dis_tutorial3')

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from dis_tutorial4_yellow_interfaces.msg import Detection, DetectionArray
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import numpy as np

# ── Tutorial parameters ───────────────────────────────────────────────────────
ECC_THR     = 100    # maximum ellipse axis length (pixels) — ignore huge ellipses
RATIO_THR   = 1.5    # maximum axis ratio — ignore very flat ellipses
CENTER_THR  = 10     # max centre distance between inner/outer ellipse (pixels)
MIN_CNT_PTS = 20     # minimum contour points to attempt ellipse fitting

# ── Depth hole check ──────────────────────────────────────────────────────────
# Real 3D ring: background seen through hole is farther than the ring band.
# 2D painted ring: hole and band are on the same flat surface → ~0 mm diff.
HOLE_DIFF_MM = 30    # minimum hole-band depth difference to accept as 3D ring

# ── Other constants ───────────────────────────────────────────────────────────
DEPTH_TOPIC  = '/oakd/rgb/preview/depth'
CLOUD_TOPIC  = '/oakd/rgb/preview/depth/points'
MARKER_FRAME = '/base_link'
EDGE_MARGIN  = 5
DEDUP_PX     = 20


class ObjectDetectionNode(Node):

    def __init__(self):
        super().__init__('object_detection_node')

        self.bridge       = CvBridge()
        self.model        = YOLO("yolov8n.pt")
        self.is_turning   = False
        self.latest_depth = None
        self.latest_cloud = None

        self.create_subscription(Odometry, '/odom',
            self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw',
            self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Image, DEPTH_TOPIC,
            self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, CLOUD_TOPIC,
            self.cloud_callback, qos_profile_sensor_data)

        self.detection_pub   = self.create_publisher(DetectionArray, '/detections', 10)
        self.ring_marker_pub = self.create_publisher(MarkerArray, '/ring_markers', 10)

        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window",   cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"v13h started — tutorial algorithm + depth hole check "
            f"(HOLE_DIFF_MM={HOLE_DIFF_MM})")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        self.is_turning = abs(msg.twist.twist.angular.z) > 0.1

    def depth_callback(self, msg):
        try:
            enc = msg.encoding.lower()
            if enc == '16uc1':
                raw = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32)
            else:
                raw = self.bridge.imgmsg_to_cv2(msg, '32FC1') * 1000.0
            raw[raw <= 0] = np.nan
            self.latest_depth = raw
        except CvBridgeError as e:
            self.get_logger().error(f"depth: {e}")

    def cloud_callback(self, msg):
        self.latest_cloud = msg

    # ── colour detection ──────────────────────────────────────────────────────

    _HSV_RANGES = {
        'red':    [((0,   80,  60),  (10,  255, 255)),
                   ((160, 80,  60),  (180, 255, 255))],
        'green':  [((35,  50,  50),  (85,  255, 255))],
        'blue':   [((100, 50,  50),  (130, 255, 255))],
        'yellow': [((20,  80,  80),  (35,  255, 255))],
    }

    def _ring_color(self, hsv, le, se):
        h, w = hsv.shape[:2]
        om = np.zeros((h, w), np.uint8); cv2.ellipse(om, le, 255, -1)
        im = np.zeros((h, w), np.uint8); cv2.ellipse(im, se, 255, -1)
        band = cv2.bitwise_and(om, cv2.bitwise_not(im))
        band_px = int(np.sum(band > 0))
        if band_px == 0:
            return 'unknown'
        best, best_n = 'unknown', 0
        for name, ranges in self._HSV_RANGES.items():
            m = np.zeros((h, w), np.uint8)
            for lo, hi in ranges:
                m = cv2.bitwise_or(m, cv2.inRange(hsv,
                                                   np.array(lo), np.array(hi)))
            n = int(np.sum(cv2.bitwise_and(band, m) > 0))
            if n > best_n:
                best_n, best = n, name
        return best if best_n / band_px >= 0.20 else 'unknown'

    # ── depth hole check ──────────────────────────────────────────────────────

    def _depth_hole_check(self, depth_mm, le, se):
        """Return True/False/None.

        True  → hole is deeper than band by HOLE_DIFF_MM → real 3D ring
        False → hole and band are at similar depth → flat / 2D surface
        None  → insufficient depth data → skip check, do not reject
        """
        if depth_mm is None:
            return None
        h, w = depth_mm.shape[:2]

        outer_mask = np.zeros((h, w), np.uint8)
        inner_mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(outer_mask, le, 255, -1)
        cv2.ellipse(inner_mask, se, 255, -1)
        band_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

        def valid(mask):
            v = depth_mm[mask > 0]
            return v[np.isfinite(v) & (v > 0)]

        band_v = valid(band_mask)
        hole_v = valid(inner_mask)

        if len(band_v) < 5:
            return None  # can't measure band depth

        if len(hole_v) < 5:
            # No valid depth through the hole → open air → 3D ring
            return True

        diff = float(np.median(hole_v)) - float(np.median(band_v))
        self.get_logger().info(
            f"    depth check: band={np.median(band_v):.0f}mm "
            f"hole={np.median(hole_v):.0f}mm diff={diff:.0f}mm "
            f"({'HOLE' if diff > HOLE_DIFF_MM else 'FLAT'})")
        return diff > HOLE_DIFF_MM

    # ── 3-D position ──────────────────────────────────────────────────────────

    def _get_3d(self, cx, cy, outer_r):
        if self.latest_cloud is None:
            return None
        h = self.latest_cloud.height; w = self.latest_cloud.width
        try:
            pts = pc2.read_points_numpy(
                self.latest_cloud, field_names=("x", "y", "z"))
            pts = pts.reshape((h, w, 3))
            for r in [outer_r // 2, outer_r // 3, 6, 3]:
                for dy, dx in [(0, r), (r, 0), (0, -r), (-r, 0)]:
                    ry, rx = cy + dy, cx + dx
                    if 0 <= ry < h and 0 <= rx < w:
                        xyz = pts[ry, rx]
                        if np.all(np.isfinite(xyz)) and not np.allclose(xyz, 0):
                            return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        except Exception:
            pass
        return None

    # ── markers ───────────────────────────────────────────────────────────────

    _RCOLORS = {
        'red':    (1.0, 0.0, 0.0), 'green':  (0.0, 0.9, 0.0),
        'blue':   (0.0, 0.4, 1.0), 'yellow': (1.0, 0.9, 0.0),
    }

    def _make_markers(self, mid, pos, color, stamp):
        r, g, b = self._RCOLORS.get(color, (0.7, 0.7, 0.7))
        s = Marker(); s.header.frame_id = MARKER_FRAME; s.header.stamp = stamp
        s.ns = 'rings'; s.id = mid * 2; s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.pose.position.x = pos[0]; s.pose.position.y = pos[1]
        s.pose.position.z = pos[2]; s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.20
        s.color.r = r; s.color.g = g; s.color.b = b; s.color.a = 1.0
        s.lifetime.sec = 3
        t = Marker(); t.header.frame_id = MARKER_FRAME; t.header.stamp = stamp
        t.ns = 'ring_labels'; t.id = mid * 2 + 1
        t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
        t.pose.position.x = pos[0]; t.pose.position.y = pos[1]
        t.pose.position.z = pos[2] + 0.25; t.pose.orientation.w = 1.0
        t.scale.z = 0.12; t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = f"Ring ({color})"; t.lifetime.sec = 3
        return [s, t]

    # ── main image callback ───────────────────────────────────────────────────

    def image_callback(self, msg):
        if self.is_turning:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e)); return

        depth_mm = self.latest_depth
        if depth_mm is not None and depth_mm.shape[:2] != cv_image.shape[:2]:
            depth_mm = cv2.resize(depth_mm,
                (cv_image.shape[1], cv_image.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        h_img, w_img = cv_image.shape[:2]
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        vis = cv_image.copy()

        det_array = DetectionArray(); det_array.header = msg.header

        # ── Step 1+2: YOLO face detection ─────────────────────────────────────
        face_bboxes = []
        for r in self.model.predict(cv_image, imgsz=(256, 320),
                                    show=False, verbose=False,
                                    classes=[0], device=''):
            for box in r.boxes:
                b = box.xyxy[0]
                if b.nelement() == 0: continue
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                face_bboxes.append((x1, y1, x2, y2))
                d = Detection(); d.label = "face"
                d.cx = (x1 + x2) // 2; d.cy = (y1 + y2) // 2
                d.bbox_x1 = x1; d.bbox_y1 = y1; d.bbox_x2 = x2; d.bbox_y2 = y2
                d.color = ""; det_array.detections.append(d)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # ── Step 2: grayscale ─────────────────────────────────────────────────
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # ── Step 3: adaptive threshold → binary ───────────────────────────────
        thresh = cv2.adaptiveThreshold(gray, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)

        # ── Step 4: contours → ellipses (filter by size + eccentricity) ───────
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        elps = []
        for cnt in contours:
            if cnt.shape[0] < MIN_CNT_PTS:
                continue
            try:
                ell = cv2.fitEllipse(cnt)
            except Exception:
                continue
            e = ell[1]                               # (minor_axis, major_axis)
            if e[0] > ECC_THR or e[1] > ECC_THR:   # too large
                continue
            ratio = e[1] / e[0] if e[0] > 0 else 999
            if ratio > RATIO_THR:                    # too flat
                continue
            elps.append(ell)

        # ── Step 5: find concentric pairs → rings ─────────────────────────────
        ring_markers = []; added = set()

        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1, e2 = elps[n], elps[m]

                # centres must be close
                dist = np.hypot(e1[0][0] - e2[0][0], e1[0][1] - e2[0][1])
                if dist >= CENTER_THR:
                    continue

                # one ellipse must be strictly larger on both axes
                if e1[1][0] >= e2[1][0] and e1[1][1] >= e2[1][1]:
                    le, se = e1, e2
                elif e2[1][0] >= e1[1][0] and e2[1][1] >= e1[1][1]:
                    le, se = e2, e1
                else:
                    continue

                cx, cy = int(le[0][0]), int(le[0][1])

                # discard rings too close to image edges
                if (cx < EDGE_MARGIN or cx > w_img - EDGE_MARGIN or
                        cy < EDGE_MARGIN or cy > h_img - EDGE_MARGIN):
                    continue

                # deduplication
                key = (cx // DEDUP_PX, cy // DEDUP_PX)
                if key in added:
                    continue

                # skip candidates inside a detected face bbox
                PAD = 15
                if any(fx1 - PAD <= cx <= fx2 + PAD and fy1 - PAD <= cy <= fy2 + PAD
                       for fx1, fy1, fx2, fy2 in face_bboxes):
                    continue

                # colour check — reject if colour is unknown
                color = self._ring_color(hsv, le, se)
                if color == 'unknown':
                    continue

                outer_r = int(max(le[1]) / 2)

                # depth hole check — reject 2D flat surfaces
                hole = self._depth_hole_check(depth_mm, le, se)
                if hole is False:
                    self.get_logger().info(
                        f"  {color} ({cx},{cy}) REJECTED — flat/2D surface")
                    continue

                added.add(key)

                self.get_logger().info(
                    f"  Ring: {color} at ({cx},{cy}) "
                    f"depth={'3D confirmed' if hole is True else 'no depth data'}")

                cv2.ellipse(vis, le, (0, 255, 0), 2)
                cv2.ellipse(vis, se, (0, 255, 0), 2)
                cv2.putText(vis, f"Ring:{color}",
                            (cx - 20, cy - outer_r - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                det = Detection(); det.label = "ring"
                det.cx = cx; det.cy = cy
                det.bbox_x1 = cx - outer_r; det.bbox_y1 = cy - outer_r
                det.bbox_x2 = cx + outer_r; det.bbox_y2 = cy + outer_r
                det.color = color; det_array.detections.append(det)

                pos = self._get_3d(cx, cy, outer_r)
                if pos:
                    mid = len(ring_markers) // 2
                    ring_markers += self._make_markers(mid, pos, color,
                                                       msg.header.stamp)

        if det_array.detections:
            self.detection_pub.publish(det_array)
        if ring_markers:
            self.ring_marker_pub.publish(MarkerArray(markers=ring_markers))

        # ── visualisation ─────────────────────────────────────────────────────
        cv2.imshow("Detected rings", vis)

        if depth_mm is not None:
            d = np.nan_to_num(depth_mm, nan=0.0); valid = d[d > 0]
            d_max = float(np.percentile(valid, 95)) if valid.size > 0 else 5000.0
            d_vis = np.clip(d / d_max * 255, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
            for det in det_array.detections:
                if det.label == "ring":
                    r = (det.bbox_x2 - det.bbox_x1) // 2
                    cv2.circle(depth_color, (det.cx, det.cy), r, (0, 255, 0), 2)
            cv2.imshow("Depth window", depth_color)

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    print("detection node version 13h")
    main()

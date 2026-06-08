#!/usr/bin/env python3
"""Ring + face detection node — v18.

Ring detection mirrors detect_rings.py exactly.
All debug images are tiled into a single "Ring Detection" window:

  Row 0: Binary Image | Detected contours | Detected rings | Depth
  Row 1: Raw Red      | Morph Red         | Raw Green      | Morph Green
  Row 2: Raw Blue     | Morph Blue        | (empty)        | (empty)
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

# ── Ellipse / ring parameters (same as detect_rings.py) ──────────────────────
ECC_THR    = 100   # maximum single ellipse axis length (pixels)
RATIO_THR  = 1.5   # maximum axis ratio
CENTER_THR = 10    # maximum centre-to-centre distance for a concentric pair

# ── Colour-purity thresholds ──────────────────────────────────────────────────
PURITY_THR        = 0.50   # channel / (R+G+B) must exceed this
MORPH_KERNEL_SIZE = 3      # structuring-element side for close→open

# ── Grid tile dimensions (ring-detection mosaic only) ────────────────────────
TILE_H = 150
TILE_W = 200
GRID_COLS = 2   # 2×2 layout: Binary/Contours top, Rings/Depth bottom

# ── HSV colour check for ring labelling ───────────────────────────────────────
COLOR_THR = 0.10   # fraction of band pixels that must match a named colour

# ── Other constants ───────────────────────────────────────────────────────────
DEPTH_TOPIC  = '/oakd/rgb/preview/depth'
CLOUD_TOPIC  = '/oakd/rgb/preview/depth/points'
MARKER_FRAME = '/base_link'
EDGE_MARGIN  = 5
DEDUP_PX     = 20

_BGR_RING = {
    'red':    (0,   0,   255),
    'green':  (0,   210, 0),
    'blue':   (255, 80,  0),
    'yellow': (0,   220, 255),
}


class ObjectDetectionNode(Node):

    def __init__(self):
        super().__init__('object_detection_node')

        self.bridge       = CvBridge()
        self.model        = YOLO("yolov8n.pt")
        self.is_turning   = False
        self.latest_cloud = None
        self.latest_depth_vis = None   # stored by depth_callback, shown in grid

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

        cv2.namedWindow("Ring Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Red mask",       cv2.WINDOW_NORMAL)
        cv2.namedWindow("Green mask",     cv2.WINDOW_NORMAL)
        cv2.namedWindow("Blue mask",      cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"v18 started  ECC_THR={ECC_THR} RATIO_THR={RATIO_THR} "
            f"CENTER_THR={CENTER_THR}  PURITY_THR={PURITY_THR}")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        self.is_turning = abs(msg.twist.twist.angular.z) > 0.1

    def cloud_callback(self, msg):
        self.latest_cloud = msg

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"depth: {e}"); return

        depth[depth == np.inf] = 0
        img = depth / 65536.0 * 255
        if img.max() > 0:
            img = img / img.max() * 255
        self.latest_depth_vis = img.astype(np.uint8)

    # ── grid helper ───────────────────────────────────────────────────────────

    def _make_grid(self, panels):
        """Tile (label, img) panels into a GRID_COLS-wide mosaic.

        img may be grayscale, BGR, or None (black tile).
        A dark header strip carries the label text.
        """
        tiles = []
        for label, img in panels:
            if img is None:
                tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
            else:
                if len(img.shape) == 2:
                    tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    tile = img.copy()
                tile = cv2.resize(tile, (TILE_W, TILE_H))
            cv2.rectangle(tile, (0, 0), (TILE_W, 18), (0, 0, 0), -1)
            cv2.putText(tile, label, (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)

        rows = (len(tiles) + GRID_COLS - 1) // GRID_COLS
        while len(tiles) < rows * GRID_COLS:
            tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))

        return np.vstack([
            np.hstack(tiles[r * GRID_COLS:(r + 1) * GRID_COLS])
            for r in range(rows)
        ])

    # ── side-by-side helper (full resolution, no downscale) ──────────────────

    @staticmethod
    def _side_by_side(img_a, label_a, img_b, label_b):
        """Hstack two images at their native resolution with a label strip each."""
        def _label(img, text):
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 18), (0, 0, 0), -1)
            cv2.putText(bgr, text, (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255), 1, cv2.LINE_AA)
            return bgr
        return np.hstack([_label(img_a, label_a), _label(img_b, label_b)])

    # ── colour-purity masks ────────────────────────────────────────────────────

    def _color_masks(self, cv_image):
        """Return (raw_tuple, morph_tuple) each being (red, green, blue) uint8 masks.

        raw_tuple  — purity threshold only, no morphology
        morph_tuple — close then open applied (MORPH_KERNEL_SIZE × MORPH_KERNEL_SIZE)
        """
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        total = r + g + b + 1e-6

        red_raw   = ((r / total) > PURITY_THR).astype(np.uint8) * 255
        green_raw = ((g / total) > PURITY_THR).astype(np.uint8) * 255
        blue_raw  = ((b / total) > PURITY_THR).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
        def morph(m):
            return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        return (
            (red_raw,   green_raw,   blue_raw),
            (morph(red_raw), morph(green_raw), morph(blue_raw)),
        )

    # ── HSV colour labelling ───────────────────────────────────────────────────

    _HSV_RANGES = {
        'red':    [((0,   60,  60),  (10,  255, 255)),
                   ((160, 60,  60),  (180, 255, 255))],
        'green':  [((35,  40,  40),  (85,  255, 255))],
        'blue':   [((100, 40,  40),  (130, 255, 255))],
        'yellow': [((18,  60,  60),  (38,  255, 255))],
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
        return best if best_n / band_px >= COLOR_THR else 'unknown'

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

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        det_array = DetectionArray(); det_array.header = msg.header

        # ── colour-purity masks (raw and morphed) ─────────────────────────────
        (red_raw, green_raw, blue_raw), \
        (red_mask, green_mask, blue_mask) = self._color_masks(cv_image)

        # ── YOLO face detection ───────────────────────────────────────────────
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

        # ── ring detection — mirrors detect_rings.py exactly ──────────────────
        gray   = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255,
                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        gray_contour = gray.copy()
        cv2.drawContours(gray_contour, contours, -1, (255, 0, 0), 1)

        # fit ellipses
        elps = []
        for cnt in contours:
            if cnt.shape[0] < 20:
                continue
            try:
                ell = cv2.fitEllipse(cnt)
            except Exception:
                continue
            ecc1, ecc2 = ell[1][0], ell[1][1]
            ratio = ecc1 / ecc2 if ecc1 > ecc2 else ecc2 / ecc1
            if ratio <= RATIO_THR and ecc1 < ECC_THR and ecc2 < ECC_THR:
                elps.append(ell)

        vis = cv_image.copy()
        h_img, w_img = cv_image.shape[:2]
        ring_markers = []

        for e in elps:
            cv2.ellipse(vis, e, (255, 255, 0), 1)
            cv2.circle(vis, (int(e[0][0]), int(e[0][1])), 1, (255, 255, 0), -1)

        added = set()
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1, e2 = elps[n], elps[m]
                dist = np.hypot(e1[0][0] - e2[0][0], e1[0][1] - e2[0][1])
                if dist >= CENTER_THR:
                    continue
                if e1[1][0] >= e2[1][0] and e1[1][1] >= e2[1][1]:
                    le, se = e1, e2
                elif e2[1][0] >= e1[1][0] and e2[1][1] >= e1[1][1]:
                    le, se = e2, e1
                else:
                    continue

                cx, cy = int(le[0][0]), int(le[0][1])
                if (cx < EDGE_MARGIN or cx > w_img - EDGE_MARGIN or
                        cy < EDGE_MARGIN or cy > h_img - EDGE_MARGIN):
                    continue

                PAD = 15
                if any(fx1-PAD <= cx <= fx2+PAD and fy1-PAD <= cy <= fy2+PAD
                       for fx1, fy1, fx2, fy2 in face_bboxes):
                    continue

                key = (cx // DEDUP_PX, cy // DEDUP_PX)
                if key in added:
                    continue
                added.add(key)

                cv2.ellipse(vis, le, (0, 255, 0), 2)
                cv2.ellipse(vis, se, (0, 255, 0), 2)

                color = self._ring_color(hsv, le, se)
                outer_r = int(max(le[1]) / 2)

                self.get_logger().info(
                    f"  Ring candidate ({cx},{cy}) color={color}")

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

        self.get_logger().info(
            f"elps={len(elps)} candidates={len(added)}")

        if det_array.detections:
            self.detection_pub.publish(det_array)
        if ring_markers:
            self.ring_marker_pub.publish(MarkerArray(markers=ring_markers))

        # ── ring-detection mosaic (Binary / Contours / Rings / Depth, 2×2) ──────
        panels = [
            ("Binary Image", thresh),
            ("Contours",     gray_contour),
            ("Rings",        vis),
            ("Depth",        self.latest_depth_vis),
        ]
        cv2.imshow("Ring Detection", self._make_grid(panels))

        # ── colour-mask windows (raw | morph, full resolution) ────────────────
        cv2.imshow("Red mask",   self._side_by_side(red_raw,   "Raw", red_mask,   "Morph"))
        cv2.imshow("Green mask", self._side_by_side(green_raw, "Raw", green_mask, "Morph"))
        cv2.imshow("Blue mask",  self._side_by_side(blue_raw,  "Raw", blue_mask,  "Morph"))
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    print("detection node version 18")
    main()

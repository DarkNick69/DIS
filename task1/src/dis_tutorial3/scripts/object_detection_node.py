#!/usr/bin/python3
"""Ring detection using HSV color-based segmentation with single dashboard display.

Single window shows grid of:
  [Raw RGB]           [HSV Color Mask]      [Morphological Ops]
  [Detected Contours] [Ring Detection]      [Depth Image]
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dis_tutorial4_yellow_interfaces.msg import Detection, DetectionArray
from ultralytics import YOLO

# ── Ring color ranges (HSV) ───────────────────────────────────────────────────
COLOR_RANGES = {
    'red': [
        ((0,   80,   80),   (10,  255, 255)),
        ((160, 80,   80),   (180, 255, 255)),
    ],
    'green': [
        ((40,  80,   80),   (90,  255, 255)),
    ],
    'blue': [
        ((100, 80,   80),   (140, 255, 255)),
    ],
    'yellow': [
        ((18,  80,   80),   (35,  255, 255)),
    ],
}

# Morphological operations
MORPH_KERNEL_SIZE = 3
MORPH_CLOSE_ITER = 3
MORPH_OPEN_ITER = 1

# Contour filtering
MIN_CONTOUR_POINTS = 5
MIN_CONTOUR_AREA = 15

# Ellipse fitting
ECC_THR = 100
RATIO_THR = 1.9
CENTER_THR = 12
ARC_COMPLETENESS_THR = 0.60

# Ring validation
MIN_RING_RATIO = 1.15
MAX_RING_RATIO = 3.5

# Grid display
TILE_H = 180
TILE_W = 240
GRID_COLS = 3

# Annotation
_C_RING = (0, 255, 255)


class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')
        
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        
        self.latest_depth_raw = None
        self.model = YOLO("yolov8n.pt")

        self.detection_pub = self.create_publisher(DetectionArray, '/detections', 10)

        cv2.namedWindow("Ring Detection Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ring Detection Dashboard", 1200, 800)
        
        self.get_logger().info(
            f"Ring detector started (HSV-based with dashboard): "
            f"ECC_THR={ECC_THR}, RATIO_THR={RATIO_THR}, "
            f"MORPH={MORPH_KERNEL_SIZE}x{MORPH_KERNEL_SIZE}")

    # ── Depth callback ────────────────────────────────────────────────────────

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"depth: {e}")
            return
        
        depth[depth == np.inf] = 0
        self.latest_depth_raw = depth

    # ── Create color mask ─────────────────────────────────────────────────────

    @staticmethod
    def _create_color_mask(hsv_image, color_ranges):
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for (h_min, s_min, v_min), (h_max, s_max, v_max) in color_ranges:
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask |= cv2.inRange(hsv_image, lower, upper)
        return mask

    @staticmethod
    def _create_combined_color_mask(hsv_image, color_dict):
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for color_name, ranges in color_dict.items():
            color_mask = RingDetector._create_color_mask(hsv_image, ranges)
            combined_mask |= color_mask
        return combined_mask

    # ── Arc-completeness ──────────────────────────────────────────────────────

    @staticmethod
    def _arc_completeness(cnt, ellipse, n_bins=36):
        pts = cnt[:, 0, :].astype(np.float32)
        cx, cy = ellipse[0]
        a, b = ellipse[1][0] / 2.0, ellipse[1][1] / 2.0
        
        if a < 1 or b < 1:
            return 0.0
        
        rad = np.deg2rad(ellipse[2])
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy
        
        thetas = np.arctan2(yr / b, xr / a)
        bin_idx = (np.degrees(thetas) % 360 * n_bins / 360).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        filled = np.zeros(n_bins, dtype=bool)
        filled[bin_idx] = True
        
        return float(filled.sum()) / n_bins

    # ── Try fit ellipse ───────────────────────────────────────────────────────

    @staticmethod
    def _try_fit_ellipse(cnt):
        """Returns (ell, completeness) on success, or a str rejection reason on failure."""
        if cnt.shape[0] < MIN_CONTOUR_POINTS:
            return "<pts"

        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            return "area"

        try:
            ell = cv2.fitEllipseAMS(cnt)
        except Exception:
            return "fit_fail"

        ecc1, ecc2 = ell[1][0], ell[1][1]

        if ecc1 < 2 or ecc2 < 2:
            return "degen"

        ratio = ecc1 / ecc2 if ecc1 > ecc2 else ecc2 / ecc1

        if ratio > RATIO_THR:
            return f"ratio({ratio:.2f}>{RATIO_THR})"
        if ecc1 >= ECC_THR or ecc2 >= ECC_THR:
            return f"big({max(ecc1,ecc2):.0f}>={ECC_THR})"

        completeness = RingDetector._arc_completeness(cnt, ell)
        return (ell, completeness)

    # ── Identify color ────────────────────────────────────────────────────────

    @staticmethod
    def _identify_contour_color(hsv_image, contour):
        h, w = hsv_image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        best_color = 'unknown'
        best_count = 0
        
        for color_name, ranges in COLOR_RANGES.items():
            color_mask = RingDetector._create_color_mask(hsv_image, ranges)
            count = np.sum(cv2.bitwise_and(mask, color_mask) > 0)
            if count > best_count:
                best_count, best_color = count, color_name
        
        return best_color

    # ── Extract ellipses ──────────────────────────────────────────────────────

    def _extract_ellipses(self, contours):
        """Returns (ellipses, rejections) where rejections maps reason→count."""
        ellipses = []
        rejections = {}
        for cnt in contours:
            result = self._try_fit_ellipse(cnt)
            if isinstance(result, str):
                rejections[result] = rejections.get(result, 0) + 1
            else:
                ell, completeness = result
                ellipses.append({
                    'ell': ell,
                    'cnt': cnt,
                    'completeness': completeness,
                })
        return ellipses, rejections

    # ── Depth hole check ─────────────────────────────────────────────────────

    @staticmethod
    def _check_ring_hole_depth(outer_ell, inner_ell, depth_image, n_samples=16):
        """Return True if depth confirms a hole (>10 cm gap) or depth is unavailable.

        Samples points on the ring band (midway between inner and outer ellipse)
        and near the centre (40 % of inner radius).  If centre samples are all
        invalid the hole is open to infinity — also accepted.
        """
        if depth_image is None:
            return True

        h, w = depth_image.shape[:2]
        cx, cy      = outer_ell[0]
        outer_a     = outer_ell[1][0] / 2.0
        outer_b     = outer_ell[1][1] / 2.0
        inner_a     = inner_ell[1][0] / 2.0
        inner_b     = inner_ell[1][1] / 2.0
        angle_rad   = np.deg2rad(outer_ell[2])
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        ring_depths   = []
        center_depths = []

        for i in range(n_samples):
            theta        = 2.0 * np.pi * i / n_samples
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Ring band: midpoint between inner and outer radii
            xr = (outer_a + inner_a) / 2.0 * cos_t
            yr = (outer_b + inner_b) / 2.0 * sin_t
            px = int(round(cx + cos_a * xr - sin_a * yr))
            py = int(round(cy + sin_a * xr + cos_a * yr))
            if 0 <= px < w and 0 <= py < h:
                d = float(depth_image[py, px])
                if np.isfinite(d) and d > 0:
                    ring_depths.append(d)

            # Centre region: 10 % of inner radius
            xr = inner_a * 0.1 * cos_t
            yr = inner_b * 0.1 * sin_t
            px = int(round(cx + cos_a * xr - sin_a * yr))
            py = int(round(cy + sin_a * xr + cos_a * yr))
            if 0 <= px < w and 0 <= py < h:
                d = float(depth_image[py, px])
                if np.isfinite(d) and d > 0:
                    center_depths.append(d)

        if not ring_depths:
            return True   # no ring depth data — skip check

        if not center_depths:
            return True   # centre has no valid depth → open hole to infinity

        if abs(np.mean(center_depths) - np.mean(ring_depths)) > 0.10:
            return True
        else:
            print(f"center: {np.mean(center_depths)}; ring: {np.mean(ring_depths)}")
            return False

    # ── Find rings ────────────────────────────────────────────────────────────

    def _find_rings(self, ellipses, hsv_image, depth_image=None):
        """Returns (rings, pair_rejections) where pair_rejections maps reason→count."""
        rings      = []
        rejections = {}
        used       = set()

        def _reject(reason):
            rejections[reason] = rejections.get(reason, 0) + 1

        for i in range(len(ellipses)):
            if i in used:
                continue

            for j in range(i + 1, len(ellipses)):
                if j in used:
                    continue

                e1 = ellipses[i]['ell']
                e2 = ellipses[j]['ell']

                dist = np.hypot(e1[0][0] - e2[0][0], e1[0][1] - e2[0][1])
                if dist >= CENTER_THR:
                    _reject(f"dist({dist:.1f}>={CENTER_THR})")
                    continue

                if e1[1][0] >= e2[1][0] and e1[1][1] >= e2[1][1]:
                    outer_idx, inner_idx = i, j
                    outer_ell, inner_ell = e1, e2
                elif e2[1][0] >= e1[1][0] and e2[1][1] >= e1[1][1]:
                    outer_idx, inner_idx = j, i
                    outer_ell, inner_ell = e2, e1
                else:
                    _reject("not_nested")
                    continue

                outer_r    = max(outer_ell[1]) / 2.0
                inner_r    = max(inner_ell[1]) / 2.0
                ring_ratio = outer_r / max(inner_r, 0.1)

                if ring_ratio < MIN_RING_RATIO or ring_ratio > MAX_RING_RATIO:
                    _reject(f"ratio({ring_ratio:.2f})")
                    continue

                if not self._check_ring_hole_depth(outer_ell, inner_ell, depth_image):
                    _reject("depth")
                    continue

                cx, cy = int(outer_ell[0][0]), int(outer_ell[0][1])
                color  = self._identify_contour_color(hsv_image, ellipses[outer_idx]['cnt'])

                rings.append({
                    'center':       (cx, cy),
                    'outer_ell':    outer_ell,
                    'inner_ell':    inner_ell,
                    'color':        color,
                    'outer_radius': int(outer_r),
                    'inner_radius': int(inner_r),
                })

                used.add(outer_idx)
                used.add(inner_idx)

        return rings, rejections

    # ── Grid display helper ───────────────────────────────────────────────────

    def _make_grid(self, panels):
        """Create grid of labeled image tiles. panels = [(label, img), ...]"""
        tiles = []
        
        for label, img in panels:
            if img is None:
                tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
            else:
                if img.dtype == np.float32:
                    # Depth image: fixed scale 0–6 m → 0–255, no per-frame normalisation
                    vis = np.clip(img * (255.0 / 6.0), 0, 255).astype(np.uint8)
                    tile = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                    tile = cv2.resize(tile, (TILE_W, TILE_H))
                elif img.ndim == 2:
                    # Binary mask: hard-threshold then nearest-neighbour resize
                    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    tile = cv2.resize(tile, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
                else:
                    tile = cv2.resize(img.copy(), (TILE_W, TILE_H))
            
            # Add label bar at bottom
            cv2.rectangle(tile, (0, TILE_H - 20), (TILE_W, TILE_H), (0, 0, 0), -1)
            cv2.putText(tile, label, (3, TILE_H - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                       (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        
        # Pad to fill grid
        rows = (len(tiles) + GRID_COLS - 1) // GRID_COLS
        while len(tiles) < rows * GRID_COLS:
            tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
        
        # Stack into grid
        grid_rows = []
        for r in range(rows):
            row = np.hstack(tiles[r * GRID_COLS:(r + 1) * GRID_COLS])
            grid_rows.append(row)
        
        return np.vstack(grid_rows)

    # ── Main image callback ───────────────────────────────────────────────────

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return
        
        h, w = cv_image.shape[:2]
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # ── Color mask ────────────────────────────────────────────────────────
        color_mask = self._create_combined_color_mask(hsv_image, COLOR_RANGES)
        
        # ── Morphological operations ──────────────────────────────────────────
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
        
        mask_opened = cv2.morphologyEx(
            color_mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)
        mask_closed = cv2.morphologyEx(
            mask_opened, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)
        
        # ── Contours ──────────────────────────────────────────────────────────
        contours, _ = cv2.findContours(
            mask_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_vis = cv_image.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 0, 0), 1)
        
        # ── Ellipses and rings ────────────────────────────────────────────────
        ellipses, rejections = self._extract_ellipses(contours)
        detected_rings, pair_rejections = self._find_rings(ellipses, hsv_image, self.latest_depth_raw)
        
        # ── Ring visualization ────────────────────────────────────────────────
        ring_output = cv_image.copy()

        for ring in detected_rings:
            cv2.ellipse(ring_output, ring['outer_ell'], _C_RING, 2)
            cv2.ellipse(ring_output, ring['inner_ell'], _C_RING, 2)

        # ── Face detection (YOLO) ─────────────────────────────────────────────
        det_array = DetectionArray()
        det_array.header = msg.header

        face_res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False,
                                      classes=[0], device='')
        n_faces = 0
        for r in face_res:
            for box in r.boxes:
                bbox = box.xyxy[0]
                if bbox.nelement() == 0:
                    continue
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cx_f = (x1 + x2) // 2
                cy_f = (y1 + y2) // 2
                cv2.rectangle(ring_output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(ring_output, (cx_f, cy_f), 4, (0, 0, 255), -1)
                fd = Detection()
                fd.label   = "face"
                fd.cx      = cx_f
                fd.cy      = cy_f
                fd.bbox_x1 = x1
                fd.bbox_y1 = y1
                fd.bbox_x2 = x2
                fd.bbox_y2 = y2
                fd.color   = ""
                det_array.detections.append(fd)
                n_faces += 1

        # Add info text
        info_text = f"Rings: {len(detected_rings)}  Faces: {n_faces}"
        cv2.putText(ring_output, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, _C_RING, 2)

        # ── Publish detections ────────────────────────────────────────────────
        for ring in detected_rings:
            cx, cy_center = ring['center']
            outer_r = ring['outer_radius']
            inner_r = ring['inner_radius']
            # Point at bottom of ring band (ring material) — the hole center would
            # return the wall depth behind the ring and place the marker at the wall
            band_r    = (outer_r + inner_r) // 2
            sample_cy = min(cy_center + band_r, h - 1)
            rd = Detection()
            rd.label   = "ring"
            rd.cx      = cx
            rd.cy      = sample_cy
            rd.bbox_x1 = cx - outer_r
            rd.bbox_y1 = cy_center - outer_r
            rd.bbox_x2 = cx + outer_r
            rd.bbox_y2 = cy_center + outer_r
            rd.color   = ring['color']
            det_array.detections.append(rd)
        if det_array.detections:
            self.detection_pub.publish(det_array)

        # ── Create dashboard grid ─────────────────────────────────────────────
        dashboard = self._make_grid([
            ("Raw RGB", cv_image),
            ("HSV Color Mask", color_mask),
            ("Morphological", mask_closed),
            ("Detected Contours", contour_vis),
            ("Ring Detection", ring_output),
            ("Depth", self.latest_depth_raw),
        ])
        
        cv2.imshow("Ring Detection Dashboard", dashboard)
        cv2.waitKey(1)
        
        # ── Logging ───────────────────────────────────────────────────────────
        ell_rej  = "  ".join(f"{r}×{n}" for r, n in sorted(rejections.items()))
        pair_rej = "  ".join(f"{r}×{n}" for r, n in sorted(pair_rejections.items()))
        self.get_logger().info(
            f"Contours={len(contours)} Ellipses={len(ellipses)} Rings={len(detected_rings)} Faces={n_faces}"
            f"  ell_rejected: [{ell_rej}]"
            f"  pair_rejected: [{pair_rej}]")
        
        for ring in detected_rings:
            self.get_logger().info(
                f"  Ring @ ({ring['center'][0]},{ring['center'][1]}): "
                f"{ring['color']}, r={ring['outer_radius']}px")


def main(args=None):
    rclpy.init(args=args)
    node = RingDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
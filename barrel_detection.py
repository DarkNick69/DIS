#!/usr/bin/env python3
"""
barrel_detection.py  — colour-first barrel inspector for dis_tutorial7

Detection pipeline per colour
------------------------------
1. HSV colour mask  (vivid S + per-colour H band)
2. Morphological cleanup
3. Contour -> area, solidity, bbox-size filters
4. Depth validity filter:
      Sample depth over every pixel of the contour (colour mask as stencil).
      Floor markings at shallow angle → mostly invalid depth (~5-15% valid)
      Real 3-D barrel             → mostly valid depth  (~50-85% valid)
      Require depth_valid_frac >= MIN_DEPTH_VALID_FRAC (0.25)
5. Depth range filter  (0.25 – 2.5 m on median valid depth)
6. Orientation from bbox ratio:
      bh/bw > ASPECT_VERT         -> vertical barrel
      bh/bw in [MIN_HORIZ, VERT)  -> CANDIDATE for horizontal barrel
      bh/bw < MIN_HORIZ           -> rejected (flat stripe)
7. Compactness guard for horizontal candidates:
      fill_ratio = contour_area / (bw*bh)
      Diagonal floor lines: ~0.15-0.25  ->  REJECTED
      Real barrels:         ~0.60-0.85  ->  ACCEPTED
      Also require bh >= MIN_HORIZ_HT_PX (hard pixel floor)
8. Spill check ONLY for accepted horizontal barrels:
      look for barrel-colour pixels in strip below+beside the bbox

Enable:  ros2 topic pub /barrel_check/enable std_msgs/msg/Bool '{data: true}'
RViz:    Add > MarkerArray > /barrel_check/markers  (Fixed frame: base_link)
"""

import math
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from dis_tutorial4_yellow_interfaces.msg import Detection, DetectionArray
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

# ---------- HSV colour ranges (OpenCV: H 0-179, S 0-255, V 0-255) -----------
COLOR_RANGES = {
    "red":    [(np.array([0,   90,  60]), np.array([9,  255, 255])),
               (np.array([170, 90,  60]), np.array([180,255, 255]))],
    "orange": [(np.array([9,  120, 130]), np.array([21, 255, 255]))],
    "yellow": [(np.array([20, 120, 110]), np.array([38, 255, 255]))],
    "green":  [(np.array([38,  70,  50]), np.array([92, 255, 255]))],
    "blue":   [(np.array([90,  60,  30]), np.array([135,255, 255]))],
    "brown":  [(np.array([8,   60,  20]), np.array([22, 200, 140]))],
    "black":  [(np.array([0,    0,   0]), np.array([180, 80,  55]))],
}

BARREL_BGR = {
    "red":    (0,   0, 220),
    "orange": (0, 140, 255),
    "yellow": (0, 220, 220),
    "green":  (0, 200,  60),
    "blue":   (220, 80,   0),
    "brown":  (30,  80, 130),
    "black":  (0, 0, 0),
}

BARREL_RGB = {
    "red":    (1.,   0.,   0.),
    "orange": (1.,   0.5,  0.),
    "yellow": (1.,   1.,   0.),
    "green":  (0.,   0.8,  0.),
    "blue":   (0.,   0.3,  1.),
    "brown":  (0.55, 0.27, 0.07),
    "black":  (0.15, 0.15, 0.15),
}

# ---------- tunables ---------------------------------------------------------
MIN_AREA        = 1200
MAX_AREA        = 60_000
MIN_DIM         = 22       # px   minimum bbox side (both axes)
SOLIDITY_MIN    = 0.62     # contour_area / convex_hull_area
ASPECT_VERT     = 1.30     # bh/bw > this  -> vertical barrel
MIN_HORIZ_RATIO = 0.20     # bh/bw >= this -> horizontal candidate; < this -> flat stripe (REJECT)
MIN_FILL_RATIO  = 0.38     # contour_area / (bw*bh) for horizontal candidates
                           #   diagonal floor stripe: ~0.15-0.25  -> REJECT
                           #   real barrel blob:      ~0.60-0.85  -> ACCEPT
MIN_HORIZ_HT_PX = 28      # px   hard minimum bbox height for horizontal barrel
MORPH_CLOSE_K   = 11
MORPH_OPEN_K    = 5
DEPTH_MIN       = 0.25     # m  reject too-close
DEPTH_MAX       = 2.5      # m  reject too-far (walls / windows)
MIN_DEPTH_VALID_FRAC = 0.25  # fraction of colour-mask pixels with valid depth
                              # floor lines at grazing angle: ~0.05-0.15
                              # solid 3-D barrel:            ~0.50-0.85
MAX_DEPTH_STD        = 0.12  # m — std-dev of valid depth within contour
                              # barrel face (uniform surface): ~0.02-0.06 m
                              # floor line (near→far gradient): ~0.12-0.40 m
MIN_FILL_RATIO_VERT  = 0.50  # fill_ratio for VERTICAL candidates
                              # real barrel (rectangular blob): ~0.65-0.85 -> ACCEPT
                              # floor perspective triangle:     ~0.35-0.50 -> REJECT
MAX_TEXTURE_STD      = 45    # grayscale std-dev inside bbox
                              # solid coloured barrel: ~10-30
                              # QR code / face photo:  ~60-120
BARREL_DIAM_M        = 0.30  # m  physical barrel diameter (used for size-vs-depth)
BARREL_HEIGHT_M      = 0.85  # m  physical barrel height
SIZE_DEPTH_FRAC      = 0.45  # min(bw,bh) must be >= this × (BARREL_DIAM_M/depth × focal)
                              # face/poster at 0.5 m: blob ~70 px, expected ≥79 px → REJECT
                              # real barrel at 0.5 m: blob ~175 px               → ACCEPT
SIZE_DEPTH_MAX_FRAC  = 2.0   # blob must NOT exceed this × expected barrel size in pixels
                              # dark wall at 1.3 m: blob ~250 px, limit ~132 px  → REJECT
                              # real barrel at 1.3 m: blob ~66 px                → ACCEPT
SPILL_STRIP_W    = 50      # px  width of the strip checked on each side of the core bbox
SPILL_THR        = 0.08    # fraction of strip that must match barrel colour
INFRAME_DEDUP_PX = 50      # px  centroids closer than this in one frame = same barrel
APPROACH_DIST   = 0.90     # m
APPROACH_SPD    = 0.12     # m/s
ANG_GAIN        = 1.2
CAMERA_HFOV     = 1.05     # OAK-D approx horizontal FOV (rad)


# ---------- data class -------------------------------------------------------
class DetectedBarrel:
    __slots__ = ("color", "orientation", "has_spill",
                 "cx_px", "cy_px", "bx", "by", "bw", "bh", "depth_m")

    def __init__(self):
        self.color = self.orientation = "unknown"
        self.has_spill = False
        self.cx_px = self.cy_px = 0
        self.bx = self.by = self.bw = self.bh = 0
        self.depth_m = None

    def __repr__(self):
        d = f"{self.depth_m:.2f}m" if self.depth_m is not None else "?m"
        return f"Barrel({self.color},{self.orientation},{d},spill={self.has_spill})"


# ---------- node -------------------------------------------------------------
class BarrelDetection(Node):

    def __init__(self):
        super().__init__("barrel_detection")

        self.bridge = CvBridge()
        self._lock  = threading.Lock()

        self.latest_bgr   = None
        self.latest_depth = None
        self.enabled      = False
        self._barrels     = []
        self._approach_target = None
        self._warned      = set()

        self._disp_cam = None
        self._disp_msk = None
        self._disp_dep = None

        self.create_subscription(Image, "/oakd/rgb/preview/image_raw",
                                 self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, "/oakd/rgb/preview/depth",
                                 self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(Bool, "/barrel_check/enable",
                                 self._enable_cb, 10)

        self._total_barrels_seen = 0
        self._total_ver_seen = 0
        self._total_hor_seen = 0
        self.create_subscription(String, '/barrel_reports',
                                 self._barrel_report_cb, 10)

        self.cmd_vel_pub    = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.warning_pub    = self.create_publisher(String, "/barrel_warning", 10)
        self.marker_pub     = self.create_publisher(MarkerArray, "/barrel_check/markers", 10)
        self.detection_pub  = self.create_publisher(DetectionArray, "/detections", 10)

        self.create_timer(0.15, self._detect_tick)
        self.create_timer(0.10, self._approach_tick)
        self.create_timer(0.10, self._display_tick)

        self.get_logger().info("BarrelDetection node ready.")

    def _rgb_cb(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return
        with self._lock:
            self.latest_bgr = bgr

    def _depth_cb(self, msg):
        try:
            d = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except CvBridgeError:
            return
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        with self._lock:
            self.latest_depth = d

    def _barrel_report_cb(self, _msg):
        self._total_barrels_seen += 1
        self._total_ver_seen+= 1
        self._total_hor_seen+=1

    def _enable_cb(self, msg: Bool):
        if msg.data and not self.enabled:
            self.enabled = True
            self._warned.clear()
            self._approach_target = None
            self.get_logger().info("[BARREL] ENABLED")
        elif not msg.data and self.enabled:
            self.enabled = False
            self._stop()
            self.get_logger().info("[BARREL] DISABLED")

    # ------ detection tick ---------------------------------------------------

    def _detect_tick(self):
        with self._lock:
            bgr   = self.latest_bgr.copy()   if self.latest_bgr   is not None else None
            depth = self.latest_depth.copy() if self.latest_depth is not None else None

        if bgr is None:
            return

        H, W = bgr.shape[:2]
        hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        focal_px = W / (2.0 * math.tan(CAMERA_HFOV / 2.0))   # pixels per radian
        debug = bgr.copy()

        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K,  MORPH_OPEN_K))

        combined_mask = np.zeros((H, W), dtype=np.uint8)
        barrels = []

        for color_name, ranges in COLOR_RANGES.items():
            mask = np.zeros((H, W), dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko)
            combined_mask |= mask

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA <= area <= MAX_AREA):
                    continue

                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                if hull_area < 1 or area / hull_area < SOLIDITY_MIN:
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw < MIN_DIM or bh < MIN_DIM:
                    continue

                cx = bx + bw // 2
                cy = by + bh // 2

                # Depth validity: sample over the entire colour-mask footprint.
                # Floor markings at grazing angle → mostly invalid depth returns.
                # Solid 3-D objects (barrels) → high fraction of valid returns.
                depth_m = None
                depth_valid_frac = 0.0
                if depth is not None:
                    mask_roi  = mask[by:by+bh, bx:bx+bw].astype(bool)
                    depth_roi = depth[by:by+bh, bx:bx+bw]
                    vals  = depth_roi[mask_roi]
                    valid = vals[(vals > 0.1) & (vals < 9.0)]
                    n_total = int(mask_roi.sum())
                    if n_total > 0:
                        depth_valid_frac = valid.size / n_total
                    if valid.size > 0:
                        depth_m = float(np.median(valid))

                if depth_m is None or not (DEPTH_MIN <= depth_m <= DEPTH_MAX):
                    continue
                if depth_valid_frac < MIN_DEPTH_VALID_FRAC:
                    continue  # floor marking — depth sensor fails at grazing angles
                # Floor lines span near→far and create a large depth gradient.
                # A real barrel face is a flat surface with nearly uniform depth.
                if valid.size > 4 and float(np.std(valid)) > MAX_DEPTH_STD:
                    continue

                # Size-vs-depth synergy: depth tells us how big a real barrel
                # MUST appear in pixels.  A face photo or coloured patch is far
                # too small for its claimed distance.
                #   face blob at 0.5 m:  ~70 px  < expected 79 px  → REJECTED
                #   real barrel at 0.5 m: ~175 px > expected 79 px  → ACCEPTED
                expected_min_px = BARREL_DIAM_M / depth_m * focal_px * SIZE_DEPTH_FRAC
                if min(bw, bh) < expected_min_px:
                    continue
                # Upper bound: blob much wider/taller than a barrel at this depth = wall/shadow.
                expected_max_w = BARREL_DIAM_M   / depth_m * focal_px * SIZE_DEPTH_MAX_FRAC
                expected_max_h = BARREL_HEIGHT_M / depth_m * focal_px * SIZE_DEPTH_MAX_FRAC
                if bw > expected_max_w or bh > expected_max_h:
                    continue

                # Texture guard: reject QR codes, face images, textured signs.
                # A solid barrel has uniform pixels (low std-dev); patterned
                # surfaces (QR, faces) have high contrast → high std-dev.
                gray_roi = cv2.cvtColor(bgr[by:by+bh, bx:bx+bw],
                                        cv2.COLOR_BGR2GRAY)
                if float(gray_roi.std()) > MAX_TEXTURE_STD:
                    continue

                # --- orientation ---
                ratio = bh / (bw + 1e-6)

                if ratio > ASPECT_VERT:
                    # Guard vertical candidates: a perspective floor triangle has
                    # bh/bw > 1.3 but only fills ~35-50% of its bounding box.
                    # A real cylindrical barrel fills ~65-85%.
                    fill_ratio = area / (bw * bh + 1e-6)
                    if fill_ratio < MIN_FILL_RATIO_VERT:
                        continue
                    orientation = "vertical"

                elif ratio >= MIN_HORIZ_RATIO:
                    # Candidate for horizontal barrel.
                    # Guard against diagonal floor stripes: they have large
                    # bounding boxes but only fill a fraction of them.
                    # A real barrel blob fills ~60-85% of its bbox;
                    # a diagonal line fills only ~15-25%.
                    fill_ratio = area / (bw * bh + 1e-6)
                    if fill_ratio < MIN_FILL_RATIO or bh < MIN_HORIZ_HT_PX:
                        continue   # floor stripe / navigation line -- not a barrel
                    orientation = "horizontal"

                else:
                    # bh/bw too small -> obviously a flat floor stripe
                    continue

                # Shrink bbox to the barrel body only, trimming spill extensions.
                # The colour mask has columns/rows with low density where the
                # spill is attached; the compact barrel body has high density.
                cbx, cby, cbw, cbh = self._find_barrel_core_bbox(mask, bx, by, bw, bh)
                ccx = cbx + cbw // 2
                ccy = cby + cbh // 2

                # In-frame dedup: the same barrel can produce multiple sub-contours
                # (top cap, body ring, etc.).  Keep only the first per location.
                if any(math.hypot(ccx - e.cx_px, ccy - e.cy_px) < INFRAME_DEDUP_PX
                       for e in barrels):
                    continue

                # Spill check — horizontal only, one side at a time.
                # We check left, right, and below the CORE bbox independently.
                # A single side exceeding SPILL_THR confirms spillage.
                has_spill = (orientation == "horizontal"
                             and self._check_spill(hsv, cbx, cby, cbw, cbh,
                                                   H, W, color_name))

                b = DetectedBarrel()
                b.color, b.orientation, b.has_spill = color_name, orientation, has_spill
                b.cx_px, b.cy_px = ccx, ccy
                b.bx, b.by, b.bw, b.bh = cbx, cby, cbw, cbh
                b.depth_m = depth_m
                barrels.append(b)

                draw_col = BARREL_BGR[color_name]
                ann_col  = (255, 255, 255) if color_name == "black" else draw_col
                thick = 3 if has_spill else 2
                cv2.rectangle(debug, (cbx, cby), (cbx + cbw, cby + cbh), ann_col, thick)
                orient_char = "V" if orientation == "vertical" else "H"
                lbl = f"{color_name} {orient_char} {depth_m:.1f}m"
                if has_spill:
                    lbl += " SPILL!"
                cv2.putText(debug, lbl, (cbx, max(12, cby - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ann_col, 1, cv2.LINE_AA)
                cv2.circle(debug, (ccx, ccy), 5, ann_col, -1)

        self._barrels = barrels

        for b in barrels:
            self.get_logger().info(f"  {b}")
            key = (b.color, b.orientation)
            if b.has_spill and key not in self._warned:
                self._warned.add(key)
                txt = f"Spill detected at {b.color} horizontal barrel!"
                self.get_logger().warn(f"[BARREL] {txt}")
                msg = String(); msg.data = txt
                self.warning_pub.publish(msg)
            if (self._approach_target is None
                    and b.orientation == "horizontal" and b.has_spill):
                self._approach_target = b

        n_v = sum(1 for b in barrels if b.orientation == "vertical")
        n_h = sum(1 for b in barrels if b.orientation == "horizontal")
        cv2.rectangle(debug, (0, H - 18), (W, H), (0, 0, 0), -1)
        cv2.putText(debug,
                    f"in-frame={len(barrels)}  vert={n_v}  horiz={n_h}"
                    f"  total={self._total_barrels_seen}"
                    f"  totalVer={self._total_ver_seen}"
                    f"  totalHor={self._total_hor_seen}",
                    (4, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        self._publish_markers(barrels, W)
        self._publish_detections(barrels)

        with self._lock:
            self._disp_cam = debug
            self._disp_msk = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            self._disp_dep = depth

    # ------ detections for object_localizer ---------------------------------

    def _publish_detections(self, barrels):
        da = DetectionArray()
        da.header.stamp    = self.get_clock().now().to_msg()
        da.header.frame_id = 'oakd_rgb_camera_optical_frame'
        for b in barrels:
            d = Detection()
            # Encode orientation and spill in label so the localizer
            # can include them in its report without a custom message type.
            # Format: "barrel_vertical" | "barrel_horizontal" | "barrel_horizontal_spill"
            label = f"barrel_{b.orientation}"
            if b.has_spill:
                label += "_spill"
            d.label   = label
            d.cx      = int(b.cx_px)
            d.cy      = int(b.cy_px)
            d.bbox_x1 = int(b.bx)
            d.bbox_y1 = int(b.by)
            d.bbox_x2 = int(b.bx + b.bw)
            d.bbox_y2 = int(b.by + b.bh)
            d.color   = b.color
            da.detections.append(d)
        self.detection_pub.publish(da)

    # ------ spill check ------------------------------------------------------

    def _find_barrel_core_bbox(self, mask, bx, by, bw, bh):
        """
        Trim the bounding box to the dense barrel body, excluding thin spill
        extensions.  A real barrel fills ~70-90% of each column/row in its
        bbox; a spill extension fills only ~20-40%.
        Falls back to the original bbox if analysis fails.
        """
        roi = (mask[by:by+bh, bx:bx+bw] > 0)
        col_sum = roi.sum(axis=0).astype(float)   # pixels per column, length = bw
        row_sum = roi.sum(axis=1).astype(float)   # pixels per row,    length = bh

        if col_sum.max() == 0:
            return bx, by, bw, bh

        dense_cols = np.where(col_sum >= col_sum.max() * 0.45)[0]
        dense_rows = np.where(row_sum >= row_sum.max() * 0.45)[0]

        if len(dense_cols) == 0 or len(dense_rows) == 0:
            return bx, by, bw, bh

        cx0 = bx + int(dense_cols[0])
        cx1 = bx + int(dense_cols[-1]) + 1
        cy0 = by + int(dense_rows[0])
        cy1 = by + int(dense_rows[-1]) + 1
        return cx0, cy0, max(1, cx1 - cx0), max(1, cy1 - cy0)

    # ------ spill check ------------------------------------------------------

    def _check_spill(self, hsv, bx, by, bw, bh, H, W, color_name):
        """
        Check each side of the barrel core bbox independently.
        Returns True if any ONE side has >= SPILL_THR fraction of barrel-colour
        pixels immediately adjacent to the barrel body.
        Sides checked: left strip, right strip, below strip.
        """
        ranges = COLOR_RANGES.get(color_name, [])

        def _frac(x0, x1, y0, y1):
            x0 = max(0, x0); x1 = min(W, x1)
            y0 = max(0, y0); y1 = min(H, y1)
            if x1 <= x0 or y1 <= y0:
                return 0.0
            strip = hsv[y0:y1, x0:x1]
            hit = np.zeros(strip.shape[:2], dtype=bool)
            for lo, hi in ranges:
                hit |= np.all((strip >= lo) & (strip <= hi), axis=2)
            return float(hit.sum()) / (strip.shape[0] * strip.shape[1])

        sw = SPILL_STRIP_W
        left  = _frac(bx - sw,      bx,       by,      by + bh)
        right = _frac(bx + bw,      bx+bw+sw, by,      by + bh)
        below = _frac(bx - sw,      bx+bw+sw, by + bh, by+bh+sw)

        return max(left, right, below) > SPILL_THR

    # ------ RViz markers -----------------------------------------------------

    def _publish_markers(self, barrels, img_w):
        ma = MarkerArray()
        for i, b in enumerate(barrels):
            if b.depth_m is None:
                continue
            angle = ((b.cx_px - img_w / 2.0) / (img_w / 2.0)) * (CAMERA_HFOV / 2.0)
            fwd   =  b.depth_m * math.cos(angle)
            lat   = -b.depth_m * math.sin(angle)
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp    = self.get_clock().now().to_msg()
            m.ns     = b.color   # color name — read by barrel_localizer
            m.id     = i + 1
            m.type   = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = fwd
            m.pose.position.y = lat
            m.pose.position.z = 0.35
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.30
            m.scale.z = 0.65 if b.orientation == "vertical" else 0.28
            r, g, bv = BARREL_RGB.get(b.color, (0.8, 0.8, 0.))
            m.color.r = r; m.color.g = g; m.color.b = bv
            m.color.a = 0.90 if b.has_spill else 0.65
            m.lifetime.sec = 1
            ma.markers.append(m)
        self.marker_pub.publish(ma)

    # ------ display ----------------------------------------------------------

    def _display_tick(self):
        with self._lock:
            cam   = self._disp_cam.copy() if self._disp_cam is not None else None
            msk   = self._disp_msk.copy() if self._disp_msk is not None else None
            depth = self._disp_dep.copy() if self._disp_dep is not None else None
            raw   = self.latest_bgr.copy() if self.latest_bgr is not None else None

        if raw is None:
            return

        TH = 300

        def rsz(img):
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w * TH / h), TH))

        p_cam = rsz(cam if cam is not None else raw)
        p_msk = rsz(msk if msk is not None else np.zeros_like(raw))

        if depth is not None:
            d_clip = np.clip(depth / 5.0, 0.0, 1.0)
            p_dep  = rsz(cv2.applyColorMap((d_clip * 255).astype(np.uint8),
                                           cv2.COLORMAP_JET))
        else:
            p_dep = np.full((TH, p_cam.shape[1], 3), 50, dtype=np.uint8)
            cv2.putText(p_dep, "no depth", (8, TH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        for panel, lbl in zip([p_cam, p_msk, p_dep],
                               ["Camera + Detections", "Colour Mask (HSV)", "Depth  0-5 m"]):
            cv2.putText(panel, lbl, (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Barrel Detection", np.hstack([p_cam, p_msk, p_dep]))
        cv2.waitKey(1)

    # ------ approach ---------------------------------------------------------

    def _approach_tick(self):
        if not self.enabled or self._approach_target is None:
            return
        t = self._approach_target
        if t.depth_m is None:
            return
        if t.depth_m <= APPROACH_DIST:
            self._stop()
            txt = f"Arrived at {t.color} barrel — SPILL WARNING issued."
            self.get_logger().warn(f"[BARREL] {txt}")
            msg = String(); msg.data = txt
            self.warning_pub.publish(msg)
            self._approach_target = None
            return
        with self._lock:
            w = self.latest_bgr.shape[1] if self.latest_bgr is not None else 320
        err = (t.cx_px - w / 2.0) / (w / 2.0)
        twist = TwistStamped()
        twist.header.stamp    = self.get_clock().now().to_msg()
        twist.header.frame_id = "base_link"
        twist.twist.angular.z = -ANG_GAIN * err
        if abs(err) < 0.30:
            twist.twist.linear.x = APPROACH_SPD
        self.cmd_vel_pub.publish(twist)

    def _stop(self):
        t = TwistStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        self.cmd_vel_pub.publish(t)


def main(args=None):
    print("idk man 3")
    rclpy.init(args=args)
    node = BarrelDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from dis_tutorial4_yellow_interfaces.msg import Detection, DetectionArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO

class ObjectDetectionNode(Node):

    def __init__(self):
        super().__init__('object_detection_node')

        self.bridge = CvBridge()

        #yolo for person detection
        self.model = YOLO("yolov8n.pt")

        #subscriber for image_raw
        self.image_sub = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        # Publisher: detections
        self.detection_pub = self.create_publisher(
            DetectionArray,
            '/detections',
            10
        )

        self.get_logger().info("object_detection_node started. Subscribed to image_raw, publishing to /detections.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        detection_array = DetectionArray()
        detection_array.header = msg.header

        #yolo person, face detection
        res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device='')
        for r in res:
            for box in r.boxes:
                bbox = box.xyxy[0]
                if bbox.nelement() == 0:
                    continue

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                det = Detection()
                det.label = "face"
                det.cx = cx
                det.cy = cy
                det.bbox_x1 = x1
                det.bbox_y1 = y1
                det.bbox_x2 = x2
                det.bbox_y2 = y2
                det.color = ""
                detection_array.detections.append(det)

        #Ring detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'green': [
                (np.array([35, 100, 100]), np.array([85, 255, 255]))
            ],
            'blue': [
                (np.array([100, 100, 100]), np.array([130, 255, 255]))
            ],
            'black': [
                (np.array([0, 0, 0]), np.array([180, 255, 50]))
            ],
        }

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.5:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                det = Detection()
                det.label = "ring"
                det.cx = x + w // 2
                det.cy = y + h // 2
                det.bbox_x1 = x
                det.bbox_y1 = y
                det.bbox_x2 = x + w
                det.bbox_y2 = y + h
                det.color = color_name
                detection_array.detections.append(det)

        # Publish detections
        if len(detection_array.detections) > 0:
            self.detection_pub.publish(detection_array)
            self.get_logger().info(f"Published {len(detection_array.detections)} detections.")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

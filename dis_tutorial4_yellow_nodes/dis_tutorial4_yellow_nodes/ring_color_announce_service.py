#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from dis_tutorial4_yellow_interfaces.srv import RingColorAnnounce
import subprocess


class RingColorAnnounceService(Node):

    def __init__(self):
        super().__init__('ring_color_announce_service')

        self.srv = self.create_service(
            RingColorAnnounce,
            '/ring_color_announce_service',
            self.announce_callback
        )

        self.get_logger().info("ring_color_announce_service started. Waiting for requests...")

    def announce_callback(self, request, response):
        color = request.color
        self.get_logger().info(f"Ring color announce requested: {color}")

        text = f"I found a {color} ring!"

        try:
            subprocess.run(
                ['espeak', '-s', '150', text],
                timeout=10
            )
            response.success = True
        except FileNotFoundError:
            self.get_logger().warn("espeak is not installed. Falling back to terminal.")
            self.get_logger().info(f"ANNOUNCEMENT: {text}")
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Announcement failed: {e}")
            response.success = False

        return response


def main(args=None):
    rclpy.init(args=args)
    node = RingColorAnnounceService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

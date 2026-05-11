#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from dis_tutorial4_yellow_interfaces.srv import Greeting
import subprocess


class GreetingService(Node):

    def __init__(self):
        super().__init__('greeting_service')

        self.srv = self.create_service(
            Greeting,
            '/greeting_service',
            self.greeting_callback
        )

        self.get_logger().info("greeting_service started. Waiting for requests...")

    def greeting_callback(self, request, response):
        name = request.name
        self.get_logger().info(f"Greeting requested for: {name}")

        text = f"Hello {name}, nice to meet you!"

        try:
            subprocess.run(
                ['espeak', '-s', '150', text],
                timeout=10
            )
            response.success = True
        except FileNotFoundError:
            self.get_logger().warn("espeak not installed. Falling back to terminal output.")
            self.get_logger().info(f"GREETING: {text}")
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Greeting failed: {e}")
            response.success = False

        return response


def main(args=None):
    rclpy.init(args=args)
    node = GreetingService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

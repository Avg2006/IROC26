#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

import math
import time


class DronePositionMonitor(Node):

    def __init__(self):

        super().__init__('drone_position_monitor')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.position_callback,
            qos_profile
        )

        # Initial reference storage
        self.init_samples = []

        self.reference_set = False

        self.ref_x = 0.0
        self.ref_y = 0.0
        self.ref_z = 0.0

        self.start_time = time.time()

        self.get_logger().info("Drone Position Monitor Started")

    def quaternion_to_euler(self, x, y, z, w):

        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch
        sinp = 2 * (w * y - z * x)

        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw)
        )

    def position_callback(self, msg):

        # Raw position
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_z = msg.pose.pose.position.z

        # Collect samples for first 3 seconds
        if not self.reference_set:

            self.init_samples.append((raw_x, raw_y, raw_z))

            elapsed = time.time() - self.start_time

            print(f"\rCalibrating reference frame... {elapsed:.1f}s",
                  end='')

            if elapsed >= 3.0:

                self.ref_x = sum(p[0] for p in self.init_samples) / len(self.init_samples)
                self.ref_y = sum(p[1] for p in self.init_samples) / len(self.init_samples)
                self.ref_z = sum(p[2] for p in self.init_samples) / len(self.init_samples)

                self.reference_set = True

                print("\nReference frame locked.\n")

            return

        # Relative position
        x = raw_x - self.ref_x
        y = raw_y - self.ref_y
        z = raw_z - self.ref_z

        # Velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        speed = math.sqrt(vx**2 + vy**2 + vz**2)

        # Orientation
        q = msg.pose.pose.orientation

        roll, pitch, yaw = self.quaternion_to_euler(
            q.x,
            q.y,
            q.z,
            q.w
        )

        # Clean output
        print("\n" + "=" * 50)

        print("LOCAL POSITION")

        print("=" * 50)

        print("Position (Relative):")
        print(f"  X : {x:.3f} m")
        print(f"  Y : {y:.3f} m")
        print(f"  Z : {z:.3f} m")

        print("\nVelocity:")
        print(f"  Vx : {vx:.3f} m/s")
        print(f"  Vy : {vy:.3f} m/s")
        print(f"  Vz : {vz:.3f} m/s")
        print(f"  Speed : {speed:.3f} m/s")

        print("\nOrientation:")
        print(f"  Roll  : {roll:.2f} deg")
        print(f"  Pitch : {pitch:.2f} deg")
        print(f"  Yaw   : {yaw:.2f} deg")

        print("=" * 50)

def main(args=None):

    rclpy.init(args=args)

    node = DronePositionMonitor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Stopping node...")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
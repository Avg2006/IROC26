import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy,qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from tf_transformations import euler_from_quaternion


class WaypointMission(Node):
    def __init__(self):
        super().__init__('waypoint_mission')

        # ── Publishers / Services ──────────────────────────────────────
        self.pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10
        )

        self.ref_odom_pub = self.create_publisher(
            PoseStamped,
            '/reference_odom',
            qos_profile_sensor_data
        )
        self.bool_pub = self.create_publisher(Bool, '/take_picture', 10)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos
        )

        # ── Calibration state ───────────────────────────────────────────
        self.init_samples = []
        self.reference_set = False
        self.ref_x = self.ref_y = self.ref_z = 0.0
        self.calib_start_time = time.time()
        self.calib_duration = 3.0
        self.locked_yaw = None          # fixed body-frame heading (radians)

        # ── General flight state ────────────────────────────────────────
        self.current_pos = [0.0, 0.0, 0.0]   # calibrated-frame (world-aligned)
        self.pos_threshold = 0.25
        self.takeoff_alt = 1.0
        self.hover_time = 5.0            # seconds per waypoint
        self.counter = 0
        self.phase = "CALIBRATE"
        self.start_time = 0.0
        self.move_timeout = 60.0

        # ── EDIT YOUR MISSION HERE ───────────────────────────────────────
        # Each entry: (forward_m, right_m, alt_m) relative to the takeoff
        # point, expressed in BODY frame (i.e. relative to however the
        # drone happened to be facing when armed).
        self.body_waypoints = [
            (0.0, 0.0, self.takeoff_alt),
            (1.0, 0.0, self.takeoff_alt),
            (0.0, 0.0, self.takeoff_alt)
        ]
        # ──────────────────────────────────────────────────────────────

        self.waypoints = None   # filled in (world/calibrated frame) after yaw lock
        self.wp_index = 0

        self.pos_type_mask = (
            PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE | PositionTarget.IGNORE_YAW
        )

        self.timer = self.create_timer(0.1, self.loop)
        self.get_logger().info("Calibrating reference frame + locking yaw for 3 seconds...")

    # =====================================================================
    # Body-frame -> world-frame conversion (uses the ONE locked yaw)
    # =====================================================================
    def body_to_world(self, forward, right, alt):
        yaw = self.locked_yaw
        world_x = forward * math.cos(yaw) + right * math.sin(yaw)
        world_y = forward * math.sin(yaw) - right * math.cos(yaw)
        return (world_x, world_y, alt)

    # =====================================================================
    # Callbacks
    # =====================================================================
    def pose_callback(self, msg):
        raw_x = msg.pose.position.x
        raw_y = msg.pose.position.y
        raw_z = msg.pose.position.z

        (_, _, yaw) = euler_from_quaternion([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

        if self.locked_yaw is None:
            self.locked_yaw = yaw
            self.get_logger().info(f"Yaw locked at {math.degrees(yaw):.1f} deg (from East)")

        if not self.reference_set:
            elapsed = time.time() - self.calib_start_time
            self.init_samples.append((raw_x, raw_y, raw_z))
            print(f"\rCalibrating... {elapsed:.1f}s", end='')
            if elapsed >= self.calib_duration:
                n = len(self.init_samples)
                self.ref_x = sum(p[0] for p in self.init_samples) / n
                self.ref_y = sum(p[1] for p in self.init_samples) / n
                self.ref_z = sum(p[2] for p in self.init_samples) / n
                self.reference_set = True

                # Build the world-frame waypoint list ONCE, using locked yaw
                self.waypoints = [
                    self.body_to_world(f, r, a) for (f, r, a) in self.body_waypoints
                ]

                self.phase = "INIT"
                print()
                self.get_logger().info(
                    f"Reference locked -> ({self.ref_x:.3f}, {self.ref_y:.3f}, {self.ref_z:.3f})"
                )
                self.get_logger().info(f"World-frame waypoints: {self.waypoints}")
            return

        self.current_pos[0] = raw_x - self.ref_x
        self.current_pos[1] = raw_y - self.ref_y
        self.current_pos[2] = raw_z - self.ref_z

        ref_odom_msg = PoseStamped()
        ref_odom_msg.header.stamp = self.get_clock().now().to_msg()
        ref_odom_msg.header.frame_id = "map"
        ref_odom_msg.pose.position.x = self.current_pos[0]
        ref_odom_msg.pose.position.y = self.current_pos[1]
        ref_odom_msg.pose.position.z = self.current_pos[2]
        self.ref_odom_pub.publish(ref_odom_msg)

    # =====================================================================
    # Helpers
    # =====================================================================
    def hold(self, target):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = self.pos_type_mask
        msg.position.x = target[0] + self.ref_x
        msg.position.y = target[1] + self.ref_y
        msg.position.z = target[2] + self.ref_z
        self.pub.publish(msg)

    def arrived(self, target):
        dist = (
            (self.current_pos[0] - target[0]) ** 2 +
            (self.current_pos[1] - target[1]) ** 2 +
            (self.current_pos[2] - target[2]) ** 2
        ) ** 0.5
        return dist < self.pos_threshold

    def trigger(self, value=False):
        msg = Bool()
        msg.data = value
        self.bool_pub.publish(msg)

    def takeoff(self, altitude):
        if self.takeoff_client.wait_for_service(timeout_sec=2.0):
            req = CommandTOL.Request()
            req.altitude = altitude
            req.min_pitch = 0.0
            req.yaw = 0.0
            req.latitude = 0.0
            req.longitude = 0.0
            future = self.takeoff_client.call_async(req)
            future.add_done_callback(self._takeoff_cb)
        else:
            self.get_logger().error("Takeoff service unavailable")

    def _takeoff_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info("Takeoff command accepted")
            self.phase = "TAKEOFF"
        else:
            self.get_logger().warn(f"Takeoff rejected (code {result.result})")

    def land(self):
        self.get_logger().info("Switching to LAND mode...")
        if self.mode_client.wait_for_service(timeout_sec=2.0):
            req = SetMode.Request()
            req.custom_mode = "LAND"
            self.mode_client.call_async(req)
        self.phase = "DONE"

    # =====================================================================
    # Main loop
    # =====================================================================
    def loop(self):
        if self.phase == "CALIBRATE":
            return

        # ── INIT: GUIDED -> arm -> takeoff ──────────────────────────────
        if self.phase == "INIT":
            if self.counter == 10:
                self.get_logger().info("Switching to GUIDED...")
                if self.mode_client.wait_for_service(timeout_sec=2.0):
                    req = SetMode.Request()
                    req.custom_mode = "GUIDED"
                    self.mode_client.call_async(req)
            elif self.counter == 30:
                self.get_logger().info("Arming...")
                if self.arm_client.wait_for_service(timeout_sec=2.0):
                    req = CommandBool.Request()
                    req.value = True
                    self.arm_client.call_async(req)
            elif self.counter == 50:
                self.get_logger().info(f"Sending takeoff to {self.takeoff_alt} m...")
                self.takeoff(self.takeoff_alt)

        # ── TAKEOFF ───────────────────────────────────────────────────────
        elif self.phase == "TAKEOFF":
            self.get_logger().info(f"Altitude = {self.current_pos[2]:.2f} m")
            if self.current_pos[2] >= (self.takeoff_alt - self.pos_threshold):
                self.get_logger().info("Takeoff altitude reached -> starting waypoint mission")
                self.wp_index = 0
                self.phase = "HOVER"
                self.start_time = time.time()
                self.trigger(True)

        # ── MOVE ─────────────────────────────────────────────────────────
        elif self.phase == "MOVE":
            target = self.waypoints[self.wp_index]
            self.hold(target)

            elapsed_move = time.time() - self.start_time
            if elapsed_move > self.move_timeout:
                self.get_logger().error(f"[wp{self.wp_index}] Timeout! Landing...")
                self.phase = "LAND"
                return

            dx = target[0] - self.current_pos[0]
            dy = target[1] - self.current_pos[1]
            dz = target[2] - self.current_pos[2]
            self.get_logger().info(
                f"[wp{self.wp_index}] moving  dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f}"
            )

            if self.arrived(target):
                self.get_logger().info(f"Reached wp{self.wp_index} -> hovering {self.hover_time}s")
                self.phase = "HOVER"
                self.start_time = time.time()
                self.trigger(True)

        # ── HOVER ────────────────────────────────────────────────────────
        elif self.phase == "HOVER":
            target = self.waypoints[self.wp_index]
            self.hold(target)
            elapsed = time.time() - self.start_time
            if elapsed < self.hover_time:
                self.get_logger().info(
                    f"[wp{self.wp_index}] hovering... {self.hover_time - elapsed:.1f}s left"
                )
            elif self.wp_index + 1 < len(self.waypoints):
                self.trigger(False)
                self.wp_index += 1
                self.get_logger().info(
                    f"--> moving to wp{self.wp_index} {self.waypoints[self.wp_index]}"
                )
                self.phase = "MOVE"
                self.start_time = time.time()
            else:
                self.trigger(False)
                self.get_logger().info("All waypoints complete -> landing")
                self.phase = "LAND"

        # ── LAND ─────────────────────────────────────────────────────────
        elif self.phase == "LAND":
            self.land()

        # ── DONE ─────────────────────────────────────────────────────────
        elif self.phase == "DONE":
            print("lala lulu land")
            pass  # node keeps spinning; kill externally

        self.counter += 1


def main():
    rclpy.init()
    node = WaypointMission()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
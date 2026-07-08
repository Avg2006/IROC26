import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from tf_transformations import euler_from_quaternion


class WaypointBoundaryMission(Node):
    def __init__(self):
        super().__init__('waypoint_boundary_mission')

        # ── Publishers / Services ──────────────────────────────────────
        self.pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10
        )
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos
        )
        # ── Boundary (yellow line) detector ──────────────────────────────
        self.sub_boundary = self.create_subscription(
            Bool, '/boundary_detected', self.boundary_callback, 10
        )
        self.boundary_detected_raw = False
        self.boundary_confirm_count = 0
        self.boundary_confirm_needed = 3
        self.boundary_confirmed = False

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
        self.takeoff_alt = 2.5
        self.hover_time = 5.0            # seconds per waypoint
        self.boundary_hover_time = 5.0   # settle time right after stopping
        self.counter = 0
        self.phase = "CALIBRATE"
        self.start_time = 0.0
        self.move_timeout = 60.0

        self.search_forward_distance = 50.0   # m, body-frame forward
        self.search_max_distance     = 10.0   # m, abort search if exceeded
        self.search_max_time         = 120.0  # s, abort search if exceeded
        self.search_target = None             # world-frame target, set once

        # ── Stop point (new local origin after boundary detection) ────────
        self.stop_origin = None   # [x, y, z] in calibrated frame


        self.post_boundary_body_waypoints = [
            (0.0, 0.0, self.takeoff_alt),   # 0: hold at the stop point
            (-1.0, 0.0, self.takeoff_alt),   # 1: forward 1 m from stop point
            (-1.0, -1.0, self.takeoff_alt),   # 2: then right 1 m
        ]
        # ──────────────────────────────────────────────────────────────

        self.post_boundary_waypoints = None   # filled in after stop (world frame)
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
    def body_to_world(self, forward, right, alt, origin=(0.0, 0.0, 0.0)):
        yaw = self.locked_yaw
        world_x = origin[0] + forward * math.cos(yaw) + right * math.sin(yaw)
        world_y = origin[1] + forward * math.sin(yaw) - right * math.cos(yaw)
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

                # Build the forward-search world target ONCE, using locked yaw
                self.search_target = self.body_to_world(
                    self.search_forward_distance, 0.0, self.takeoff_alt
                )

                self.phase = "INIT"
                print()
                self.get_logger().info(
                    f"Reference locked -> ({self.ref_x:.3f}, {self.ref_y:.3f}, {self.ref_z:.3f})"
                )
                self.get_logger().info(f"Forward search target: {self.search_target}")
            return

        self.current_pos[0] = raw_x - self.ref_x
        self.current_pos[1] = raw_y - self.ref_y
        self.current_pos[2] = raw_z - self.ref_z

    def boundary_callback(self, msg):
        self.boundary_detected_raw = bool(msg.data)

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
                self.get_logger().info("Takeoff altitude reached -> searching forward for boundary")
                self.phase = "SEARCH_FORWARD"
                self.start_time = time.time()

        # ── SEARCH_FORWARD: fly toward big forward target until boundary ──
        elif self.phase == "SEARCH_FORWARD":
            self.hold(self.search_target)

            # Debounce the boundary flag: require N consecutive True reads
            if self.boundary_detected_raw:
                self.boundary_confirm_count += 1
            else:
                self.boundary_confirm_count = 0
            self.boundary_confirmed = (
                self.boundary_confirm_count >= self.boundary_confirm_needed
            )

            dist_travelled = math.sqrt(
                self.current_pos[0] ** 2 + self.current_pos[1] ** 2
            )
            elapsed = time.time() - self.start_time

            self.get_logger().info(
                f"[search] dist={dist_travelled:.2f} m  boundary_raw={self.boundary_detected_raw}  "
                f"confirm={self.boundary_confirm_count}/{self.boundary_confirm_needed}",
                throttle_duration_sec=0.5,
            )

            if self.boundary_confirmed:
                self.stop_origin = [
                    self.current_pos[0], self.current_pos[1], self.current_pos[2]
                ]
                self.get_logger().info(
                    f"Boundary confirmed -> stopping at {self.stop_origin} -> BOUNDARY_HOVER"
                )
                self.phase = "BOUNDARY_HOVER"
                self.start_time = time.time()
                return

            # Failsafes: never seen the line -> land rather than fly forever
            if dist_travelled > self.search_max_distance:
                self.get_logger().error(
                    f"Search distance limit ({self.search_max_distance} m) exceeded, "
                    "no boundary seen -> LAND"
                )
                self.phase = "LAND"
                return
            if elapsed > self.search_max_time:
                self.get_logger().error(
                    f"Search time limit ({self.search_max_time} s) exceeded, "
                    "no boundary seen -> LAND"
                )
                self.phase = "LAND"
                return

        # ── BOUNDARY_HOVER: settle at the stop point, latch it as origin ──
        elif self.phase == "BOUNDARY_HOVER":
            self.hold(self.stop_origin)
            elapsed = time.time() - self.start_time
            if elapsed < self.boundary_hover_time:
                self.get_logger().info(
                    f"Settling at boundary stop... {self.boundary_hover_time - elapsed:.1f}s left"
                )
            else:
                # Build the second waypoint list, in world frame, anchored
                # at the stop point -- using the SAME locked yaw.
                self.post_boundary_waypoints = [
                    self.body_to_world(f, r, a, origin=self.stop_origin)
                    for (f, r, a) in self.post_boundary_body_waypoints
                ]
                self.wp_index = 0
                self.get_logger().info(
                    f"Post-boundary waypoints (world frame): {self.post_boundary_waypoints}"
                )
                self.phase = "HOVER2"
                self.start_time = time.time()

        # ── MOVE2 ────────────────────────────────────────────────────────
        elif self.phase == "MOVE2":
            target = self.post_boundary_waypoints[self.wp_index]
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
                self.phase = "HOVER2"
                self.start_time = time.time()

        # ── HOVER2 ───────────────────────────────────────────────────────
        elif self.phase == "HOVER2":
            target = self.post_boundary_waypoints[self.wp_index]
            self.hold(target)
            elapsed = time.time() - self.start_time
            if elapsed < self.hover_time:
                self.get_logger().info(
                    f"[wp{self.wp_index}] hovering... {self.hover_time - elapsed:.1f}s left"
                )
            elif self.wp_index + 1 < len(self.post_boundary_waypoints):
                self.wp_index += 1
                self.get_logger().info(
                    f"--> moving to wp{self.wp_index} {self.post_boundary_waypoints[self.wp_index]}"
                )
                self.phase = "MOVE2"
                self.start_time = time.time()
            else:
                self.get_logger().info("All post-boundary waypoints complete -> landing")
                self.phase = "LAND"

        # ── LAND ─────────────────────────────────────────────────────────
        elif self.phase == "LAND":
            self.land()

        # ── DONE ─────────────────────────────────────────────────────────
        elif self.phase == "DONE":
            pass  # node keeps spinning; kill externally

        self.counter += 1


def main():
    rclpy.init()
    node = WaypointBoundaryMission()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
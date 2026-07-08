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


class WaypointFullMission(Node):
    def __init__(self):
        super().__init__('waypoint_full_mission')

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
        # ── Line / corner detectors ──────────────────────────────────────
        self.sub_boundary = self.create_subscription(
            Bool, '/boundary_detected', self.boundary_callback, 10
        )
        self.sub_corner = self.create_subscription(
            Bool, '/corner_detected', self.corner_callback, 10
        )
        self.boundary_detected_raw = False
        self.corner_detected_raw = False

        # Debounce: require N consecutive True reads before trusting a flag
        self.detect_confirm_needed = 1
        self.boundary_confirm_count = 0
        self.corner_confirm_count = 0

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
        self.stop_hover_time = 2.0       # settle time after boundary / corner stop
        self.counter = 0
        self.phase = "CALIBRATE"
        self.start_time = 0.0
        self.move_timeout = 60.0

        # ── HOME (latched right after takeoff) ────────────────────────────
        self.home_position = None   # [x, y, z] in calibrated frame

        # ── Forward search leg (until yellow line) ─────────────────────────
        self.search_forward_distance = 50.0
        self.search_max_distance = 10.0
        self.search_max_time = 120.0
        self.forward_search_target = None

        # ── Boundary stop point ─────────────────────────────────────────
        self.boundary_origin = None   # [x, y, z], latched when line seen

        # ── Right search leg (until corner) ────────────────────────────────
        self.search_right_distance = 50.0
        self.right_search_target = None

        # ── Corner stop point -> becomes waypoint-list origin ──────────────
        self.corner_origin = None   # [x, y, z], latched when corner seen


        # BODY frame (forward_m, right_m, alt_m). alt_m is absolute.
        self.mission_body_waypoints = [
            (0.0, 0.0, self.takeoff_alt),
            (-1.0, 0.0, self.takeoff_alt),
            (-2.0, 0.0, self.takeoff_alt),
            (-3.0, 0.0, self.takeoff_alt),
            (-3.0, -1.0, self.takeoff_alt),
            (-2.0, -1.0, self.takeoff_alt),
            (-1.0, -1.0, self.takeoff_alt),
            (0.0, -1.0, self.takeoff_alt),
            (0.0, -2.0, self.takeoff_alt),
            (-1.0, -2.0, self.takeoff_alt),
            (-2.0, -2.0, self.takeoff_alt),
            (-3.0, -2.0, self.takeoff_alt),
            (-3.0, -3.0, self.takeoff_alt),
            (-2.0, -3.0, self.takeoff_alt),
            (-1.0, -3.0, self.takeoff_alt),
            (0.0, -3.0, self.takeoff_alt),
            (0.0, -4.0, self.takeoff_alt),
            (-2.0, -4.0, self.takeoff_alt),
            (-1.0, -4.0, self.takeoff_alt),
            (-3.0, -4.0, self.takeoff_alt),
            (-3.0, -5.0, self.takeoff_alt),
            (-2.0, -5.0, self.takeoff_alt),
            (-1.0, -5.0, self.takeoff_alt),
            (0.0, -5.0, self.takeoff_alt),
            (0.0, -6.0, self.takeoff_alt),
            (-1.0, -6.0, self.takeoff_alt),
            (-2.0, -6.0, self.takeoff_alt),
            (-3.0, -6.0, self.takeoff_alt),
            (-3.0, -7.0, self.takeoff_alt),
            (-2.0, -7.0, self.takeoff_alt),
            (-1.0, -7.0, self.takeoff_alt),
            (0.0, -7.0, self.takeoff_alt)
        ]
        # ──────────────────────────────────────────────────────────────

        self.mission_waypoints = None   # filled in after corner stop (world frame)
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

                self.forward_search_target = self.body_to_world(
                    self.search_forward_distance, 0.0, self.takeoff_alt
                )

                self.phase = "INIT"
                print()
                self.get_logger().info(
                    f"Reference locked -> ({self.ref_x:.3f}, {self.ref_y:.3f}, {self.ref_z:.3f})"
                )
            return

        self.current_pos[0] = raw_x - self.ref_x
        self.current_pos[1] = raw_y - self.ref_y
        self.current_pos[2] = raw_z - self.ref_z

    def boundary_callback(self, msg):
        self.boundary_detected_raw = bool(msg.data)

    def corner_callback(self, msg):
        self.corner_detected_raw = bool(msg.data)

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
                self.home_position = [
                    self.current_pos[0], self.current_pos[1], self.takeoff_alt
                ]
                self.get_logger().info(
                    f"Takeoff altitude reached, HOME latched at {self.home_position} "
                    "-> searching forward for yellow line"
                )
                self.phase = "SEARCH_FORWARD"
                self.start_time = time.time()

        # ── SEARCH_FORWARD: fly forward until yellow line ──────────────────
        elif self.phase == "SEARCH_FORWARD":
            self.hold(self.forward_search_target)

            if self.boundary_detected_raw:
                self.boundary_confirm_count += 1
            else:
                self.boundary_confirm_count = 0
            confirmed = self.boundary_confirm_count >= self.detect_confirm_needed

            dist_travelled = math.sqrt(
                self.current_pos[0] ** 2 + self.current_pos[1] ** 2
            )
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f"[search-fwd] dist={dist_travelled:.2f} m  "
                f"boundary_confirm={self.boundary_confirm_count}/{self.detect_confirm_needed}",
                throttle_duration_sec=0.5,
            )

            if confirmed:
                self.boundary_origin = [
                    self.current_pos[0], self.current_pos[1], self.current_pos[2]
                ]
                self.get_logger().info(
                    f"Yellow line confirmed -> stopping at {self.boundary_origin} "
                    "-> BOUNDARY_HOVER"
                )
                self.phase = "BOUNDARY_HOVER"
                self.start_time = time.time()
                return

            if dist_travelled > self.search_max_distance:
                self.get_logger().error("Forward search distance limit exceeded -> LAND")
                self.phase = "LAND"
                return
            if elapsed > self.search_max_time:
                self.get_logger().error("Forward search time limit exceeded -> LAND")
                self.phase = "LAND"
                return

        # ── BOUNDARY_HOVER: settle at the line, then search right ──────────
        elif self.phase == "BOUNDARY_HOVER":
            self.hold(self.boundary_origin)
            elapsed = time.time() - self.start_time
            if elapsed < self.stop_hover_time:
                self.get_logger().info(
                    f"Settling at yellow line... {self.stop_hover_time - elapsed:.1f}s left"
                )
            else:
                self.right_search_target = self.body_to_world(
                    0.0, self.search_right_distance, self.takeoff_alt,
                    origin=self.boundary_origin,
                )
                self.get_logger().info(
                    f"Searching right for corner, target={self.right_search_target}"
                )
                self.corner_confirm_count = 0
                self.phase = "SEARCH_RIGHT"
                self.start_time = time.time()

        # ── SEARCH_RIGHT: fly right (from boundary stop) until corner ──────
        elif self.phase == "SEARCH_RIGHT":
            self.hold(self.right_search_target)

            if self.corner_detected_raw:
                self.corner_confirm_count += 1
            else:
                self.corner_confirm_count = 0
            confirmed = self.corner_confirm_count >= self.detect_confirm_needed

            dist_travelled = math.sqrt(
                (self.current_pos[0] - self.boundary_origin[0]) ** 2 +
                (self.current_pos[1] - self.boundary_origin[1]) ** 2
            )
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f"[search-right] dist={dist_travelled:.2f} m  "
                f"corner_confirm={self.corner_confirm_count}/{self.detect_confirm_needed}",
                throttle_duration_sec=0.5,
            )

            if confirmed:
                self.corner_origin = [
                    self.current_pos[0], self.current_pos[1], self.current_pos[2]
                ]
                self.get_logger().info(
                    f"Corner confirmed -> stopping at {self.corner_origin} -> CORNER_HOVER"
                )
                self.phase = "CORNER_HOVER"
                self.start_time = time.time()
                return

            if dist_travelled > self.search_max_distance:
                self.get_logger().error("Right search distance limit exceeded -> LAND")
                self.phase = "LAND"
                return
            if elapsed > self.search_max_time:
                self.get_logger().error("Right search time limit exceeded -> LAND")
                self.phase = "LAND"
                return

        # ── CORNER_HOVER: settle, latch corner as waypoint-list origin ─────
        elif self.phase == "CORNER_HOVER":
            self.hold(self.corner_origin)
            elapsed = time.time() - self.start_time
            if elapsed < self.stop_hover_time:
                self.get_logger().info(
                    f"Settling at corner... {self.stop_hover_time - elapsed:.1f}s left"
                )
            else:
                self.mission_waypoints = [
                    self.body_to_world(f, r, a, origin=self.corner_origin)
                    for (f, r, a) in self.mission_body_waypoints
                ]
                self.wp_index = 0
                self.get_logger().info(
                    f"Mission waypoints (world frame): {self.mission_waypoints}"
                )
                self.phase = "HOVER2"
                self.start_time = time.time()

        # ── MOVE2 ────────────────────────────────────────────────────────
        elif self.phase == "MOVE2":
            target = self.mission_waypoints[self.wp_index]
            self.hold(target)

            elapsed_move = time.time() - self.start_time
            if elapsed_move > self.move_timeout:
                self.get_logger().error(f"[wp{self.wp_index}] Timeout! -> RETURN_HOME")
                self.phase = "RETURN_HOME"
                self.start_time = time.time()
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
            target = self.mission_waypoints[self.wp_index]
            self.hold(target)
            elapsed = time.time() - self.start_time
            if elapsed < self.hover_time:
                self.get_logger().info(
                    f"[wp{self.wp_index}] hovering... {self.hover_time - elapsed:.1f}s left"
                )
            elif self.wp_index + 1 < len(self.mission_waypoints):
                self.wp_index += 1
                self.get_logger().info(
                    f"--> moving to wp{self.wp_index} {self.mission_waypoints[self.wp_index]}"
                )
                self.phase = "MOVE2"
                self.start_time = time.time()
            else:
                self.get_logger().info("All mission waypoints complete -> returning home")
                self.phase = "RETURN_HOME"
                self.start_time = time.time()

        # ── RETURN_HOME ──────────────────────────────────────────────────
        elif self.phase == "RETURN_HOME":
            self.hold(self.home_position)
            elapsed = time.time() - self.start_time

            dx = self.home_position[0] - self.current_pos[0]
            dy = self.home_position[1] - self.current_pos[1]
            self.get_logger().info(
                f"[return-home] dx={dx:+.2f} dy={dy:+.2f}", throttle_duration_sec=0.5
            )

            if self.arrived(self.home_position):
                self.get_logger().info("Reached home -> landing")
                self.phase = "LAND"
                return

            if elapsed > self.move_timeout:
                self.get_logger().error("Return-home timeout -> landing here")
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
    node = WaypointFullMission()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
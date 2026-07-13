import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from tf_transformations import euler_from_quaternion

from config import MissionConfig
from state import MissionState
from helpers import MissionHelpersMixin


class WaypointFullMission(Node, MissionHelpersMixin):
    def __init__(self):
        super().__init__('waypoint_full_mission')
        self.config = MissionConfig()
        self.state = MissionState()

        # ── Fixed waypoint mission (BODY frame: forward, right, alt) ──
        # Converted to world frame at the corner, using the ALIGNED yaw.
        self.hover_time = 5.0
        self.mission_body_waypoints = [
            (0.0, 0.0, self.config.takeoff_alt),
            (-1.0, -1.0, self.config.takeoff_alt),
            (-2.0, -1.0, self.config.takeoff_alt),
            (-3.0, -1.0, self.config.takeoff_alt),
            (-3.0, -2.0, self.config.takeoff_alt)
        ]
        self.mission_waypoints = None   # filled at CORNER_HOVER (world frame)
        self.wp_index = 0

        # ── Publishers / Services ──────────────────────────────────────
        self.pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.bool_pub = self.create_publisher(Bool, '/take_picture', 10)
        self.ref_odom_pub = self.create_publisher(PoseStamped, '/reference_odom', qos_profile_sensor_data)

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # ── Subscribers ────────────────────────────────────────────────
        self.sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos)
        self.sub_boundary = self.create_subscription(Bool, '/boundary_detected', self.boundary_callback, 10)
        self.sub_front_back_offset = self.create_subscription(Point, '/front_back_offset', self.front_back_offset_callback, 10)
        self.sub_side_offset = self.create_subscription(Point, '/side_line_offset', self.side_offset_callback, 10)
        self.sub_is_front_back = self.create_subscription(Bool, '/boundary_is_front_back', self.is_front_back_callback, 10)
        self.sub_corner = self.create_subscription(Bool, '/corner_detected', self.corner_callback, 10)
        self.sub_test_confirm = self.create_subscription(Bool, '/mission_test_confirm', self.test_confirm_callback, 10)

        self.timer = self.create_timer(0.1, self.loop)
        self.get_logger().info("Calibrating reference frame + locking yaw for 3 seconds...")

    # =====================================================================
    # Body->world conversion using the ALIGNED yaw (target_yaw).
    # Same math as helpers.body_to_world, but rotated by the arena-aligned
    # heading instead of the takeoff locked yaw. This is the ONLY
    # difference vs the plain waypoint mission.
    # =====================================================================
    def body_to_world_aligned(self, forward, right, alt, origin=(0.0, 0.0, 0.0)):
        yaw = self.state.target_yaw if self.state.target_yaw is not None else self.state.locked_yaw
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
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w
        ])
        self.state.current_yaw = yaw

        if self.state.locked_yaw is None:
            self.state.locked_yaw = yaw
            self.get_logger().info(f"Yaw locked at {math.degrees(yaw):.1f} deg (from East)")

        if not self.state.reference_set:
            elapsed = time.time() - self.state.calib_start_time
            self.state.init_samples.append((raw_x, raw_y, raw_z))
            print(f"\rCalibrating... {elapsed:.1f}s", end='')
            if elapsed >= self.config.calib_duration:
                n = len(self.state.init_samples)
                self.state.ref_x = sum(p[0] for p in self.state.init_samples) / n
                self.state.ref_y = sum(p[1] for p in self.state.init_samples) / n
                self.state.ref_z = sum(p[2] for p in self.state.init_samples) / n
                self.state.reference_set = True

                self.state.forward_search_target = self.body_to_world(
                    self.config.search_forward_distance, 0.0, self.config.takeoff_alt
                )
                print()
                self.get_logger().info(f"Reference locked -> ({self.state.ref_x:.3f}, {self.state.ref_y:.3f}, {self.state.ref_z:.3f})")

                if self.config.test_mode:
                    self.get_logger().info("test_mode enabled -> TEST (publish True on /mission_test_confirm to proceed)")
                    self.state.phase = "TEST"
                    self.state.test_last_log_time = time.time()
                else:
                    self.state.phase = "INIT"
            return

        self.state.current_pos[0] = raw_x - self.state.ref_x
        self.state.current_pos[1] = raw_y - self.state.ref_y
        self.state.current_pos[2] = raw_z - self.state.ref_z

        self.publish_reference_odom(msg.pose.orientation)

    def boundary_callback(self, msg):
        self.state.boundary_detected_raw = bool(msg.data)

    def front_back_offset_callback(self, msg):
        self.state.front_back_offset_z = msg.z
        self.state.front_back_offset_y = msg.y

    def side_offset_callback(self, msg):
        self.state.side_offset_z = msg.z
        self.state.side_offset_y = msg.y

    def is_front_back_callback(self, msg):
        self.state.boundary_is_front_back = bool(msg.data)

    def corner_callback(self, msg):
        self.state.corner_detected_raw = bool(msg.data)

    def test_confirm_callback(self, msg):
        if bool(msg.data):
            self.state.test_confirmed = True

    # =====================================================================
    # Main Mission State Machine Loop
    # =====================================================================
    def loop(self):
        if self.state.phase == "CALIBRATE":
            return

        # ── TEST (Ground testing: inspect odom, no arm/takeoff/setpoints) ───
        if self.state.phase == "TEST":
            now = time.time()
            if now - self.state.test_last_log_time >= self.config.test_log_interval:
                self.state.test_last_log_time = now
                self.get_logger().info(
                    "[TEST] ref=({:.3f}, {:.3f}, {:.3f}) | "
                    "current_pos=({:.3f}, {:.3f}, {:.3f}) | "
                    "yaw_locked={:.1f}deg current_yaw={:.1f}deg".format(
                        self.state.ref_x, self.state.ref_y, self.state.ref_z,
                        self.state.current_pos[0], self.state.current_pos[1], self.state.current_pos[2],
                        math.degrees(self.state.locked_yaw) if self.state.locked_yaw is not None else float('nan'),
                        math.degrees(self.state.current_yaw),
                    )
                )

            if self.state.test_confirmed:
                self.get_logger().info("TEST confirmed -> proceeding to INIT")
                self.state.phase = "INIT"
                self.state.counter = 0
            return

        # ── INIT ──────────────────────────────────────────────────────────
        if self.state.phase == "INIT":
            if self.state.counter == 10:
                self.get_logger().info("Switching to GUIDED...")
                if self.mode_client.wait_for_service(timeout_sec=2.0):
                    req = SetMode.Request()
                    req.custom_mode = "GUIDED"
                    self.mode_client.call_async(req)
            elif self.state.counter == 30:
                self.get_logger().info("Arming...")
                if self.arm_client.wait_for_service(timeout_sec=2.0):
                    req = CommandBool.Request()
                    req.value = True
                    self.arm_client.call_async(req)
            elif self.state.counter == 50:
                self.get_logger().info(f"Sending takeoff to {self.config.takeoff_alt} m...")
                self.takeoff(self.config.takeoff_alt)

        # ── TAKEOFF ───────────────────────────────────────────────────────
        elif self.state.phase == "TAKEOFF":
            self.get_logger().info(f"Altitude = {self.state.current_pos[2]:.2f} m")
            if self.state.current_pos[2] >= (self.config.takeoff_alt - self.config.pos_threshold):
                self.state.home_position = [self.state.current_pos[0], self.state.current_pos[1], self.config.takeoff_alt]
                self.state.target_yaw = self.state.locked_yaw
                self.get_logger().info("Takeoff altitude reached, HOME latched -> searching forward")
                self.state.phase = "SEARCH_FORWARD"
                self.state.start_time = time.time()

        # ── SEARCH_FORWARD ────────────────────────────────────────────────
        elif self.state.phase == "SEARCH_FORWARD":
            self.hold(self.state.forward_search_target, self.state.target_yaw)

            self.state.boundary_confirm_count = self.state.boundary_confirm_count + 1 if self.state.boundary_detected_raw else 0
            confirmed = self.state.boundary_confirm_count >= self.config.detect_confirm_needed

            dist_travelled = math.sqrt(self.state.current_pos[0] ** 2 + self.state.current_pos[1] ** 2)
            elapsed = time.time() - self.state.start_time

            if confirmed:
                self.state.boundary_origin = list(self.state.current_pos)
                self.state.yaw_arena_offset_set = False
                self.log_waypoint("boundary_line_found", self.state.boundary_origin)
                self.get_logger().info("Yellow line confirmed -> ALIGN_PERPENDICULAR")
                self.state.phase = "ALIGN_PERPENDICULAR"
                self.state.start_time = time.time()
                return

            if dist_travelled > self.config.search_max_distance or elapsed > self.config.search_max_time:
                self.get_logger().error("Search limits exceeded -> LAND")
                self.state.phase = "LAND"
                return

        # ── ALIGN_PERPENDICULAR ───────────────────────────────────────────
        elif self.state.phase == "ALIGN_PERPENDICULAR":
            self.hold(self.state.boundary_origin, self.state.target_yaw)

            # --- 1. SET THE TARGET YAW ---
            if not self.state.yaw_arena_offset_set:
                # SEARCH_FORWARD only ever confirms on the front/back line,
                # so that's the correct source for the yaw error here.
                err_yaw = self.state.front_back_offset_z

                # Normalize error to [-pi/2, pi/2]
                err_yaw = (err_yaw + math.pi / 2) % math.pi - math.pi / 2

                self.state.yaw_arena_offset = self.state.locked_yaw - self.state.current_yaw + err_yaw

                # Normalize offset to [-pi, pi]
                self.state.yaw_arena_offset = (self.state.yaw_arena_offset + math.pi) % (2 * math.pi) - math.pi

                self.state.yaw_arena_offset_set = True
                self.state.target_yaw = self.state.locked_yaw - self.state.yaw_arena_offset

                # Normalize target yaw to [-pi, pi]
                self.state.target_yaw = (self.state.target_yaw + math.pi) % (2 * math.pi) - math.pi

                self.state.start_time = time.time()
                return

            elapsed = time.time() - self.state.start_time

            # --- 2. WAIT FOR YAW TO SETTLE (no sampling) ---
            if elapsed < self.config.align_hover_time:
                return

            # --- 3. LATCH CURRENT POSITION ---
            self.state.boundary_origin = list(self.state.current_pos)

            self.log_waypoint("boundary_origin_aligned", self.state.boundary_origin)
            self.get_logger().info(
                f"Alignment complete -> origin latched at "
                f"({self.state.boundary_origin[0]:.2f}, {self.state.boundary_origin[1]:.2f}) "
                f"-> BOUNDARY_HOVER"
            )
            self.state.phase = "BOUNDARY_HOVER"
            self.state.start_time = time.time()

        # ── BOUNDARY_HOVER ────────────────────────────────────────────────
        elif self.state.phase == "BOUNDARY_HOVER":
            self.hold(self.state.boundary_origin, self.state.target_yaw)
            elapsed = time.time() - self.state.start_time

            if elapsed >= self.config.stop_hover_time:
                # Search right IN THE ALIGNED FRAME so we track parallel
                # to the boundary line.
                self.state.right_search_target = self.body_to_world_aligned(
                    0.0, self.config.search_right_distance, self.config.takeoff_alt,
                    origin=self.state.boundary_origin,
                )
                self.get_logger().info(
                    f"Searching right for corner, target="
                    f"({self.state.right_search_target[0]:.2f}, {self.state.right_search_target[1]:.2f})"
                )
                self.state.corner_confirm_count = 0
                self.state.phase = "SEARCH_RIGHT"
                self.state.start_time = time.time()

        # ── SEARCH_RIGHT (simple, like code 2: fly right until corner) ────
        elif self.state.phase == "SEARCH_RIGHT":
            self.hold(self.state.right_search_target, self.state.target_yaw)

            self.state.corner_confirm_count = self.state.corner_confirm_count + 1 if self.state.corner_detected_raw else 0
            confirmed = self.state.corner_confirm_count >= self.config.corner_confirm_needed

            dist_travelled = math.sqrt(
                (self.state.current_pos[0] - self.state.boundary_origin[0]) ** 2 +
                (self.state.current_pos[1] - self.state.boundary_origin[1]) ** 2
            )
            elapsed = time.time() - self.state.start_time

            if confirmed:
                self.state.corner_origin = list(self.state.current_pos)
                self.log_corner(1, self.state.corner_origin, dist_travelled)
                self.log_waypoint("corner1_found", self.state.corner_origin)
                self.get_logger().info("Corner confirmed -> CORNER_HOVER")
                self.state.phase = "CORNER_HOVER"
                self.state.start_time = time.time()
                return

            if dist_travelled > self.config.search_max_distance or elapsed > self.config.search_max_time:
                self.get_logger().error("Right search limits exceeded -> LAND")
                self.state.phase = "LAND"
                return

        # ── CORNER_HOVER: settle, build waypoint list in ALIGNED frame ────
        elif self.state.phase == "CORNER_HOVER":
            self.hold(self.state.corner_origin, self.state.target_yaw)
            elapsed = time.time() - self.state.start_time

            if elapsed >= self.config.stop_hover_time:
                self.mission_waypoints = [
                    self.body_to_world_aligned(f, r, a, origin=self.state.corner_origin)
                    for (f, r, a) in self.mission_body_waypoints
                ]
                self.wp_index = 0
                self.get_logger().info(
                    f"Mission waypoints latched at corner (aligned frame, "
                    f"yaw={math.degrees(self.state.target_yaw):.1f} deg) -> HOVER2"
                )
                self.state.phase = "HOVER2"
                self.state.start_time = time.time()

        # ── MOVE2 ────────────────────────────────────────────────────────
        elif self.state.phase == "MOVE2":
            target = self.mission_waypoints[self.wp_index]
            self.hold(target, self.state.target_yaw)

            elapsed_move = time.time() - self.state.start_time
            if elapsed_move > self.config.move_timeout:
                self.get_logger().error(f"[wp{self.wp_index}] Timeout! -> RETURN_HOME")
                self.state.phase = "RETURN_HOME"
                self.state.start_time = time.time()
                return

            if self.arrived(target):
                self.log_waypoint(f"wp{self.wp_index}", list(target))
                self.trigger(True)
                self.get_logger().info(f"Reached wp{self.wp_index} -> hovering {self.hover_time}s")
                self.state.phase = "HOVER2"
                self.state.start_time = time.time()

        # ── HOVER2 ───────────────────────────────────────────────────────
        elif self.state.phase == "HOVER2":
            target = self.mission_waypoints[self.wp_index]
            self.hold(target, self.state.target_yaw)
            elapsed = time.time() - self.state.start_time
            if elapsed < self.hover_time:
                pass
            elif self.wp_index + 1 < len(self.mission_waypoints):
                self.trigger(False)
                self.wp_index += 1
                self.get_logger().info(f"--> moving to wp{self.wp_index}")
                self.state.phase = "MOVE2"
                self.state.start_time = time.time()
            else:
                self.trigger(False)
                self.get_logger().info("All mission waypoints complete -> returning home")
                self.state.phase = "RETURN_HOME"
                self.state.start_time = time.time()

        # ── RETURN_HOME ──────────────────────────────────────────────────
        elif self.state.phase == "RETURN_HOME":
            self.hold(self.state.home_position, self.state.target_yaw)
            elapsed = time.time() - self.state.start_time

            if self.arrived(self.state.home_position):
                self.log_waypoint("home_reached", self.state.home_position)
                self.get_logger().info("Reached home -> landing")
                self.state.phase = "LAND"
                return

            if elapsed > self.config.move_timeout:
                self.get_logger().error("Return-home timeout -> landing here")
                self.state.phase = "LAND"

        # ── LAND / DONE ──────────────────────────────────────────────────
        elif self.state.phase == "LAND":
            self.land()

        self.state.counter += 1


def main():
    rclpy.init()
    node = WaypointFullMission()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
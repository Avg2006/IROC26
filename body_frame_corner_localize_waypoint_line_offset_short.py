import math
import os
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Point, PoseArray
from std_msgs.msg import Bool, Int32, Float32
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
        self.ref_odom_pub = self.create_publisher(
            PoseStamped,
            '/reference_odom',
            qos_profile_sensor_data
        )
        # Rising-edge camera trigger
        self.bool_pub = self.create_publisher(Bool, '/take_picture', 10)
        # Signals that the ArUco search phase has begun
        self.mission_complete_pub = self.create_publisher(Bool, '/mission_complete', 10)
        # Signals transfer status to the GUI (0=inflight, 1=transferring, 2=done)
        self.transfer_pub = self.create_publisher(Int32, '/transfer_status', 10)
        # Tells the charging rig it is safe to start charging
        self.charging_pub = self.create_publisher(Bool, '/chargingOkay', 10)
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
        # ── Boundary offset (distance from yellow line) ────────────────────
        self.sub_offset = self.create_subscription(
            Point, '/front_back_offset', self.offset_callback, qos
        )
        self.boundary_offset = Point()   # last raw msg (x, y, z)

        # ── ArUco detector ──────────────────────────────────────────────
        self.sub_aruco = self.create_subscription(
            PoseArray, '/aruco_poses', self.aruco_callback, qos
        )
        # Pack voltage reported by the ESP on the charging rig
        self.sub_voltage = self.create_subscription(
            Float32, '/esp/total_voltage', self.voltage_callback, 10
        )

        self.boundary_detected_raw = False
        self.corner_detected_raw = False
        self.min_move_before_boundary = 1.0  # meters from takeoff point
        self.boundary_detection_started = False

        # ── Line-offset hold (maintain fixed standoff from yellow line) ────
        self.line_offset_setpoint = -0.20  # target y reading -> 20 cm standoff
        self.offset_zero_eps = 0.02        # |y| below this counts as "zero" (line lost)
        self.offset_lost_wait = 1.0        # seconds of zero before we call it lost
        self.offset_backup_dist = 0.5      # meters to back off once truly lost
        self.line_follow_gain = 1.2        # m of forward-correction per m of offset error
        self.max_follow_correction = 1.0   # clamp correction so a bad reading can't fling us
        self.offset_sign = -1.0
        self.last_valid_offset_y = 0.0
        self.offset_zero_since = None
        self.offset_lost_confirmed = False

        # Debounce: require N consecutive True reads before trusting a flag
        self.detect_confirm_needed = 2
        self.detect_corner_confirm_needed = 1
        self.boundary_confirm_count = 0
        self.corner_confirm_count = 0

        # ── Calibration state ───────────────────────────────────────────
        self.init_samples = []
        self.reference_set = False
        self.ref_x = self.ref_y = self.ref_z = 0.0
        self.calib_start_time = time.time()
        self.calib_duration = 3.0
        self.locked_yaw = None          # fixed body-frame heading (radians)
        self.current_yaw = 0.0
        self.target_yaw = None
        self.yaw_arena_offset = 0.0
        self.yaw_arena_offset_set = False
        self.align_hover_time = 2.0    # seconds to let yaw settle before latching

        # ── General flight state ────────────────────────────────────────
        self.current_pos = [0.0, 0.0, 0.0]   # calibrated-frame (world-aligned)
        self.pos_threshold = 0.25
        self.takeoff_alt = 2.0
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
        self.search_right_distance = 10.0
        self.right_search_target = None

        # ── Corner stop point -> becomes waypoint-list origin ──────────────
        self.corner_origin = None   # [x, y, z], latched when corner seen

        self.yaw_kar_bc = False


        # BODY frame (forward_m, right_m, alt_m). alt_m is absolute
        # self.mission_body_waypoints = [
        #     (0.0, 0.0, self.takeoff_alt),
        #     (-1.0, -0.0, self.takeoff_alt),
        #     # (-3.0, -3.0, self.takeoff_alt)
        # ]

        self.mission_body_waypoints = [
            (-0.0, -0.0, self.takeoff_alt),
            (-1.0, -0.0, self.takeoff_alt),
            (-2.0, -0.0, self.takeoff_alt),
            (-3.0, -0.0, self.takeoff_alt),
            (-4.0, -0.0, self.takeoff_alt),
            (-5.0, -0.0, self.takeoff_alt),
            (-5.0, -1.0, self.takeoff_alt),
            (-4.0, -1.0, self.takeoff_alt),
            (-3.0, -1.0, self.takeoff_alt),
            (-2.0, -1.0, self.takeoff_alt),
            (-1.0, -1.0, self.takeoff_alt),
            (-0.0, -1.0, self.takeoff_alt),
            (-0.0, -2.0, self.takeoff_alt),
            (-1.0, -2.0, self.takeoff_alt),
            (-2.0, -2.0, self.takeoff_alt),
            (-3.0, -2.0, self.takeoff_alt),
            (-4.0, -2.0, self.takeoff_alt),
            (-5.0, -2.0, self.takeoff_alt),
            (-5.0, -3.0, self.takeoff_alt),
            (-4.0, -3.0, self.takeoff_alt),
            (-3.0, -3.0, self.takeoff_alt),
            (-2.0, -3.0, self.takeoff_alt),
            (-1.0, -3.0, self.takeoff_alt),
            (-0.0, -3.0, self.takeoff_alt),
            (-0.0, -4.0, self.takeoff_alt),
            (-1.0, -4.0, self.takeoff_alt),
            (-2.0, -4.0, self.takeoff_alt),
            (-3.0, -4.0, self.takeoff_alt),
            (-4.0, -4.0, self.takeoff_alt),
            (-5.0, -4.0, self.takeoff_alt),
            (-5.0, -5.0, self.takeoff_alt),
            (-4.0, -5.0, self.takeoff_alt),
            (-3.0, -5.0, self.takeoff_alt),
            (-2.0, -5.0, self.takeoff_alt),
            (-1.0, -5.0, self.takeoff_alt),
            (-0.0, -5.0, self.takeoff_alt),
            (-0.0, -6.0, self.takeoff_alt),
            (-1.0, -6.0, self.takeoff_alt),
            (-2.0, -6.0, self.takeoff_alt),
            (-3.0, -6.0, self.takeoff_alt),
            (-4.0, -6.0, self.takeoff_alt),
            (-5.0, -6.0, self.takeoff_alt),
            (-5.0, -7.0, self.takeoff_alt),
            (-4.0, -7.0, self.takeoff_alt),
            (-3.0, -7.0, self.takeoff_alt),
            (-2.0, -7.0, self.takeoff_alt),
            (-1.0, -7.0, self.takeoff_alt),
            (-0.0, -7.0, self.takeoff_alt)
        ]
        # ──────────────────────────────────────────────────────────────

        self.mission_waypoints = None   # filled in after corner stop (world frame)
        self.wp_index = 0

        # ── Mid-mission re-align (whenever a waypoint sits back at the line) ─
        self.line_realign_search_time = 2.0   # seconds to look for the line before giving up
        self.line_realign_confirm_count = 0
        self.line_realign_applied = False

        # ── ArUco / precision-landing state ─────────────────────────────
        self.align_threshold = 0.1
        self.land_alt = 0.55
        self.descend_step = 0.3
        self.hover_duration = 0.5        # seconds to confirm alignment before descending
        self.aruco_detected = False
        self.error_x = 0.0
        self.error_y = 0.0
        self.last_aruco_global = None    # [x, y, z] in calibrated frame
        self.last_aruco_cb_time = None
        self.aruco_stale_timeout = 1.0
        self.descent_target = None
        self.temp_pos = None             # XY held during ARUCO_HOVER_ALIGNED
        self.align_alt = None            # altitude latched for ARUCO_ALIGN (not re-read each tick)

        # ── Spiral / vicinity search around home before giving up ──────────
        self.search_radius = 0.18        # metres
        self.search_dwell = 0.5          # seconds to hover at each search point
        self.search_offsets = []
        self.search_index = 0
        self.search_home = None

        self.mission_complete_sent = False

        # ── Post-land image transfer ────────────────────────────────────
        self.scp_transfer_done = False

        # ── Charging (CHARGING_STARTED) ─────────────────────────────────
        # battery_voltage tracks /esp/total_voltage; initial_battery_voltage is
        # latched on entry to CHARGING_STARTED.  We leave the state once the
        # pack has gained charge_delta_min AND cleared charge_voltage_min.
        # /chargingOkay is published every loop tick: True only while in
        # CHARGING_STARTED, False in every other phase.
        self.battery_voltage = None
        self.initial_battery_voltage = None
        self.charge_delta_min = 0.55     # volts gained since charging began
        self.charge_voltage_min = 15.5  # absolute volts

        # ── Global per-phase watchdog ────────────────────────────────────
        # If ANY phase runs longer than phase_timeout without transitioning,
        # something is stuck (lost detection, marker never found, a service
        # call that never resolves, etc). The watchdog forces a LAND, which
        # naturally flows into AFTER_LAND and runs the SCP transfer the same
        # way a normal landing would.
        self.phase_timeout = 120.0   # 2 minutes
        self.phase_entry_time = time.time()
        self._watchdog_last_phase = None
        self.watchdog_exempt_phases = {
            "CALIBRATE",         # fixed 3s calibration window
            "LAND",               # already landing, nothing to time out to
            "AFTER_LAND",         # settle + SCP transfer; handles its own timing
            "CHARGING_STARTED",   # battery charging legitimately takes a while
            "DONE",
        }

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
        yaw = self.locked_yaw - self.yaw_arena_offset
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

        self.current_yaw = yaw

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
                    self.search_forward_distance, -0.0, self.takeoff_alt
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

        ref_odom_msg = PoseStamped()
        ref_odom_msg.header.stamp = self.get_clock().now().to_msg()
        ref_odom_msg.header.frame_id = "map"
        ref_odom_msg.pose.position.x = self.current_pos[0]
        ref_odom_msg.pose.position.y = self.current_pos[1]
        ref_odom_msg.pose.position.z = self.current_pos[2]
        self.ref_odom_pub.publish(ref_odom_msg)

    def boundary_callback(self, msg):
        distance_from_takeoff = math.hypot(self.current_pos[0], self.current_pos[1])
        if distance_from_takeoff < self.min_move_before_boundary:
            return
        if not self.boundary_detection_started:
            self.boundary_detection_started = True
            self.get_logger().info("Yellow line detection start")
        self.boundary_detected_raw = bool(msg.data)

    def corner_callback(self, msg):
        self.corner_detected_raw = bool(msg.data)

    def offset_callback(self, msg):
        self.boundary_offset = msg

    def aruco_callback(self, msg):
        if msg.poses:
            self.aruco_detected = True
            self.error_x = msg.poses[0].position.x
            self.error_y = msg.poses[0].position.y

            self.last_aruco_global = [
                self.current_pos[0] + self.error_x,
                self.current_pos[1] - self.error_y,
                self.current_pos[2],
            ]
            self.last_aruco_cb_time = time.time()

            self.get_logger().info(
                f"ArUco global target -> ({self.last_aruco_global[0]:.3f}, "
                f"{self.last_aruco_global[1]:.3f})",
                throttle_duration_sec=1.0
            )
        else:
            self.aruco_detected = False

    def voltage_callback(self, msg):
        self.battery_voltage = float(msg.data)

    def update_line_offset(self):
        y = self.boundary_offset.y
        now = time.time()

        if abs(y) > self.offset_zero_eps:
            self.last_valid_offset_y = y
            self.offset_zero_since = None
            self.offset_lost_confirmed = False
            return y

        # Reading is (near) zero this cycle
        if self.offset_zero_since is None:
            self.offset_zero_since = now

        if (now - self.offset_zero_since) < self.offset_lost_wait:
            return self.last_valid_offset_y

        # Zero for a full second straight -> genuinely lost the line
        self.offset_lost_confirmed = True
        return 0.0

    # =====================================================================
    # Helpers
    # =====================================================================
    def hold(self, target, yaw=None):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        if (yaw is not None) and self.yaw_kar_bc:
            msg.type_mask = (
                PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW_RATE
            )
            msg.yaw = yaw
        else:
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

    def xy_aligned(self):
        return (self.error_x ** 2 + self.error_y ** 2) ** 0.5 < self.align_threshold

    def trigger(self, value=False):
        """Rising edge (False->True) tells the camera node to save one frame."""
        msg = Bool()
        msg.data = value
        self.bool_pub.publish(msg)

    def publish_mission_complete(self, value: bool):
        msg = Bool()
        msg.data = value
        self.mission_complete_pub.publish(msg)

    def publish_charging_okay(self, value: bool):
        msg = Bool()
        msg.data = value
        self.charging_pub.publish(msg)

    def publish_transfer_status(self, state: int):
        """Publish transfer status to the GUI.
        0 = inflight
        1 = transferring
        2 = transfer complete
        """
        msg = Int32()
        msg.data = state
        self.transfer_pub.publish(msg)

    def _is_line_realign_wp(self, idx):
        return math.isclose(self.mission_body_waypoints[idx][0], 0.0, abs_tol=1e-6)

    def _recompute_remaining_waypoints(self, from_index):
        for i in range(from_index, len(self.mission_body_waypoints)):
            f, r, a = self.mission_body_waypoints[i]
            self.mission_waypoints[i] = self.body_to_world(f, r, a, origin=self.corner_origin)

    def _build_search_offsets(self, home_x, home_y, home_z):
        pts = [[home_x, home_y, home_z]]
        for i in range(8):
            angle = 2 * math.pi * i / 8
            pts.append([
                home_x + self.search_radius * math.cos(angle),
                home_y + self.search_radius * math.sin(angle),
                home_z,
            ])
        return pts

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
            self.publish_transfer_status(0)   # GUI state 0: inflight
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
    # Post-land image transfer
    # =====================================================================
    def _transfer_images(self):
        """
        SCP images from the Xavier's image folder to the base station using
        sshpass for password-based authentication (no SSH key needed).

        Requires sshpass to be installed on the Xavier:
            sudo apt install sshpass

        Credentials/paths are read from scp_config.py placed next to this
        script.
        """
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scp_config.py"
        )

        if not os.path.isfile(config_path):
            self.get_logger().error(
                f"SCP config not found at {config_path} — skipping transfer"
            )
            return

        # Load config as a plain namespace (avoids a full import system)
        cfg = {}
        with open(config_path) as f:
            exec(f.read(), cfg)  # noqa: S102  (intentional; file is local)

        bs_user  = cfg.get("BS_USER", "")
        bs_host  = cfg.get("BS_HOST", "")
        password = cfg.get("BS_PASSWORD", "")
        bs_dest = cfg.get("BS_DEST_DIR", "")
        image_dir = cfg.get("XAVIER_IMAGE_DIR", "")

        if not all([bs_user, bs_host, bs_dest, image_dir, password]):
            self.get_logger().error(
                "scp_config.py is missing one or more required fields "
                "(BS_USER, BS_HOST, BS_DEST_DIR, XAVIER_IMAGE_DIR, BS_PASSWORD) "
                "— skipping"
            )
            return

        # sshpass feeds the password non-interactively to scp
        # -r  : recursive (transfers entire folder)
        # -q  : quiet (suppress progress meter in log)
        # -o StrictHostKeyChecking=no : skip interactive host-key prompt on first connect
        cmd = [
            "sshpass", "-p", password,
            "scp", "-r", "-q", "-o", "StrictHostKeyChecking=no",
            image_dir,
            f"{bs_user}@{bs_host}:{bs_dest}",
        ]

        # Log without exposing the password
        safe_cmd = cmd[:2] + ["****"] + cmd[3:]
        self.get_logger().info(f"Running: {' '.join(safe_cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,   # 2-minute hard cap; raise if folder is large
            )
            if result.returncode == 0:
                self.get_logger().info("Image transfer complete.")
                self.publish_transfer_status(2)
            else:
                self.get_logger().error(
                    f"SCP failed (exit {result.returncode}): {result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            self.get_logger().error("SCP timed out after 120 s — transfer incomplete")
        except FileNotFoundError:
            self.get_logger().error(
                "sshpass not found — install it with: sudo apt install sshpass"
            )
        except Exception as exc:
            self.get_logger().error(f"SCP exception: {exc}")

    # =====================================================================
    # Main loop
    # =====================================================================
    def loop(self):
        # /chargingOkay is published every tick, in every phase:
        # True only while sitting on the pad in CHARGING_STARTED, False otherwise.
        self.publish_charging_okay(self.phase == "CHARGING_STARTED")

        # ── Global per-phase watchdog (2-minute cap on any single phase) ───
        now = time.time()
        if self.phase != self._watchdog_last_phase:
            # Phase just changed (or this is the first tick) — start the clock.
            self._watchdog_last_phase = self.phase
            self.phase_entry_time = now
        elif (
            self.phase not in self.watchdog_exempt_phases
            and (now - self.phase_entry_time) > self.phase_timeout
        ):
            self.get_logger().error(
                f"[watchdog] Phase '{self.phase}' exceeded "
                f"{self.phase_timeout:.0f}s without transitioning -> forcing LAND"
            )
            self.phase = "LAND"
            self._watchdog_last_phase = self.phase
            self.phase_entry_time = now

        if self.phase == "CALIBRATE":
            return

        self.align_threshold = 0.1 + 0.2 * (
            (self.current_pos[2] - self.land_alt) / (self.takeoff_alt - self.land_alt)
        )

        if (
            self.aruco_detected
            and self.last_aruco_cb_time is not None
            and (time.time() - self.last_aruco_cb_time) > self.aruco_stale_timeout
        ):
            self.aruco_detected = False
            self.get_logger().warn("ArUco callback silent -> marking lost")

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
                self.target_yaw = self.locked_yaw
                self.get_logger().info(
                    f"Takeoff altitude reached, HOME latched at {self.home_position} "
                    "-> searching forward for yellow line"
                )
                self.phase = "SEARCH_FORWARD"
                self.target_yaw = self.current_yaw
                self.locked_yaw = self.current_yaw
                self.start_time = time.time()

        # ── SEARCH_FORWARD: fly forward until yellow line ──────────────────
        elif self.phase == "SEARCH_FORWARD":
            self.forward_search_target = self.body_to_world(
                    self.search_forward_distance, 0.0, self.takeoff_alt
                )
            self.hold(self.forward_search_target, self.target_yaw)

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
                self.yaw_kar_bc = True;
                self.boundary_origin = [
                    self.current_pos[0], self.current_pos[1], self.current_pos[2]
                ]
                self.yaw_arena_offset_set = False
                self.get_logger().info(
                    f"Yellow line confirmed -> stopping at {self.boundary_origin} "
                    "-> ALIGN_PERPENDICULAR"
                )
                self.phase = "ALIGN_PERPENDICULAR"
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

        # ── ALIGN_PERPENDICULAR: rotate to square up with the boundary line ─
        elif self.phase == "ALIGN_PERPENDICULAR":
            self.hold(self.boundary_origin, self.target_yaw)

            if not self.yaw_arena_offset_set:
                err_yaw = self.boundary_offset.z
                err_yaw = (err_yaw + math.pi / 2) % math.pi - math.pi / 2

                self.yaw_arena_offset = self.locked_yaw - self.current_yaw + err_yaw
                self.yaw_arena_offset = (self.yaw_arena_offset + math.pi) % (2 * math.pi) - math.pi

                self.yaw_arena_offset_set = True
                self.target_yaw = self.locked_yaw - self.yaw_arena_offset
                self.target_yaw = (self.target_yaw + math.pi) % (2 * math.pi) - math.pi

                self.start_time = time.time()
                return

            elapsed = time.time() - self.start_time
            if elapsed < self.align_hover_time:
                return

            self.boundary_origin = list(self.current_pos)
            self.get_logger().info(
                f"Alignment complete -> origin latched at "
                f"({self.boundary_origin[0]:.2f}, {self.boundary_origin[1]:.2f}) "
                "-> BOUNDARY_HOVER"
            )
            self.phase = "BOUNDARY_HOVER"
            self.start_time = time.time()

        # ── BOUNDARY_HOVER: settle at the line, then search right ──────────
        elif self.phase == "BOUNDARY_HOVER":
            self.hold(self.boundary_origin, self.target_yaw)
            elapsed = time.time() - self.start_time
            if elapsed < self.stop_hover_time:
                self.get_logger().info(
                    f"Settling at yellow line... {self.stop_hover_time - elapsed:.1f}s left"
                )
            else:
                self.last_valid_offset_y = self.boundary_offset.y
                self.offset_zero_since = None
                self.offset_lost_confirmed = False

                self.get_logger().info(
                    f"Searching right for corner, holding offset setpoint "
                    f"{self.line_offset_setpoint:.3f}"
                )
                self.corner_confirm_count = 0
                self.phase = "SEARCH_RIGHT"
                self.start_time = time.time()

        # ── SEARCH_RIGHT: fly right (from boundary stop) until corner,
        #                 correcting forward position to hold line offset ──
        elif self.phase == "SEARCH_RIGHT":
            offset_y = self.update_line_offset()

            if self.offset_lost_confirmed:
                backup_target = self.body_to_world(
                    self.offset_backup_dist, 0.0, self.takeoff_alt,
                    origin=self.current_pos,
                )
                self.hold(backup_target, self.target_yaw)
                self.get_logger().warn(
                    "Line offset lost for 1s straight -> backing off 0.5 m to reacquire",
                    throttle_duration_sec=1.0,
                )
                if abs(self.boundary_offset.y) > self.offset_zero_eps:
                    self.offset_lost_confirmed = False
                    self.offset_zero_since = None

                elapsed = time.time() - self.start_time
                if elapsed > self.search_max_time:
                    self.get_logger().error("Right search time limit exceeded -> LAND")
                    self.phase = "LAND"
                return

            error = offset_y - self.line_offset_setpoint
            forward_correction = self.offset_sign * max(
                -self.max_follow_correction,
                min(self.max_follow_correction, error * self.line_follow_gain),
            )
            self.right_search_target = self.body_to_world(
                forward_correction, self.search_right_distance, self.takeoff_alt,
                origin=self.boundary_origin,
            )
            self.hold(self.right_search_target, self.target_yaw)

            if self.corner_detected_raw:
                self.corner_confirm_count += 1
            else:
                self.corner_confirm_count = 0
            confirmed = self.corner_confirm_count >= self.detect_corner_confirm_needed

            dist_travelled = math.sqrt(
                (self.current_pos[0] - self.boundary_origin[0]) ** 2 +
                (self.current_pos[1] - self.boundary_origin[1]) ** 2
            )
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f"[search-right] dist={dist_travelled:.2f} m  offset_y={offset_y:.3f}  "
                f"setpoint={self.line_offset_setpoint:.3f}  error={error:+.3f}  "
                f"corner_confirm={self.corner_confirm_count}/{self.detect_corner_confirm_needed}",
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
            self.hold(self.corner_origin, self.target_yaw)
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
                if self._is_line_realign_wp(self.wp_index):
                    self.line_realign_confirm_count = 0
                    self.line_realign_applied = False
                    self.phase = "LINE_REALIGN"
                else:
                    self.phase = "HOVER2"
                self.start_time = time.time()

        # ── MOVE2 ────────────────────────────────────────────────────────
        elif self.phase == "MOVE2":
            target = self.mission_waypoints[self.wp_index]
            self.hold(target, self.target_yaw)

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
                self.trigger(True)   # rising edge -> camera snapshot
                if self._is_line_realign_wp(self.wp_index):
                    self.get_logger().info(
                        f"Reached wp{self.wp_index} (back at line) -> checking for realign"
                    )
                    self.line_realign_confirm_count = 0
                    self.line_realign_applied = False
                    self.phase = "LINE_REALIGN"
                else:
                    self.get_logger().info(f"Reached wp{self.wp_index} -> hovering {self.hover_time}s")
                    self.phase = "HOVER2"
                self.start_time = time.time()

        # ── LINE_REALIGN: at a waypoint back near the boundary, re-square
        #                 to the yellow line if it's visible, then continue ──
        elif self.phase == "LINE_REALIGN":
            target = self.mission_waypoints[self.wp_index]
            self.hold(target, self.target_yaw)
            elapsed = time.time() - self.start_time

            if self.boundary_detected_raw:
                self.line_realign_confirm_count += 1
            else:
                self.line_realign_confirm_count = 0
            confirmed = self.line_realign_confirm_count >= self.detect_confirm_needed

            if confirmed and not self.line_realign_applied:
                err_yaw = self.boundary_offset.z
                err_yaw = (err_yaw + math.pi / 2) % math.pi - math.pi / 2

                self.yaw_arena_offset = self.locked_yaw - self.current_yaw + err_yaw
                self.yaw_arena_offset = (self.yaw_arena_offset + math.pi) % (2 * math.pi) - math.pi
                self.target_yaw = self.locked_yaw - self.yaw_arena_offset
                self.target_yaw = (self.target_yaw + math.pi) % (2 * math.pi) - math.pi

                self._recompute_remaining_waypoints(self.wp_index)
                self.line_realign_applied = True
                self.get_logger().info(
                    f"[wp{self.wp_index}] yellow line visible -> re-aligned perpendicular, "
                    "remaining waypoints recomputed"
                )
                self.start_time = time.time()
                return

            settle_needed = self.align_hover_time if self.line_realign_applied else self.line_realign_search_time
            if elapsed >= settle_needed:
                if not self.line_realign_applied:
                    self.get_logger().info(
                        f"[wp{self.wp_index}] no yellow line seen -> continuing without realign"
                    )
                self.get_logger().info(f"--> hovering {self.hover_time}s")
                self.phase = "HOVER2"
                self.start_time = time.time()

        # ── HOVER2 ───────────────────────────────────────────────────────
        elif self.phase == "HOVER2":
            self.trigger(False)  # reset latch
            target = self.mission_waypoints[self.wp_index]
            self.hold(target, self.target_yaw)
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
                if not self.mission_complete_sent:
                    self.publish_mission_complete(True)
                    self.mission_complete_sent = True
                self.phase = "RETURN_HOME"
                self.start_time = time.time()

        # ── RETURN_HOME ──────────────────────────────────────────────────
        elif self.phase == "RETURN_HOME":
            self.hold(self.home_position, self.target_yaw)
            elapsed = time.time() - self.start_time

            dx = self.home_position[0] - self.current_pos[0]
            dy = self.home_position[1] - self.current_pos[1]
            self.get_logger().info(
                f"[return-home] dx={dx:+.2f} dy={dy:+.2f}", throttle_duration_sec=0.5
            )

            if self.aruco_detected:
                self.get_logger().info("ArUco detected en-route home -> ARUCO_ALIGN")
                self.phase = "ARUCO_ALIGN"
                return

            if self.arrived(self.home_position):
                self.get_logger().info("Reached home, no ArUco yet -> ARUCO_SPIRAL_SEARCH")
                self.search_home = list(self.home_position)
                self.search_offsets = self._build_search_offsets(*self.search_home)
                self.search_index = 0
                self.start_time = time.time()
                self.phase = "ARUCO_SPIRAL_SEARCH"
                return

            if elapsed > self.move_timeout:
                self.get_logger().error("Return-home timeout -> ARUCO_SPIRAL_SEARCH here")
                self.search_home = list(self.current_pos)
                self.search_offsets = self._build_search_offsets(*self.search_home)
                self.search_index = 0
                self.start_time = time.time()
                self.phase = "ARUCO_SPIRAL_SEARCH"

        # ── ARUCO_SPIRAL_SEARCH: sweep near home to account for drift ──────
        elif self.phase == "ARUCO_SPIRAL_SEARCH":
            if self.aruco_detected:
                self.get_logger().info(
                    f"ArUco detected during spiral search "
                    f"(point {self.search_index}/{len(self.search_offsets) - 1}) "
                    "-> ARUCO_ALIGN"
                )
                self.phase = "ARUCO_ALIGN"
                return

            if not self.search_offsets:
                self.get_logger().warn(
                    "ARUCO_SPIRAL_SEARCH entered with no offsets -> ARUCO_HOVER_WAIT"
                )
                self.phase = "ARUCO_HOVER_WAIT"
                return

            target = self.search_offsets[self.search_index]
            elapsed = time.time() - self.start_time
            self.hold(target, self.target_yaw)

            dx = target[0] - self.current_pos[0]
            dy = target[1] - self.current_pos[1]
            self.get_logger().info(
                f"[spiral {self.search_index + 1}/{len(self.search_offsets)}] "
                f"dx={dx:+.2f} dy={dy:+.2f}  dwell={elapsed:.1f}/{self.search_dwell:.1f}s",
                throttle_duration_sec=0.5,
            )

            if self.arrived(target) and elapsed >= self.search_dwell:
                self.search_index += 1
                if self.search_index < len(self.search_offsets):
                    self.get_logger().info(
                        f"--> spiral: moving to point {self.search_index + 1}"
                        f"/{len(self.search_offsets)}"
                    )
                    self.start_time = time.time()
                else:
                    self.get_logger().info(
                        "Spiral search complete, ArUco not found -> ARUCO_HOVER_WAIT"
                    )
                    self.hold(self.search_home, self.target_yaw)
                    self.phase = "ARUCO_HOVER_WAIT"

        # ── ARUCO_HOVER_WAIT: hold at home, keep scanning for the marker ────
        elif self.phase == "ARUCO_HOVER_WAIT":
            self.hold([self.home_position[0],self.home_position[1],min(self.current_pos[2]+0.05, self.takeoff_alt)], self.target_yaw)
            if self.aruco_detected:
                self.get_logger().info("ArUco detected -> ARUCO_ALIGN")
                self.phase = "ARUCO_ALIGN"
            else:
                self.get_logger().info(
                    "Hovering at home... waiting for ArUco marker",
                    throttle_duration_sec=1.0,
                )

        # ── ARUCO_ALIGN ──────────────────────────────────────────────────
        elif self.phase == "ARUCO_ALIGN":
            if not self.aruco_detected:
                self.align_alt = None
                if self.last_aruco_global is not None:
                    self.get_logger().warn("ArUco lost during ALIGN -> ARUCO_RECOVER")
                    self.phase = "ARUCO_RECOVER"
                else:
                    self.get_logger().warn("ArUco lost, no global target -> ARUCO_HOVER_WAIT")
                    self.phase = "ARUCO_HOVER_WAIT"
                return

            if self.align_alt is None:
                self.align_alt = self.current_pos[2]

            self.get_logger().info(
                f"Aligning | err_x={self.error_x:+.3f}  err_y={self.error_y:+.3f}  "
                f"alt={self.current_pos[2]:.2f}"
            )

            error_x = max(self.error_x / 2, -0.05) if self.error_x < 0 else min(self.error_x / 2, 0.05)
            error_y = max(self.error_y / 2, -0.05) if self.error_y < 0 else min(self.error_y / 2, 0.05)

            target = [
                self.current_pos[0] + error_x,
                self.current_pos[1] - error_y,
                self.align_alt,
            ]
            self.hold(target, self.target_yaw)

            if self.xy_aligned():
                self.align_alt = None
                self.get_logger().info("Aligned -> ARUCO_HOVER_ALIGNED")
                self.start_time = time.time()
                self.temp_pos = [
                    self.current_pos[0], self.current_pos[1], self.current_pos[2]
                ]
                self.phase = "ARUCO_HOVER_ALIGNED"

        # ── ARUCO_HOVER_ALIGNED ──────────────────────────────────────────
        elif self.phase == "ARUCO_HOVER_ALIGNED":
            self.hold(self.temp_pos, self.target_yaw)
            elapsed = time.time() - self.start_time
            remaining = self.hover_duration - elapsed
            self.get_logger().info(
                f"Hover hold... {remaining:.1f}s left  alt={self.current_pos[2]:.2f}"
            )

            if elapsed >= self.hover_duration:
                if self.current_pos[2] <= self.land_alt:
                    self.get_logger().info("Below land threshold -> LAND")
                    self.phase = "LAND"
                else:
                    new_alt = max(self.current_pos[2] - self.descend_step, 0.4)
                    self.descent_target = [self.temp_pos[0], self.temp_pos[1], new_alt]
                    self.get_logger().info(f"Descending to {new_alt:.2f} m -> ARUCO_DESCEND")
                    self.phase = "ARUCO_DESCEND"

        # ── ARUCO_DESCEND ────────────────────────────────────────────────
        elif self.phase == "ARUCO_DESCEND":
            self.hold(self.descent_target, self.target_yaw)
            dz = abs(self.current_pos[2] - self.descent_target[2])
            self.get_logger().info(
                f"Descending... delta_z={dz:.2f}  alt={self.current_pos[2]:.2f}"
            )

            if not self.aruco_detected:
                self.get_logger().warn("ArUco lost during DESCEND -> ARUCO_RECOVER")
                self.phase = "ARUCO_RECOVER"
                return

            if self.arrived(self.descent_target):
                self.get_logger().info("Descent step complete -> ARUCO_ALIGN")
                self.phase = "ARUCO_ALIGN"

        # ── ARUCO_RECOVER ────────────────────────────────────────────────
        elif self.phase == "ARUCO_RECOVER":
            self.hold(self.last_aruco_global, self.target_yaw)
            dist = (
                (self.current_pos[0] - self.last_aruco_global[0]) ** 2 +
                (self.current_pos[1] - self.last_aruco_global[1]) ** 2
            ) ** 0.5
            self.get_logger().info(
                f"Recovering to last ArUco global pos... dist={dist:.2f}  "
                f"alt={self.current_pos[2]:.2f}"
            )

            if self.aruco_detected:
                self.get_logger().info("ArUco reacquired -> ARUCO_ALIGN")
                self.phase = "ARUCO_ALIGN"
                return

            if self.arrived(self.last_aruco_global):
                self.get_logger().info(
                    "Reached last known ArUco pos, still no marker -> ARUCO_HOVER_WAIT"
                )
                self.phase = "ARUCO_HOVER_WAIT"

        # ── LAND ─────────────────────────────────────────────────────────
        elif self.phase == "LAND":
            self.land()
            self.phase = "AFTER_LAND"
            self.start_time = time.time()

        # ── AFTER_LAND: wait for motors to stop, then SCP images ──────────
        elif self.phase == "AFTER_LAND":
            elapsed = time.time() - self.start_time
            # Give the drone a few seconds to fully settle before touching the FS
            settle_time = 5.0
            if elapsed < settle_time:
                self.get_logger().info(
                    f"Settling after land... {settle_time - elapsed:.1f}s",
                    throttle_duration_sec=1.0,
                )
                return

            if not self.scp_transfer_done:
                self.get_logger().info("Starting image transfer to base station...")
                self.publish_transfer_status(1)   # GUI state 1: transferring
                self._transfer_images()
                self.scp_transfer_done = True

            self.get_logger().info("Transfer done -> CHARGING_STARTED")
            self.initial_battery_voltage = self.battery_voltage
            self.phase = "CHARGING_STARTED"
            self.start_time = time.time()

        # ── CHARGING_STARTED: sit on the pad until the pack has charged ────
        # /chargingOkay is held True for the whole phase by the publisher at
        # the top of loop().
        elif self.phase == "CHARGING_STARTED":
            if self.battery_voltage is None:
                self.get_logger().warn(
                    "Waiting for /esp/total_voltage...",
                    throttle_duration_sec=2.0,
                )
                return

            # No reading had arrived when we entered the state — latch the
            # first one we get as the baseline.
            if self.initial_battery_voltage is None:
                self.initial_battery_voltage = self.battery_voltage
                self.get_logger().info(
                    f"Initial battery voltage = {self.initial_battery_voltage:.2f} V"
                )

            delta = self.battery_voltage - self.initial_battery_voltage
            self.get_logger().info(
                f"Charging... V={self.battery_voltage:.2f} "
                f"(start {self.initial_battery_voltage:.2f}, "
                f"delta {delta:+.2f}/{self.charge_delta_min:.2f}, "
                f"need > {self.charge_voltage_min:.2f})",
                throttle_duration_sec=2.0,
            )

            if delta > self.charge_delta_min and self.battery_voltage > self.charge_voltage_min:
                self.get_logger().info(
                    f"Charge complete at {self.battery_voltage:.2f} V -> DONE"
                )
                self.phase = "DONE"

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
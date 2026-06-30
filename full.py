'''
+x -> right
+y -> forward
+z -> up (altitude)

All movement uses position-hold on /mavros/setpoint_raw/local.
No velocity control — eliminates the velocity->position handoff race condition.

Mission:
  takeoff to 2.5 m, hover 5 s
  forward 1 m -> hover 5 s -> back to centre -> hover 5 s
  right   1 m -> hover 5 s -> back to centre -> hover 5 s
  back    1 m -> hover 5 s -> back to centre -> hover 5 s
  left    1 m -> hover 5 s -> back to centre -> hover 5 s
  climb to 2.0 m -> hover 5 s
  down  to 1.5 m -> hover 5 s
  down  to 1.0 m -> hover 5 s
  up    to 1.5 m -> hover 5 s
  [return home — last waypoint]
    → /mission_complete = True published immediately on departure
    → while flying: actively scan for ArUco
      - if ArUco detected mid-flight → abort waypoint nav, switch to ARUCO_ALIGN
      - if arrived at home with no ArUco → HOVER_WAIT (keep scanning)
    → once aligned → descend in steps → precision land
'''

import rclpy
import time
import subprocess
import os
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf_transformations import euler_from_quaternion
from std_msgs.msg import Bool
import math


class droneControl(Node):
    def __init__(self):
        super().__init__('drone_control')

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10
        )
        # Rising-edge camera trigger (aruco_line_combined.py listens here)
        self.bool_pub = self.create_publisher(Bool, '/take_picture', 10)
        # Signals that the ArUco search phase has begun
        self.mission_complete_pub = self.create_publisher(Bool, '/mission_complete', 10)
        # Signals that transfer is complete
        # self.transfer_pub = self.create_publisher(Bool, '/transfer_complete', 10)

        # /transfer_complete


        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            qos
        )
        self.sub_aruco = self.create_subscription(
            PoseArray,
            '/aruco_poses',
            self.aruco_callback,
            qos
        )

        
        # ── Services ──────────────────────────────────────────────────────────
        self.arm_client     = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client    = self.create_client(SetMode,     '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL,  '/mavros/cmd/takeoff')

        # ── Calibration ───────────────────────────────────────────────────────
        self.init_samples     = []
        self.reference_set    = False
        self.ref_x = self.ref_y = self.ref_z = 0.0
        self.calib_start_time = time.time()
        self.calib_duration   = 3.0
        self.locked_yaw       = None

        # ── General state ─────────────────────────────────────────────────────
        self.current_pos   = [0.0, 0.0, 0.0]
        self.pos_threshold = 0.25
        self.takeoff_alt   = 2.0
        self.counter       = 0
        self.phase         = "CALIBRATE"
        self.start_time    = 0.0
        self.move_timeout  = 20.0

        # ── Waypoint mission (file 1) ─────────────────────────────────────────
        self.hover_time = 3  # seconds per waypoint hover

        # Mission waypoints at 2.5 m altitude: (x_right, y_forward, z_alt)
        # Index 0–15 are the survey pattern; index 16 is the final "return home"
        # waypoint that triggers the ArUco search.
        self.waypoints = [
            (0.0,   0.0,  2.0),  
            (0.0,   1.0,  2.0),   
            (0.0,   2.0,  2.0),   
            (0.0,   3.0,  2.0),   
            (-1.0,   3.0,  2.0),   
            (-1.0,  2.0,  2.0),   
            (-1.0,   1.0,  2.0),   
            (-1.0,  0.0,  2.0),   
            (-2.0,   0.0,  2.0),   #  8  back to centre
            (-2.0,   1.0,  2.0),   #  9  (reuse centre — matches original count)
            (-2.0,  2.0,  2.0),   # 10  climb/down to 2.0 m
            (-2.0,   3.0,  2.0),
            (0.0,   0.0,  2.0),   # 16  ← LAST WAYPOINT — return home / ArUco search
        ]
        self.LAST_WP_INDEX = len(self.waypoints) - 1  # = 16
        self.wp_index = 0

        # Set True once we depart toward the last waypoint (fires /mission_complete)
        self.mission_complete_sent = False

        # type_mask: position only, ignore vel/accel/yaw-rate; keep YAW
        self.pos_type_mask = (
            PositionTarget.IGNORE_VX      |
            PositionTarget.IGNORE_VY      |
            PositionTarget.IGNORE_VZ      |
            PositionTarget.IGNORE_AFX     |
            PositionTarget.IGNORE_AFY     |
            PositionTarget.IGNORE_AFZ     |
            PositionTarget.IGNORE_YAW_RATE|
            PositionTarget.IGNORE_YAW
        )

        # ── ArUco / precision-landing state (file 2) ──────────────────────────
        self.align_threshold    = 0.1
        self.land_alt           = 0.5
        self.descend_step       = 0.3
        self.hover_duration     = 0.5   # seconds to confirm alignment before descending
        self.aruco_detected     = False
        self.error_x            = 0.0
        self.error_y            = 0.0
        self.last_aruco_global  = None  # [x, y, z] in calibrated frame
        self.last_aruco_cb_time = None
        self.aruco_stale_timeout = 1.0
        self.descent_target     = None
        self.temp_pos           = None  # XY held during HOVER_ALIGNED

        # ── Post-land image transfer ───────────────────────────────────────────
        self.scp_transfer_done = False

        self.timer = self.create_timer(0.1, self.loop)
        self.get_logger().info("Calibrating reference frame for 3 seconds…")

    # =========================================================================
    # Callbacks
    # =========================================================================

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
            self.get_logger().info(f"Yaw locked at {math.degrees(yaw):.1f}°")

        if not self.reference_set:
            elapsed = time.time() - self.calib_start_time
            self.init_samples.append((raw_x, raw_y, raw_z))
            print(f"\rCalibrating… {elapsed:.1f}s", end='')

            if elapsed >= self.calib_duration:
                n = len(self.init_samples)
                self.ref_x = sum(p[0] for p in self.init_samples) / n
                self.ref_y = sum(p[1] for p in self.init_samples) / n
                self.ref_z = sum(p[2] for p in self.init_samples) / n
                self.reference_set = True
                self.phase = "INIT"
                print()
                self.get_logger().info(
                    f"Reference locked → ({self.ref_x:.3f}, "
                    f"{self.ref_y:.3f}, {self.ref_z:.3f})"
                )
            return

        self.current_pos[0] = raw_x - self.ref_x
        self.current_pos[1] = raw_y - self.ref_y
        self.current_pos[2] = raw_z - self.ref_z

    def aruco_callback(self, msg):
        if msg.poses:
            self.aruco_detected = True
            self.error_x = msg.poses[0].position.x
            self.error_y = msg.poses[0].position.y

            # Convert camera-frame error to global calibrated position
            self.last_aruco_global = [
                self.current_pos[0] + self.error_x,
                self.current_pos[1] - self.error_y,
                self.current_pos[2],
            ]
            self.last_aruco_cb_time = time.time()

            self.get_logger().info(
                f"ArUco global target → ({self.last_aruco_global[0]:.3f}, "
                f"{self.last_aruco_global[1]:.3f})",
                throttle_duration_sec=1.0
            )
        else:
            self.aruco_detected = False

    # =========================================================================
    # Helpers
    # =========================================================================

    def hold(self, target):
        """Publish a position setpoint in the calibrated frame."""
        msg = PositionTarget()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask        = self.pos_type_mask
        # msg.yaw              = math.radians(90)
        msg.position.x       = target[0] + self.ref_x
        msg.position.y       = target[1] + self.ref_y
        msg.position.z       = target[2] + self.ref_z
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
        """
        Publish a Bool to /take_picture.
        Rising edge (False→True) : camera saves one frame immediately.
        False                    : resets the latch for the next trigger.
        """
        msg = Bool()
        msg.data = value
        self.bool_pub.publish(msg)

    def publish_mission_complete(self, value: bool):
        msg = Bool()
        msg.data = value
        self.mission_complete_pub.publish(msg)

    # =========================================================================
    # Services
    # =========================================================================

    def takeoff(self, altitude):
        if self.takeoff_client.wait_for_service(timeout_sec=2.0):
            req           = CommandTOL.Request()
            req.altitude  = altitude
            req.min_pitch = 0.0
            req.yaw       = 0.0
            req.latitude  = 0.0
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
        self.get_logger().info("Switching to LAND mode…")
        if self.mode_client.wait_for_service(timeout_sec=2.0):
            req = SetMode.Request()
            req.custom_mode = "LAND"
            self.mode_client.call_async(req)
        self.phase = "DONE"
    # =========================================================================
    # Post-land image transfer
    # =========================================================================

    def _transfer_images(self):
        """
        SCP all images from the Xavier's image folder to the base station
        using sshpass for password-based authentication (no SSH key needed).

        Requires sshpass to be installed on the Xavier:
            sudo apt install sshpass

        Credentials are read from scp_config.py placed next to this script.
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

        bs_user   = cfg.get("BS_USER", "")
        bs_host   = cfg.get("BS_HOST", "")
        bs_dest   = cfg.get("BS_DEST_DIR", "")
        image_dir = cfg.get("XAVIER_IMAGE_DIR", "")
        password  = cfg.get("BS_PASSWORD", "")

        if not all([bs_user, bs_host, bs_dest, image_dir, password]):
            self.get_logger().error(
                "scp_config.py is missing one or more required fields "
                "(BS_USER, BS_HOST, BS_DEST_DIR, XAVIER_IMAGE_DIR, BS_PASSWORD) — skipping"
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

    # =========================================================================
    # Main loop
    # =========================================================================

    def loop(self):
        if self.phase == "CALIBRATE":
            return

        self.align_threshold = 0.1 + 0.2*((self.current_pos[2]-self.land_alt)/(self.takeoff_alt - self.land_alt))
        # ── Stale ArUco check ─────────────────────────────────────────────────
        if (
            self.aruco_detected
            and self.last_aruco_cb_time is not None
            and (time.time() - self.last_aruco_cb_time) > self.aruco_stale_timeout
        ):
            self.aruco_detected = False
            self.get_logger().warn("ArUco callback silent → marking lost")

        # ── INIT: GUIDED → arm → takeoff ─────────────────────────────────────
        if self.phase == "INIT":
            if self.counter == 10:
                self.get_logger().info("Switching to GUIDED…")
                if self.mode_client.wait_for_service(timeout_sec=2.0):
                    req = SetMode.Request()
                    req.custom_mode = "GUIDED"
                    self.mode_client.call_async(req)

            elif self.counter == 30:
                self.get_logger().info("Arming…")
                if self.arm_client.wait_for_service(timeout_sec=2.0):
                    req = CommandBool.Request()
                    req.value = True
                    self.arm_client.call_async(req)

            elif self.counter == 50:
                self.get_logger().info(f"Sending takeoff to {self.takeoff_alt} m…")
                self.takeoff(self.takeoff_alt)

        # ── TAKEOFF ───────────────────────────────────────────────────────────
        elif self.phase == "TAKEOFF":
            self.get_logger().info(f"Altitude = {self.current_pos[2]:.2f} m")
            if self.current_pos[2] >= (self.takeoff_alt - self.pos_threshold):
                self.get_logger().info("Takeoff altitude reached → starting mission")
                self.wp_index = 0
                self.phase = "MOVE"
                self.start_time = time.time()

        # ── MOVE ─────────────────────────────────────────────────────────────
        # Handles both normal survey waypoints AND the last waypoint.
        # When flying toward the last waypoint, /mission_complete is published
        # and the ArUco detector is checked every tick.
        elif self.phase == "MOVE":
            target = self.waypoints[self.wp_index]
            self.hold(target)

            # ── Publish /mission_complete on departure toward last waypoint ───
            if self.wp_index == self.LAST_WP_INDEX and not self.mission_complete_sent:
                self.get_logger().info(
                    "Heading to last waypoint — publishing /mission_complete = True"
                )
                self.publish_mission_complete(True)
                self.mission_complete_sent = True

            # ── Timeout guard ─────────────────────────────────────────────────
            elapsed_move = time.time() - self.start_time
            if elapsed_move > self.move_timeout:
                self.get_logger().error(
                    f"[wp{self.wp_index}] Timeout! Landing…"
                )
                self.phase = "LAND_SAFE"
                return

            dx = target[0] - self.current_pos[0]
            dy = target[1] - self.current_pos[1]
            dz = target[2] - self.current_pos[2]
            self.get_logger().info(
                f"[wp{self.wp_index}] moving  dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f}"
            )

            # ── ArUco intercept on last waypoint leg ──────────────────────────
            if self.wp_index == self.LAST_WP_INDEX and self.aruco_detected:
                self.get_logger().info(
                    "ArUco detected en-route to last waypoint → ARUCO_ALIGN"
                )
                self.phase = "ARUCO_ALIGN"
                return

            # ── Normal arrival ────────────────────────────────────────────────
            if self.arrived(target):
                self.trigger(True)   # rising edge → camera snapshot

                if self.wp_index == self.LAST_WP_INDEX:
                    # Reached home but no ArUco yet — hover and keep scanning
                    self.get_logger().info(
                        "Arrived at last waypoint, no ArUco yet → ARUCO_HOVER_WAIT"
                    )
                    self.phase = "ARUCO_HOVER_WAIT"
                else:
                    self.get_logger().info(
                        f"Reached wp{self.wp_index} → hovering {self.hover_time}s"
                    )
                    self.phase = "HOVER"
                    self.start_time = time.time()

        # ── HOVER (survey waypoints 0–15) ─────────────────────────────────────
        elif self.phase == "HOVER":
            self.trigger(False)  # reset latch
            target = self.waypoints[self.wp_index]
            elapsed = time.time() - self.start_time
            self.hold(target)

            if elapsed < self.hover_time:
                self.get_logger().info(
                    f"[wp{self.wp_index}] hovering… {self.hover_time - elapsed:.1f}s left"
                )
            elif self.wp_index + 1 < len(self.waypoints):
                self.wp_index += 1
                self.get_logger().info(
                    f"--> moving to wp{self.wp_index} {self.waypoints[self.wp_index]}"
                )
                self.phase = "MOVE"
                self.start_time = time.time()
            else:
                # Shouldn't happen — last wp is handled in MOVE, but safety net
                self.get_logger().info("Mission complete → landing")
                self.phase = "LAND_SAFE"

        # =====================================================================
        # ── ArUco precision-landing phases (from file 2) ─────────────────────
        # =====================================================================

        # ── ARUCO_HOVER_WAIT: arrived at home, scanning for marker ───────────
        elif self.phase == "ARUCO_HOVER_WAIT":
            self.hold(self.waypoints[self.LAST_WP_INDEX])
            if self.aruco_detected:
                self.get_logger().info("ArUco detected → ARUCO_ALIGN")
                self.phase = "ARUCO_ALIGN"
            else:
                self.get_logger().info(
                    "Hovering at home… waiting for ArUco marker",
                    throttle_duration_sec=1.0
                )

        # ── ARUCO_ALIGN ───────────────────────────────────────────────────────
        elif self.phase == "ARUCO_ALIGN":
            if not self.aruco_detected:
                if self.last_aruco_global is not None:
                    self.get_logger().warn("ArUco lost during ALIGN → ARUCO_RECOVER")
                    self.phase = "ARUCO_RECOVER"
                else:
                    self.get_logger().warn("ArUco lost, no global target → ARUCO_HOVER_WAIT")
                    self.phase = "ARUCO_HOVER_WAIT"
                return

            self.get_logger().info(
                f"Aligning | err_x={self.error_x:+.3f}  err_y={self.error_y:+.3f}  "
                f"alt={self.current_pos[2]:.2f}"
            )

            # Clamp correction to ±0.1 m per tick
            error_x = max(self.error_x / 2, -0.05) if self.error_x < 0 else min(self.error_x / 2, 0.05)
            error_y = max(self.error_y / 2, -0.05) if self.error_y < 0 else min(self.error_y / 2, 0.05)

            target = [
                self.current_pos[0] + error_x,
                self.current_pos[1] - error_y,
                self.current_pos[2],
            ]
            self.hold(target)

            if self.xy_aligned():
                self.get_logger().info("Aligned → ARUCO_HOVER_ALIGNED")
                self.start_time = time.time()
                self.temp_pos = [
                    self.current_pos[0],
                    self.current_pos[1],
                    self.current_pos[2],
                ]
                self.phase = "ARUCO_HOVER_ALIGNED"

        # ── ARUCO_HOVER_ALIGNED ───────────────────────────────────────────────
        elif self.phase == "ARUCO_HOVER_ALIGNED":
            self.hold(self.temp_pos)
            elapsed   = time.time() - self.start_time
            remaining = self.hover_duration - elapsed
            self.get_logger().info(
                f"Hover hold… {remaining:.1f}s left  alt={self.current_pos[2]:.2f}"
            )

            if elapsed >= self.hover_duration:
                if self.current_pos[2] <= self.land_alt:
                    self.get_logger().info("Below land threshold → LAND")
                    self.phase = "LAND"
                else:
                    new_alt = max(self.current_pos[2] - self.descend_step, 0.4)
                    self.descent_target = [self.temp_pos[0], self.temp_pos[1], new_alt]
                    self.get_logger().info(f"Descending to {new_alt:.2f} m → ARUCO_DESCEND")
                    self.phase = "ARUCO_DESCEND"

        # ── ARUCO_DESCEND ─────────────────────────────────────────────────────
        elif self.phase == "ARUCO_DESCEND":
            self.hold(self.descent_target)
            dz = abs(self.current_pos[2] - self.descent_target[2])
            self.get_logger().info(
                f"Descending… delta_z={dz:.2f}  alt={self.current_pos[2]:.2f}"
            )

            if not self.aruco_detected:
                self.get_logger().warn("ArUco lost during DESCEND → ARUCO_RECOVER")
                self.phase = "ARUCO_RECOVER"
                return

            if self.arrived(self.descent_target):
                self.get_logger().info("Descent step complete → ARUCO_ALIGN")
                self.phase = "ARUCO_ALIGN"

        # ── ARUCO_RECOVER ─────────────────────────────────────────────────────
        elif self.phase == "ARUCO_RECOVER":
            self.hold(self.last_aruco_global)
            dist = (
                (self.current_pos[0] - self.last_aruco_global[0]) ** 2 +
                (self.current_pos[1] - self.last_aruco_global[1]) ** 2
            ) ** 0.5
            self.get_logger().info(
                f"Recovering to last ArUco global pos… dist={dist:.2f}  "
                f"alt={self.current_pos[2]:.2f}"
            )

            if self.aruco_detected:
                self.get_logger().info("ArUco reacquired → ARUCO_ALIGN")
                self.phase = "ARUCO_ALIGN"
                return

            if self.arrived(self.last_aruco_global):
                self.get_logger().info(
                    "Reached last known ArUco pos, still no marker → ARUCO_HOVER_WAIT"
                )
                self.phase = "ARUCO_HOVER_WAIT"

        # ── LAND (precision land after descent steps) ─────────────────────────
        elif self.phase == "LAND":
            self.land()
            self.phase = "AFTER_LAND"
            self.start_time = time.time()

        # ── LAND_SAFE (timeout / emergency fallback) ──────────────────────────
        elif self.phase == "LAND_SAFE":
            self.land()
            self.phase = "AFTER_LAND"
            self.start_time = time.time()
        
        # ── AFTER_LAND: wait for motors to fully stop, then SCP images ────────
        elif self.phase == "AFTER_LAND":
            elapsed = time.time() - self.start_time
            # Give the drone a few seconds to fully settle before touching the FS
            settle_time = 5.0
            if elapsed < settle_time:
                self.get_logger().info(
                    f"Settling after land... {settle_time - elapsed:.1f}s",
                    throttle_duration_sec=1.0
                )
                return

            if not self.scp_transfer_done:
                self.get_logger().info("Starting image transfer to base station...")
                self._transfer_images()
                self.scp_transfer_done = True

            self.phase = "DONE"


        # ── DONE ─────────────────────────────────────────────────────────────
        elif self.phase == "DONE":
            pass  # node keeps spinning; kill externally

        self.counter += 1


def main():
    rclpy.init()
    node = droneControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


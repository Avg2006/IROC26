
import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from rclpy.qos import QoSProfile, ReliabilityPolicy


class droneControl(Node):
    def __init__(self):
        super().__init__('drone_control')

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 10
        )

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
        self.arm_client    = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client   = self.create_client(SetMode,     '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # ── Calibration ───────────────────────────────────────────────────────
        self.init_samples    = []
        self.reference_set   = False
        self.ref_x = self.ref_y = self.ref_z = 0.0
        self.calib_start_time = time.time()
        self.calib_duration   = 3.0

        # ── State ─────────────────────────────────────────────────────────────
        self.current_pos     = [0.0, 0.0, 0.0]   # calibrated frame
        self.pos_threshold   = 0.15               # metres, arrival tolerance
        self.align_threshold = 0.1                # metres, XY alignment tolerance
        self.takeoff_alt     = 2.0
        self.land_alt        = 0.3                # below this → switch to LAND
        self.descend_step    = 0.3                # metres per descent step
        self.hover_duration  = 1.5                # seconds to hold after align
        self.kp              = 0.5                # proportional gain for alignment

        self.counter     = 0
        self.phase       = "CALIBRATE"
        self.start_time  = 0.0

        # ArUco
        self.aruco_detected = False
        self.error_x = 0.0   # lateral error  (camera frame → drone body +x)
        self.error_y = 0.0   # longitudinal error

        # type_mask: ignore everything except XYZ position
        self.pos_type_mask = (
            PositionTarget.IGNORE_VX  |
            PositionTarget.IGNORE_VY  |
            PositionTarget.IGNORE_VZ  |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW |
            PositionTarget.IGNORE_YAW_RATE
        )

        self.timer = self.create_timer(0.1, self.loop)
        self.get_logger().info("Calibrating reference frame for 3 seconds…")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def pose_callback(self, msg):
        raw_x = msg.pose.position.x
        raw_y = msg.pose.position.y
        raw_z = msg.pose.position.z

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
            # error_x / error_y = displacement of marker in camera/body frame
            self.error_x = msg.poses[0].position.x
            self.error_y = msg.poses[0].position.y
        else:
            self.aruco_detected = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def hold(self, target):
        """Publish a position setpoint in the calibrated frame."""
        msg = PositionTarget()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask       = self.pos_type_mask
        msg.position.x      = target[0] + self.ref_x
        msg.position.y      = target[1] + self.ref_y
        msg.position.z      = target[2] + self.ref_z
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

    # ── Services ──────────────────────────────────────────────────────────────

    def takeoff(self, altitude):
        if self.takeoff_client.wait_for_service(timeout_sec=2.0):
            req = CommandTOL.Request()
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

    # ── Main loop ─────────────────────────────────────────────────────────────

    def loop(self):
        if self.phase == "CALIBRATE":
            return

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

        # ── TAKEOFF: wait until cruise altitude reached ───────────────────────
        elif self.phase == "TAKEOFF":
            self.get_logger().info(f"Altitude = {self.current_pos[2]:.2f} m")
            if self.current_pos[2] >= (self.takeoff_alt - self.pos_threshold):
                self.get_logger().info("Cruise altitude reached → HOVER_WAIT")
                self.phase = "HOVER_WAIT"

        # ── HOVER_WAIT: hold position; wait for ArUco ────────────────────────
        elif self.phase == "HOVER_WAIT":
            self.hold([self.current_pos[0], self.current_pos[1], self.current_pos[2]])
            if self.aruco_detected:
                self.get_logger().info("ArUco detected → ALIGN")
                self.phase = "ALIGN"
            else:
                self.get_logger().info("Hovering… waiting for ArUco marker", throttle_duration_sec=1.0)

        # ── ALIGN: proportional XY correction toward marker ──────────────────
        elif self.phase == "ALIGN":
            if not self.aruco_detected:
                self.get_logger().warn("ArUco lost! Holding position…")
                self.hold([self.current_pos[0], self.current_pos[1], self.current_pos[2]])
                return

            # Compute corrected XY target (stay at current altitude)
            target = [
                self.current_pos[0] - self.error_y/2,  # cam Y  → body X
                self.current_pos[1] - self.error_x/2,  # cam X  → body Y
                self.current_pos[2],
            ]
            self.hold(target)

            self.get_logger().info(
                f"Aligning | err_x={self.error_x:+.3f}  err_y={self.error_y:+.3f}  "
                f"alt={self.current_pos[2]:.2f}"
            )

            if self.xy_aligned():
                self.get_logger().info("Aligned → HOVER_ALIGNED")
                self.start_time = time.time()
                self.phase = "HOVER_ALIGNED"

        # ── HOVER_ALIGNED: hold 1.5 s then decide next action ────────────────
        elif self.phase == "HOVER_ALIGNED":
            self.hold([self.current_pos[0], self.current_pos[1], self.current_pos[2]])
            elapsed = time.time() - self.start_time
            remaining = self.hover_duration - elapsed
            self.get_logger().info(f"Hover hold… {remaining:.1f}s left  alt={self.current_pos[2]:.2f}")

            if elapsed >= self.hover_duration:
                if self.current_pos[2] < self.land_alt:
                    self.get_logger().info("Below land threshold → LAND")
                    self.phase = "LAND"
                else:
                    new_alt = max(
                        self.current_pos[2] - self.descend_step,
                        0.0
                    )
                    self.get_logger().info(f"Descending to {new_alt:.2f} m → DESCEND")
                    self.descent_target = [
                        self.current_pos[0],
                        self.current_pos[1],
                        new_alt,
                    ]
                    self.phase = "DESCEND"

        # ── DESCEND: move to new altitude, then re-align ─────────────────────
        elif self.phase == "DESCEND":
            self.hold(self.descent_target)
            dz = abs(self.current_pos[2] - self.descent_target[2])
            self.get_logger().info(f"Descending… delta_z={dz:.2f}  alt={self.current_pos[2]:.2f}")

            if self.arrived(self.descent_target):
                self.get_logger().info("Descent complete → ALIGN")
                self.phase = "ALIGN"

        # ── LAND ─────────────────────────────────────────────────────────────
        elif self.phase == "LAND":
            self.land()

        self.counter += 1


def main():
    rclpy.init()
    node = droneControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
'''
+x -> right
+y -> forward
+z -> up (altitude)

Movement and hover both use position-hold on /mavros/setpoint_raw/local.
No velocity control — eliminates the velocity->position handoff race condition.

Mission:
  takeoff to 1.5 m, hover 5 s
  forward 1 m -> hover 5 s -> back to centre -> hover 5 s
  right   1 m -> hover 5 s -> back to centre -> hover 5 s
  back    1 m -> hover 5 s -> back to centre -> hover 5 s
  left    1 m -> hover 5 s -> back to centre -> hover 5 s
  climb to 2.0 m -> hover 5 s
  down  to 1.5 m -> hover 5 s
  down  to 1.0 m -> hover 5 s
  up    to 1.5 m -> hover 5 s
  land
'''
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from rclpy.qos import QoSProfile, ReliabilityPolicy
from time import time


class droneControl(Node):
    def __init__(self):
        super().__init__('drone_control')

        # Publishers
        self.pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Subscriber
        self.sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            qos
        )

        # Services
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # Calibration
        self.init_samples = []
        self.reference_set = False
        self.ref_x = 0.0
        self.ref_y = 0.0
        self.ref_z = 0.0
        self.calib_start_time = time()
        self.calib_duration = 3.0

        # State
        self.current_pos = [0.0, 0.0, 0.0]
        self.pos_threshold = 0.25
        self.takeoff_alt = 1.5
        self.counter = 0
        self.phase = "CALIBRATE"
        self.hover_time = 5.0
        self.start_time = 0.0

        # Mission waypoints: (x_right, y_forward, z_alt)
        self.waypoints = [
            (0.0,  0.0, 1.5),   # 0  settle at centre after takeoff
            (0.0,  10.0, 1.5),   # 1  forward 1 m
            (3.0,  10.0, 1.5),   # 2  back to centre
            (3.0,  0.0, 1.5),   # 3  right 1 m
            (6.0,  0.0, 1.5),   # 4  back to centre
            (6.0, 10.0, 1.5),   # 5  back 1 m
            (9.0,  10.0, 1.5),   # 6  back to centre
            (0.0, 0.0, 1.5),   # 7  left 1 m
            # (0.0,  0.0, 1.5),   # 8  back to centre
            # (0.0,  0.0, 2.0),   # 9  climb to 2.0 m
            # (0.0,  0.0, 1.5),   # 10 down to 1.5 m
            # (0.0,  0.0, 1.0),   # 11 down to 1.0 m
            # (0.0,  0.0, 1.5),   # 12 up to 1.5 m
        ]
        self.wp_index = 0

        # type_mask: position only, ignore vel/accel/yaw
        self.pos_type_mask = (
            PositionTarget.IGNORE_VX |
            PositionTarget.IGNORE_VY |
            PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW |
            PositionTarget.IGNORE_YAW_RATE
        )

        self.timer = self.create_timer(0.1, self.loop)
        self.get_logger().info("Calibrating reference frame for 3 seconds...")

    # ─── Calibration ─────────────────────────────────────────────────────────────
    def pose_callback(self, msg):
        raw_x = msg.pose.position.x
        raw_y = msg.pose.position.y
        raw_z = msg.pose.position.z

        if not self.reference_set:
            elapsed = time() - self.calib_start_time
            self.init_samples.append((raw_x, raw_y, raw_z))
            print(f"\rCalibrating... {elapsed:.1f}s", end='')

            if elapsed >= self.calib_duration:
                self.ref_x = sum(p[0] for p in self.init_samples) / len(self.init_samples)
                self.ref_y = sum(p[1] for p in self.init_samples) / len(self.init_samples)
                self.ref_z = sum(p[2] for p in self.init_samples) / len(self.init_samples)
                self.reference_set = True
                self.phase = "INIT"
                print()
                self.get_logger().info(
                    f"Reference locked -> ref=({self.ref_x:.3f}, {self.ref_y:.3f}, {self.ref_z:.3f})"
                )
            return

        self.current_pos[0] = raw_x - self.ref_x
        self.current_pos[1] = raw_y - self.ref_y
        self.current_pos[2] = raw_z - self.ref_z

    # ─── Position hold helper ────────────────────────────────────────────────────
    def hold(self, target):
        """Publish a position setpoint in the calibrated frame."""
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = self.pos_type_mask
        msg.position.x = target[0] + self.ref_x
        msg.position.y = target[1] + self.ref_y
        msg.position.z = target[2] + self.ref_z
        self.pub.publish(msg)

    # ─── Arrival check ───────────────────────────────────────────────────────────
    def arrived(self, target):
        dist = (
            (self.current_pos[0] - target[0]) ** 2 +
            (self.current_pos[1] - target[1]) ** 2 +
            (self.current_pos[2] - target[2]) ** 2
        ) ** 0.5
        return dist < self.pos_threshold

    # ─── Services ────────────────────────────────────────────────────────────────
    def takeoff(self, altitude=1.5):
        if self.takeoff_client.wait_for_service(timeout_sec=2.0):
            req = CommandTOL.Request()
            req.altitude = altitude
            req.min_pitch = 0.0
            req.yaw = 0.0
            req.latitude = 0.0
            req.longitude = 0.0
            future = self.takeoff_client.call_async(req)
            future.add_done_callback(self.takeoff_response_cb)
        else:
            self.get_logger().error("Takeoff service not available")

    def takeoff_response_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info("Takeoff command accepted")
            self.phase = "TAKEOFF"
        else:
            self.get_logger().warn(f"Takeoff failed, result code: {result.result}")

    def land(self):
        self.get_logger().info("Landing...")
        if self.mode_client.wait_for_service(timeout_sec=2.0):
            req = SetMode.Request()
            req.custom_mode = "LAND"
            self.mode_client.call_async(req)
        self.phase = "DONE"

    # ─── Main loop ───────────────────────────────────────────────────────────────
    def loop(self):
        if self.phase == "CALIBRATE":
            return

        # INIT: GUIDED -> arm -> takeoff
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
                self.get_logger().info("Sending takeoff command...")
                self.takeoff(altitude=self.takeoff_alt)

        # TAKEOFF: wait until target altitude reached
        elif self.phase == "TAKEOFF":
            self.get_logger().info(f"Altitude = {self.current_pos[2]:.2f}")
            if self.current_pos[2] > (self.takeoff_alt - self.pos_threshold):
                self.get_logger().info(f"Reached {self.takeoff_alt} m -> starting mission")
                self.wp_index = 0
                self.phase = "MOVE"

        # MOVE: command target position and wait until arrived
        elif self.phase == "MOVE":
            target = self.waypoints[self.wp_index]
            self.hold(target)   # keep publishing every loop tick

            dx = target[0] - self.current_pos[0]
            dy = target[1] - self.current_pos[1]
            dz = target[2] - self.current_pos[2]
            self.get_logger().info(
                f"[wp{self.wp_index}] moving  dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f}"
            )

            if self.arrived(target):
                self.get_logger().info(
                    f"Reached wp{self.wp_index} --> hovering {self.hover_time}s"
                )
                self.phase = "HOVER"
                self.start_time = time()

        # HOVER: hold position for hover_time seconds, then advance
        elif self.phase == "HOVER":
            target = self.waypoints[self.wp_index]
            elapsed = time() - self.start_time
            self.hold(target)   # keep publishing every loop tick

            if elapsed < self.hover_time:
                self.get_logger().info(
                    f"[wp{self.wp_index}] hovering... {self.hover_time - elapsed:.1f}s left"
                )
            elif self.wp_index + 1 < len(self.waypoints):
                self.wp_index += 1
                self.get_logger().info(
                    f"--> moving to wp{self.wp_index} {self.waypoints[self.wp_index]}"
                )
                self.phase = "MOVE"
            else:
                self.get_logger().info("Mission complete --> Landing")
                self.phase = "LAND"

        # LAND
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
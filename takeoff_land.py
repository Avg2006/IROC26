'''
+x -> right
+y -> forward
+z -> up (altitude)

Movement uses VELOCITY control on /mavros/setpoint_velocity/cmd_vel.
Hover uses position-hold on /mavros/setpoint_raw/local (your proven method).

Mission:
  takeoff to 1.5 m, hover 5 s
  forward 1 m -> hover 5 s -> back to centre -> hover 5 s
  right   1 m -> hover 5 s -> back to centre -> hover 5 s
  back    1 m -> hover 5 s -> back to centre -> hover 5 s
  right   1 m -> hover 5 s -> back to centre -> hover 5 s
  climb to 2.0 m -> hover 5 s
  down  to 1.5 m -> hover 5 s
  down  to 1.0 m -> hover 5 s
  up    to 1.5 m -> hover 5 s
  land
'''
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from rclpy.qos import QoSProfile, ReliabilityPolicy
from time import time


class droneControl(Node):
    def __init__(self):
        super().__init__('drone_control')

        # Publishers
        self.pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.pubVel = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

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

        # Velocity-control gains / limits
        self.kp = 0.8            # error (m) -> velocity (m/s)
        self.move_speed = 0.3    # max horizontal speed (m/s)
        self.climb_speed = 0.3   # max vertical speed   (m/s)

        # Mission waypoints in the calibrated frame: (x_right, y_forward, z_alt)
        # A hover of self.hover_time is performed at every waypoint.
        self.waypoints = [
            (0.0,  0.0, 1.5),   # 0  settle at centre after takeoff
            (0.0,  1.0, 1.5),   # 1  forward 1 m
            (0.0,  0.0, 1.5),   # 2  back to centre
            (1.0,  0.0, 1.5),   # 3  right 1 m
            (0.0,  0.0, 1.5),   # 4  back to centre
            (0.0, -1.0, 1.5),   # 5  back 1 m
            (0.0,  0.0, 1.5),   # 6  back to centre
            (-1.0,  0.0, 1.5),   # 7 left 1 m
            (0.0,  0.0, 1.5),   # 8  back to centre
            (0.0,  0.0, 2.0),   # 9  climb to 2.0 m
            (0.0,  0.0, 1.5),   # 10 down to 1.5 m
            (0.0,  0.0, 1.0),   # 11 down to 1.0 m
            (0.0,  0.0, 1.5),   # 12 up to 1.5 m
        ]
        self.wp_index = 0

        # Reusable type_mask: position only, ignore vel/accel/yaw
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

    # ─── Calibration ────────────────────────────────────────────────────────────
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

    # ─── Movement helpers ─────────────────────────────────────────────────────────
    def _make_pos_target(self, x, y, z):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = self.pos_type_mask
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        return msg

    def hold(self, target):
        """Position-hold at a waypoint (target given in calibrated frame)."""
        msg = self._make_pos_target(
            target[0] + self.ref_x,
            target[1] + self.ref_y,
            target[2] + self.ref_z
        )
        self.pub.publish(msg)

    def publish_velocity(self, vx, vy, vz):
        """Velocity command in local ENU frame (x=right, y=forward, z=up)."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        self.pubVel.publish(msg)

    def _axis_velocity(self, error, max_speed):
        v = self.kp * error
        if v > max_speed:
            v = max_speed
        elif v < -max_speed:
            v = -max_speed
        return v

    def drive_towards(self, target):
        """P-controller: drive to a waypoint using velocity setpoints (slows near target)."""
        vx = self._axis_velocity(target[0] - self.current_pos[0], self.move_speed)
        vy = self._axis_velocity(target[1] - self.current_pos[1], self.move_speed)
        vz = self._axis_velocity(target[2] - self.current_pos[2], self.climb_speed)
        self.publish_velocity(vx, vy, vz)

    def distance(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) ** 0.5

    def arrived(self, target):
        return self.distance(self.current_pos, target) < self.pos_threshold

    # ─── Services ──────────────────────────────────────────────────────────────────
    def takeoff(self, altitude=1.0):
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

    # ─── Main loop ───────────────────────────────────────────────────────────────────
    def loop(self):
        if self.phase == "CALIBRATE":
            return

        # Steps 1-3: GUIDED -> arm -> takeoff (timed by counter)
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

        # Step 4: climb until target altitude reached
        elif self.phase == "TAKEOFF":
            self.get_logger().info(f"Altitude = {self.current_pos[2]:.2f}")
            if self.current_pos[2] > (self.takeoff_alt - self.pos_threshold):
                self.get_logger().info(f"Reached {self.takeoff_alt} m -> starting mission")
                self.wp_index = 0
                self.phase = "HOVER"
                self.start_time = time()

        # Hover at the current waypoint, then advance to the next
        elif self.phase == "HOVER":
            target = self.waypoints[self.wp_index]
            elapsed = time() - self.start_time
            if elapsed < self.hover_time:
                self.get_logger().info(
                    f"[wp{self.wp_index}] hovering... {self.hover_time - elapsed:.1f}s left"
                )
                self.hold(target)
            elif self.wp_index + 1 < len(self.waypoints):
                self.wp_index += 1
                self.get_logger().info(
                    f"--> moving to wp{self.wp_index} {self.waypoints[self.wp_index]}"
                )
                self.phase = "MOVE"
            else:
                self.get_logger().info("Mission complete --> Landing")
                self.phase = "LAND"

        # Move toward the current waypoint using velocity
        elif self.phase == "MOVE":
            target = self.waypoints[self.wp_index]
            if self.arrived(target):
                self.publish_velocity(0.0, 0.0, 0.0)   # brake
                self.get_logger().info(f"Reached wp{self.wp_index} --> hover {self.hover_time}s")
                self.phase = "HOVER"
                self.start_time = time()
            else:
                ex = target[0] - self.current_pos[0]
                ey = target[1] - self.current_pos[1]
                ez = target[2] - self.current_pos[2]
                self.get_logger().info(
                    f"[wp{self.wp_index}] moving  dx={ex:+.2f} dy={ey:+.2f} dz={ez:+.2f}"
                )
                self.drive_towards(target)

        # Land
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
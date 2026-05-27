'''
+x -> right
+y -> forward
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
        self.goal_pos = [1.5, 1.5, 2.0]
        self.pos_threshold = 0.25
        self.takeoff_alt = 1.5
        self.counter = 0
        self.phase = "CALIBRATE"
        self.hover_time = 10.0
        self.start_time = 0.0

        self.forwar_motion_start_coord = None
        self.forward_dist = 1.2

        # Error info
        self.errorX = None
        self.errorY = None

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
        # print(f"x: {self.current_pos[0]:.2f}, y: {self.current_pos[1]:.2f}, z: {self.current_pos[2]:.2f}")

    # ─── Movement helpers ────────────────────────────────────────────────────────

    def _make_pos_target(self, x, y, z):
        """Build a PositionTarget msg with yaw ignored (no heading change)."""
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = self.pos_type_mask
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z
        return msg

    # def go_to_pos(self, pos):
    #     msg = self._make_pos_target(
    #         pos[0] + self.ref_x,
    #         pos[1] + self.ref_y,
    #         pos[2] + self.ref_z
    #     )
    #     self.pub.publish(msg)

    def hover(self):
        msg = self._make_pos_target(
            self.current_pos[0] + self.ref_x,
            self.current_pos[1] + self.ref_y,
            self.current_pos[2] + self.ref_z
        )
        self.pub.publish(msg)

    def move_with_vel(self, vel,dir):
        velMsg = TwistStamped()
        if dir == "x":
            velMsg.twist.linear.x = vel
        elif dir == "y":
            velMsg.twist.linear.y = vel
        elif dir == "z":
            velMsg.twist.linear.z = vel
        self.pubVel.publish(velMsg)

    def distance(self, initial, final):
        return ((initial[0] - final[0])**2 + (initial[1] - final[1])**2 + (initial[2] - final[2])**2)**0.5

    # ─── Services ────────────────────────────────────────────────────────────────

    def takeoff(self, altitude=1.0):
        if self.takeoff_client.wait_for_service(timeout_sec=2.0):
            req = CommandTOL.Request()
            req.altitude = altitude     # relative to home; lat/lon ignored in GPS-denied
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
        self.phase = "LAND"

    # ─── Main loop ───────────────────────────────────────────────────────────────

    def loop(self):
        if self.phase == "CALIBRATE":
            return

        # Step 1: Switch to GUIDED
        if self.phase == "INIT" and self.counter == 10:
            self.get_logger().info("Switching to GUIDED...")
            if self.mode_client.wait_for_service(timeout_sec=2.0):
                req = SetMode.Request()
                req.custom_mode = "GUIDED"
                self.mode_client.call_async(req)

        # Step 2: Arm
        if self.phase == "INIT" and self.counter == 30:
            self.get_logger().info("Arming...")
            if self.arm_client.wait_for_service(timeout_sec=2.0):
                req = CommandBool.Request()
                req.value = True
                self.arm_client.call_async(req)

        # Step 3: Takeoff via service
        if self.phase == "INIT" and self.counter == 50:
            self.get_logger().info("Sending takeoff command...")
            self.takeoff(altitude=self.takeoff_alt)

        # Step 4: Monitor altitude
        if self.phase == "TAKEOFF":
            self.get_logger().info(f"Altitude = {self.current_pos[2]:.2f}")
            if self.current_pos[2] > (self.takeoff_alt - self.pos_threshold):
                self.get_logger().info(f"Reached {self.takeoff_alt}m -> Hovering for {self.hover_time}s")
                self.phase = "HOVER1"
                self.start_time = time()

        # Step 5: Hover
        if self.phase == "HOVER1":
            elapsed = time() - self.start_time
            if elapsed < self.hover_time:
                self.get_logger().info(f"Hovering... {(self.hover_time - elapsed):.1f}s left")
                self.hover()
            else:
                self.get_logger().info("Hover complete -> Moving Forward")
                self.forwar_motion_start_coord = self.current_pos[1]
                self.phase = "FORWARD"
        
        if self.phase == "FORWARD":
            if self.current_pos[1] - self.forwar_motion_start_coord < self.forward_dist -self.pos_threshold:
                self.get_logger().info(f"distance left to move {self.forward_dist - (self.current_pos[1] - self.forwar_motion_start_coord)} m")
                self.move_with_vel(0.3,"y")
            else: 
                self.get_logger().info(f"GOAL REACHED -> Hovering for {self.hover_time} s")
                self.phase = "HOVER2"
                self.start_time = time()
        
        if self.phase == "HOVER2":
            elapsed = time() - self.start_time
            if elapsed < self.hover_time:
                self.get_logger().info(f"Hovering... {(self.hover_time - elapsed):.1f}s left")
                self.hover()
            else:
                self.get_logger().info("Hover complete -> Landig")
                self.phase = "LAND" 

        # Step 6: Land
        if self.phase == "LAND":
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
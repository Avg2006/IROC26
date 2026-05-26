import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from rclpy.qos import QoSProfile, ReliabilityPolicy
from time import time


class droneControl(Node):
    def __init__(self):
        super().__init__('drone_control')

        # Publishers
        self.pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
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
        self.calib_duration = 3.0           # seconds to collect samples

        # State
        self.current_pos = [0.0, 0.0, 0.0]  # always relative to reference after calibration
        self.goal_pos = [1.5, 1.5, 2.0]
        self.pos_threshold = 0.25
        self.takeoff_alt = 1.5
        self.counter = 0
        self.phase = "CALIBRATE"            # start in CALIBRATE phase
        self.kp = -0.2
        self.hover_time = 20.0
        self.start_time = 0.0

        # Error info
        self.errorX = None
        self.errorY = None

        self.timer = self.create_timer(0.1, self.loop)
        self.get_logger().info("Calibrating reference frame for 3 seconds...")

    # ─── Calibration ────────────────────────────────────────────────────────────

    def pose_callback(self, msg):
        raw_x = msg.pose.position.x
        raw_y = msg.pose.position.y
        raw_z = msg.pose.position.z

        # Collect samples during calibration
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

        # After calibration: store relative position
        self.current_pos[0] = raw_x - self.ref_x
        self.current_pos[1] = raw_y - self.ref_y
        self.current_pos[2] = raw_z - self.ref_z
        print(f"x: {self.current_pos[0]:.2f}, y: {self.current_pos[1]:.2f}, z: {self.current_pos[2]:.2f}")

    # ─── Movement helpers ────────────────────────────────────────────────────────

    def go_to_pos(self, pos):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Convert relative target back to raw frame for MAVROS
        msg.pose.position.x = pos[0] + self.ref_x
        msg.pose.position.y = pos[1] + self.ref_y
        msg.pose.position.z = pos[2] + self.ref_z
        self.pub.publish(msg)

    def move_with_vel(self, vel):
        velMsg = TwistStamped()
        velMsg.twist.linear.x = vel[0]
        velMsg.twist.linear.y = vel[1]
        velMsg.twist.linear.z = vel[2]
        self.pubVel.publish(velMsg)

    def hover(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.current_pos[0] + self.ref_x
        msg.pose.position.y = self.current_pos[1] + self.ref_y
        msg.pose.position.z = self.current_pos[2] + self.ref_z
        self.pub.publish(msg)
        
    def distance(self, initial, final):
        return ((initial[0] - final[0])**2 + (initial[1] - final[1])**2 + (initial[2] - final[2])**2)**0.5

    # ─── Services ────────────────────────────────────────────────────────────────

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
        self.phase = "LAND"

    # ─── Main loop ───────────────────────────────────────────────────────────────

    def loop(self):
        # Wait until calibration is done before doing anything
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
                self.get_logger().info("Reached 1m -> Hovering for 10 seconds")
                self.phase = "HOVER"
                self.start_time = time()

        # Step 5: Hover
        if self.phase == "HOVER":
            elapsed = time() - self.start_time
            if elapsed < self.hover_time:
                self.get_logger().info(f"Hovering... {elapsed:.1f}s elapsed")
                self.hover()
            else:
                self.get_logger().info("Hover complete -> Landing")
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
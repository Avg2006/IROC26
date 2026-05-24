import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, TwistStamped
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.qos import QoSProfile, ReliabilityPolicy
from time import time
from datetime import datetime


class droneControl(Node):
    def __init__(self):
        super().__init__('drone_control')

        # Publisher
        self.pub = self.create_publisher(PoseStamped,'/mavros/setpoint_position/local',10)
        self.pubVel = self.create_publisher(TwistStamped,'/mavros/setpoint_velocity/cmd_vel',10)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Subscriber
        self.sub = self.create_subscription(PoseStamped,'/mavros/local_position/odom',self.pose_callback,qos)

        # Services
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # State 
        self.current_pos = [0.0,0.0,0.0]
        self.goal_pos = [1.5,1.5,2.0]
        self.pos_threshold = 0.1
        self.align_threshold = 0.1
        self.takeoff_alt = 1.0
        self.counter = 0
        self.phase = "INIT"
        self.kp = -0.2
        self.hover_time = 5.0
        self.start_time = 0.0


        #error info
        self.errorX = None
        self.errorY = None

        #vel info
        self.velx = None
        self.vely = None

        self.timer = self.create_timer(0.1, self.loop)

    def pose_callback(self, msg):
        self.current_pos[0] = msg.pose.position.x
        self.current_pos[1] = msg.pose.position.y
        self.current_pos[2] = msg.pose.position.z
        print(f"x: {msg.pose.position.x}, y:{msg.pose.position.y}, z:{msg.pose.position.z}")

    def poseArray_callback(self, msg):
        self.errorX = msg.poses[0].position.x
        self.errorY = msg.poses[0].position.y

    def go_to_pos(self, pos):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        self.pub.publish(msg)

    def hover(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.current_pos[0]
        msg.pose.position.y = self.current_pos[1]
        msg.pose.position.z = self.current_pos[2]
        self.pub.publish(msg)

    def move_with_vel(self,vel):
        velMsg = TwistStamped()
        velMsg.twist.linear.x = vel[0]
        velMsg.twist.linear.y = vel[1]
        velMsg.twist.linear.z = vel[2]
        self.pubVel.publish(velMsg)
    
    def distance(self,initial,final):
        return ((initial[0]-final[0])**2+(initial[1]-final[1])**2+(initial[2]-final[2])**2)**0.5

    def land(self):
        self.get_logger().info("Landing...")
        if self.mode_client.wait_for_service(timeout_sec=2.0):
            req = SetMode.Request()
            req.custom_mode = "AUTO.LAND"
            self.mode_client.call_async(req)
        self.phase = "LAND"

    

    def loop(self):
        now = datetime.now()
        if self.phase == "INIT" and self.counter == 40:
            self.get_logger().info("Switching to OFFBOARD...")
            if self.mode_client.wait_for_service(timeout_sec=2.0):
                req = SetMode.Request()
                req.custom_mode = "GUIDED"
                self.mode_client.call_async(req)
                self.get_logger().info("Arming!!")

        if self.phase == "INIT" and self.counter == 100:
            self.get_logger().info("Arming...")
            if self.arm_client.wait_for_service(timeout_sec=2.0):
                req = CommandBool.Request()
                req.value = True
                self.arm_client.call_async(req)
                self.phase = "TAKEOFF"
                self.get_logger().info("Taking Off")

        if self.phase == "TAKEOFF":
            self.go_to_pos([self.current_pos[0],self.current_pos[1],self.takeoff_alt])
            self.get_logger().info(f"Altitude = {self.current_pos[2]:.2f}")

            if self.current_pos[2] > (self.takeoff_alt-self.pos_threshold):
                self.get_logger().info("Reached Altitude -> Hovering for 10 second")
                self.phase = "HOVER"
                self.start_time = now.second


        if self.phase == "HOVER":
            
            if (now.second - self.start_time < self.hover_time):
                self.get_logger().info(f"Hovering of {now.second - self.start_time} seconds left")
                self.hover()
            else:
                self.get_logger().info("Hover Completed -> Landing")
                self.phase = "LAND"


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

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import cv2
import tf_transformations
from collections import deque
import yaml
import math
import os
from datetime import datetime
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry 
from ros2_aruco_interfaces.msg import ArucoMarkers
# from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from std_msgs.msg import Bool, Float32, String, Int32

# ============================================================
# HELPER FUNCTIONS (MATH & GEOMETRY)
# ============================================================

def line_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2((y2 - y1), (x2 - x1)))

def line_length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def angle_difference(a1, a2):
    diff = abs(a1 - a2)
    if diff > 180:
        diff = 360 - diff
    return diff

def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    if abs(denominator) < 1e-6:
        return None
    px = (((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator)
    py = (((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator)
    return int(px), int(py)

def point_to_line_distance(px, py, line):
    x1, y1, x2, y2 = line
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    if denominator == 0:
        return 9999
    return numerator / denominator

def merge_two_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    angle1 = math.atan2(y2 - y1, x2 - x1)
    angle2 = math.atan2(y4 - y3, x4 - x3)
    if abs(angle1 - angle2) > math.pi / 2:
        if angle2 < angle1:
            angle2 += math.pi
        else:
            angle2 -= math.pi
    avg_angle = (angle1 + angle2) / 2
    dx = math.cos(avg_angle)
    dy = math.sin(avg_angle)
    pdx, pdy = -dy, dx
    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    t_vals = [p[0] * dx + p[1] * dy for p in pts]
    p_vals = [p[0] * pdx + p[1] * pdy for p in pts]
    p_mid = sum(p_vals) / 4
    t_min, t_max = min(t_vals), max(t_vals)
    rx1 = int(t_min * dx + p_mid * pdx)
    ry1 = int(t_min * dy + p_mid * pdy)
    rx2 = int(t_max * dx + p_mid * pdx)
    ry2 = int(t_max * dy + p_mid * pdy)
    return (rx1, ry1, rx2, ry2)


# ============================================================
# MAIN ROS2 NODE
# ============================================================

class CombinedPerceptionNode(Node):

    def __init__(self):
        super().__init__("combined_perception_node")

        # ---------------- 1. ARUCO INITIALIZATION ---------------- #
        
        self.camera_info_path = os.path.expanduser("~/iroc_2026-scripts/zebronics_HD_new.yaml") 
        
        self.declare_parameter("marker_size", 0.09)
        self.declare_parameter("aruco_dictionary_id", "DICT_5X5_100")
        self.declare_parameter("camera_frame", "")

        self.mission_complete = False
        
        self.c = 0
        self.prev_time = 0
        self.window_size = 5
        self.position_history = {}
        self.vel_history = {}
        
        self.marker_size = 0.09
        dictionary_name = "DICT_5X5_100"
        self.camera_frame = self.get_parameter("camera_frame").value

        self.intrinsic_mat = None
        self.distortion = None
        self.load_camera_info(self.camera_info_path)

        try:
            dictionary_id = getattr(cv2.aruco, dictionary_name)
        except AttributeError:
            raise

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        # ---------------- 2. LINE DETECTION INITIALIZATION ---------------- #
        self.current_x = 0
        self.current_y = 0
        self.drone_height_meters = 2.0
        self.fx = 1430
        self.fy = 1426
        self.focal_length_pixels = (self.fx + self.fy) / 2

        self.corner_history_size = 4
        self.corner_history = []

        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        self.min_line_length_meters = 0.10
        self.max_lines = 10
        self.angle_threshold = 45
        self.parallel_angle_threshold = 30
        self.parallel_dist_thresh_meters = 0.33

        # ---------------- 3. ROS & OPENCV SETUP ---------------- #
        camera_index = 0
        self.cap = cv2.VideoCapture(camera_index) # Change to 2 if needed for your specific setup
        
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
        else:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Change this to match your ROS timer
            self.cap.set(cv2.CAP_PROP_FPS, 15) 
            
            self.get_logger().info("Camera successfully opened in 1080p at 15 FPS")
        #frame capturing and self.take_picture subscriber
        self.save_dir = os.path.expanduser("~/captured_frames")
        os.makedirs(self.save_dir, exist_ok=True)

        self.take_picture_sub = self.create_subscription(
            Bool, '/take_picture', self.take_picture_cb, 10
        )
        self.picture_trigger_state = False
        self._latest_frame = None  # cache the latest frame

        # ArUco Publishers
        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)
        self.vel_pub = self.create_publisher(Odometry, 'aruco_vel', 10)
        self.aruco_detected_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        
        # Line Detection Publishers
        self.boundary_pub = self.create_publisher(Int32, '/boundary_detected', 10)
        self.line1_dist_pub = self.create_publisher(Float32, '/distance1', 10)
        self.line2_dist_pub = self.create_publisher(Float32, '/distance2', 10)
        self.corner_pub = self.create_publisher(String, '/corner_type', 10)

        # Shared Image Publisher
        self.image_pub = self.create_publisher(CompressedImage, "aruco_debug_image/compressed", 10)

        # Height Subscriber
        self.height_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.height_callback, 10)
        self.mission_complete_sub = self.create_subscription(Bool,'/mission_complete',self.mission_complete_cb,10)

        # Create a timer to fetch frames at 20 Hz
        timer_period = 1.0 / 20.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def mission_complete_cb(self,msg):
        self.mission_complete = msg.data


    # ---------------- YAML LOADER ---------------- #

    def load_camera_info(self, file_path):
        try:
            with open(file_path, 'r') as file:
                calib_data = yaml.safe_load(file)
                self.intrinsic_mat = np.array(calib_data['camera_matrix']['data']).reshape((3, 3))
                self.distortion = np.array(calib_data['distortion_coefficients']['data'])
                self.get_logger().info("Camera info loaded successfully from file.")
        except Exception as e:
            self.get_logger().error(f"Failed to load camera info from {file_path}. Error: {e}")
            self.intrinsic_mat = np.eye(3, dtype=np.float32)
            self.distortion = np.zeros(5, dtype=np.float32)


    # ---------------- IMAGE COMPRESSION ---------------- #

    def publish_compressed_image(self, img, timestamp, frame_id):
        small_image = cv2.resize(img, (640, 360))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, encoded_image = cv2.imencode('.jpg', small_image, encode_param)
        
        if success:
            msg = CompressedImage()
            msg.header.stamp = timestamp
            msg.header.frame_id = frame_id
            msg.format = "jpeg"
            msg.data = encoded_image.tobytes()
            self.image_pub.publish(msg)

    # ---------------- DRONE HEIGHT ---------------- #

    def height_callback(self, msg):
        self.drone_height_meters = msg.pose.position.z
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        if self.drone_height_meters < 0.05:
            self.drone_height_meters = 0.05

    # TAKING THE PICTURE#
    def take_picture_cb(self, msg):
        current_signal = msg.data
        if current_signal and not self.picture_trigger_state:
            self.picture_trigger_state = True
            if self._latest_frame is None:
                self.get_logger().info("Cannot take picture: No frame available.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.save_dir, f"frame_{timestamp}.jpg")
            
            cv2.imwrite(filename, self._latest_frame)

            # Write metadata sidecar file
            meta = {
                "timestamp": timestamp,
                "position": {
                    "x": self.current_x,
                    "y": self.current_y,
                    "z": self.drone_height_meters,
                },
            }

            meta_path = filename.replace(".jpg", ".yaml")
            with open(meta_path, "w") as f:
                yaml.dump(meta, f)

            self.get_logger().info(f"Saved frame → {filename}")
        elif not current_signal:
            self.picture_trigger_state = False


    # ---------------- LINE DETECTION HELPERS ---------------- #

    def lines_are_similar(self, line1, line2, center_x, center_y):
        angle1 = line_angle(*line1)
        angle2 = line_angle(*line2)
        diff = angle_difference(angle1, angle2)
        if diff > self.parallel_angle_threshold:
            return False

        d1 = point_to_line_distance(center_x, center_y, line1)
        d2 = point_to_line_distance(center_x, center_y, line2)
        
        parallel_distance_threshold = (self.parallel_dist_thresh_meters * self.focal_length_pixels) / self.drone_height_meters
        
        dist = abs(d1 - d2)
        if dist > parallel_distance_threshold:
            return False
        return True

    def update_corner_history(self, corner):
        self.corner_history.append(corner)
        if len(self.corner_history) > self.corner_history_size:
            self.corner_history.pop(0)

    def get_stable_corner(self):
        if len(self.corner_history) == 0:
            return None
        pts = np.array(self.corner_history, dtype=np.float32)
        (x, y), radius = cv2.minEnclosingCircle(pts)
        return (int(x), int(y)), int(radius)

    def classify_corner(self, line1, line2, corner):
        px, py = corner
        angle1 = abs(line_angle(*line1))
        
        if angle1 < 45:
            horizontal, vertical = line1, line2
        else:
            horizontal, vertical = line2, line1

        hx1, hy1, hx2, hy2 = horizontal
        d1 = math.hypot(hx1 - px, hy1 - py)
        d2 = math.hypot(hx2 - px, hy2 - py)
        
        if d1 > d2: hvx, hvy = hx1 - px, hy1 - py
        else:       hvx, hvy = hx2 - px, hy2 - py

        vx1, vy1, vx2, vy2 = vertical
        d1 = math.hypot(vx1 - px, vy1 - py)
        d2 = math.hypot(vx2 - px, vy2 - py)
        
        if d1 > d2: vvx, vvy = vx1 - px, vy1 - py
        else:       vvx, vvy = vx2 - px, vy2 - py

        cross = (hvx * vvy - hvy * vvx)
        return "RIGHT" if cross < 0 else "LEFT"


    # ---------------- 1. PROCESS LINE DETECTION ---------------- #

    def process_line_detection(self, frame):
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # Dynamic Threshold calculation based on height
        min_line_length_pixels = (self.min_line_length_meters * self.focal_length_pixels) / self.drone_height_meters
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(mask, 50, 150)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, 
                                minLineLength=min_line_length_pixels, maxLineGap=20)

        raw_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = line_length(x1, y1, x2, y2)
                if length >= min_line_length_pixels:
                    raw_lines.append((x1, y1, x2, y2, length))

        raw_lines = sorted(raw_lines, key=lambda l: l[4], reverse=True)
        merged_lines = []

        for line_data in raw_lines:
            candidate = line_data[:4]
            was_merged = False
            for i, accepted in enumerate(merged_lines):
                if self.lines_are_similar(candidate, accepted, center_x, center_y):
                    merged_lines[i] = merge_two_lines(accepted, candidate)
                    was_merged = True
                    break
            if not was_merged:
                merged_lines.append(candidate)
            if len(merged_lines) >= self.max_lines:
                break

        line_detected = 1 if len(merged_lines) > 0 else 0
        min_distance_pixels_parallel = 999999
        min_distance_pixels_perpendicular = 999999
        closest_line_parallel = None
        closest_line_perpendicular = None

        for line in merged_lines:
            x1, y1, x2, y2 = line
            # VISUALIZATION: Draw the detected boundary lines on the frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3) # Yellow line

            dist_pixels = point_to_line_distance(center_x, center_y, line)
            if (x2 - x1) != 0 and abs((y2 - y1) / (x2 - x1)) <= 1:
                if dist_pixels < min_distance_pixels_parallel:
                    min_distance_pixels_parallel = dist_pixels
                    closest_line_parallel = line
            else:
                if dist_pixels < min_distance_pixels_perpendicular:
                    min_distance_pixels_perpendicular = dist_pixels
                    closest_line_perpendicular = line

        corners = []
        for i in range(len(merged_lines)):
            for j in range(i + 1, len(merged_lines)):
                line1 = merged_lines[i]
                line2 = merged_lines[j]
                if angle_difference(line_angle(*line1), line_angle(*line2)) < self.angle_threshold:
                    continue
                pt = intersection(line1, line2)
                if pt is None: continue
                px, py = pt
                if 0 <= px < width and 0 <= py < height:
                    corners.append((px, py, line1, line2))

        corner_type = "NONE"
        if len(corners) > 0:
            selected_corner = min(corners, key=lambda c: math.hypot(c[0] - center_x, c[1] - center_y))
            px, py, line1, line2 = selected_corner
            
            self.update_corner_history((px, py))
            stable_result = self.get_stable_corner()

            if stable_result is not None:
                (sx, sy), radius = stable_result
                corner_type = self.classify_corner(line1, line2, (sx, sy))
                
                # VISUALIZATION: Draw the stable corner point and label
                cv2.circle(frame, (sx, sy), radius + 5, (255, 0, 255), 3)
                cv2.putText(frame, f"{corner_type} CORNER", (sx + 15, sy - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

        parallel_distance_meters = 0.0
        perpendicular_distance_meters = 0.0

        if closest_line_parallel is not None:
            parallel_distance_meters = (min_distance_pixels_parallel * self.drone_height_meters) / self.focal_length_pixels
        if closest_line_perpendicular is not None:
            perpendicular_distance_meters = (min_distance_pixels_perpendicular * self.drone_height_meters) / self.focal_length_pixels

        return line_detected, parallel_distance_meters, perpendicular_distance_meters, corner_type


    # ---------------- 2. PROCESS ARUCO DETECTION ---------------- #

    def process_aruco_detection(self, cv_image, current_time, frame_id):
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, marker_ids, _ = self.detector.detectMarkers(gray_image)
        
        isdetected = Bool()
        isdetected.data = False
        
        if marker_ids is None:
            self.aruco_detected_pub.publish(isdetected)
            return

        cv2.aruco.drawDetectedMarkers(cv_image, corners, marker_ids)

        for corner in corners:
            pts = corner.reshape(4,2).astype(int)
            cv2.polylines(cv_image, [pts], True, (0,255,0), thickness=4)

        markers_msg = ArucoMarkers()
        pose_array = PoseArray()
        vel_msg = Odometry()

        markers_msg.header.frame_id = frame_id
        pose_array.header.frame_id = frame_id
        markers_msg.header.stamp = current_time
        pose_array.header.stamp = current_time

        half_size = self.marker_size / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        for i, marker_id in enumerate(marker_ids):
            corner = corners[i]
            success, rvec, tvec = cv2.solvePnP(object_points, corner, self.intrinsic_mat, self.distortion)
            if not success: continue

            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = tf_transformations.euler_from_matrix(
                np.vstack((np.hstack((R, np.zeros((3,1)))), [0,0,0,1]))
            )

            roll = np.degrees(roll)
            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)

            cv2.drawFrameAxes(cv_image, self.intrinsic_mat, self.distortion, rvec, tvec, self.marker_size * 0.5)

            pose = Pose()
            m_id = int(marker_id[0])
            raw_x, raw_y, raw_z = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])

            if m_id not in self.position_history:
                self.position_history[m_id] = deque(maxlen=self.window_size)

            self.position_history[m_id].append((raw_x, raw_y, raw_z))
            history = self.position_history[m_id]
            
            pose.position.x = float(np.median([p[0] for p in history]))
            pose.position.y = float(np.median([p[1] for p in history]))
            pose.position.z = float(np.median([p[2] for p in history]))

            # Euler angles in degrees stored in orientation fields (x: roll, y: pitch, z: yaw)
            pose.orientation.x = float(roll)
            pose.orientation.y = float(pitch)
            pose.orientation.z = float(yaw)
            pose.orientation.w = 1.0

            # ArUco Text Background
            cv2.rectangle(cv_image, (10,10), (280,190), (0,0,0), -1)
            cv2.rectangle(cv_image, (10,10), (280,190), (0,255,255), 2)

            lines = [
                f"ID : {m_id}",
                f"X : {pose.position.x:.3f} m",
                f"Y : {pose.position.y:.3f} m",
                f"Z : {pose.position.z:.3f} m",
                f"Roll  : {roll:.1f}",
                f"Pitch : {pitch:.1f}",
                f"Yaw   : {yaw:.1f}"
            ]

            y = 35
            for line in lines:
                cv2.putText(cv_image, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                y += 25

            pose_array.poses.append(pose)
            markers_msg.poses.append(pose)
            markers_msg.marker_ids.append(m_id)
            
            time_in_seconds = current_time.sec + (current_time.nanosec * 1e-9)
            dt = time_in_seconds - self.prev_time
            
            if dt > 0 and self.c == 1:
                raw_vx = (pose.position.x - self.prev_x)/dt
                raw_vy = (pose.position.y - self.prev_y)/dt
                raw_vz = (pose.position.z - self.prev_z)/dt
                
                if m_id not in self.vel_history:
                    self.vel_history[m_id] = deque(maxlen=self.window_size)
                self.vel_history[m_id].append((raw_vx, raw_vy, raw_vz))
                
                history2 = self.vel_history[m_id]
                vel_msg.twist.twist.linear.x = float(np.median([p[0] for p in history2]))
                vel_msg.twist.twist.linear.y = float(np.median([p[1] for p in history2]))
                vel_msg.twist.twist.linear.z = float(np.median([p[2] for p in history2]))
                
            self.prev_x = pose.position.x
            self.prev_y = pose.position.y
            self.prev_z = pose.position.z
            self.prev_time = time_in_seconds
            self.c = 1

        self.poses_pub.publish(pose_array)
        self.markers_pub.publish(markers_msg)
        self.vel_pub.publish(vel_msg)
        
        isdetected.data = True
        self.aruco_detected_pub.publish(isdetected)


    # ---------------- MAIN TIMER LOOP ---------------- #

    def timer_callback(self):
        ret, cv_image = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame from camera")
            return
        self._latest_frame = cv_image.copy() 

        detected_msg = Int32()
        d1_msg = Float32()
        d2_msg = Float32()
        c_msg = String()
        current_time = self.get_clock().now().to_msg()
        frame_id = self.camera_frame if self.camera_frame != "" else "camera_link"
        if not self.mission_complete:

            # 1. Process Boundary/Lines (Modifies cv_image by drawing lines)
            detected, dist1, dist2, c_type = self.process_line_detection(cv_image)
            
            # Update Values
            detected_msg.data = detected
            d1_msg.data = float(dist1)
            d2_msg.data = float(dist2)
            c_msg.data = c_type

            not_detected = Bool()
            not_detected.data = False
            self.aruco_detected_pub.publish(not_detected)
                        
        else: 
            # 2. Process ArUco (Modifies the SAME cv_image by drawing markers)
            self.process_aruco_detection(cv_image, current_time, frame_id)
            detected_msg.data = 0
            d1_msg.data = float(0)
            d2_msg.data = float(0)
            c_msg.data = "NONE"

        self.boundary_pub.publish(detected_msg)
        self.line1_dist_pub.publish(d1_msg)
        self.line2_dist_pub.publish(d2_msg)
        self.corner_pub.publish(c_msg)

        # 3. Publish the combined, drawn image
        self.publish_compressed_image(cv_image, current_time, frame_id)
        
    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)        
    node = CombinedPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

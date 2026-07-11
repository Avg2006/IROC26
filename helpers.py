import os
import json
import time
import math
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import CommandTOL, SetMode
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

class MissionHelpersMixin:
    """
    Mixin class containing helper functions.
    Requires the host class to have self.config and self.state initialized.
    """

    def body_to_world(self, forward, right, alt, origin=(0.0, 0.0, 0.0)):
        yaw = self.state.locked_yaw - self.state.yaw_arena_offset
        world_x = origin[0] + forward * math.cos(yaw) + right * math.sin(yaw)
        world_y = origin[1] + forward * math.sin(yaw) - right * math.cos(yaw)
        return (world_x, world_y, alt)

    def update_line_offset(self):
        """
        Debounce the boundary-line offset (state.front_back_offset_y).

        The drone faces the boundary line head-on after ALIGN_PERPENDICULAR,
        so while sliding sideways along it during SEARCH_RIGHT the line is
        still detected as the front/back (parallel) line, not the side line
        -- hence front_back_offset_y is the correct standoff signal here.

        y != 0  -> trusted immediately, used as the live offset.
        y == 0  -> NOT trusted immediately. Starts a grace timer and keeps
                   returning the last good offset during that window, so a
                   single dropped frame doesn't cause a jerk. Only once y
                   has read 0 for `config.offset_lost_wait` seconds straight
                   do we set state.offset_lost_confirmed = True and return 0.0.
        """
        y = self.state.front_back_offset_y
        now = time.time()

        if abs(y) > self.config.offset_zero_eps:
            self.state.last_valid_offset_y = y
            self.state.offset_zero_since = None
            self.state.offset_lost_confirmed = False
            return y

        # Reading is (near) zero this cycle
        if self.state.offset_zero_since is None:
            self.state.offset_zero_since = now

        if (now - self.state.offset_zero_since) < self.config.offset_lost_wait:
            # Inside grace period: don't trust the zero yet
            return self.state.last_valid_offset_y

        # Zero for a full window straight -> genuinely lost the line
        self.state.offset_lost_confirmed = True
        return 0.0

    def hold(self, target, yaw=None):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        
        if yaw is not None:
            msg.type_mask = (
                PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
                PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW_RATE
            )
            msg.yaw = yaw
        else:
            msg.type_mask = self.config.pos_type_mask
            
        msg.position.x = target[0] + self.state.ref_x
        msg.position.y = target[1] + self.state.ref_y
        msg.position.z = target[2] + self.state.ref_z
        self.pub.publish(msg)

    def publish_reference_odom(self, orientation):
        """
        Publishes current position relative to the base station (calibration
        origin, i.e. the ground point where CALIBRATE averaged ref_x/y/z) on
        /reference_odom. Runs independent of mission phase -- including TEST --
        as long as the reference has been set.
        """
        ref_msg = PoseStamped()
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.header.frame_id = "base_station"
        ref_msg.pose.position.x = self.state.current_pos[0]
        ref_msg.pose.position.y = self.state.current_pos[1]
        ref_msg.pose.position.z = self.state.current_pos[2]
        ref_msg.pose.orientation = orientation
        self.ref_odom_pub.publish(ref_msg)

    def arrived(self, target):
        dist = (
            (self.state.current_pos[0] - target[0]) ** 2 +
            (self.state.current_pos[1] - target[1]) ** 2 +
            (self.state.current_pos[2] - target[2]) ** 2
        ) ** 0.5
        return dist < self.config.pos_threshold

    def flip_sweep_direction(self):
        """
        Reverses the forward/backward travel vector. 
        Because the yaw is locked, flipping this sign naturally causes 
        the drone to sweep back and forth across the arena grid.
        """
        self.state.sweep_dir_forward *= -1.0
        
    def log_waypoint(self, label, target):
        entry = {
            "label": label,
            "phase": self.state.phase,
            "edge_number": self.state.current_row,
            "step_index": self.state.sweep_step_index,
            "target": [round(float(t), 3) for t in target],
            "actual_position": [round(p, 3) for p in self.state.current_pos],
            "timestamp": time.time(),
        }
        self.state.waypoint_log.append(entry)

    def log_corner(self, corner_number, position, dist_from_prev, confirmed=True, reason="corner_detected"):
        entry = {
            "corner_number": corner_number,
            "position": [round(p, 3) for p in position],
            "timestamp": time.time(),
            "dist_from_prev_corner": round(dist_from_prev, 3),
            "confirm_count": self.state.corner_confirm_count,
            "confirmed": confirmed,
            "reason": reason,
            "edge_number": self.state.current_row,
        }
        self.state.corner_log.append(entry)
        self.get_logger().info(
            f"[corner-log] #{corner_number} at {entry['position']} "
            f"(edge {self.state.current_row}, confirmed={confirmed}, reason={reason})"
        )

    def save_debug_logs(self):
        try:
            log_dir = os.path.expanduser('~/ascend_logs')
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(log_dir, f'mission_debug_{ts}.json')
            with open(path, 'w') as f:
                json.dump({"waypoints": self.state.waypoint_log, "corners": self.state.corner_log}, f, indent=2)
            self.get_logger().info(f"Saved debug log to {path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save debug logs: {e}")

    def trigger(self, value=False):
        msg = Bool()
        msg.data = value
        self.bool_pub.publish(msg)

    def takeoff(self, altitude):
        if self.takeoff_client.wait_for_service(timeout_sec=2.0):
            req = CommandTOL.Request()
            req.altitude = altitude
            future = self.takeoff_client.call_async(req)
            future.add_done_callback(self._takeoff_cb)
        else:
            self.get_logger().error("Takeoff service unavailable")

    def _takeoff_cb(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info("Takeoff command accepted")
            self.state.phase = "TAKEOFF"
        else:
            self.get_logger().warn(f"Takeoff rejected (code {result.result})")

    def land(self):
        self.get_logger().info("Switching to LAND mode...")
        if self.mode_client.wait_for_service(timeout_sec=2.0):
            req = SetMode.Request()
            req.custom_mode = "LAND"
            self.mode_client.call_async(req)
            
        if not self.state._logs_saved:
            self.save_debug_logs()
            self.state._logs_saved = True
            
        self.state.phase = "DONE"
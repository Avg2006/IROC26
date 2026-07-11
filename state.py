import time

class MissionState:
    def __init__(self):
        # ── Live Sensor / Detection Data ──
        self.boundary_detected_raw = False
        self.corner_detected_raw = False
        self.front_back_offset_z = 0.0
        self.front_back_offset_y = 0.0
        self.side_offset_z = 0.0
        self.side_offset_y = 0.0
        self.boundary_is_front_back = True

        # ── Line-following (SEARCH_RIGHT) State ──
        self.line_offset_setpoint = None
        self.last_valid_offset_y = 0.0
        self.offset_zero_since = None
        self.offset_lost_confirmed = False
        
        # ── Debounce Counters ──
        self.boundary_confirm_count = 0
        self.corner_confirm_count = 0

        # ── Calibration State ──
        self.init_samples = []
        self.reference_set = False
        self.ref_x = 0.0
        self.ref_y = 0.0
        self.ref_z = 0.0
        self.calib_start_time = time.time()
        
        # ── Yaw / Heading State ──
        self.locked_yaw = None
        self.current_yaw = 0.0
        self.target_yaw = None
        self.yaw_arena_offset = 0.0
        self.yaw_arena_offset_set = False

        # ── General Flight State ──
        self.current_pos = [0.0, 0.0, 0.0]
        self.counter = 0
        self.phase = "CALIBRATE"
        self.start_time = 0.0

        # ── Mission Waypoints / Latched Origins ──
        self.home_position = None
        self.forward_search_target = None
        self.boundary_origin = None
        self.right_search_target = None
        self.corner_origin = None
        
        # ── Alignment State ──
        self.align_samples = []

        # ── Lawnmower Scanning State ──
        self.sweep_step_index = 0
        self.current_row = 1
        self.sweep_dir_forward = -1.0
        self.sweep_origin = None
        self.sweep_target = None
        self.row_origin = None           # Latch point when a boundary is hit
        self.cleared_start_boundary = False
        self.sweep_scan_start_time = 0.0

        # ── Corner Counting (Sweep Phase) ──
        self.corners_found = 1            # corner 1 already located pre-sweep
        self.corner_seen_this_row = False

        # ── Logging State ──
        self.corner_log = []
        self.waypoint_log = []
        self._logs_saved = False

        # ── Ground Testing State ──
        self.test_last_log_time = 0.0
        self.test_confirmed = False
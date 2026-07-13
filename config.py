from mavros_msgs.msg import PositionTarget

class MissionConfig:
    def __init__(self):
        # ── Ground Testing ──
        self.test_mode = True           # If True, drop into TEST phase after calibration
        self.test_log_interval = 1.0     # Seconds between odom diagnostic prints in TEST

        # ── Calibration & Thresholds ──
        self.calib_duration = 3.0
        self.pos_threshold = 0.25
        self.takeoff_alt = 2.5
        self.stop_hover_time = 2.0
        self.move_timeout = 60.0
        self.align_hover_time = 2.0

        # ── Search Forward ──
        self.search_forward_distance = 50.0
        self.search_max_distance = 10.0
        self.search_max_time = 120.0

        # ── Search Right ──
        self.search_right_distance = 50.0

        # ── Line-Offset Following (SEARCH_RIGHT) ──
        # side_offset.y == 0 normally means "no side line seen". But a
        # single bad frame can also read 0 while the line is genuinely
        # still there. So a zero reading is NOT trusted immediately -- it
        # must stay zero for `offset_lost_wait` seconds straight before
        # we treat the line as actually lost.
        self.offset_zero_eps = 0.02        # |y| below this counts as "zero" (line lost)
        self.offset_lost_wait = 1.0        # seconds of zero before we call it lost
        self.offset_backup_dist = 0.5      # meters to back off once truly lost
        self.line_follow_gain = 1.0        # m of forward-correction per m of offset error
        self.max_follow_correction = 1.0   # clamp correction so a bad reading can't fling us
        # NOTE: this now drives off front_back_offset.y (the drone faces the
        # boundary line head-on during SEARCH_RIGHT, so it's still detected
        # as the front/back line, not the side line). The sign/magnitude
        # behavior here was tuned against side_offset.y previously -- this
        # is UNVERIFIED against front_back_offset.y and should be checked
        # against live telemetry. +1.0 or -1.0 here flips which way a
        # correction pushes the drone -- if line-following drifts the wrong
        # direction in testing, flip this sign, nothing else.
        self.offset_sign = -1.0

        # ── Lawnmower Scan Parameters ──
        self.sweep_step = 1.0
        self.sweep_dir_right = 1.0       # The constant lateral step direction (e.g., always right)
        self.sweep_hover_time = 2.0      # Settle + photo time at each 1m step
        self.sweep_max_distance = 50.0   # Safety cap for a single row's length
        self.sweep_max_time = 180.0      # Safety cap for a single row's duration
        
        # ── Perimeter & Validation ──
        self.max_rows = 9
        self.detect_confirm_needed = 1
        self.corner_confirm_needed = 3
        self.min_sweep_confirm_distance = 1.5 * self.sweep_step
        
        # ── Masks ──
        self.pos_type_mask = (
            PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE | PositionTarget.IGNORE_YAW
        )
# ── SCP transfer credentials ─────────────────────────────────────────────────
# Place this file in the same directory as full_mission.py.
# It is read at runtime by _transfer_images(); never commit it to version control.
# ─────────────────────────────────────────────────────────────────────────────

# Username on the base station computer
BS_USER = "kanak"

# IP address or hostname of the base station (must be reachable from the Xavier)
BS_HOST = "192.168.0.100"

# Absolute path on the base station where images will be copied into
BS_DEST_DIR = "/home/kanak/iroc_2026_images_rcv"

# Absolute path on the Xavier where the camera node saves images
XAVIER_IMAGE_DIR = "/media/nvidia/galileo/home/nvidia/captured_frames/"

# Password for the base station user account
BS_PASSWORD = "Something8"

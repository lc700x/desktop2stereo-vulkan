import os


GUI_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(GUI_DIR)
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "desktop2stereo.log")
GUI_PID_FILE = os.path.join(LOG_DIR, "gui.pid")
GUI_LOCK_FILE = os.path.join(LOG_DIR, "gui.instance.lock")
GUI_ACTIVATION_FILE = os.path.join(LOG_DIR, "gui.activate.request")
RUNTIME_PID_FILE = os.path.join(LOG_DIR, "runtime.pid")
RUNTIME_LOCK_FILE = os.path.join(LOG_DIR, "runtime.instance.lock")
GUI_READY_FILE = os.path.join(LOG_DIR, "gui_ready.flag")
STOP_REQUEST_FILE = os.path.join(LOG_DIR, "stop.request")
STREAM_CALIBRATION_STATE_FILE = os.path.join(LOG_DIR, "stream_calibration_state.json")
STREAM_CALIBRATION_PROFILE_FILE = os.path.join(LOG_DIR, "stream_calibration_profile.json")

# Kept as an alias for code that still references the old diagnostic log name.
DIAG_LOG = LOG_FILE

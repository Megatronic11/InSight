"""
to run in terminal: uvicorn main:app --host 0.0.0.0 --port 8000

workflow: CAL_1(c1 l1) -> CAL_2(c2,l1) -> CAL_3(last c,l1) -> CAL_4 (c1,l2) -> pic (SNAPSHOT) -> tap to start (READY) -> READING -> ENDED (shows csv and heatmap download buttons)
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import base64
import json
import time
import io
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#all tuning constants
MODEL_PATH = "hand_landmarker.task"
COUNTDOWN_SECS = 8      # seconds per calibration point
SMOOTH_WINDOW = 8      # frames averaged for speed display
LINE_PAUSE_SECS = 1.2    # stillness = new line
READING_LEAD_SECS  = 10     # countdown before reading starts
SNAPSHOT_WAIT_SECS = 8      # "remove hand" countdown

#reversal thresholds
REVERSAL_MIN_CHARS = 2      # reversal must cover >2 chars (Hughes said >0.5cm)
FORWARD_MIN_CHARS = 1      # min forward chars before reversal is eligible
LINE_RETURN_Y_THRESH = 0.8    # y-delta < this * line_height = not a line return

#heat map colors *in blue, green, red instead of RGB*
COLOR_FAST = (56, 56, 255)    # red
COLOR_MED = (86, 201, 100)   # green
COLOR_SLOW = (0, 255, 255)    # yellow
COLOR_UNREAD = (80,  80,  80)
FAST_THRESH = 5.0 #chars per second
MED_THRESH = 3.0 #chars per second
#these thresholds are currently coded based on the speeds for young adults, younger children have less data so it is hard to find without experimentlly doing data find

#regression overlay
REG_BASE_ALPHA = 0.35
REG_ALPHA_STEP = 0.15
REG_MAX_ALPHA = 0.90
REG_COLOR = (209, 96, 59)

#mediapipe setup
base_options = python.BaseOptions(model_asset_path = MODEL_PATH)
mp_options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 1,
    running_mode=vision.RunningMode.IMAGE,
)
#loading the model into the memory (runs once during sturtup)
detector = vision.HandLandmarker.create_from_options(mp_options)

#webserver hosting
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

#route to index.html when running
@app.get("/")
async def root():
    return FileResponse("static/index.html")

#SESSION SETUP!!!

#reversal - detecting one regression and stores it
class Reversal:
    def __init__(self, start_bucket, end_bucket, extent_chars):
        self.start_bucket = start_bucket   # (line_idx, char_in_line)
        self.end_bucket = end_bucket
        self.extent_chars = extent_chars
        

# states for the session itself - always starts at cal 1
class Session:
    def __init__(self):
        # states (goes through top workflow)
        self.state = "CAL_1"
        self.cal_start = time.time()

        #each of these initializes the state variables
        
        #calibration 
        self.cal_x1 = None
        self.cal_y1 = None
        self.pixels_per_char  = None
        self.cal_x_last = None
        self.chars_per_line = None
        self.line_height_px    = None

        #pic 
        self.snapshot_start = None
        self.page_snapshot= None

        #during actual reading 
        self.countdown_start = None
        self.speed_buffer = []
        self.chars_per_sec = 0.0
        self.prev_cx = None
        self.prev_cy = None
        self.prev_time = None
        self.last_move_time = None
        self.current_line_y = None

        self.current_line = 0
        self.line_start_time = None
        self.line_chars = 0.0
        self.line_records = []

        #heat map
        #(line, char) -> speed [cps, ...]
        self.bucket_speeds = {}

        #regressions (called reversals so people who don't know term can understand code easily)
        self.reversals  = []
        self.rev_state = "FORWARD"
        self.rev_start_bucket = None
        self.forward_chars = 0.0
        self.rev_chars = 0.0

        #storing tap data to start reading state
        self.last_tap_time = 0.0

        #this is to make sure that line changes are not counted as reversals because of the backwards x movement
        self.just_changed_line = False
        self.line_change_cooldown = 0


    #converting pixel coordinates to a coordinate grid
    # -> based on the calibration
    def pixel_to_bucket(self, cx, cy):
        if self.pixels_per_char is None or self.line_height_px is None:
            return None
        line_idx = max(0, round((cy - self.cal_y1) / self.line_height_px))
        char_in_line = max(0, round((cx - self.cal_x1) / self.pixels_per_char))
        return (line_idx, char_in_line)

    def record_bucket_speed(self, bucket, cps):
        if bucket is None:
            return
        self.bucket_speeds.setdefault(bucket, []).append(cps)

    #pixels to characters using calib points
    def update_reversal(self, pixel_dist, cx, cy, current_line_y):
        if self.pixels_per_char is None:
            return
        char_dist = pixel_dist / self.pixels_per_char

        # basically if regression and also line change then 
        # -- its not a regression!
        if self.prev_cy is not None:
            dy = abs(cy - self.prev_cy)
            if dy > LINE_RETURN_Y_THRESH * self.line_height_px:
                self.rev_state     = "FORWARD"
                self.rev_chars     = 0.0
                self.forward_chars = 0.0
                return

        # logic to count as a regression from
        # -- literature (paper ref: Hughes et al. 2014)
        if self.rev_state == "FORWARD":
            if char_dist > 0:
                self.forward_chars += char_dist
            elif char_dist < -0.1:
                if self.forward_chars >= FORWARD_MIN_CHARS:
                    self.rev_state      = "REVERSING"
                    self.rev_chars      = abs(char_dist)
                    self.rev_start_bucket = self.pixel_to_bucket(cx, cy)
                self.forward_chars = 0.0

        elif self.rev_state == "REVERSING":
            if char_dist < 0:
                self.rev_chars += abs(char_dist)
            elif char_dist > 0.1:
                if self.rev_chars >= REVERSAL_MIN_CHARS:
                    end_bucket = self.pixel_to_bucket(cx, cy)
                    self.reversals.append(Reversal(
                        start_bucket  = self.rev_start_bucket,
                        end_bucket    = end_bucket,
                        extent_chars  = round(self.rev_chars, 2),
                    ))
                self.rev_state     = "FORWARD"
                self.rev_chars     = 0.0
                self.forward_chars = char_dist

# gets hand position and runs coordinate code
def get_landmark8(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        tip = result.hand_landmarks[0][8]
        return int(tip.x * w), int(tip.y * h), True
    return w // 2, h // 2, False

#converts JPEG data from url to numpy for processing
def decode_frame(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

#takes openCV frame and sends it back to the website text
def encode_frame(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

# takes png into text data as well
def encode_png(frame):
    _, buf = cv2.imencode(".png", frame)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

#all of the following visual feedback onto the frame
#before the browser displays it
def draw_ring(frame, cx, cy, fraction):
    r = 40
    cv2.circle(frame, (cx, cy), r, (60, 60, 60), 3)
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0,
                int(360 * fraction), (0, 220, 255), 5)
    cv2.circle(frame, (cx, cy), 8, (0, 220, 255), -1)
    secs = str(max(1, int((1.0 - fraction) * COUNTDOWN_SECS) + 1))
    cv2.putText(frame, secs, (cx - 10, cy - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)

def draw_fingertip(frame, cx, cy, color=(0, 255, 180)):
    cv2.circle(frame, (cx, cy), 14, color, -1)
    cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)

def draw_vline(frame, x, lbl, color):
    cv2.line(frame, (x, 0), (x, frame.shape[0]), color, 2)
    cv2.putText(frame, lbl, (x + 5, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def draw_hline(frame, y, lbl, color):
    cv2.line(frame, (0, y), (frame.shape[1], y), color, 1)
    cv2.putText(frame, lbl, (5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def label(frame, text, color=(255, 220, 60)):
    cv2.putText(frame, text, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

def build_regression_counts(session):
    #mapping coordinate to total number of regressions
    counts = {}
    for rev in session.reversals:
        sb = rev.start_bucket
        eb = rev.end_bucket
        if sb is None or eb is None:
            continue
        # Both buckets on same line
        if sb[0] == eb[0]:
            lo = min(sb[1], eb[1])
            hi = max(sb[1], eb[1])
            for c in range(lo, hi + 1):
                key = (sb[0], c)
                counts[key] = counts.get(key, 0) + 1
        else:
            # Multi-line reversal — mark start and end buckets only
            counts[sb] = counts.get(sb, 0) + 1
            counts[eb] = counts.get(eb, 0) + 1
    return counts

# heat map generation
def generate_heatmap(session):
    if session.page_snapshot is None:
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 30
    else:
        canvas = session.page_snapshot.copy()

    h_c, w_c = canvas.shape[:2]
    reg_counts = build_regression_counts(session)

    all_avgs = []
    for speeds in session.bucket_speeds.values():
        if speeds:
            all_avgs.append(sum(speeds) / len(speeds))
    global_max = max(all_avgs) if all_avgs else 1.0
    global_min = min(all_avgs) if all_avgs else 0.0

    char_w = max(2, int(session.pixels_per_char))
    char_h = max(2, int(session.line_height_px * 0.75))
    hw = char_w // 2  # half width  — for centering
    hh = char_h // 2  # half height — for centering

    #figure out which chars were actually read based on trackng data
    visited = set(session.bucket_speeds.keys()) | set(reg_counts.keys())

    for bucket in visited:
        line_idx, char_in_line = bucket

        # center of each bucket in cam coord
        cx = int(session.cal_x1 + char_in_line * session.pixels_per_char)
        cy = int(session.cal_y1 + line_idx      * session.line_height_px)

        #rectangle centered on those coords (drawing each bucket!)
        x1, x2 = cx - hw, cx + hw
        y1, y2 = cy - hh, cy + hh

        if x1 < 0 or y1 < 0 or x2 > w_c or y2 > h_c:
            continue

        #if there is no data for this maek it flag as unread
        speeds = session.bucket_speeds.get(bucket, [])
        if not speeds:
            color = COLOR_UNREAD
            alpha = 0.20
        else:
            avg   = sum(speeds) / len(speeds)
            color = (COLOR_FAST if avg >= FAST_THRESH
                     else COLOR_MED if avg >= MED_THRESH
                     else COLOR_SLOW)
            norm  = (avg - global_min) / (global_max - global_min + 0.000001)
            #0.000001 is bascially to make sure we never divide by zero just in case max is min
            #this block of code just like on the range of speeds what is this bucket fall in
            alpha = 0.55 - norm * 0.3
            #and then this makes the thickness and opacity of the block adjust to that (ie. slower --> darker)

        # speed color overlay
        roi = canvas[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        canvas[y1:y2, x1:x2] = roi

        #regression darkness
        reg_n = reg_counts.get(bucket, 0)
        if reg_n > 0:
            reg_alpha = min(REG_MAX_ALPHA,
                            REG_BASE_ALPHA + (reg_n - 1) * REG_ALPHA_STEP)
            roi2     = canvas[y1:y2, x1:x2]
            overlay2 = roi2.copy()
            cv2.rectangle(overlay2, (0, 0), (x2 - x1, y2 - y1), REG_COLOR, -1)
            #setting darkness based on how many regs over same chars
            cv2.addWeighted(overlay2, reg_alpha, roi2, 1 - reg_alpha, 0, roi2)
            canvas[y1:y2, x1:x2] = roi2

    #key at bottom for readers/teachesr to know what they see
    legend = [
        (COLOR_FAST, f"Fast  (>{FAST_THRESH:.2f} c/s)"),
        (COLOR_MED,  f"Med   ({MED_THRESH:.2f}–{FAST_THRESH:.2f} c/s)"),
        (COLOR_SLOW, f"Slow  (<{MED_THRESH:.2f} c/s)"),
        (COLOR_UNREAD, "Not read"),
        (REG_COLOR,    "Regression (darker=more)"),
    ]
    lx = 10
    ly = h_c - 10 - len(legend) * 22
    for col, txt in legend:
        cv2.rectangle(canvas, (lx, ly), (lx + 16, ly + 14), col, -1)
        cv2.putText(canvas, txt, (lx + 22, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ly += 22

    return canvas


#csv generation
def generate_csv(session):
    buf    = io.StringIO()
    writer = csv.writer(buf)

    #line by line summaries
    writer.writerow(["=== PER LINE SUMMARY ==="])
    writer.writerow(["line_number", "chars_read", "duration_sec",
                     "chars_per_sec", "chars_per_min"])
    for rec in session.line_records:
        dur = rec["duration_sec"]
        cps = rec["chars_read"] / dur if dur > 0 else 0
        writer.writerow([rec["line"] + 1,
                         f"{rec['chars_read']:.1f}",
                         f"{dur:.2f}",
                         f"{cps:.3f}",
                         f"{cps*60:.1f}"])
    total_chars = sum(r["chars_read"] for r in session.line_records)
    total_dur = sum(r["duration_sec"] for r in session.line_records)
    total_cps = total_chars / total_dur if total_dur > 0 else 0
    writer.writerow([])
    writer.writerow(["TOTAL", f"{total_chars:.1f}", f"{total_dur:.2f}",
                     f"{total_cps:.3f}", f"{total_cps*60:.1f}"])

    #char by char speed tracking
    writer.writerow([])
    writer.writerow(["=== PER POSITION DATA ==="])
    writer.writerow(["line_idx", "char_in_line", "avg_cps", "avg_cpm"])
    for (line_idx, char_in_line), speeds in sorted(session.bucket_speeds.items()):
        avg = sum(speeds) / len(speeds) if speeds else 0
        writer.writerow([line_idx, char_in_line,
                         f"{avg:.3f}", f"{avg*60:.1f}"])

    #regressions (where they happened, how many chars they spanned)
    writer.writerow([])
    writer.writerow(["=== REGRESSIONS ==="])
    writer.writerow(["reversal_number", "start_line", "start_char_in_line",
                     "end_line", "end_char_in_line", "extent_chars"])
    for i, rev in enumerate(session.reversals):
        sb = rev.start_bucket or ("?", "?")
        eb = rev.end_bucket   or ("?", "?")
        writer.writerow([i + 1, sb[0], sb[1], eb[0], eb[1],
                         f"{rev.extent_chars:.2f}"])
    writer.writerow([])
    writer.writerow(["total_reversals", len(session.reversals)])

    # overall session data (genreal date and time, total chars read, page data like chars per line etc)
    writer.writerow([])
    writer.writerow(["=== METADATA ==="])
    writer.writerow(["pixels_per_char",   f"{session.pixels_per_char:.2f}"])
    writer.writerow(["chars_per_line",    session.chars_per_line])
    writer.writerow(["line_height_px",    f"{session.line_height_px:.2f}"])
    writer.writerow(["reversal_min_chars",REVERSAL_MIN_CHARS])
    writer.writerow(["forward_min_chars", FORWARD_MIN_CHARS])
    writer.writerow(["session_date",      datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    return buf.getvalue()

# state machine! handles each state according
# to session state (defined earlier)
# cal 1-4, snapshot, ready, countdown, reading, end
def process_frame(session, frame):
    h, w, _ = frame.shape
    cx, cy, finger = get_landmark8(frame)

    status = {
        "state": session.state,
        "chars_per_sec": round(session.chars_per_sec, 3),
        "chars_per_min": round(session.chars_per_sec * 60, 1),
        "current_line": session.current_line,
        "finger_detected": finger,
        "reversal_count": len(session.reversals),
    }

    # CAL_1 
    if session.state == "CAL_1":
        elapsed = time.time() - session.cal_start
        fraction = min(1.0, elapsed / COUNTDOWN_SECS)
        if finger:
            draw_ring(frame, cx, cy, fraction)
            draw_fingertip(frame, cx, cy, (255, 200, 0))
        label(frame, "Cal 1/4 — finger on char 1, line 1")
        if elapsed >= COUNTDOWN_SECS:
            if finger:
                session.cal_x1 = cx
                session.cal_y1 = cy
                session.state = "CAL_2"
                session.cal_start = time.time()
                status["beep"] = "low"
            else:
                session.cal_start = time.time()
        status["fraction"] = fraction

    # CAL_2
    elif session.state == "CAL_2":
        elapsed  = time.time() - session.cal_start
        fraction = min(1.0, elapsed / COUNTDOWN_SECS)
        draw_vline(frame, session.cal_x1, "C1", (0, 255, 100))
        if finger:
            draw_ring(frame, cx, cy, fraction)
            draw_fingertip(frame, cx, cy, (255, 200, 0))
        label(frame, "Cal 2/4 — finger on char 2, line 1")
        if elapsed >= COUNTDOWN_SECS:
            if finger:
                ppc = abs(cx - session.cal_x1)
                if ppc < 3:
                    session.state = "CAL_1"
                    session.cal_start = time.time()
                    status["beep"] = "error"
                else:
                    session.pixels_per_char = ppc
                    session.state = "CAL_3"
                    session.cal_start = time.time()
                    status["beep"] = "low"
            else:
                session.cal_start = time.time()
        status["fraction"] = fraction

    #CAL_3
    elif session.state == "CAL_3":
        elapsed  = time.time() - session.cal_start
        fraction = min(1.0, elapsed / COUNTDOWN_SECS)
        draw_vline(frame, session.cal_x1, "C1", (0, 255, 100))
        if finger:
            draw_ring(frame, cx, cy, fraction)
            draw_fingertip(frame, cx, cy, (255, 200, 0))
        label(frame, "Cal 3/4 — finger on LAST char of line 1")
        if elapsed >= COUNTDOWN_SECS:
            if finger:
                measured_cpl = round(abs(cx - session.cal_x1) /
                                     session.pixels_per_char) + 1
                if measured_cpl < 2:
                    session.state = "CAL_1"
                    session.cal_start = time.time()
                    status["beep"] = "error"
                else:
                    session.cal_x_last = cx
                    session.chars_per_line = measured_cpl
                    session.state = "CAL_4"
                    session.cal_start = time.time()
                    status["beep"]  = "low"
            else:
                session.cal_start = time.time()
        status["fraction"] = fraction

    #CAL_4
    elif session.state == "CAL_4":
        elapsed  = time.time() - session.cal_start
        fraction = min(1.0, elapsed / COUNTDOWN_SECS)
        draw_vline(frame, session.cal_x1, "C1",  (0, 255, 100))
        draw_vline(frame, session.cal_x_last, "end", (0, 180, 255))
        draw_hline(frame, session.cal_y1, "L1",  (0, 200, 255))
        if finger:
            draw_ring(frame, cx, cy, fraction)
            draw_fingertip(frame, cx, cy, (255, 200, 0))
        label(frame, "Cal 4/4 — finger on char 1, line 2")
        if elapsed >= COUNTDOWN_SECS:
            if finger:
                lh = abs(cy - session.cal_y1)
                if lh < 3:
                    session.state = "CAL_1"
                    session.cal_start = time.time()
                    status["beep"] = "error"
                else:
                    session.line_height_px = lh
                    session.state = "SNAPSHOT"
                    session.snapshot_start = time.time()
                    status["beep"] = "high"
            else:
                session.cal_start = time.time()
        status["fraction"] = fraction

    #snapshot 
    elif session.state == "SNAPSHOT":
        elapsed   = time.time() - session.snapshot_start
        remaining = max(0, SNAPSHOT_WAIT_SECS - elapsed)

        if remaining > 0:
            # still counting down — show overlay on display only
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.putText(frame, str(int(remaining) + 1),
                        (w // 2 - 20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 220, 255), 4, cv2.LINE_AA)
            label(frame, "Remove hand — taking page photo!", (0, 220, 255))
        else:
            # =countdown done — save frame with ZERO drawings on it
            session.page_snapshot = frame.copy()
            session.state = "READY"
            status["beep"] = "low"
            status["snapshot_taken"] = True

        status["snapshot_remaining"] = remaining

    #ready
    elif session.state == "READY":
        draw_vline(frame, session.cal_x1,     "start", (0, 255, 100))
        draw_vline(frame, session.cal_x_last, "end",   (0, 180, 255))
        if finger:
            draw_fingertip(frame, cx, cy, (100, 200, 255))
        label(frame, "Tap screen when ready to start", (100, 200, 255))

    # countdown
    elif session.state == "COUNTDOWN":
        elapsed = time.time() - session.countdown_start
        remaining = max(0, READING_LEAD_SECS - elapsed)
        if finger:
            draw_fingertip(frame, cx, cy, (100, 200, 255))
        cv2.putText(frame, str(int(remaining) + 1),
                    (w // 2 - 30, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5, cv2.LINE_AA)
        label(frame, "Get ready to read!", (255, 255, 255))
        status["countdown_remaining"] = remaining
        if elapsed >= READING_LEAD_SECS:
            session.state           = "READING"
            session.line_start_time = time.time()
            session.prev_time       = time.time()
            session.last_move_time  = time.time()
            status["beep"]          = "high"

    # reading
    elif session.state == "READING":
        draw_vline(frame, session.cal_x1,     "start", (0, 255, 100))
        draw_vline(frame, session.cal_x_last, "end",   (0, 180, 255))

        if finger:
            draw_fingertip(frame, cx, cy)
            curr_time = time.time()
            dt = curr_time - session.prev_time if session.prev_time else 0

            if dt > 0 and session.prev_cx is not None:
                pixel_dist = cx - session.prev_cx

                #regression tracking: excluding all the line changes
                if session.line_change_cooldown > 0:
                    session.line_change_cooldown -= 1
                else:
                    session.update_reversal(pixel_dist, cx, cy, session.current_line_y)

                if pixel_dist > 1:
                    session.last_move_time = curr_time
                    if pixel_dist < session.pixels_per_char * 10:
                        raw_cps = (pixel_dist / session.pixels_per_char) / dt
                        session.line_chars += pixel_dist / session.pixels_per_char
                        session.speed_buffer.append(raw_cps)
                        if len(session.speed_buffer) > SMOOTH_WINDOW:
                            session.speed_buffer.pop(0)

                # record bucket every frame finger is detected, not just on forward movement
                if session.chars_per_sec > 0:
                    bucket = session.pixel_to_bucket(cx, cy)
                    session.record_bucket_speed(bucket, session.chars_per_sec)

            if session.speed_buffer:
                session.chars_per_sec = (sum(session.speed_buffer) /
                                         len(session.speed_buffer))

            # line break on pause
            # set anchor Y for current line when reading starts
            if session.current_line_y is None:
                session.current_line_y = cy

            #amount y cord drops by
            y_drop = cy - session.current_line_y

            #too strict! x reset
            #x_reset   = cx < (session.cal_x1 + session.pixels_per_char * 3)
            # also not that helpful could do just like inline didnt need var for it 
            # new_line_by_y = y_drop > (session.line_height_px * 0.7) and x_reset

            # Fallback: long pause - didn;t work; but dont have the nergy to delete lol
            # new_line_by_pause = (session.last_move_time and
                # time.time() - session.last_move_time > LINE_PAUSE_SECS and
                # session.line_chars > 0)

            # Line break: Y has dropped by ~1 line height (moved to next line)
            # AND finger has moved back toward the start (X reset)
            #this signifies a line change
            if y_drop > (session.line_height_px * 0.6) and session.line_chars > 0:
                duration = time.time() - session.line_start_time
                session.line_records.append({
                    "line":         session.current_line,
                    "chars_read":   session.line_chars,
                    "duration_sec": duration,
                })
                session.current_line += 1
                session.current_line_y = cy
                session.line_start_time = time.time()
                session.line_chars = 0.0
                session.speed_buffer.clear()
                session.chars_per_sec = 0.0
                status["new_line"] = session.current_line
                #cools down for this many frames after line change so that regression isn't counted when the line changed
                session.line_change_cooldown = 6 

            session.prev_cx = cx
            session.prev_cy = cy
            session.prev_time = curr_time

        cps = session.chars_per_sec
        cpm = cps * 60
        color = (80, 255, 120) if cps < 2 else (0, 220, 255) if cps < 5 else (0, 140, 255)
        label(frame,
              f"Line {session.current_line+1}  "
              f"{cps:.2f} c/s  {cpm:.0f} c/min  |  rev: {len(session.reversals)}",
              color)
        status["chars_per_sec"]  = round(cps, 3)
        status["chars_per_min"]  = round(cpm, 1)
        status["reversal_count"] = len(session.reversals)

    # end
    elif session.state == "ENDED":
        label(frame, "Session ended — check download", (0, 255, 180))

    status["cx"] = cx
    status["cy"] = cy
    return frame, status

# connecting to the browser and creating a new session 
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    print("[WS] Client connected — starting at CAL_1")

    try:
        # getting feedback and processing it (ie. tap)
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # type input
            if msg.get("type") == "select_stimulus":
                await ws.send_text(json.dumps({"type": "stimulus_selected"}))
                continue


            # tap input
            if msg.get("type") == "tap":
                if session.state == "READY":
                    session.state           = "COUNTDOWN"
                    session.countdown_start = time.time()
                    await ws.send_text(json.dumps({"type": "countdown_started"}))

                elif session.state == "READING":
                    session.state = "ENDED"
                    duration = time.time() - session.line_start_time
                    if session.line_chars > 0:
                        session.line_records.append({
                            "line":         session.current_line,
                            "chars_read":   session.line_chars,
                            "duration_sec": duration,
                        })
                    csv_data = generate_csv(session)
                    heatmap = generate_heatmap(session)
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                    await ws.send_text(json.dumps({
                        "type":             "session_ended",
                        "csv":              csv_data,
                        "csv_filename":     f"session_{timestamp}.csv",
                        "heatmap":          encode_png(heatmap),
                        "heatmap_filename": f"heatmap_{timestamp}.png",
                        "num_lines":        len(session.line_records),
                        "num_reversals":    len(session.reversals),
                    }))
                    print(f"[WS] Ended — {len(session.line_records)} lines, "
                          f"{len(session.reversals)} reversals")
                continue

            # frame (general updating)
            if msg.get("type") == "frame" and session.state != "ENDED":
                frame = decode_frame(msg.get("data", ""))
                if frame is None:
                    continue
                try:
                    annotated, status = process_frame(session, frame)
                    await ws.send_text(json.dumps({
                        "type": "frame",
                        "image": encode_frame(annotated),
                        "status": status,
                    }))
                except Exception as e:
                    print(f"[Frame error] {e}")

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
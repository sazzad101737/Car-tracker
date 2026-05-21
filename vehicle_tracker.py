import cv2
import numpy as np
from collections import defaultdict, deque

VIDEO_PATH = "traffic.mp4"
SCALE = 0.6
MIN_AREA = 1500
MAX_AREA = 80000
LINE_A_FRAC = 0.42
LINE_B_FRAC = 0.58
REAL_DISTANCE_M = 10.0
MAX_LOST_FRAMES = 8
IOU_THRESHOLD = 0.25

line_y_A = None
line_y_B = None


def iou(boxA, boxB):
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - inter
    return inter / union


def centroid(x, y, w, h):
    return (x + w // 2, y + h // 2)


def classify_vehicle(w, h):
    area = w * h
    if area < 4000:
        return "Motorbike", (255, 200, 0)
    elif area < 14000:
        return "Car", (0, 255, 100)
    else:
        return "Truck/Bus", (0, 100, 255)


class Track:
    _next_id = 1

    def __init__(self, box, fps):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.cx, self.cy = centroid(*box)
        self.lost = 0
        self.path = deque(maxlen=30)
        self.path.append((self.cx, self.cy))
        self.crossed_A = False
        self.crossed_B = False
        self.time_A = None
        self.time_B = None
        self.speed_kmh = None
        self.fps = fps
        self.label, self.color = classify_vehicle(box[2], box[3])
        self.counted = False

    def update(self, box, frame_time):
        self.box = box
        self.cx, self.cy = centroid(*box)
        self.path.append((self.cx, self.cy))
        self.lost = 0
        self.label, self.color = classify_vehicle(box[2], box[3])
        self._check_lines(frame_time)

    def _check_lines(self, frame_time):
        if not self.crossed_A and line_y_A is not None:
            if abs(self.cy - line_y_A) < 10:
                self.crossed_A = True
                self.time_A = frame_time
        if self.crossed_A and not self.crossed_B and line_y_B is not None:
            if abs(self.cy - line_y_B) < 10:
                self.crossed_B = True
                self.time_B = frame_time
                self._estimate_speed()

    def _estimate_speed(self):
        if self.time_A and self.time_B:
            elapsed = self.time_B - self.time_A
            if elapsed > 0:
                self.speed_kmh = round((REAL_DISTANCE_M / elapsed) * 3.6, 1)


def main():
    global line_y_A, line_y_B

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)

    tracks = []
    counts = defaultdict(int)
    direction_count = {"Up-Down": 0, "Down-Up": 0}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_time = frame_idx / fps

        h0, w0 = frame.shape[:2]
        frame = cv2.resize(frame, (int(w0 * SCALE), int(h0 * SCALE)))
        H, W = frame.shape[:2]

        if line_y_A is None:
            line_y_A = int(H * LINE_A_FRAC)
            line_y_B = int(H * LINE_B_FRAC)

        mask = bg_sub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append((x, y, w, h))

        matched_det = set()
        matched_track = set()

        for ti, track in enumerate(tracks):
            best_iou = IOU_THRESHOLD
            best_di = -1
            for di, det in enumerate(detections):
                if di in matched_det:
                    continue
                score = iou(track.box, det)
                if score > best_iou:
                    best_iou = score
                    best_di = di
            if best_di >= 0:
                track.update(detections[best_di], frame_time)
                matched_det.add(best_di)
                matched_track.add(ti)
            else:
                track.lost += 1

        for di, det in enumerate(detections):
            if di not in matched_det:
                tracks.append(Track(det, fps))

        alive = []
        for track in tracks:
            if track.lost > MAX_LOST_FRAMES:
                if not track.counted and track.crossed_A:
                    counts[track.label] += 1
                    track.counted = True
                    if len(track.path) >= 2:
                        dy = track.path[-1][1] - track.path[0][1]
                        d = "Up-Down" if dy > 0 else "Down-Up"
                        direction_count[d] += 1
                continue
            alive.append(track)
        tracks = alive

        cv2.line(frame, (0, line_y_A), (W, line_y_A), (255, 255, 0), 2)
        cv2.line(frame, (0, line_y_B), (W, line_y_B), (0, 200, 255), 2)
        cv2.putText(frame, "Line A", (5, line_y_A - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "Line B", (5, line_y_B - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        for track in tracks:
            x, y, w, h = track.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), track.color, 2)
            pts = list(track.path)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                col = tuple(int(c * alpha) for c in track.color)
                cv2.line(frame, pts[i - 1], pts[i], col, 1)
            spd_txt = f" {track.speed_kmh}km/h" if track.speed_kmh else ""
            cv2.putText(frame, f"#{track.id} {track.label}{spd_txt}", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, track.color, 1)

        total = sum(counts.values())
        panel_lines = [
            f"Total: {total}",
            f"Cars: {counts.get('Car', 0)}",
            f"Motorbikes: {counts.get('Motorbike', 0)}",
            f"Trucks/Bus: {counts.get('Truck/Bus', 0)}",
            f"Up-Down: {direction_count['Up-Down']}",
            f"Down-Up: {direction_count['Down-Up']}",
            f"Active: {len(tracks)}",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (230, 22 + 22 * len(panel_lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for i, line in enumerate(panel_lines):
            cv2.putText(frame, line, (14, 28 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        cv2.imshow("Vehicle Tracker - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n===== FINAL COUNT =====")
    print(f"Total: {sum(counts.values())}")
    for label, n in counts.items():
        print(f"  {label}: {n}")
    print(f"Up-Down: {direction_count['Up-Down']}")
    print(f"Down-Up: {direction_count['Down-Up']}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

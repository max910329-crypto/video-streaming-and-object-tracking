# track.py — Minimal skeleton: YOLO detection + Hungarian assignment
# pip install ultralytics opencv-python numpy scipy

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# ========== 你只要改這幾個「預設參數」就好 ==========
INPUT_VIDEO  = "easy_9.mp4"
OUTPUT_VIDEO = "easy_9_out.mp4"

YOLO_WEIGHTS = "yolo11x.pt"   # 可換 yolov8s.pt
IMG_SIZE     = 512

CONF_TH      = 0.30            # 偵測置信門檻
IOU_GATE     = 0.05           # 太低不配
MAX_AGE      = 45              # 連續幾幀沒配到就刪軌跡
MIN_HITS     = 6               # 新軌跡連續幾幀配到才確立

GATE_CHI2_4D = 9.49     # 95% χ² gate for z=[cx,cy,w,h]
Q_pos = 0.02            # 過程雜訊（動作快/低FPS可拉到 0.05）
R_pos = 1.5             # 量測雜訊（偵測不穩/遠距可拉到 2.0）
LAMBDA_IOU = 1.0          # 成本裡 IoU 權重

TRACE_LEN    = 250              # 軌跡長度；0=不畫

# ==== 追加的參數（保持只用 IoU，不用馬氏距離）====
EDGE_MARGIN    = 40     # 邊界區寬度(px)
ENTER_FRAMES   = 3      # 連續在畫面內(非邊界)幾幀才算正式「入場」
MIN_SPEED_PX   = 0.6    # 新生確認前的平均速度下限（過低像背景就不確認）
NEW_TRACK_TH   = 0.80   # 新生更嚴格（用 conf 判斷；你有需要時調）
# 成本 = w1*(1-IOU) + w2*center_norm；center_norm = 中心距離 / 影像對角線
LAMBDA_IOU     = 0.7
LAMBDA_CENTER  = 0.3

# ================================================
def _diag_len(W, H):
    return (W*W + H*H) ** 0.5

def _center_norm(a, b, diag):
    # a,b: [x1,y1,x2,y2]
    cax, cay = 0.5*(a[0]+a[2]), 0.5*(a[1]+a[3])
    cbx, cby = 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])
    return ((cax-cbx)**2 + (cay-cby)**2) ** 0.5 / (diag + 1e-6)

def _in_edge_zone(bbox, W, H):
    x1,y1,x2,y2 = bbox
    return (x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or
            x2 > W - EDGE_MARGIN or y2 > H - EDGE_MARGIN)

def _hsv_to_bgr(h, s, v):
    hsv = np.uint8([[[int(h*179)%180, int(s*255), int(v*255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def vivid_color_for_id(tid: int):
    """用黃金比例跳色，避免顏色彼此太像；高飽和高亮度。"""
    step = 0.61803398875
    h = (step * (tid*7 + 3)) % 1.0
    return _hsv_to_bgr(h, 0.95, 1.0)

def iou_xyxy(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union


def bbox_center(b):
    return (0.5*(b[0]+b[2]), 0.5*(b[1]+b[3]))

def bbox_from_state(x):
    cx, cy, w, h = x[0], x[1], max(1.0, x[2]), max(1.0, x[3])
    x1, y1 = int(cx - w/2), int(cy - h/2)
    x2, y2 = int(cx + w/2), int(cy + h/2)
    return [x1, y1, x2, y2]

class KalmanBox:
    def __init__(self, cx, cy, w, h, dt=1.0, q=0.02, r=1.5):
        # x: [cx, cy, w, h, vx, vy, vw, vh]^T
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = dt           # position += velocity*dt
        self.H = np.zeros((4, 8))
        self.H[0,0]=self.H[1,1]=self.H[2,2]=self.H[3,3]=1.0

        # 初始不確定性
        self.P = np.eye(8)*10.0
        self.P[4:,4:] *= 100.0

        # 過程雜訊 Q、量測雜訊 R（以尺度比例調整）
        self.Q = np.eye(8)*q
        self.Q[4:,4:] *= 10*q
        self.R = np.eye(4)*r

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        # z = [cx, cy, w, h]
        z = np.asarray(z, dtype=float)
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return y, S

    def mahalanobis(self, z):
        z = np.asarray(z, dtype=float)
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        m2 = float(y.T @ np.linalg.inv(S) @ y)  # squared distance
        return m2


class Detector:
    """最簡 YOLO 人偵測器（只保留 person=cls 0）"""
    def __init__(self, weights=YOLO_WEIGHTS, imgsz=IMG_SIZE, conf=CONF_TH, device=0):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf  = conf
        self.device = device
        self.PERSON = 0

    def detect(self, frame):
        res = self.model(
            frame, imgsz=self.imgsz, conf=self.conf,
            verbose=False, device=self.device, classes=[self.PERSON], half=True
        )[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            out.append([int(x1), int(y1), int(x2), int(y2)])
        return out


class Track:
    _next_id = 1
    def __init__(self, bbox):
        self.disp_id = None
        self.frames_inside = 0       # 連續在非邊界的幀數
        self.has_entered   = False   # 是否已正式入場
        self.prev_center   = bbox_center(bbox)  # 前一幀中心
        self.speed_hist    = []      # 最近位移量（用來過濾背景假軌）
        self.id  = Track._next_id; Track._next_id += 1
        self.bbox = bbox
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.trace = [] if TRACE_LEN > 0 else None
        if self.trace is not None:
            cx, cy = bbox_center(bbox); self.trace.append((int(cx), int(cy)))
        self.color = vivid_color_for_id(self.id)
        cx, cy = bbox_center(bbox)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.kf = KalmanBox(cx, cy, w, h, dt=1.0, q=Q_pos, r=R_pos)
        self.kf.predict()  # 初始化一次


class Tracker:
    """Kalman + 匈牙利追蹤器（使用馬氏距離 gate；不再用 center gate）"""
    def __init__(self, iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS):
        self.next_disp_id = 1
        self.tracks = []
        self.iou_gate = iou_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.total_count = 0  # 確立過的 ID 數

    def _too_close_to_existing(self, det, H=None, W=None):
        """抑制明顯重複的新生：
        1) 若偵測框緊貼畫面邊界（容易是殘影/離場殘框），先不新生
        2) 若與任一已存在軌跡的『預測框』IoU 很高，視為同一人，不新生
        """
        x1, y1, x2, y2 = det
        # --- 1) 邊界守門（避免剛進場/出場抖動誤新生）---
        if H is not None and W is not None:
            edge = int(0.04 * min(H, W))  # 4% 畫面邊界
            if x1 <= edge or y1 <= edge or (W - x2) <= edge or (H - y2) <= edge:
                return True

        # --- 2) 對已存在軌跡的「KF 預測框」做重疊檢查 ---
        for t in self.tracks:
            if hasattr(t, "pred_bbox"):
                if iou_xyxy(t.pred_bbox, det) > 0.60:  # 門檻可微調 0.55~0.70
                    return True
        return False

    def _cost_matrix(self, dets, H, W):
        T, D = len(self.tracks), len(dets)
        if T == 0 or D == 0:
            return np.zeros((T, D), dtype=float)

        C = np.full((T, D), 1e6, dtype=float)
        diag = _diag_len(W, H)             # NEW

        for i, t in enumerate(self.tracks):
            pb = t.pred_bbox
            for j, d in enumerate(dets):
                IoU = iou_xyxy(pb, d)
                if IoU < IOU_GATE:
                    continue
                cnorm = _center_norm(pb, d, diag)           # NEW
                # 成本：IoU 為主、中心距離為輔助（仍然不碰馬氏距離）
                C[i, j] = LAMBDA_IOU * (1.0 - IoU) + LAMBDA_CENTER * cnorm
        return C

    def update(self, frame, dets):
        H, W = frame.shape[:2]

        # 0) 每幀先 predict 一次（所有軌跡）
        for t in self.tracks:
            t.kf.predict()
            t.pred_bbox = bbox_from_state(t.kf.x)

        # 1) 匹配（匈牙利）
        matches, unmatched_t, unmatched_d = [], list(range(len(self.tracks))), list(range(len(dets)))
        if self.tracks and dets:
            C = self._cost_matrix(dets, H, W)  # ← 不再呼叫 predict
            r, c = linear_sum_assignment(C)
            unmatched_t = set(range(len(self.tracks)))
            unmatched_d = set(range(len(dets)))
            for i, j in zip(r, c):
                if C[i, j] >= 1e6:
                    continue
                matches.append((i, j))
                unmatched_t.discard(i)
                unmatched_d.discard(j)
            unmatched_t = sorted(list(unmatched_t))
        unmatched_d = sorted(list(unmatched_d))

        # 2) 更新匹配成功的軌跡
        for ti, dj in matches:
            t = self.tracks[ti]
            d = dets[dj]
            cx = 0.5*(d[0]+d[2]); cy = 0.5*(d[1]+d[3])
            w  = max(1.0, d[2]-d[0]); h = max(1.0, d[3]-d[1])

            t.kf.update([cx, cy, w, h])
            t.bbox = bbox_from_state(t.kf.x)   # 用濾波後狀態更新輸出框
            # 維護速度與入場
            cx, cy = bbox_center(t.bbox)
            if t.prev_center is not None:
                dx = cx - t.prev_center[0]
                dy = cy - t.prev_center[1]
                t.speed_hist.append((dx*dx + dy*dy) ** 0.5)
                if len(t.speed_hist) > 10:
                    t.speed_hist.pop(0)
            t.prev_center = (cx, cy)

            if not _in_edge_zone(t.bbox, W, H):
                t.frames_inside += 1
                if t.frames_inside >= ENTER_FRAMES:
                    t.has_entered = True
            else:
                t.frames_inside = 0

            t.hits += 1
            t.time_since_update = 0
            if t.trace is not None:
                cx, cy = bbox_center(t.bbox); t.trace.append((int(cx), int(cy)))
                if len(t.trace) > TRACE_LEN:
                    t.trace.pop(0)

        # 3) 老化未匹配的舊軌跡
        alive = []
        for idx, trk in enumerate(self.tracks):
            if idx in unmatched_t:
                trk.time_since_update += 1
                trk.bbox = trk.pred_bbox      # 用預測框繼續顯示（但不要把預測點寫入 trace）
                cx, cy = bbox_center(trk.bbox)
                if trk.prev_center is not None:
                    dx = cx - trk.prev_center[0]
                    dy = cy - trk.prev_center[1]
                    trk.speed_hist.append((dx*dx + dy*dy) ** 0.5)
                    if len(trk.speed_hist) > 10:
                        trk.speed_hist.pop(0)
                trk.prev_center = (cx, cy)

                if not _in_edge_zone(trk.bbox, W, H):
                    trk.frames_inside += 1
                    if trk.frames_inside >= ENTER_FRAMES:
                        trk.has_entered = True
                else:
                    trk.frames_inside = 0

            if (not trk.confirmed) and (trk.hits >= self.min_hits):
                trk.confirmed = True
                if trk.disp_id is None:
                    trk.disp_id = self.next_disp_id
                    self.next_disp_id += 1
                self.total_count += 1

            if trk.time_since_update <= self.max_age:
                alive.append(trk)
        self.tracks = alive
        def _too_close_to_existing(self, det):
            for t in self.tracks:
                if hasattr(t, "pred_bbox"):
                    if iou_xyxy(t.pred_bbox, det) > 0.30:  # 只用 IoU 檢查接近
                        return True
            return False

        # 4) 為未匹配的 detection 新增軌跡（加出生抑制）
        for dj in unmatched_d:
            d = dets[dj]
            if not self._too_close_to_existing(d, H, W):
                self.tracks.append(Track(d))



    def draw(self, frame):
        vis = frame
        for t in self.tracks:
            if not t.confirmed:
                continue
            x1, y1, x2, y2 = map(int, t.bbox)
            # ★ 彩色框（用各自的 t.color），外加黑色描邊讓對比更好
            cv2.rectangle(vis, (x1, y1), (x2, y2), t.color, 2)          # 彩色框

            # ★ 文字永遠白色
            cv2.putText(vis, f"{t.disp_id}", (x1+4, y1-5),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # ===== 高效漸透明尾跡（單次疊圖）=====
            if t.trace and len(t.trace) > 1:
                n = len(t.trace)
                thickness = 2
                min_alpha, max_alpha = 0.15, 0.85

                # 單張 overlay，畫完一次性疊回
                overlay = np.zeros_like(vis, dtype=np.uint8)

                # 彩色主線畫在 overlay，顏色已預乘 alpha
                for k in range(1, n):
                    p1, p2 = t.trace[k - 1], t.trace[k]
                    a = k / n
                    alpha = min_alpha + (max_alpha - min_alpha) * a  # 由舊→新 漸亮
                    col = tuple(int(c * alpha) for c in t.color)     # 預乘 alpha
                    cv2.line(overlay, p1, p2, col, thickness, cv2.LINE_AA)

                # 一次性把 overlay 疊回 vis（overlay 已預乘，不需再縮權重）
                cv2.add(overlay, vis, dst=vis)


        return vis


def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {INPUT_VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    det = Detector()
    tracker = Tracker()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)            # [[x1,y1,x2,y2], ...]
        tracker.update(frame, dets)         # 更新軌跡
        vis = tracker.draw(frame)           # 畫框與ID
        cv2.putText(vis, f"People count: {tracker.total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
        writer.write(vis)

    cap.release()
    writer.release()
    print(f"Saved: {OUTPUT_VIDEO} | total people seen: {tracker.total_count}")


if __name__ == "__main__":
    main()

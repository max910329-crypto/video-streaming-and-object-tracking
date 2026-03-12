# -*- coding: utf-8 -*-
# track_note.py — 在不改變邏輯的前提下，為每一行加入繁體中文註解與函式用途說明。
# 注意：僅新增註解與說明，不修改任何可執行語句。

# track.py — Minimal skeleton: YOLO detection + Hungarian assignment
# pip install ultralytics opencv-python numpy scipy
# （空行）

# 導入模組：cv2
import cv2
# 一般語句
import numpy as np
# 從模組 ultralytics 導入：YOLO
from ultralytics import YOLO
# 從模組 scipy.optimize 導入：linear_sum_assignment
from scipy.optimize import linear_sum_assignment
# （空行）

# （空行）

# ========== 你只要改這幾個「預設參數」就好 ==========
# 設定參數/變數 INPUT_VIDEO：輸入影片檔名/路徑（要被追蹤的人物影片）
INPUT_VIDEO  = "easy_9.mp4"
# 設定參數/變數 OUTPUT_VIDEO：輸出結果影片檔名（畫上框與ID後的成品）
OUTPUT_VIDEO = "easy_9_out.mp4"
# （空行）

# 設定參數/變數 YOLO_WEIGHTS：YOLO 權重檔路徑（可換其他模型權重）
YOLO_WEIGHTS = "yolo11x.pt"   # 可換 yolov8s.pt
# 設定參數/變數 IMG_SIZE：YOLO 推論的輸入尺寸（縮放大小）
IMG_SIZE     = 512
# （空行）

# 設定參數/變數 CONF_TH：偵測信心分數門檻（過低會出現較多誤檢）
CONF_TH      = 0.30            # 偵測置信門檻
# 設定參數/變數 IOU_GATE：IoU 篩選門檻；低於此值的配對視為不合理
IOU_GATE     = 0.05           # 太低不配
# 設定參數/變數 MAX_AGE：連續幾幀沒有匹配就刪除軌跡（老化）
MAX_AGE      = 45              # 連續幾幀沒配到就刪軌跡
# 設定參數/變數 MIN_HITS：新軌跡至少連續匹配幾幀才算『確立』
MIN_HITS     = 6               # 新軌跡連續幾幀配到才確立
# （空行）

# 設定參數/變數 GATE_CHI2_4D：4維量測的卡方門檻（95%），常用於馬氏距離gate
GATE_CHI2_4D = 9.49     # 95% χ² gate for z=[cx,cy,w,h]
# 設定參數/變數 Q_pos：卡爾曼過程雜訊強度（動作快/低FPS可拉高）
Q_pos = 0.02            # 過程雜訊（動作快/低FPS可拉到 0.05）
# 設定參數/變數 R_pos：量測雜訊強度（偵測不穩/距離遠可拉高）
R_pos = 1.5             # 量測雜訊（偵測不穩/遠距可拉到 2.0）
# 設定參數/變數 LAMBDA_IOU：成本函式中 IoU 的權重（數值越高越注重IoU）
LAMBDA_IOU = 1.0          # 成本裡 IoU 權重
# （空行）

# 設定參數/變數 TRACE_LEN：尾跡要保留的點數（0代表不畫尾跡）
TRACE_LEN    = 250              # 軌跡長度；0=不畫
# （空行）

# ==== 追加的參數（保持只用 IoU，不用馬氏距離）====
# 設定參數/變數 EDGE_MARGIN：畫面邊緣保護帶寬度（像素）
EDGE_MARGIN    = 40     # 邊界區寬度(px)
# 設定參數/變數 ENTER_FRAMES：在非邊界連續幾幀才視為正式入場
ENTER_FRAMES   = 3      # 連續在畫面內(非邊界)幾幀才算正式「入場」
# 設定參數/變數 MIN_SPEED_PX：新生確認前的平均速度下限（過低視為背景）
MIN_SPEED_PX   = 0.6    # 新生確認前的平均速度下限（過低像背景就不確認）
# 設定參數/變數 NEW_TRACK_TH：新生更嚴格門檻（例如基於conf或其他條件）
NEW_TRACK_TH   = 0.80   # 新生更嚴格（用 conf 判斷；你有需要時調）
# 成本 = w1*(1-IOU) + w2*center_norm；center_norm = 中心距離 / 影像對角線
# 設定參數/變數 LAMBDA_IOU：成本函式中 IoU 的權重（數值越高越注重IoU）
LAMBDA_IOU     = 0.7
# 設定參數/變數 LAMBDA_CENTER：成本函式中中心距離的權重
LAMBDA_CENTER  = 0.3
# （空行）

# ================================================
# 函式說明：計算影像寬高的對角線長度，用於將中心距離正規化到 0~1。
# 定義函式 _diag_len：計算影像寬高的對角線長度，用於將中心距離正規化到 0~1。
def _diag_len(W, H):
    # 回傳函式結果
    return (W*W + H*H) ** 0.5
# （空行）

# 函式說明：計算兩框中心之間的像素距離，並除以影像對角線得到正規化距離。
# 定義函式 _center_norm：計算兩框中心之間的像素距離，並除以影像對角線得到正規化距離。
def _center_norm(a, b, diag):
    # a,b: [x1,y1,x2,y2]
    # 設定變數 cax, cay
    cax, cay = 0.5*(a[0]+a[2]), 0.5*(a[1]+a[3])
    # 設定變數 cbx, cby
    cbx, cby = 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])
    # 回傳函式結果
    return ((cax-cbx)**2 + (cay-cby)**2) ** 0.5 / (diag + 1e-6)
# （空行）

# 函式說明：判斷給定框是否落在畫面邊界區域內（避免入/離場抖動造成誤新生）。
# 定義函式 _in_edge_zone：判斷給定框是否落在畫面邊界區域內（避免入/離場抖動造成誤新生）。
def _in_edge_zone(bbox, W, H):
    # 設定變數 x1,y1,x2,y2
    x1,y1,x2,y2 = bbox
    # 回傳函式結果
    return (x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or
            # 一般語句
            x2 > W - EDGE_MARGIN or y2 > H - EDGE_MARGIN)
# （空行）

# 函式說明：把 HSV 色彩值轉成 OpenCV 使用的 BGR 顏色。
# 定義函式 _hsv_to_bgr：把 HSV 色彩值轉成 OpenCV 使用的 BGR 顏色。
def _hsv_to_bgr(h, s, v):
    # 設定變數 hsv
    hsv = np.uint8([[[int(h*179)%180, int(s*255), int(v*255)]]])
    # 設定變數 bgr
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    # 回傳函式結果
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
# （空行）

# 函式說明：根據追蹤ID用黃金比例在色相上取樣，產生高飽和亮色以區分不同ID。
# 定義函式 vivid_color_for_id：根據追蹤ID用黃金比例在色相上取樣，產生高飽和亮色以區分不同ID。
def vivid_color_for_id(tid: int):
    # 一般語句
    """用黃金比例跳色，避免顏色彼此太像；高飽和高亮度。"""
    # 設定變數 step
    step = 0.61803398875
    # 設定變數 h
    h = (step * (tid*7 + 3)) % 1.0
    # 回傳函式結果
    return _hsv_to_bgr(h, 0.95, 1.0)
# （空行）

# 函式說明：計算兩個 xyxy 框之間的 IoU（交集比上聯集）。
# 定義函式 iou_xyxy：計算兩個 xyxy 框之間的 IoU（交集比上聯集）。
def iou_xyxy(a, b):
    # 設定變數 xA, yA
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    # 設定變數 xB, yB
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    # 設定變數 inter
    inter = max(0, xB - xA) * max(0, yB - yA)
    # 設定變數 areaA
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    # 設定變數 areaB
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    # 設定變數 union
    union = areaA + areaB - inter + 1e-6
    # 回傳函式結果
    return inter / union
# （空行）

# （空行）

# 函式說明：回傳框中心點 (cx, cy)。
# 定義函式 bbox_center：回傳框中心點 (cx, cy)。
def bbox_center(b):
    # 回傳函式結果
    return (0.5*(b[0]+b[2]), 0.5*(b[1]+b[3]))
# （空行）

# 函式說明：把卡爾曼狀態 [cx,cy,w,h] 轉回對應的 xyxy 邊界框。
# 定義函式 bbox_from_state：把卡爾曼狀態 [cx,cy,w,h] 轉回對應的 xyxy 邊界框。
def bbox_from_state(x):
    # 設定變數 cx, cy, w, h
    cx, cy, w, h = x[0], x[1], max(1.0, x[2]), max(1.0, x[3])
    # 設定變數 x1, y1
    x1, y1 = int(cx - w/2), int(cy - h/2)
    # 設定變數 x2, y2
    x2, y2 = int(cx + w/2), int(cy + h/2)
    # 回傳函式結果
    return [x1, y1, x2, y2]
# （空行）

# 類別 KalmanBox：建立用於邊界框的 8 維卡爾曼濾波器（位置、尺寸與其速度）。
# 定義類別 KalmanBox（追蹤邏輯結構）
class KalmanBox:
    # 函式說明：建立用於邊界框的 8 維卡爾曼濾波器（位置、尺寸與其速度）。
    # 定義函式 KalmanBox.__init__：建立用於邊界框的 8 維卡爾曼濾波器（位置、尺寸與其速度）。
    def __init__(self, cx, cy, w, h, dt=1.0, q=0.02, r=1.5):
        # x: [cx, cy, w, h, vx, vy, vw, vh]^T
        # 設定變數 self.x
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        # 設定變數 self.F
        self.F = np.eye(8)
        # for 迴圈：遍歷集合元素
        for i in range(4):
            # 設定變數 self.F[i, i+4]
            self.F[i, i+4] = dt           # position += velocity*dt
        # 設定變數 self.H
        self.H = np.zeros((4, 8))
        # 設定變數 self.H[0,0]
        self.H[0,0]=self.H[1,1]=self.H[2,2]=self.H[3,3]=1.0
# （空行）

        # 初始不確定性
        # 設定變數 self.P
        self.P = np.eye(8)*10.0
        # 設定變數 self.P[4:,4:] *
        self.P[4:,4:] *= 100.0
# （空行）

        # 過程雜訊 Q、量測雜訊 R（以尺度比例調整）
        # 設定變數 self.Q
        self.Q = np.eye(8)*q
        # 設定變數 self.Q[4:,4:] *
        self.Q[4:,4:] *= 10*q
        # 設定變數 self.R
        self.R = np.eye(4)*r
# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 predict
    def predict(self):
        # 設定變數 self.x
        self.x = self.F @ self.x
        # 設定變數 self.P
        self.P = self.F @ self.P @ self.F.T + self.Q
        # 回傳函式結果
        return self.x.copy()
# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 update
    def update(self, z):
        # z = [cx, cy, w, h]
        # 設定變數 z
        z = np.asarray(z, dtype=float)
        # 設定變數 y
        y  = z - self.H @ self.x
        # 設定變數 S
        S  = self.H @ self.P @ self.H.T + self.R
        # 設定變數 K
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        # 設定變數 self.x
        self.x = self.x + K @ y
        # 設定變數 self.P
        self.P = (np.eye(8) - K @ self.H) @ self.P
        # 回傳函式結果
        return y, S
# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 mahalanobis
    def mahalanobis(self, z):
        # 設定變數 z
        z = np.asarray(z, dtype=float)
        # 設定變數 y
        y  = z - self.H @ self.x
        # 設定變數 S
        S  = self.H @ self.P @ self.H.T + self.R
        # 設定變數 m2
        m2 = float(y.T @ np.linalg.inv(S) @ y)  # squared distance
        # 回傳函式結果
        return m2
# （空行）

# （空行）

# 類別 Detector：載入 YOLO 模型，設定輸入尺寸與信心門檻，並限定只偵測 person 類別。
# 定義類別 Detector（追蹤邏輯結構）
class Detector:
    # 設定變數 """最簡 YOLO 人偵測器（只保留 person
    """最簡 YOLO 人偵測器（只保留 person=cls 0）"""
    # 函式說明：載入 YOLO 模型，設定輸入尺寸與信心門檻，並限定只偵測 person 類別。
    # 定義函式 Detector.__init__：載入 YOLO 模型，設定輸入尺寸與信心門檻，並限定只偵測 person 類別。
    def __init__(self, weights=YOLO_WEIGHTS, imgsz=IMG_SIZE, conf=CONF_TH, device=0):
        # 載入 YOLO 模型權重
        self.model = YOLO(weights)
        # 設定變數 self.imgsz
        self.imgsz = imgsz
        # 設定變數 self.conf
        self.conf  = conf
        # 設定變數 self.device
        self.device = device
        # 設定變數 self.PERSON
        self.PERSON = 0
# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 detect
    def detect(self, frame):
        # 設定變數 res
        res = self.model(
            # 設定變數 frame, imgsz
            frame, imgsz=self.imgsz, conf=self.conf,
            # 設定變數 verbose
            verbose=False, device=self.device, classes=[self.PERSON], half=True
        # 一般語句
        )[0]
        # 設定變數 out
        out = []
        # for 迴圈：遍歷集合元素
        for b in res.boxes:
            # 設定變數 x1, y1, x2, y2
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            # 一般語句
            out.append([int(x1), int(y1), int(x2), int(y2)])
        # 回傳函式結果
        return out
# （空行）

# （空行）

# 類別 Track：初始化單一追蹤目標：紀錄ID、KF、顏色、命中次數、尾跡等。
# 定義類別 Track（追蹤邏輯結構）
class Track:
    # 設定變數 _next_id
    _next_id = 1
    # 函式說明：初始化單一追蹤目標：紀錄ID、KF、顏色、命中次數、尾跡等。
    # 定義函式 Track.__init__：初始化單一追蹤目標：紀錄ID、KF、顏色、命中次數、尾跡等。
    def __init__(self, bbox):
        # 設定變數 self.disp_id
        self.disp_id = None
        # 設定變數 self.frames_inside
        self.frames_inside = 0       # 連續在非邊界的幀數
        # 設定變數 self.has_entered
        self.has_entered   = False   # 是否已正式入場
        # 設定變數 self.prev_center
        self.prev_center   = bbox_center(bbox)  # 前一幀中心
        # 設定變數 self.speed_hist
        self.speed_hist    = []      # 最近位移量（用來過濾背景假軌）
        # 設定變數 self.id
        self.id  = Track._next_id; Track._next_id += 1
        # 設定變數 self.bbox
        self.bbox = bbox
        # 設定變數 self.hits
        self.hits = 1
        # 設定變數 self.time_since_update
        self.time_since_update = 0
        # 設定變數 self.confirmed
        self.confirmed = False
        # 設定變數 self.trace
        self.trace = [] if TRACE_LEN > 0 else None
        # 條件判斷 if
        if self.trace is not None:
            # 設定變數 cx, cy
            cx, cy = bbox_center(bbox); self.trace.append((int(cx), int(cy)))
        # 設定變數 self.color
        self.color = vivid_color_for_id(self.id)
        # 設定變數 cx, cy
        cx, cy = bbox_center(bbox)
        # 設定變數 w, h
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        # 建立針對框的卡爾曼濾波器
        self.kf = KalmanBox(cx, cy, w, h, dt=1.0, q=Q_pos, r=R_pos)
        # 卡爾曼濾波器預測
        self.kf.predict()  # 初始化一次
# （空行）

# （空行）

# 類別 Tracker：建立追蹤器：保存所有 Track、設定IoU門檻、老化、確立條件。
# 定義類別 Tracker（追蹤邏輯結構）
class Tracker:
    # 一般語句
    """Kalman + 匈牙利追蹤器（使用馬氏距離 gate；不再用 center gate）"""
    # 函式說明：建立追蹤器：保存所有 Track、設定IoU門檻、老化、確立條件。
    # 定義函式 Tracker.__init__：建立追蹤器：保存所有 Track、設定IoU門檻、老化、確立條件。
    def __init__(self, iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS):
        # 設定變數 self.next_disp_id
        self.next_disp_id = 1
        # 設定變數 self.tracks
        self.tracks = []
        # 設定變數 self.iou_gate
        self.iou_gate = iou_gate
        # 設定變數 self.max_age
        self.max_age = max_age
        # 設定變數 self.min_hits
        self.min_hits = min_hits
        # 設定變數 self.total_count
        self.total_count = 0  # 確立過的 ID 數
# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 _too_close_to_existing
    def _too_close_to_existing(self, det, H=None, W=None):
        # 一般語句
        """抑制明顯重複的新生：
        # 一般語句
        1) 若偵測框緊貼畫面邊界（容易是殘影/離場殘框），先不新生
        # 一般語句
        2) 若與任一已存在軌跡的『預測框』IoU 很高，視為同一人，不新生
        # 一般語句
        """
        # 設定變數 x1, y1, x2, y2
        x1, y1, x2, y2 = det
        # --- 1) 邊界守門（避免剛進場/出場抖動誤新生）---
        # 條件判斷 if
        if H is not None and W is not None:
            # 設定變數 edge
            edge = int(0.04 * min(H, W))  # 4% 畫面邊界
            # 設定變數 if x1 <
            if x1 <= edge or y1 <= edge or (W - x2) <= edge or (H - y2) <= edge:
                # 回傳函式結果
                return True
# （空行）

        # --- 2) 對已存在軌跡的「KF 預測框」做重疊檢查 ---
        # for 迴圈：遍歷集合元素
        for t in self.tracks:
            # 條件判斷 if
            if hasattr(t, "pred_bbox"):
                # 條件判斷 if
                if iou_xyxy(t.pred_bbox, det) > 0.60:  # 門檻可微調 0.55~0.70
                    # 回傳函式結果
                    return True
        # 回傳函式結果
        return False
# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 _cost_matrix
    def _cost_matrix(self, dets, H, W):
        # 設定變數 T, D
        T, D = len(self.tracks), len(dets)
        # 條件判斷 if
        if T == 0 or D == 0:
            # 設定變數 return np.zeros((T, D), dtype
            return np.zeros((T, D), dtype=float)
# （空行）

        # 設定變數 C
        C = np.full((T, D), 1e6, dtype=float)
        # 設定變數 diag
        diag = _diag_len(W, H)             # NEW
# （空行）

        # for 迴圈：遍歷集合元素
        for i, t in enumerate(self.tracks):
            # 設定變數 pb
            pb = t.pred_bbox
            # for 迴圈：遍歷集合元素
            for j, d in enumerate(dets):
                # 設定變數 IoU
                IoU = iou_xyxy(pb, d)
                # 條件判斷 if
                if IoU < IOU_GATE:
                    # 一般語句
                    continue
                # 設定變數 cnorm
                cnorm = _center_norm(pb, d, diag)           # NEW
                # 成本：IoU 為主、中心距離為輔助（仍然不碰馬氏距離）
                # 設定變數 C[i, j]
                C[i, j] = LAMBDA_IOU * (1.0 - IoU) + LAMBDA_CENTER * cnorm
        # 回傳函式結果
        return C
# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 update
    def update(self, frame, dets):
        # 設定變數 H, W
        H, W = frame.shape[:2]
# （空行）

        # 0) 每幀先 predict 一次（所有軌跡）
        # for 迴圈：遍歷集合元素
        for t in self.tracks:
            # 卡爾曼濾波器預測
            t.kf.predict()
            # 設定變數 t.pred_bbox
            t.pred_bbox = bbox_from_state(t.kf.x)
# （空行）

        # 1) 匹配（匈牙利）
        # 設定變數 matches, unmatched_t, unmatched_d
        matches, unmatched_t, unmatched_d = [], list(range(len(self.tracks))), list(range(len(dets)))
        # 條件判斷 if
        if self.tracks and dets:
            # 設定變數 C
            C = self._cost_matrix(dets, H, W)  # ← 不再呼叫 predict
            # 呼叫匈牙利演算法進行最佳匹配
            r, c = linear_sum_assignment(C)
            # 設定變數 unmatched_t
            unmatched_t = set(range(len(self.tracks)))
            # 設定變數 unmatched_d
            unmatched_d = set(range(len(dets)))
            # for 迴圈：遍歷集合元素
            for i, j in zip(r, c):
                # 設定變數 if C[i, j] >
                if C[i, j] >= 1e6:
                    # 一般語句
                    continue
                # 一般語句
                matches.append((i, j))
                # 一般語句
                unmatched_t.discard(i)
                # 一般語句
                unmatched_d.discard(j)
            # 設定變數 unmatched_t
            unmatched_t = sorted(list(unmatched_t))
        # 設定變數 unmatched_d
        unmatched_d = sorted(list(unmatched_d))
# （空行）

        # 2) 更新匹配成功的軌跡
        # for 迴圈：遍歷集合元素
        for ti, dj in matches:
            # 設定變數 t
            t = self.tracks[ti]
            # 設定變數 d
            d = dets[dj]
            # 設定變數 cx
            cx = 0.5*(d[0]+d[2]); cy = 0.5*(d[1]+d[3])
            # 設定變數 w
            w  = max(1.0, d[2]-d[0]); h = max(1.0, d[3]-d[1])
# （空行）

            # 卡爾曼濾波器更新（餵入量測）
            t.kf.update([cx, cy, w, h])
            # 設定變數 t.bbox
            t.bbox = bbox_from_state(t.kf.x)   # 用濾波後狀態更新輸出框
            # 維護速度與入場
            # 設定變數 cx, cy
            cx, cy = bbox_center(t.bbox)
            # 條件判斷 if
            if t.prev_center is not None:
                # 設定變數 dx
                dx = cx - t.prev_center[0]
                # 設定變數 dy
                dy = cy - t.prev_center[1]
                # 一般語句
                t.speed_hist.append((dx*dx + dy*dy) ** 0.5)
                # 條件判斷 if
                if len(t.speed_hist) > 10:
                    # 一般語句
                    t.speed_hist.pop(0)
            # 設定變數 t.prev_center
            t.prev_center = (cx, cy)
# （空行）

            # 條件判斷 if
            if not _in_edge_zone(t.bbox, W, H):
                # 設定變數 t.frames_inside +
                t.frames_inside += 1
                # 設定變數 if t.frames_inside >
                if t.frames_inside >= ENTER_FRAMES:
                    # 設定變數 t.has_entered
                    t.has_entered = True
            # 條件判斷 else
            else:
                # 設定變數 t.frames_inside
                t.frames_inside = 0
# （空行）

            # 設定變數 t.hits +
            t.hits += 1
            # 設定變數 t.time_since_update
            t.time_since_update = 0
            # 條件判斷 if
            if t.trace is not None:
                # 設定變數 cx, cy
                cx, cy = bbox_center(t.bbox); t.trace.append((int(cx), int(cy)))
                # 條件判斷 if
                if len(t.trace) > TRACE_LEN:
                    # 一般語句
                    t.trace.pop(0)
# （空行）

        # 3) 老化未匹配的舊軌跡
        # 設定變數 alive
        alive = []
        # for 迴圈：遍歷集合元素
        for idx, trk in enumerate(self.tracks):
            # 條件判斷 if
            if idx in unmatched_t:
                # 設定變數 trk.time_since_update +
                trk.time_since_update += 1
                # 設定變數 trk.bbox
                trk.bbox = trk.pred_bbox      # 用預測框繼續顯示（但不要把預測點寫入 trace）
                # 設定變數 cx, cy
                cx, cy = bbox_center(trk.bbox)
                # 條件判斷 if
                if trk.prev_center is not None:
                    # 設定變數 dx
                    dx = cx - trk.prev_center[0]
                    # 設定變數 dy
                    dy = cy - trk.prev_center[1]
                    # 一般語句
                    trk.speed_hist.append((dx*dx + dy*dy) ** 0.5)
                    # 條件判斷 if
                    if len(trk.speed_hist) > 10:
                        # 一般語句
                        trk.speed_hist.pop(0)
                # 設定變數 trk.prev_center
                trk.prev_center = (cx, cy)
# （空行）

                # 條件判斷 if
                if not _in_edge_zone(trk.bbox, W, H):
                    # 設定變數 trk.frames_inside +
                    trk.frames_inside += 1
                    # 設定變數 if trk.frames_inside >
                    if trk.frames_inside >= ENTER_FRAMES:
                        # 設定變數 trk.has_entered
                        trk.has_entered = True
                # 條件判斷 else
                else:
                    # 設定變數 trk.frames_inside
                    trk.frames_inside = 0
# （空行）

            # 設定變數 if (not trk.confirmed) and (trk.hits >
            if (not trk.confirmed) and (trk.hits >= self.min_hits):
                # 設定變數 trk.confirmed
                trk.confirmed = True
                # 條件判斷 if
                if trk.disp_id is None:
                    # 設定變數 trk.disp_id
                    trk.disp_id = self.next_disp_id
                    # 設定變數 self.next_disp_id +
                    self.next_disp_id += 1
                # 設定變數 self.total_count +
                self.total_count += 1
# （空行）

            # 設定變數 if trk.time_since_update <
            if trk.time_since_update <= self.max_age:
                # 一般語句
                alive.append(trk)
        # 設定變數 self.tracks
        self.tracks = alive
        # 函式說明：函式用途說明：略
        # 定義函式 _too_close_to_existing
        def _too_close_to_existing(self, det):
            # for 迴圈：遍歷集合元素
            for t in self.tracks:
                # 條件判斷 if
                if hasattr(t, "pred_bbox"):
                    # 條件判斷 if
                    if iou_xyxy(t.pred_bbox, det) > 0.30:  # 只用 IoU 檢查接近
                        # 回傳函式結果
                        return True
            # 回傳函式結果
            return False
# （空行）

        # 4) 為未匹配的 detection 新增軌跡（加出生抑制）
        # for 迴圈：遍歷集合元素
        for dj in unmatched_d:
            # 設定變數 d
            d = dets[dj]
            # 條件判斷 if
            if not self._too_close_to_existing(d, H, W):
                # 一般語句
                self.tracks.append(Track(d))
# （空行）

# （空行）

# （空行）

    # 函式說明：函式用途說明：略
    # 定義函式 draw
    def draw(self, frame):
        # 設定變數 vis
        vis = frame
        # for 迴圈：遍歷集合元素
        for t in self.tracks:
            # 條件判斷 if
            if not t.confirmed:
                # 一般語句
                continue
            # 設定變數 x1, y1, x2, y2
            x1, y1, x2, y2 = map(int, t.bbox)
            # ★ 彩色框（用各自的 t.color），外加黑色描邊讓對比更好
            # 在影像上畫出彩色邊框
            cv2.rectangle(vis, (x1, y1), (x2, y2), t.color, 2)          # 彩色框
# （空行）

            # ★ 文字永遠白色
            # 在影像上寫字（顯示ID或統計資訊）
            cv2.putText(vis, f"{t.disp_id}", (x1+4, y1-5),
                 # 一般語句
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
# （空行）

            # ===== 高效漸透明尾跡（單次疊圖）=====
            # 條件判斷 if
            if t.trace and len(t.trace) > 1:
                # 設定變數 n
                n = len(t.trace)
                # 設定變數 thickness
                thickness = 2
                # 設定變數 min_alpha, max_alpha
                min_alpha, max_alpha = 0.15, 0.85
# （空行）

                # 單張 overlay，畫完一次性疊回
                # 建立與原圖大小相同的全零影像（overlay用）
                overlay = np.zeros_like(vis, dtype=np.uint8)
# （空行）

                # 彩色主線畫在 overlay，顏色已預乘 alpha
                # for 迴圈：遍歷集合元素
                for k in range(1, n):
                    # 設定變數 p1, p2
                    p1, p2 = t.trace[k - 1], t.trace[k]
                    # 設定變數 a
                    a = k / n
                    # 設定變數 alpha
                    alpha = min_alpha + (max_alpha - min_alpha) * a  # 由舊→新 漸亮
                    # 設定變數 col
                    col = tuple(int(c * alpha) for c in t.color)     # 預乘 alpha
                    # 在影像上畫線（尾跡）
                    cv2.line(overlay, p1, p2, col, thickness, cv2.LINE_AA)
# （空行）

                # 一次性把 overlay 疊回 vis（overlay 已預乘，不需再縮權重）
                # 設定變數 cv2.add(overlay, vis, dst
                cv2.add(overlay, vis, dst=vis)
# （空行）

# （空行）

        # 回傳函式結果
        return vis
# （空行）

# （空行）

# 函式說明：主程式：讀取影片逐幀偵測與追蹤，寫回輸出影片並顯示人數統計。
# 定義函式 main：主程式：讀取影片逐幀偵測與追蹤，寫回輸出影片並顯示人數統計。
def main():
    # 開啟輸入影片串流
    cap = cv2.VideoCapture(INPUT_VIDEO)
    # 條件判斷 if
    if not cap.isOpened():
        # 拋出例外
        raise RuntimeError(f"Failed to open {INPUT_VIDEO}")
    # 設定變數 fps
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # 設定變數 W
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 設定變數 H
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 建立影片編碼器 fourcc（mp4v）
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
# （空行）

    # 設定變數 det
    det = Detector()
    # 設定變數 tracker
    tracker = Tracker()
# （空行）

    # while 迴圈
    while True:
        # 設定變數 ok, frame
        ok, frame = cap.read()
        # 條件判斷 if
        if not ok:
            # 一般語句
            break
        # 設定變數 dets
        dets = det.detect(frame)            # [[x1,y1,x2,y2], ...]
        # 更新追蹤器（匹配與維護軌跡）
        tracker.update(frame, dets)         # 更新軌跡
        # 設定變數 vis
        vis = tracker.draw(frame)           # 畫框與ID
        # 在影像上寫字（顯示ID或統計資訊）
        cv2.putText(vis, f"People count: {tracker.total_count}", (20, 40),
                    # 一般語句
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
        # 一般語句
        writer.write(vis)
# （空行）

    # 釋放資源（關閉影片/寫入器）
    cap.release()
    # 釋放資源（關閉影片/寫入器）
    writer.release()
    # 輸出訊息到終端機
    print(f"Saved: {OUTPUT_VIDEO} | total people seen: {tracker.total_count}")
# （空行）

# （空行）

# 條件判斷 if
if __name__ == "__main__":
    # 一般語句
    main()

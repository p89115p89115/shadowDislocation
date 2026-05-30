import os
# ==============================================================================
# 🌟 核心終極優化：必須在 import cv2 之前設定！
# ==============================================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;50000|analyzeduration;10000|probesize;32"

import cv2
from ultralytics import YOLO
import threading
import time
import numpy as np
import torch

# ==========================================
# 🛡️ 核心：建立一個永遠只拿最新畫面的即時抓圖器 (保持不變)
# ==========================================
class RTSPStreamReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.frame = None
        self.started = False
        self.lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True 
        self.thread.start()
        return self

    def update(self):
        while self.started:
            success, frame = self.cap.read()
            if not success:
                print("❌ 無法讀取 RTSP 串流")
                self.started = False
                break
            with self.lock:
                self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame

    def stop(self):
        self.started = False
        self.cap.release()

# ==========================================
# 🚀 星空高精準真實人影系統 (絕無殘影版)
# ==========================================
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("💡 成功啟用 NVIDIA CUDA 顯示卡硬體加速！")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("💡 成功啟用 Apple Silicon MPS 圖像加速！")
else:
    DEVICE = "cpu"
    print("⚠️ 未偵測到獨立顯卡，使用 CPU 運算。")

# 初始化高精準 AI 實例分割模型
print("⚡ 初始化 AI 實例分割大腦 (YOLOv8s-Seg)...")
model_seg = YOLO("yolov8s-seg.pt").to(DEVICE)

# 啟動零延遲抓圖器
stream_url = "rtsp://localhost:8554/cam"
reader = RTSPStreamReader(stream_url).start()

# 🔥 【黃金視覺與去殘影設定】
BLUR_STRENGTH = 25                  # 模糊程度 (奇數): 控制影子邊緣的柔和度（越小越銳利）
DILATE_KERNEL_SIZE = 9              # 輪廓外擴程度
PROCESSING_WIDTH = 640              # 處理解析度（若顯卡效能夠強，可調高至 960 獲得極致精準度）

# 信心度平衡點：0.30 可以有效防止漏抓，同時避免過多雜訊
GLOBAL_CONF_THRESH = 0.30           

# 初始化形態學運算核心 (單張影像去噪填充，完全不依賴歷史影格)
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))   # 用於填補內部空洞
open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))    # 用於剔除背景亂閃的雜訊

# --- 🌌 載入本地星空背景 ---
STARRY_SKY_FILENAME = "sky.png"
if os.path.exists(STARRY_SKY_FILENAME):
    starry_sky_img = cv2.imread(STARRY_SKY_FILENAME)
    print("🌌 成功載入本地星空背景 (sky.png)！")
else:
    print(f"⚠️ 找不到 {STARRY_SKY_FILENAME}！將自動使用全黑背景代替。")
    starry_sky_img = None

starry_sky_canvas_resized = None

print(f"👤 零延遲無殘影星空剪影系統啟動中...")
print("⚡ 按 'q' 鍵退出程式")

while reader.started:
    success, frame = reader.read()
    if not success:
        continue

    # 降低處理解析度 (維持流暢度)
    h_orig, w_orig = frame.shape[:2]
    aspect_ratio = h_orig / w_orig
    p_height = int(PROCESSING_WIDTH * aspect_ratio)
    frame_resized = cv2.resize(frame, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_NEAREST)

    # 動態調整星空背景大小
    if starry_sky_canvas_resized is None or starry_sky_canvas_resized.shape[:2] != (p_height, PROCESSING_WIDTH):
        if starry_sky_img is not None:
            starry_sky_canvas_resized = cv2.resize(starry_sky_img, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_AREA)
        else:
            starry_sky_canvas_resized = np.zeros((p_height, PROCESSING_WIDTH, 3), dtype=np.uint8)

    # 執行 Seg 辨識
    results_seg = model_seg(frame_resized, conf=GLOBAL_CONF_THRESH, classes=[0], stream=True, verbose=False)
    
    # 建立當前影格的獨立遮罩
    mask_resized_seg = np.zeros((p_height, PROCESSING_WIDTH), dtype=np.uint8)

    for result_seg in results_seg:
        if result_seg.masks is not None:
            masks = result_seg.masks.data.cpu().numpy()
            combined_mask_seg = np.any(masks, axis=0)
            mask_raw_seg_255 = (combined_mask_seg * 255).astype(np.uint8)
            mask_resized_seg = cv2.resize(mask_raw_seg_255, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_NEAREST)

    # --- 🔥 完全沒有歷史影格干涉的「即時後處理」 ---
    if np.any(mask_resized_seg > 0):
        # 1. 形態學開運算：直接切斷、濾除背景中單點亂閃的錯誤小噪點
        cleaned_mask = cv2.morphologyEx(mask_resized_seg, cv2.MORPH_OPEN, open_kernel)
        
        # 2. 形態學閉運算：將人體內部由於反光、衣服顏色導致的未偵測空洞強制填滿
        filled_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, close_kernel)
        
        # 3. 邊緣膨脹加粗
        dilated_mask = cv2.dilate(filled_mask, dilate_kernel, iterations=1)

        # 4. 高斯模糊柔邊
        ksize = int(BLUR_STRENGTH)
        if ksize % 2 == 0: ksize += 1
        smoothed_mask = cv2.GaussianBlur(dilated_mask, (ksize, ksize), 0)

        # --- 🌌 星空背景與純黑影子 Alpha 權重混合 ---
        alpha = cv2.bitwise_not(smoothed_mask).astype(float) / 255.0
        alpha_3ch = cv2.merge([alpha, alpha, alpha])

        black_canvas = np.zeros_like(starry_sky_canvas_resized)
        blended = starry_sky_canvas_resized.astype(float) * alpha_3ch + black_canvas.astype(float) * (1.0 - alpha_3ch)
        annotated_frame_resized = blended.astype(np.uint8)
    else:
        # 當前影格沒人，立刻切換回純星空（絕不留戀任何上個畫面）
        annotated_frame_resized = starry_sky_canvas_resized

    # 放大回原始顯示尺寸
    final_output = cv2.resize(annotated_frame_resized, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Zero Latency Dislocation Shadow Monitor", final_output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

reader.stop()
cv2.destroyAllWindows()
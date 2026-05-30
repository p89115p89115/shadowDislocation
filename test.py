import os
# ==============================================================================
# 🌟 核心終極優化：必須在 import cv2 之前設定！
# ==============================================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;50000|analyzeduration;10000|probesize;32"

import cv2
import threading
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
# 🚀 全域變數：接收新版 MediaPipe 異步回傳的遮罩結果
# ==========================================
latest_mask = None
mask_lock = threading.Lock()

def receive_result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_mask
    if result.segmentation_masks is not None and len(result.segmentation_masks) > 0:
        raw_mask_mp = result.segmentation_masks[0]
        mask_np = raw_mask_mp.numpy_view()
        with mask_lock:
            latest_mask = mask_np.copy()
    else:
        with mask_lock:
            latest_mask = None

# ==========================================
# 🚀 星空新版 MediaPipe 完美真實人影系統
# ==========================================
print("⚡ 初始化最新版 MediaPipe Tasks 骨架去背引擎...")

MODEL_PATH = "pose_landmarker_full.task"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型檔！請確認檔案存在。")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_segmentation_masks=True,
    min_pose_detection_confidence=0.75,   
    min_pose_presence_confidence=0.75,
    min_tracking_confidence=0.75,
    result_callback=receive_result_callback
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# 啟動零延遲抓圖器
stream_url = "rtsp://localhost:8554/cam"
reader = RTSPStreamReader(stream_url).start()

# 🔥 【黃金視覺與防抖設定】
BLUR_STRENGTH = 67                  # 高斯模糊強度
DILATE_KERNEL_SIZE = 9              # 輪廓外擴程度
PROCESSING_WIDTH = 640              # 處理解析度
HORROR_MODE = False                 # 恐怖影子模式開關

# 初始化形態學運算核心
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# --- 🌌 載入本地星空背景 ---
STARRY_SKY_FILENAME = "sky.png"
if os.path.exists(STARRY_SKY_FILENAME):
    starry_sky_img = cv2.imread(STARRY_SKY_FILENAME)
    print("🌌 成功載入本地星空背景 (sky.png)！")
else:
    print(f"⚠️ 找不到 {STARRY_SKY_FILENAME}！將自動使用全黑背景代替。")
    starry_sky_img = None

starry_sky_canvas_resized = None

print(f"👤 零延遲、極低敏感抗抖動星空剪影系統啟動中...")
print("⚡ 按 'h' 鍵開啟/關閉 💀 恐怖軀體大撕裂模式")
print("⚡ 按 'q' 鍵退出程式")

while reader.started:
    success, frame = reader.read()
    if not success:
        continue

    # 監聽鍵盤按鍵
    key = cv2.waitKey(1) & 0xFF
    if key == ord("h"):
        HORROR_MODE = not HORROR_MODE
        print(f"💀 恐怖影子模式: {'開啟' if HORROR_MODE else '關閉'}")
    elif key == ord("q"):
        break

    h_orig, w_orig = frame.shape[:2]
    aspect_ratio = h_orig / w_orig
    p_height = int(PROCESSING_WIDTH * aspect_ratio)
    frame_resized = cv2.resize(frame, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_NEAREST)

    if starry_sky_canvas_resized is None or starry_sky_canvas_resized.shape[:2] != (p_height, PROCESSING_WIDTH):
        if starry_sky_img is not None:
            starry_sky_canvas_resized = cv2.resize(starry_sky_img, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_AREA)
        else:
            starry_sky_canvas_resized = np.zeros((p_height, PROCESSING_WIDTH, 3), dtype=np.uint8)

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    with mask_lock:
        current_mask = latest_mask.copy() if latest_mask is not None else None

    # 1. 初始化基礎遮罩
    raw_mask_255 = np.zeros((p_height, PROCESSING_WIDTH), dtype=np.uint8)
    if current_mask is not None:
        resized_raw_mask = cv2.resize(current_mask, (PROCESSING_WIDTH, p_height))
        raw_mask_255 = (resized_raw_mask > 0.65).astype(np.uint8) * 255

    # 2. 創建最終用於渲染的平滑遮罩
    smooth_mask = np.zeros_like(raw_mask_255)

    if np.any(raw_mask_255 > 0):
        closed_mask = cv2.morphologyEx(raw_mask_255, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 3000:
                epsilon = 0.003 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                cv2.drawContours(smooth_mask, [approx_contour], -1, 255, -1)

                # ==================================================================
                # 💀 恐怖雙向十字錯位影子模式 (完美移位軀體) 💀
                # ==================================================================
                if HORROR_MODE:
                    x_b, y_b, w_b, h_b = cv2.boundingRect(approx_contour)
                    
                    # 🌟 核心修正：將切片高度擴大到 60% (h_b * 0.60)，確保整條胸膛、雙肩與頭部被一併切下來位移
                    slice_height = int(h_b * 0.60)
                    y_start = y_b
                    y_end = min(p_height, y_b + slice_height)
                    
                    if y_end > y_start and w_b > 0:
                        upper_body_patch = smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH].copy()
                        # 清空原本位置的上半身
                        smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH] = 0
                        
                        # 🌟 控制：【左右位移 50 像素】與【上下位移 -40 像素】
                        offset_x = 50 
                        offset_y = -40  # 負值往上移，正值往下移
                        
                        M_cross = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                        shifted_upper_body = cv2.warpAffine(upper_body_patch, M_cross, (PROCESSING_WIDTH, y_end - y_start))
                        
                        # 重新貼合
                        smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH] = cv2.bitwise_or(
                            smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH], shifted_upper_body
                        )

                    # 效果 2：下半身觸手巨大化延伸
                    y_mid = y_b + int(h_b * 0.5)
                    if p_height > y_mid:
                        lower_body_patch = smooth_mask[y_mid:p_height, 0:PROCESSING_WIDTH].copy()
                        
                        pts1 = np.float32([[0, 0], [PROCESSING_WIDTH, 0], [0, p_height - y_mid], [PROCESSING_WIDTH, p_height - y_mid]])
                        pts2 = np.float32([[-30, 0], [PROCESSING_WIDTH + 30, 0], [-80, p_height - y_mid + 40], [PROCESSING_WIDTH + 80, p_height - y_mid + 40]])
                        
                        M_claw = cv2.getPerspectiveTransform(pts1, pts2)
                        stretched_lower_body = cv2.warpPerspective(lower_body_patch, M_claw, (PROCESSING_WIDTH, p_height - y_mid))
                        
                        smooth_mask[y_mid:p_height, 0:PROCESSING_WIDTH] = cv2.bitwise_or(
                            smooth_mask[y_mid:p_height, 0:PROCESSING_WIDTH], stretched_lower_body
                        )
                # ==================================================================

    # --- 🌌 影像混合與柔邊渲染 ---
    if np.any(smooth_mask > 0):
        dilated_mask = cv2.dilate(smooth_mask, dilate_kernel, iterations=1)

        ksize = int(BLUR_STRENGTH)
        if ksize % 2 == 0: ksize += 1
        smoothed_mask = cv2.GaussianBlur(dilated_mask, (ksize, ksize), 0)

        alpha = cv2.bitwise_not(smoothed_mask).astype(float) / 255.0
        alpha_3ch = cv2.merge([alpha, alpha, alpha])

        black_canvas = np.zeros_like(starry_sky_canvas_resized)
        blended = starry_sky_canvas_resized.astype(float) * alpha_3ch + black_canvas.astype(float) * (1.0 - alpha_3ch)
        annotated_frame_resized = blended.astype(np.uint8)
    else:
        annotated_frame_resized = starry_sky_canvas_resized

    final_output = cv2.resize(annotated_frame_resized, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Zero Latency Dislocation Shadow Monitor", final_output)

landmarker.close()
reader.stop()
cv2.destroyAllWindows()
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

# 🌟 💀 【核心模式與動畫狀態機】 💀 🌟
HORROR_MODE = False                 # H鍵開關
SHOW_MODE = False                   # A鍵劇場

ANIMATION_DURATION = 3.0            # 慢慢分離 3 秒
TOTAL_SHOW_TIME = 5.0               # A鍵劇場總長 5 秒
show_timer = 0.0                    # 劇場計時器
current_progress = 0.0              # 當前動畫進度 (0.0 ~ 1.0)
last_time = time.time()             # 時間戳記

# 軀體位移極限目標
MAX_OFFSET_X = 50
MAX_OFFSET_Y = -40

# 🌟 斷手飛走極限目標 (讓雙手以更大幅度往兩側與上方飛走)
MAX_HAND_FLY_X = 140
MAX_HAND_FLY_Y = -60

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
red_filter_canvas = None

print(f"👤 零延遲、極低敏感抗抖動星空剪影系統啟動中...")
print("⚡ [按 H] 開啟/關閉手動常駐恐怖模式（背景去紅，純肉身撕裂與斷手飛走）")
print("⚡ [按 A] 觸發 5 秒驚悚劇場（漸進 -> 背景變紅 -> 全黑1秒 -> 瞬間歸位）")
print("⚡ [按 Q] 退出程式")

while reader.started:
    success, frame = reader.read()
    if not success:
        continue

    # 計算 Delta Time
    now = time.time()
    dt = now - last_time
    last_time = now

    # 監聽鍵盤按鍵
    key = cv2.waitKey(1) & 0xFF
    if key == ord("h"):
        if not SHOW_MODE:
            HORROR_MODE = not HORROR_MODE
            print(f"💀 手動恐怖模式: {'開啟' if HORROR_MODE else '關閉'}")
    elif key == ord("a"):
        if not HORROR_MODE and not SHOW_MODE:
            SHOW_MODE = True
            show_timer = 0.0
            print("🎬 💥 觸發 5 秒自動驚悚劇場演出！")
    elif key == ord("q"):
        break

    # 動態狀態機
    force_blackout = False  

    if SHOW_MODE:
        show_timer += dt
        if show_timer <= ANIMATION_DURATION:
            current_progress = show_timer / ANIMATION_DURATION
        else:
            current_progress = 1.0
            
        if show_timer >= TOTAL_SHOW_TIME:
            force_blackout = True    
            SHOW_MODE = False        
            current_progress = 0.0   
    else:
        if HORROR_MODE:
            if current_progress < 1.0:
                current_progress += dt / ANIMATION_DURATION
                if current_progress > 1.0: current_progress = 1.0
        else:
            if current_progress > 0.0:
                current_progress -= dt / ANIMATION_DURATION
                if current_progress < 0.0: current_progress = 0.0

    h_orig, w_orig = frame.shape[:2]
    aspect_ratio = h_orig / w_orig
    p_height = int(PROCESSING_WIDTH * aspect_ratio)
    frame_resized = cv2.resize(frame, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_NEAREST)

    # 【大招處理：劇場滿 5 秒觸發全黑斷訊 1 秒】
    if force_blackout:
        black_screen = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        cv2.imshow("Zero Latency Dislocation Shadow Monitor", black_screen)
        cv2.waitKey(1000)        
        last_time = time.time()  
        continue

    # 動態調整背景畫布
    if starry_sky_canvas_resized is None or starry_sky_canvas_resized.shape[:2] != (p_height, PROCESSING_WIDTH):
        if starry_sky_img is not None:
            starry_sky_canvas_resized = cv2.resize(starry_sky_img, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_AREA)
        else:
            starry_sky_canvas_resized = np.zeros((p_height, PROCESSING_WIDTH, 3), dtype=np.uint8)
        red_filter_canvas = np.full_like(starry_sky_canvas_resized, (0, 0, 230), dtype=np.uint8)

    # 🌟 核心分流：如果是 H 鍵手動模式，不染紅背景 (維持原始星空)；如果是 A 鍵劇場，則染紅背景
    if SHOW_MODE:
        red_alpha = current_progress * 0.45
        dynamic_sky_background = cv2.addWeighted(
            starry_sky_canvas_resized, 1.0 - red_alpha, 
            red_filter_canvas, red_alpha, 0
        )
    else:
        # H 模式下，red_alpha 直接歸 0，背景為純藍星空
        dynamic_sky_background = starry_sky_canvas_resized.copy()

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    with mask_lock:
        current_mask = latest_mask.copy() if latest_mask is not None else None

    # 初始化基礎遮罩
    raw_mask_255 = np.zeros((p_height, PROCESSING_WIDTH), dtype=np.uint8)
    if current_mask is not None:
        resized_raw_mask = cv2.resize(current_mask, (PROCESSING_WIDTH, p_height))
        raw_mask_255 = (resized_raw_mask > 0.65).astype(np.uint8) * 255

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
                # 💀 恐怖雙向十字錯位與斷手飛走演算法 💀
                # ==================================================================
                if current_progress > 0.0:
                    x_b, y_b, w_b, h_b = cv2.boundingRect(approx_contour)
                    
                    slice_height = int(h_b * 0.60)
                    y_start = y_b
                    y_end = min(p_height, y_b + slice_height)
                    
                    if y_end > y_start and w_b > 0:
                        # 1. 擷取並分離整塊上半身
                        upper_body_patch = smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH].copy()
                        smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH] = 0
                        
                        # 2. 🌟 建立手部獨立截斷遮罩 🌟
                        # 以人影中軸線 (x_b + w_b // 2) 為基準，左右各向外切除作為「左手」與「右手」
                        center_x = x_b + (w_b // 2)
                        hand_width_boundary = int(w_b * 0.22) # 切出左右兩側外圍約 22% 的範圍作為手部
                        
                        left_hand_mask = np.zeros_like(upper_body_patch)
                        right_hand_mask = np.zeros_like(upper_body_patch)
                        core_body_mask = upper_body_patch.copy()
                        
                        # 定義左右手的物理區域
                        left_bound = max(0, center_x - hand_width_boundary)
                        right_bound = min(PROCESSING_WIDTH, center_x + hand_width_boundary)
                        
                        # 從軀幹主體中把手切出來
                        left_hand_mask[:, 0:left_bound] = upper_body_patch[:, 0:left_bound]
                        right_hand_mask[:, right_bound:PROCESSING_WIDTH] = upper_body_patch[:, right_bound:PROCESSING_WIDTH]
                        
                        # 軀幹主體只保留核心，手部區域抹除（完成肉體截斷）
                        core_body_mask[:, 0:left_bound] = 0
                        core_body_mask[:, right_bound:PROCESSING_WIDTH] = 0
                        
                        # 3. 計算各部位的動態動畫位移
                        # (A) 核心軀幹與頭部：正常的慢速十字分離
                        offset_body_x = int(MAX_OFFSET_X * current_progress)
                        offset_body_y = int(MAX_OFFSET_Y * current_progress)
                        M_body = np.float32([[1, 0, offset_body_x], [0, 1, offset_body_y]])
                        shifted_core_body = cv2.warpAffine(core_body_mask, M_body, (PROCESSING_WIDTH, y_end - y_start))
                        
                        # (B) 🌟 左手：強行向左前方狂飛出去 (X軸為負向，速度加倍)
                        offset_l_x = int(-MAX_HAND_FLY_X * current_progress) + offset_body_x
                        offset_l_y = int(MAX_HAND_FLY_Y * current_progress) + offset_body_y
                        M_left_hand = np.float32([[1, 0, offset_l_x], [0, 1, offset_l_y]])
                        shifted_left_hand = cv2.warpAffine(left_hand_mask, M_left_hand, (PROCESSING_WIDTH, y_end - y_start))
                        
                        # (C) 🌟 右手：強行向右前方狂飛出去 (X軸為正向，速度加倍)
                        offset_r_x = int(MAX_HAND_FLY_X * current_progress) + offset_body_x
                        offset_r_y = int(MAX_HAND_FLY_Y * current_progress) + offset_body_y
                        M_right_hand = np.float32([[1, 0, offset_r_x], [0, 1, offset_r_y]])
                        shifted_right_hand = cv2.warpAffine(right_hand_mask, M_right_hand, (PROCESSING_WIDTH, y_end - y_start))
                        
                        # 4. 將軀幹、左斷手、右斷手透過 bitwise_or 重新組裝回主畫布
                        upper_body_final = cv2.bitwise_or(shifted_core_body, shifted_left_hand)
                        upper_body_final = cv2.bitwise_or(upper_body_final, shifted_right_hand)
                        
                        smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH] = cv2.bitwise_or(
                            smooth_mask[y_start:y_end, 0:PROCESSING_WIDTH], upper_body_final
                        )

                    # 效果 2：下半身觸手巨大化延伸 (維持原樣)
                    y_mid = y_b + int(h_b * 0.5)
                    if p_height > y_mid:
                        lower_body_patch = smooth_mask[y_mid:p_height, 0:PROCESSING_WIDTH].copy()
                        
                        pts1 = np.float32([[0, 0], [PROCESSING_WIDTH, 0], [0, p_height - y_mid], [PROCESSING_WIDTH, p_height - y_mid]])
                        
                        extend_w1 = 30 * current_progress
                        extend_w2 = 80 * current_progress
                        extend_h = 40 * current_progress
                        
                        pts2 = np.float32([
                            [-extend_w1, 0], 
                            [PROCESSING_WIDTH + extend_w1, 0], 
                            [-extend_w2, p_height - y_mid + extend_h], 
                            [PROCESSING_WIDTH + extend_w2, p_height - y_mid + extend_h]
                        ])
                        
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

        black_canvas = np.zeros_like(dynamic_sky_background)
        blended = dynamic_sky_background.astype(float) * alpha_3ch + black_canvas.astype(float) * (1.0 - alpha_3ch)
        annotated_frame_resized = blended.astype(np.uint8)
    else:
        annotated_frame_resized = dynamic_sky_background

    final_output = cv2.resize(annotated_frame_resized, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Zero Latency Dislocation Shadow Monitor", final_output)

landmarker.close()
reader.stop()
cv2.destroyAllWindows()
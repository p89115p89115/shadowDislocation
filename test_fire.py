import os
# ==============================================================================
# 🌟 核心終極優化：必須在 import cv2 之前設定！
# 徹底關閉 FFmpeg 的內部緩衝區（Buffer），縮短協議分析時間，強制走最快的 UDP 協議
# ==============================================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;50000|analyzeduration;10000|probesize;32"

import cv2
from ultralytics import YOLO
import threading
import time
import numpy as np
import torch  # 用於精準啟用顯示卡硬體加速

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
# 🚀 🚀 星空黑影模式主程式開始
# ==========================================
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("💡 成功啟用 NVIDIA CUDA 顯示卡硬體加速！")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("💡 成功啟用 Apple Silicon MPS 圖像加速！")
else:
    DEVICE = "cpu"
    print("⚠️ 未偵測到獨立顯卡，使用 CPU 運算（已更換輕量模型以維持速度）。")

print("⚡ 正在載入超快速 Nano 版本骨架模型...")
model_pose = YOLO("yolov8n-pose.pt").to(DEVICE) 

# 啟動零延遲抓圖器
stream_url = "rtsp://localhost:8554/cam"
reader = RTSPStreamReader(stream_url).start()

# 🔥 【黃金視覺與效能最佳化設定】
# 🌟 為了讓影子看起來更像自然的“黑影人”，可以稍微調高模糊和膨脹
BLUR_STRENGTH = 40                  # 柔邊強度（調高以獲得更柔軟的黑影）
DILATE_KERNEL_SIZE = 15             # 膨脹係數（調高以讓影子稍微大一點，掩盖矩形边界）
PROCESSING_WIDTH = 640              # 處理解析度
HORROR_MODE = False                 # 恐怖影子模式開關
OUTLINE_SHADOW_MODE = False         # 🌟 新增：人體輪廓黑影人模式開關

# 🌟 信心度門檻優化：將門檻調低，大幅提升捕捉敏感度（防漏抓）
GLOBAL_CONF_THRESH = 0.25           

# 初始化膨脹結構元素
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))

# --- 🌌 載入本地星空背景 ---
STARRY_SKY_FILENAME = "sky.png"
if os.path.exists(STARRY_SKY_FILENAME):
    starry_sky_img = cv2.imread(STARRY_SKY_FILENAME)
    print("🌌 成功載入本地星空背景 (sky.png)！")
else:
    print(f"⚠️ 找不到 {STARRY_SKY_FILENAME}！將自動使用全黑背景代替。")
    starry_sky_img = None

starry_sky_canvas_resized = None

print(f"👤 零延遲幾何小黑人系統啟動中 (處理解析度: {PROCESSING_WIDTH}px)...")
print("⚡ 按 'h' 開啟/關閉恐怖錯位影子模式")
print("⚡ 按 'o' 開啟/關閉人體輪廓黑影人模式 (柔邊黑塊)")

# 🌟 建立全身 EMA 全域抗抖動緩衝暫存
smooth_kpts = None
EMA_ALPHA = 0.35  

while reader.started:
    success, frame = reader.read()
    if not success:
        continue

    key = cv2.waitKey(1) & 0xFF
    if key == ord("h"):
        HORROR_MODE = not HORROR_MODE
        # 互斥：開啟恐怖模式時，關閉輪廓影子模式
        if HORROR_MODE: OUTLINE_SHADOW_MODE = False
        print(f"💀 恐怖影子模式: {'開啟' if HORROR_MODE else '關閉'}")
    elif key == ord("o"):
        OUTLINE_SHADOW_MODE = not OUTLINE_SHADOW_MODE
        # 互斥：開啟輪廓影子模式時，關閉恐怖模式
        if OUTLINE_SHADOW_MODE: HORROR_MODE = False
        print(f"👤 輪廓黑影人模式: {'開啟' if OUTLINE_SHADOW_MODE else '關閉'}")
    elif key == ord("q"):
        break

    h_orig, w_orig = frame.shape[:2]
    aspect_ratio = h_orig / w_orig
    p_height = int(PROCESSING_WIDTH * aspect_ratio)
    frame_resized = cv2.resize(frame, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_NEAREST)

    # 動態調整星空背景大小以符合當前視窗比例
    if starry_sky_canvas_resized is None or starry_sky_canvas_resized.shape[:2] != (p_height, PROCESSING_WIDTH):
        if starry_sky_img is not None:
            starry_sky_canvas_resized = cv2.resize(starry_sky_img, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_AREA)
        else:
            starry_sky_canvas_resized = np.zeros((p_height, PROCESSING_WIDTH, 3), dtype=np.uint8)

    # 🌟 優化 1：調低推論信心度（conf=0.25），讓人一出現就立刻被捕捉
    outputs_pose = model_pose(frame_resized, conf=GLOBAL_CONF_THRESH, verbose=False)
    
    # 建立遮罩（人體區域）
    mask = np.zeros((p_height, PROCESSING_WIDTH), dtype=np.uint8)

    # 判定是否有抓到人
    if len(outputs_pose) > 0:
        res_pose = outputs_pose[0]
        
        # 🌟 核心分流 ──【模式 A：人體輪廓黑影人模式】
        if OUTLINE_SHADOW_MODE:
            # 直接使用 YOLO 偵測到的所有 Bounding Boxes 建立遮罩
            if res_pose.boxes is not None:
                for box in res_pose.boxes.xyxy:
                    # 取得矩陣座標
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    # 在 mask 上畫一個充满的白色矩形，代表黑影的範圍
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        
        # ──【模式 B：幾何/恐怖小黑人模式】(你原本的精采演算法)
        # 只有在 len(res_pose.keypoints.data) > 0 時才執行
        elif res_pose.keypoints is not None and len(res_pose.keypoints.data) > 0:
            # 取得第一個偵測到的人的關鍵點
            kpts_raw = res_pose.keypoints.data[0].cpu().numpy() 
            
            # EMA 濾波平滑
            if smooth_kpts is None:
                smooth_kpts = kpts_raw[:, :2]
            else:
                smooth_kpts = EMA_ALPHA * kpts_raw[:, :2] + (1 - EMA_ALPHA) * smooth_kpts

            # 建立整數座標與信心度字典
            pts = {i: (int(smooth_kpts[i][0]), int(smooth_kpts[i][1])) for i in range(17)}
            conf = {i: kpts_raw[i][2] for i in range(17)} 

            # 幾何小灰人基礎位置初始化
            head_center = pts[0] 
            
            # 動態計算頭部圓形半徑
            r_val = 35.0
            if conf[3] > GLOBAL_CONF_THRESH and conf[4] > GLOBAL_CONF_THRESH:
                r_val = np.linalg.norm(np.array(pts[3]) - np.array(pts[4])) * 0.75
                r_val = max(20.0, min(r_val, 70.0))

            l_wrist_target = pts[9]
            l_ankle_target = pts[15]
            r_ankle_target = pts[16]

            # 恐怖模式幾何變更
            if HORROR_MODE:
                if conf[10] > GLOBAL_CONF_THRESH:
                    head_center = pts[10] 
                
                if conf[5] > GLOBAL_CONF_THRESH and conf[7] > GLOBAL_CONF_THRESH and conf[9] > GLOBAL_CONF_THRESH:
                    dir_vec = np.array(pts[7]) - np.array(pts[5])
                    extend_vec = dir_vec * 3.0 
                    l_wrist_target = (int(pts[9][0] + extend_vec[0]), int(pts[9][1] + extend_vec[1]))
                
                l_ankle_target = pts[16]
                r_ankle_target = pts[15]

            # --- 1. 畫頭部 ---
            if conf[0] > GLOBAL_CONF_THRESH:
                actual_center = (head_center[0], int(head_center[1] - r_val * 0.2))
                cv2.circle(mask, actual_center, int(r_val), 255, -1) 

            # --- 2. 畫軀幹 ---
            has_shoulders = conf[5] > GLOBAL_CONF_THRESH and conf[6] > GLOBAL_CONF_THRESH
            has_hips = conf[11] > GLOBAL_CONF_THRESH and conf[12] > GLOBAL_CONF_THRESH
            
            if has_shoulders and has_hips:
                torso_poly = np.array([pts[5], pts[6], pts[12], pts[11]], dtype=np.int32)
                cv2.fillPoly(mask, [torso_poly], 255) 
            elif has_shoulders:
                mid_shoulder = ((pts[5][0] + pts[6][0]) // 2, (pts[5][1] + pts[6][1]) // 2)
                virtual_hip = (mid_shoulder[0], mid_shoulder[1] + int(r_val * 2.5))
                cv2.line(mask, mid_shoulder, virtual_hip, 255, int(r_val * 1.8))
                cv2.line(mask, pts[5], pts[6], 255, 15)

            # --- 3. 畫右手臂 ---
            if conf[6] > GLOBAL_CONF_THRESH and conf[8] > GLOBAL_CONF_THRESH: 
                cv2.line(mask, pts[6], pts[8], 255, 14) 
            if conf[8] > GLOBAL_CONF_THRESH and conf[10] > GLOBAL_CONF_THRESH: 
                cv2.line(mask, pts[8], pts[10], 255, 12) 

            # --- 4. 畫左手臂 ---
            if conf[5] > GLOBAL_CONF_THRESH and conf[7] > GLOBAL_CONF_THRESH and conf[9] > GLOBAL_CONF_THRESH:
                if HORROR_MODE:
                    dir_vec = np.array(pts[7]) - np.array(pts[5])
                    extend_vec = dir_vec * 3.0
                    break_1 = (int(pts[7][0] + extend_vec[0] * 0.3), int(pts[7][1] + extend_vec[1] * 0.3))
                    break_2 = (int(pts[7][0] + extend_vec[0] * 0.7), int(pts[7][1] + extend_vec[1] * 0.7))
                    
                    cv2.line(mask, pts[7], break_1, 255, 8)
                    cv2.line(mask, break_2, l_wrist_target, 255, 8)

                    dir_len = np.linalg.norm(dir_vec) + 1e-5
                    dir_norm = dir_vec / dir_len
                    perp_norm = np.array([-dir_norm[1], dir_norm[0]]) 
                    
                    spread_factors = [-0.5, -0.18, 0.18, 0.5] 
                    length_factors = [0.85, 1.2, 1.1, 0.8]     

                    for s_f, l_f in zip(spread_factors, length_factors):
                        f_dir = dir_norm * l_f + perp_norm * s_f
                        f_dir_norm = f_dir / (np.linalg.norm(f_dir) + 1e-5)
                        tip = (int(l_wrist_target[0] + f_dir_norm[0] * 35), int(l_wrist_target[1] + f_dir_norm[1] * 35))
                        
                        pts_claw = np.array([l_wrist_target, 
                                              (int(l_wrist_target[0] + f_dir_norm[0]*10 - perp_norm[0]*4), int(l_wrist_target[1] + f_dir_norm[1]*10 - perp_norm[1]*4)),
                                              tip, 
                                              (int(l_wrist_target[0] + f_dir_norm[0]*10 + perp_norm[0]*4), int(l_wrist_target[1] + f_dir_norm[1]*10 + perp_norm[1]*4))], dtype=np.int32)
                        cv2.fillPoly(mask, [pts_claw], 255) 
                else:
                    cv2.line(mask, pts[5], pts[7], 255, 14)
                    cv2.line(mask, pts[7], pts[9], 255, 12)

            # --- 5. 畫右腿 ---
            if conf[12] > GLOBAL_CONF_THRESH and conf[14] > GLOBAL_CONF_THRESH: 
                cv2.line(mask, pts[12], pts[14], 255, 18)
            if conf[14] > GLOBAL_CONF_THRESH and (conf[15] > GLOBAL_CONF_THRESH or conf[16] > GLOBAL_CONF_THRESH): 
                cv2.line(mask, pts[14], r_ankle_target, 255, 14)

            # --- 6. 畫左腿 ---
            if conf[11] > GLOBAL_CONF_THRESH and conf[13] > GLOBAL_CONF_THRESH: 
                cv2.line(mask, pts[11], pts[13], 255, 18)
            if conf[13] > GLOBAL_CONF_THRESH and (conf[15] > GLOBAL_CONF_THRESH or conf[16] > GLOBAL_CONF_THRESH): 
                cv2.line(mask, pts[13], l_ankle_target, 255, 14)

    # --- 後處理與 Alpha 權重混合渲染 (保持不變) ---
    # 這個部分是通用的，不論是矩形輪廓還是骨架連線，都應用相同的柔邊效果
    if np.any(mask > 0):
        # 邊緣膨脹與高斯模糊，做出柔邊影子效果
        dilated_mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        ksize = int(BLUR_STRENGTH)
        if ksize <= 0: ksize = 15
        elif ksize % 2 == 0: ksize += 1 
        smoothed_mask = cv2.GaussianBlur(dilated_mask, (ksize, ksize), 0)

        # 計算 Alpha 權重通道：
        # 我們要把「人」的部分變黑色（值=0），「背景」部分保留星空（值=1）
        # 所以對模糊後的 mask 取反向（bitwise_not）
        alpha = cv2.bitwise_not(smoothed_mask).astype(float) / 255.0
        alpha_3ch = cv2.merge([alpha, alpha, alpha])

        # 建立純黑的人影畫布
        black_canvas = np.zeros_like(starry_sky_canvas_resized)

        # 矩陣混合公式：星空背景 * alpha + 純黑人影 * (1.0 - alpha)
        # 注意：不填入彩色人物 frame_resized，而是填入純黑 black_canvas
        blended = starry_sky_canvas_resized.astype(float) * alpha_3ch + black_canvas.astype(float) * (1.0 - alpha_3ch)
        annotated_frame_resized = blended.astype(np.uint8)
    else:
        # 畫面上沒人時，直接輸出完整的星空背景
        annotated_frame_resized = starry_sky_canvas_resized

    # 還原回原本串流的大小並顯示
    final_output = cv2.resize(annotated_frame_resized, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Zero Latency Dislocation Shadow Monitor", final_output)

reader.stop()
cv2.destroyAllWindows()
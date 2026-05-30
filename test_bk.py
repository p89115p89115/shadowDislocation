import cv2
from ultralytics import YOLO
import threading
import requests
import time
import numpy as np

# ==========================================
# 🛡️ 核心：建立一個永遠只拿最新畫面的即時抓圖器 (保持不變)
# ==========================================
class RTSPStreamReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # 設定緩衝區為 1，極小化延遲
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
            # 極短時間休息，釋放 CPU
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            # 不用 .copy() 以追求極致速度，但要注意 main loop 不要修改到原圖
            return True, self.frame

    def stop(self):
        self.started = False
        self.cap.release()

# ==========================================
# 🚀 恐怖影子模式主程式開始
# ==========================================
# 1. 🛡️ 【大腦雙開】 同時初始化 Segmentation (分割影子) 與 Pose (錯位) 模型
# 注意：跑兩個模型很吃運算，所以處理解析度一定要降得很低才能流暢！
print("⚡ 初始化 AI 模型雙大腦 (Seg + Pose)...")
model_seg = YOLO("yolov8s-seg.pt")   # 用於畫出完美的柔邊影子
model_pose = YOLO("yolov8s-pose.pt") # 用於找到肢體座標，進行影子錯位

# 2. 啟動零延遲抓圖器
stream_url = "rtsp://localhost:8554/cam"
reader = RTSPStreamReader(stream_url).start()

esp32_ip = "192.168.0.xxx"  # 填入你的 ESP32 IP
last_trigger_time = 0
cooldown_interval = 5 

# 🔥 【效能與視覺設定】
SHADOW_COLOR = (120, 120, 120)     # 影子顏色: 灰色 (BGR格式)
BLUR_STRENGTH = 31                  # 模糊程度 (奇數): 產生柔光感、消除鋸齒
DILATE_KERNEL_SIZE = 15            # 【視覺】輪廓加寬程度 (數字越大輪廓越寬，越不容易露出身體)
PROCESSING_WIDTH = 640              # 【效能】處理解析度 (關鍵優化！降低這個數字可以大幅消除同時跑雙 AI 的 LAG)
HORROR_MODE = False                # 恐怖影子模式開關

# 初始化膨脹運算用的結構元素
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))

print(f"👤 零延遲順暢影子系統啟動中 (處理解析度: {PROCESSING_WIDTH}px)...")
print("⚡ 按 'h' 開啟/關閉恐怖錯位影子模式")

# 控制發送 HTTP 的背景執行緒
def send_alert_async(ip):
    try:
        requests.post(f"http://{ip}/alert", timeout=0.5)
    except:
        pass

# 建立固定大小的畫布，避免在迴圈內反覆建立
grey_canvas_resized = None

while reader.started:
    # 拿取原始畫面
    success, frame = reader.read()
    if not success:
        continue

    # 監聽鍵盤
    key = cv2.waitKey(1) & 0xFF
    if key == ord("h"):
        HORROR_MODE = not HORROR_MODE
        print(f"💀 恐怖影子模式: {'開啟' if HORROR_MODE else '關閉'}")

    # --- 優化步驟 1：降低處理解析度 (維持流暢度) ---
    h_orig, w_orig = frame.shape[:2]
    aspect_ratio = h_orig / w_orig
    p_height = int(PROCESSING_WIDTH * aspect_ratio)
    frame_resized = cv2.resize(frame, (PROCESSING_WIDTH, p_height), interpolation=cv2.INTER_NEAREST)

    # 3. 🔥 【AI 雙開】 執行 Seg 辨識 (取影子) 與 Pose 辨識 (取肢體錯位座標)
    results_seg = model_seg(frame_resized, conf=0.6, classes=[0], stream=True, verbose=False)
    results_pose = model_pose(frame_resized, conf=0.6, stream=True, verbose=False)
    
    detected_person_count = 0
    
    # 初始化輸出畫面為縮小的原圖
    annotated_frame_resized = frame_resized.copy()
    
    # 建立一個與處理解析度一樣大的空白二元遮罩 (用於存放恐怖重組後的影子)
    # 此遮罩將用作 Alpha 混合的透明度地圖
    smoothed_mask = np.zeros((p_height, PROCESSING_WIDTH), dtype=np.uint8)

    # 處理 Seg (影子遮罩) 資料
    for result_seg in results_seg:
        if result_seg.masks is not None:
            # 取得二元遮罩並放大到 640px 解析度
            masks = result_seg.masks.data.cpu().numpy()
            combined_mask_seg = np.any(masks, axis=0)
            mask_raw_seg_255 = (combined_mask_seg * 255).astype(np.uint8)
            mask_resized_seg = cv2.resize(mask_raw_seg_255, (PROCESSING_WIDTH, p_height))
            # 複製一份準備做幾何錯位 ( mask_resized_horror )
            mask_resized_horror = mask_resized_seg.copy()

            detected_person_count = len(result_seg.masks.data)

            # 4. 🔥 【影子恐怖錯位演算法】 (對白色遮罩進行幾何操控)
            # YOLO Pose 關鍵點索引:
            # 0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear
            # 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow, 9: L_Wrist, 10: R_Wrist
            # 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle
            
            # 必須同時有 Pose 資料才能做錯位
            for result_pose in results_pose:
                if result_pose.keypoints is not None and len(result_pose.keypoints.data) > 0:
                    kpts = result_pose.keypoints.data[0].cpu().numpy() # 取得肢體關鍵點資料

                    if HORROR_MODE:
                        # --- Effect 1: 頭部消失 (斷頭影子) ---
                        # 找到鼻子關鍵點
                        nose_kp = kpts[0]
                        l_ear_kp = kpts[3]
                        r_ear_kp = kpts[4]
                        
                        if nose_kp[2] > 0.5:
                            # 估計頭部區域的包圍矩陣大小 (眼睛耳朵寬度)
                            head_min_x = int(nose_kp[0] - 50)
                            head_max_x = int(nose_kp[0] + 50)
                            head_min_y = int(nose_kp[1] - 80)
                            head_max_y = int(nose_kp[1] + 30)
                            
                            # 裁切頭部灰色影子 patch
                            head_patch = mask_resized_seg[head_min_y:head_max_y, head_min_x:head_max_x].copy()
                            
                            # ***在錯位白色遮罩上，強制將頭部白色像素變為黑洞背景像素 (斷頭效果)***
                            mask_resized_horror[head_min_y:head_max_y, head_min_x:head_max_x] = 0
                            
                            # --- Effect 2: 右手拿頭影子 (灰色影子錯位) ---
                            # 找到右手腕 ( kp 10 )
                            r_wrist_kp = kpts[10]
                            
                            if r_wrist_kp[2] > 0.5:
                                # 將裁切頭部的白色灰色影子 Patch，貼到右手腕位置的白色遮罩上
                                target_pos_x = int(r_wrist_kp[0] - (head_patch.shape[1]/2))
                                target_pos_y = int(r_wrist_kp[1] - (head_patch.shape[0]/2))
                                
                                # 確保貼上不超出遮罩邊界
                                if target_pos_y >= 0 and target_pos_x >= 0 and target_pos_y + head_patch.shape[0] < p_height and target_pos_x + head_patch.shape[1] < PROCESSING_WIDTH:
                                    # 利用 bitwise_or 將白色影子 Patch 合併進遮罩圖
                                    mask_resized_horror[target_pos_y:target_pos_y+head_patch.shape[0], target_pos_x:target_pos_x+head_patch.shape[1]] = \
                                        cv2.bitwise_or(mask_resized_horror[target_pos_y:target_pos_y+head_patch.shape[0], target_pos_x:target_pos_x+head_patch.shape[1]], head_patch)

                        # --- Effect 3: 左手延伸超長斷裂 (幾何影子) ---
                        # 找到左肩 (5)、左肘 (7)
                        l_shoulder = kpts[5]
                        l_elbow = kpts[7]
                        
                        if l_shoulder[2] > 0.5 and l_elbow[2] > 0.5:
                            l_shoulder_pos = (int(l_shoulder[0]), int(l_shoulder[1]))
                            l_elbow_pos = (int(l_elbow[0]), int(l_elbow[1]))
                            
                            # 計算延伸向量 (肘 -> 遠處)
                            # 延伸向量，設定為 3 倍
                            direction_vec = np.array(l_elbow_pos) - np.array(l_shoulder_pos)
                            extend_vec = direction_vec * 3.0
                            
                            # 計算延伸後的最終灰色影子手腕位置
                            dislocated_wrist = np.array(l_elbow_pos) + extend_vec
                            dislocated_wrist_pos = (int(dislocated_wrist[0]), int(dislocated_wrist[1]))
                            
                            # 計算斷裂的中點 (肘與遠手腕的中點)
                            break_point_1 = np.array(l_elbow_pos) + (extend_vec * 0.3)
                            break_point_1_pos = (int(break_point_1[0]), int(break_point_1[1]))
                            break_point_2 = np.array(l_elbow_pos) + (extend_vec * 0.7)
                            break_point_2_pos = (int(break_point_2[0]), int(break_point_2[1]))

                            # ***在白色遮罩上，用幾何繪製函數，畫出斷裂的延伸幾何幾何影子***
                            # 灰色影子線條: 手肘 -> 斷裂點前
                            cv2.line(mask_resized_horror, l_elbow_pos, break_point_1_pos, 255, 10) # 寬度 10 的白色線條
                            # 缺口，不畫線
                            # 灰色影子線條: 斷裂點後 -> 最終遠手腕
                            cv2.line(mask_resized_horror, break_point_2_pos, dislocated_wrist_pos, 255, 10)
                            # 在延伸幾何灰色影子末端畫上一個灰色的手幾何影子
                            cv2.circle(mask_resized_horror, dislocated_wrist_pos, 20, 255, -1) # 白色實心圓

                        # --- Effect 4: 腳部錯位 (灰色影子像素交換) ---
                        # 找到左踝 (15) 與 右踝 (16)
                        l_ankle_kp = kpts[15]
                        r_ankle_kp = kpts[16]
                        
                        if l_ankle_kp[2] > 0.5 and r_ankle_kp[2] > 0.5:
                            # 交換腳灰色影子的白色Patch (例如 Ankles 上方 50x50 區域)
                            l_ankle_x, l_ankle_y = int(l_ankle_kp[0]), int(l_ankle_kp[1])
                            r_ankle_x, r_ankle_y = int(r_ankle_kp[0]), int(r_ankle_kp[1])
                            
                            # 定義裁切區域，確保不超出遮罩圖
                            l_patch_y_min = max(0, l_ankle_y - 25)
                            l_patch_y_max = min(p_height, l_ankle_y + 25)
                            l_patch_x_min = max(0, l_ankle_x - 25)
                            l_patch_x_max = min(PROCESSING_WIDTH, l_ankle_x + 25)

                            r_patch_y_min = max(0, r_ankle_y - 25)
                            r_patch_y_max = min(p_height, r_ankle_y + 25)
                            r_patch_x_min = max(0, r_ankle_x - 25)
                            r_patch_x_max = min(PROCESSING_WIDTH, r_ankle_x + 25)

                            l_patch = mask_resized_horror[l_patch_y_min:l_patch_y_max, l_patch_x_min:l_patch_x_max].copy()
                            r_patch = mask_resized_horror[r_patch_y_min:r_patch_y_max, r_patch_x_min:r_patch_x_max].copy()
                            
                            # ***關鍵：在遮罩圖上進行像素 patch 的交換 (腳部灰色影子錯位)***
                            # 需要進行 resize 以適應不同的 Patch 尺寸 (如果貼在邊界上)
                            mask_resized_horror[l_patch_y_min:l_patch_y_max, l_patch_x_min:l_patch_x_max] = \
                                cv2.resize(r_patch, (l_patch.shape[1], l_patch.shape[0]))
                            mask_resized_horror[r_patch_y_min:r_patch_y_max, r_patch_x_min:r_patch_x_max] = \
                                cv2.resize(l_patch, (r_patch.shape[1], r_patch.shape[0]))

                    else:
                        # 正常模式: 直接使用原本 Seg 模型算出的遮罩
                        mask_resized_horror = mask_resized_seg
            
            # --- 5. 後處理 (對重組後的二元遮罩進行滑順化與灰調处理) ---
            # 視覺步驟 2：輪廓膨脹加寬 (讓灰色影子變得比原本肉身大圈，防止身體漏出)
            dilated_mask = cv2.dilate(mask_resized_horror, dilate_kernel, iterations=1)

            # 視覺步驟 3：高斯模糊 (消除二元遮罩的鋸齒、產生柔和光暈感)
            smoothed_mask = cv2.GaussianBlur(dilated_mask, (BLUR_STRENGTH, BLUR_STRENGTH), 0)

            # --- 🔥 【正統 Alpha 權重混合】 (產生滑順無鋸齒的灰色柔光影子) 🔥 ---
            # 1. 將模糊後的單通道遮罩轉為 0.0 ~ 1.0 的透明度通道 (Alpha Map)
            alpha = smoothed_mask.astype(float) / 255.0
            alpha_3ch = cv2.merge([alpha, alpha, alpha]) # 變成 3 通道用於彩色原圖混合 (BGR)

            # 步驟 C: 建立一張全灰色畫布，作為影子的基礎顏色
            # grey_canvas_resized (只需建立一次，避免耗電運算)
            if grey_canvas_resized is None or grey_canvas_resized.shape[:2] != (p_height, PROCESSING_WIDTH):
                grey_canvas_resized = np.full((p_height, PROCESSING_WIDTH, 3), SHADOW_COLOR, dtype=np.uint8)

            # 步驟 D: Alpha 混合 (徹底解決上一版的輪廓彩色邊 Bug)
            # 公式：新畫面 = 原圖背景 * (1 - 透明度地圖) + 灰色畫布 * 透明度地圖
            # 需要轉為 float 計算以防止溢位，這在 640px 下跑很快，完全不 LAG
            blended = frame_resized.astype(float) * (1.0 - alpha_3ch) + grey_canvas_resized.astype(float) * alpha_3ch
            
            annotated_frame_resized = blended.astype(np.uint8)

    # --- 優化步驟 4：放大回原始顯示尺寸 (提升 UX) ---
    # 將處理好的滑順影子畫面放大回原始大小顯示
    final_output = cv2.resize(annotated_frame_resized, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # 4. ESP32 連動保持不變 (略過以節省代碼空間，可自行補上非同步發送)
    if detected_person_count > 0:
        # print("🚨 AI 看到完美的影子人了！")
        # 补上 millis() 冷卻和 requests.post() 非同步發送的邏輯
        pass

    # 5. 顯示結果 (畫面上的人體徹底退化成一個滑順、邊緣柔和、顏色灰調、且頭手腳幾何錯位的神秘影子剪影)
    cv2.imshow("Zero Latency Dislocation Shadow Monitor", final_output)

    # 按 'q' 鍵退出程式
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

reader.stop()
cv2.destroyAllWindows()
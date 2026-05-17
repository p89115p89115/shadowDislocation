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
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        self.cap.release()

# ==========================================
# 🚀 主程式開始
# ==========================================
# 1. 載入專業的 "分割 (Segmentation)" 模型
model = YOLO("yolov8s-seg.pt")

# 2. 啟動零延遲抓圖器
stream_url = "rtsp://localhost:8554/cam"
reader = RTSPStreamReader(stream_url).start()

esp32_ip = "192.168.0.xxx"  # 填入你的 ESP32 IP
last_trigger_time = 0
cooldown_interval = 5 

# 🔥 【視覺設定】
# 影子的顏色 (BGR格式): (100, 100, 100) 是中灰色，(0,0,0)是純黑
SHADOW_COLOR = (120, 120, 120) 
# 模糊程度 (必須是奇數，數字越大越模糊、光暈越散): 建議 15~51 之間
BLUR_STRENGTH = 31 

print("👤 零延遲灰調柔光影子辨識系統已啟動...")

# 控制發送 HTTP 的背景執行緒 (保持不變)
def send_alert_async(ip):
    try:
        requests.post(f"http://{ip}/alert", timeout=0.5)
    except:
        pass

while reader.started:
    # 拿取「絕對最新」的畫面
    success, frame = reader.read()
    if not success:
        continue

    # 取得畫面尺寸
    h, w, _ = frame.shape

    # 3. 執行 AI 實例分割辨識
    results = model(frame, conf=0.6, classes=[0], stream=True, verbose=False)
    detected_person_count = 0
    
    # 最終輸出的畫面預設為原圖
    annotated_frame = frame.copy()

    for result in results:
        # 如果有偵測到人體遮罩
        if result.masks is not None:
            detected_person_count = len(result.masks.data)
            
            # 取得原始遮罩資料並轉為二元 mask (0 或 255)
            masks = result.masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0)
            
            # 🔥 【修正上次 Bug】將 0/1 轉為 0/255 並縮放回原圖大小
            mask_255 = (combined_mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_255, (w, h))

            # *** 🛡️ 進階視覺處理：灰調 + 柔光去鋸齒 ***

            # 步驟 A: 對遮罩進行高斯模糊。
            # 這會把鋒利的邊緣變成灰色漸層，消除鋸齒並產生散發光暈的效果。
            # BLUR_STRENGTH 愈大，光暈愈散。
            smoothed_mask = cv2.GaussianBlur(mask_resized, (BLUR_STRENGTH, BLUR_STRENGTH), 0)

            # 步驟 B: 將模糊後的遮罩歸一化為 0.0 ~ 1.0 之間的透明度 (Alpha 通道)
            # 並且需要把單通道的遮罩擴展為 3 通道 (BGR)，以便與彩色原圖混合
            alpha = smoothed_mask.astype(float) / 255.0
            alpha_3ch = cv2.merge([alpha, alpha, alpha]) # 變成 (h, w, 3)

            # 步驟 C: 建立一張全灰色的畫布，作為影子的基礎顏色
            grey_canvas = np.full((h, w, 3), SHADOW_COLOR, dtype=np.uint8)

            # 步驟 D: 進行 Alpha 混合 (將灰色畫布貼在原圖上)
            # 邏輯: 最終圖 = 原圖 * (1 - Alpha) + 灰色畫布 * Alpha
            # 需要轉為 float 計算以防止溢位
            frame_float = frame.astype(float)
            grey_canvas_float = grey_canvas.astype(float)

            # 混合計算
            blended = cv2.convertScaleAbs(frame_float * (1.0 - alpha_3ch) + grey_canvas_float * alpha_3ch)
            
            annotated_frame = blended

    # 4. 判斷是否通知 ESP32 亮燈 (保持不變)
    if detected_person_count > 0:
        current_time = time.time()
        if current_time - last_trigger_time > cooldown_interval:
            print(f"🚨 AI 看到影子人了！通知 ESP32...")
            last_trigger_time = current_time
            t = threading.Thread(target=send_alert_async, args=(esp32_ip,))
            t.daemon = True
            t.start()

    # 5. 顯示結果 (人體會變成滑順、邊緣柔和的灰色剪影)
    cv2.imshow("Zero Latency Smooth Shadow Monitor", annotated_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

reader.stop()
cv2.destroyAllWindows()
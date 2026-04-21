import cv2
import mediapipe as mp
import threading
import time
import os
import math
import warnings

warnings.filterwarnings("ignore")

# Inisialisasi MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Menggunakan FaceMesh karena jauh lebih akurat mendeteksi titik hidung daripada FaceDetection biasa
mp_face_mesh = mp.solutions.face_mesh 

# --- Fungsi untuk memutar video dengan kecepatan normal ---
def play_video(video_path):
    video_cap = cv2.VideoCapture(video_path)
    # Ambil FPS asli video agar durasi tidak speed-up
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default jika tidak terdeteksi
    
    delay = int(1000 / fps)
    
    cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
    
    while video_cap.isOpened():
        ret, v_frame = video_cap.read()
        if not ret:
            break
        
        cv2.imshow("Output Video", v_frame)
        # Gunakan delay berdasarkan FPS asli
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    video_cap.release()
    cv2.destroyWindow("Output Video")

# --- Program Utama ---
cap = cv2.VideoCapture(0)
last_trigger_time = 0

# Inisialisasi model
with mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2) as hands, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.7) as face_mesh:
    
    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses deteksi
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        nose_coords = None
        # Ambil titik hidung (Landmark nomor 1 adalah ujung hidung di FaceMesh)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                nose_coords = (int(nose.x * w), int(nose.y * h))
                #cv2.circle(frame, nose_coords, 5, (0, 255, 255), -1)

        tangan_di_hidung = False
        tangan_melambai = False

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Titik 0 (Pergelangan) dan Titik 8 (Ujung Telunjuk)
                wrist = hand_landmarks.landmark[0]
                index_tip = hand_landmarks.landmark[8]
                index_base = hand_landmarks.landmark[5]

                # 1. CEK APAKAH TANGAN DI HIDUNG
                if nose_coords:
                    dist_to_nose = math.hypot((wrist.x * w) - nose_coords[0], (wrist.y * h) - nose_coords[1])
                    if dist_to_nose < 120: # Ambang batas jarak ke hidung
                        tangan_di_hidung = True

                # 2. CEK APAKAH TANGAN MELAMBAI (Telapak lurus terbuka ke depan)
                # Syarat: Ujung jari telunjuk jauh di atas pangkalnya (tangan tegak)
                if index_tip.y < index_base.y:
                    tangan_melambai = True

        # --- LOGIKA TRIGGER FINAL ---
        # Video muncul jika: Ada tangan di hidung DAN ada tangan melambai lurus
        if tangan_di_hidung and tangan_melambai:
            cv2.putText(frame, "POSE KICAU MANIA AKTIF!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            if time.time() - last_trigger_time > 7: # Jeda 7 detik agar tidak bertumpuk
                if os.path.exists("nicaw.mp4"):
                    threading.Thread(target=play_video, args=("nicaw.mp4",)).start()
                    last_trigger_time = time.time()

        cv2.putText(frame, "Pose: Tangan di hidung + Satunya melambai depan", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Kicau Mania Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
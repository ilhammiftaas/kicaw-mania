import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import threading
import time
import os
import math
import warnings

warnings.filterwarnings("ignore")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Fungsi untuk memainkan suara ---
def play_audio(text):
    filename = f"voice_{text.replace(' ', '_').lower()}.mp3"
    tts = gTTS(text=text, lang='id')
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()
    os.remove(filename)


# --- FUNGSI BARU: Mengubah posisi tangan menjadi array 5 angka (0 atau 1) ---
def get_fingers_state(landmarks):
    fingers = []

    # 1. JEMPOL (Thumb)
    # Menggunakan rumus jarak (Euclidean) dari Jempol ke pangkal Kelingking (Titik 17)
    # Jika jarak ujung jempol (4) lebih jauh dari sendi jempol (3), berarti jempol terbuka
    thumb_tip_x, thumb_tip_y = landmarks.landmark[4].x, landmarks.landmark[4].y
    thumb_ip_x, thumb_ip_y = landmarks.landmark[3].x, landmarks.landmark[3].y
    pinky_base_x, pinky_base_y = landmarks.landmark[17].x, landmarks.landmark[17].y
    
    dist_tip = math.hypot(thumb_tip_x - pinky_base_x, thumb_tip_y - pinky_base_y)
    dist_ip = math.hypot(thumb_ip_x - pinky_base_x, thumb_ip_y - pinky_base_y)
    
    if dist_tip > dist_ip:
        fingers.append(1) # Jempol Terbuka
    else:
        fingers.append(0) # Jempol Ditekuk

    # 2. EMPAT JARI LAINNYA (Telunjuk, Tengah, Manis, Kelingking)
    # Bandingkan Ujung Jari (Tip) dengan Sendi Tengahnya (PIP)
    # Titik Tip: 8, 12, 16, 20. Titik PIP: 6, 10, 14, 18.
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    
    for tip, pip in zip(tip_ids, pip_ids):
        # Jika Y ujung lebih kecil dari Y sendi tengah, berarti jari berdiri tegak
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Hasil akhirnya adalah array seperti: [Jempol, Telunjuk, Tengah, Manis, Kelingking]
    return fingers


# --- Fungsi Mencocokkan Pola ---
def detect_gesture(fingers_state):
    # Pola Gestur: [Jempol, Telunjuk, Tengah, Manis, Kelingking]
    # 1 = Berdiri/Terbuka, 0 = Ditekuk/Tertutup

    if fingers_state == [1, 1, 1, 1, 1]: return "Halo"
    if fingers_state == [0, 1, 1, 1, 1]: return "Tanintut"
    if fingers_state == [0, 0, 1, 1, 1]: return "...."
    if fingers_state == [0, 1, 1, 1, 0]: return "Annisuy"
    if fingers_state == [0, 1, 1, 0, 0]: return "dan teman saya"
    if fingers_state == [1, 1, 0, 0, 0]: return "...."
    if fingers_state == [0, 1, 0, 0, 0]: return "nama saya"
    if fingers_state == [0, 1, 0, 0, 1]: return "Terimakasih"
    if fingers_state == [1, 0, 0, 0, 1]: return "...."
    if fingers_state == [0, 0, 0, 0, 0]: return "Ilham"

    return None


# --- Program utama ---
cap = cv2.VideoCapture(0)

last_gesture = None
last_time = 0

# Kamus Warna
gesture_colors = {
    "Halo": (255, 0, 0),               # Biru
    "nama saya": (0, 255, 0),          # Hijau
    "Ilham": (0, 165, 255),            # Oranye
    "Terimakasih": (255, 0, 255),      # Ungu
    "dan teman saya": (255, 255, 0),   # Cyan
    "Annisuy": (203, 192, 255),        # Pink
    "....": (0, 255, 255),            # Kuning
    "....": (0, 0, 255),             # Merah
    "Tanintut": (255, 255, 255),       # Putih
    "....": (128, 0, 128)             # Merah Marun
}

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        gesture = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 1. Ekstrak jari menjadi kode angka 0 dan 1
                fingers_state = get_fingers_state(hand_landmarks)
                
                # 2. Cocokkan kodenya dengan nama gestur
                gesture = detect_gesture(fingers_state)

        if gesture:
            text = gesture
            color = gesture_colors.get(text, (255, 255, 255))
            font = cv2.FONT_HERSHEY_SIMPLEX

            x, y = 50, 80

            cv2.putText(frame, text, (x + 2, y + 2), font, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), font, 1.2, color, 2, cv2.LINE_AA)

            if gesture != last_gesture and time.time() - last_time > 2.5:
                threading.Thread(target=play_audio, args=(text,)).start()
                last_gesture = gesture
                last_time = time.time()

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

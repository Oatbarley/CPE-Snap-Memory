import cv2
import mediapipe as mp
import time
import math
import threading
import requests
import io
import os
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import ttk
import qrcode
import numpy as np

# Configuration
SERVER_URL = 'https://lindsy-unsquelched-livia.ngrok-free.dev'
FRAME_PATH = r'D:\University\Year 3\OPH Project\Final\frame.png' #Fix your Path frame.png
COUNTDOWN_SECONDS = 5
HAND_HOLD_SECONDS = 3
TARGET_SIZE = (1000, 1000)

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Face High-Five Capture (Client)')
        self.root.configure(bg="#FFFFFF")

        # ====== LEFT PANEL (QR + History) ======
        left_frame = ttk.Frame(root)
        left_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        ttk.Label(left_frame, text="📷 Capture Info", font=('Helvetica', 14, 'bold')).pack(pady=(0, 10))

        self.qr_label = ttk.Label(left_frame)
        self.qr_label.pack(pady=(0, 10))

        self.qr_status_var = tk.StringVar(value='Latest: -')
        ttk.Label(left_frame, textvariable=self.qr_status_var).pack()

        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(left_frame, text='🕒 History:', font=('Helvetica', 12, 'bold')).pack(anchor='w')
        self.history_list = tk.Listbox(left_frame, height=15, width=30)
        self.history_list.pack(pady=5)
        self.history_list.bind("<Double-1>", self.show_qr_from_history)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text='Refresh', command=self.refresh_history).pack(side='left', padx=2)
        ttk.Button(btn_frame, text='Download', command=self.download_selected).pack(side='left', padx=2)
        # ====== Center PANEL (Camera) ======
        center_frame = ttk.Frame(root)
        center_frame.grid(row=0, column=1, sticky="ns", padx=50, pady=50)
        self.video_canvas = tk.Canvas(center_frame, width=1000, height=1000, bg='white')
        self.video_canvas.pack()
        # ====== RIGHT PANEL (STATUS) ======
        right_frame = ttk.Frame(root)
        right_frame.grid(row=0, column=2, padx=10, pady=10)

        self.countdown_var = tk.StringVar(value='Ready')
        self.countdown_label = ttk.Label(right_frame, textvariable=self.countdown_var,
                                         font=('Helvetica', 30, 'bold'), foreground='green')
        self.countdown_label.pack(pady=15)

        # ====== Setup ======
        #ปรับเป็น 0 เมื่อใช้หล้อง webcam ปรับเป็น 1 เมื่อใช้กล้องนอก
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.running = True

        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.frame_overlay = Image.open(FRAME_PATH).convert('RGBA') if os.path.exists(FRAME_PATH) else None

        self.stage = "idle"
        self.stage_start = None
        self.cooldown_until = 0

        self.update_video()
        self.root.after(2000, self.refresh_history)

    # ===== Utility Functions =====
    def update_status(self, text, color='green'):
        self.countdown_label.configure(foreground=color)
        self.countdown_var.set(text)

    def refresh_history(self):
        try:
            r = requests.get(SERVER_URL + '/images', timeout=3)
            if r.ok:
                files = r.json().get('files', [])
                self.history_list.delete(0, tk.END)
                for f in reversed(files): 
                    self.history_list.insert(tk.END, f)
        except Exception as e:
            print('History refresh error', e)

    def download_selected(self):
        sel = self.history_list.curselection()
        if not sel:
            return
        fname = self.history_list.get(sel[0])
        url = f"{SERVER_URL}/download/{fname}"
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                os.makedirs('downloads', exist_ok=True)
                with open(os.path.join('downloads', fname), 'wb') as f:
                    f.write(r.content)
                self.update_status(f'Downloaded {fname}', 'blue')
        except Exception as e:
            print('Download error', e)

    def generate_qr(self, filename):
        url = f"{SERVER_URL}/download/{filename}"
        qr = qrcode.QRCode(box_size=4)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image().convert('RGB').resize((200, 200), Image.LANCZOS)
        self.qr_img = ImageTk.PhotoImage(img)
        self.qr_label.configure(image=self.qr_img)
        self.qr_status_var.set(f'Latest: {filename}')

    def show_qr_from_history(self, event):
        sel = self.history_list.curselection()
        if not sel:
            return
        self.generate_qr(self.history_list.get(sel[0]))

    # ===== Image Overlay =====
    def overlay_frame(self, pil_img):
        enhancer = ImageEnhance.Brightness(pil_img)
        bright_img = enhancer.enhance(1.2)

        if self.frame_overlay is None:
            return bright_img

        frame_canvas = Image.new("RGBA", self.frame_overlay.size, (0, 0, 0, 0))
        x_offset = (self.frame_overlay.width - bright_img.width) // 2
        y_offset = (self.frame_overlay.height - bright_img.height) // 10
        frame_canvas.paste(bright_img, (x_offset, y_offset), mask=bright_img.convert("RGBA"))
        out = Image.alpha_composite(frame_canvas, self.frame_overlay)
        return out

    def resize_to_aspect(self, image_bgr):
        h, w, _ = image_bgr.shape
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
        return image_bgr[y1:y1 + side, x1:x1 + side]

    # ===== Gesture Detection =====
    @staticmethod
    def angle_between_points(a, b, c):
        ba, bc = (a.x - b.x, a.y - b.y), (c.x - b.x, c.y - b.y)
        dot, mag = ba[0] * bc[0] + ba[1] * bc[1], math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        return math.acos(max(-1, min(1, dot / mag))) if mag != 0 else 0

    @staticmethod
    def fingers_separated(lm, threshold=0.05):
        x_vals = [lm[i].x for i in [
            mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]]
        return all(abs(x_vals[i + 1] - x_vals[i]) >= threshold for i in range(4))

    @staticmethod
    def palm_facing_camera(lm):
        return lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y < lm[mp_hands.HandLandmark.WRIST].y

    @staticmethod
    def check_high_five(hand_landmarks):
        if not hand_landmarks:
            return False
        try:
            lm = hand_landmarks.landmark
            finger_joints = [
                (mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_TIP),
                (mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_TIP),
                (mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                (mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_TIP),
                (mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_TIP)]

            for b, m, t in finger_joints:
                if math.degrees(App.angle_between_points(lm[b], lm[m], lm[t])) < 160:
                    return False

            if not App.palm_facing_camera(lm):
                return False
            if not App.fingers_separated(lm):
                return False

            return True
        except:
            return False

    # ===== Upload Logic =====
    def upload_image(self, pil_img):
        self.stage = "uploading"
        self.update_status("Uploading...", 'blue')
        buf = io.BytesIO()
        pil_img.convert('RGB').save(buf, format='PNG')
        buf.seek(0)
        try:
            r = requests.post(SERVER_URL + '/upload', files={'file': ('upload.png', buf, 'image/png')}, timeout=10)
            if r.ok:
                fname = r.json().get('filename')
                self.update_status(fname, 'green')
                self.generate_qr(fname)
                self.refresh_history()
            else:
                self.update_status('Upload failed', 'red')
        except Exception as e:
            print('Upload exception', e)
            self.update_status('Upload error', 'red')
        finally:
            self.stage = 'cooldown'
            self.cooldown_until = time.time() + 5

    # ===== Main Video Loop =====
    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_gui = frame.copy()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_detection.process(image_rgb)
            hand_results = self.hands.process(image_rgb)

            # Handle stage logic
            if self.stage == 'cooldown':
                if time.time() >= self.cooldown_until:
                    self.stage = 'idle'
                    self.update_status('Ready!!', 'green')
                else:
                    self.update_status('Cooldown...', 'orange')
            else:
                if self.stage == 'idle':
                    detected = False
                    if hand_results.multi_hand_landmarks:
                        for h in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame_gui, h, mp_hands.HAND_CONNECTIONS)
                            if self.check_high_five(h):
                                self.stage = 'hand_hold'
                                self.stage_start = time.time()
                                detected = True
                                break
                    if not detected:
                        self.stage = 'idle'
                    self.update_status('Ready!!', 'green')

                elif self.stage == 'hand_hold':
                    if hand_results.multi_hand_landmarks:
                        remaining = max(0, HAND_HOLD_SECONDS - (time.time() - self.stage_start))
                        self.update_status(f'Hold hand: {int(remaining) + 1}s', 'red')
                        for h in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame_gui, h, mp_hands.HAND_CONNECTIONS)
                        if remaining <= 0:
                            self.stage = 'countdown'
                            self.stage_start = time.time()
                    else:
                        self.stage = 'idle'

                elif self.stage == 'countdown':
                    remaining = max(0, COUNTDOWN_SECONDS - (time.time() - self.stage_start))
                    self.update_status(f'Taking photo in: {int(remaining) + 1}s', 'green')
                    if remaining <= 0:
                        crop = self.resize_to_aspect(frame)
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(TARGET_SIZE, Image.LANCZOS)
                        out = self.overlay_frame(pil_img)
                        os.makedirs('local_captures', exist_ok=True)
                        out.save(os.path.join('local_captures', 'last_capture.png'))
                        threading.Thread(target=self.upload_image, args=(out,), daemon=True).start()
                        self.stage = 'idle'

            preview = self.resize_to_aspect(frame_gui)
            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)).resize(TARGET_SIZE, Image.LANCZOS))
            self.video_canvas.create_image(0, 0, anchor='nw', image=img_tk)
            self.video_canvas.imgtk = img_tk

        self.root.after(30, self.update_video)

    def stop(self):
        self.running = False
        self.face_detection.close()
        self.hands.close()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.protocol('WM_DELETE_WINDOW', lambda: (app.stop(), root.destroy()))
    root.mainloop()

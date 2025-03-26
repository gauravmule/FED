import cv2
import threading
import time
import queue
from deepface import DeepFace
from database_setup import get_db_connection
from flask import session
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.emotion_summary = {"total_faces": 0, "emotions": {}}
        self.face_tracker = {}
        self.frame_queue = queue.Queue(maxsize=3)
        self.processed_frame = None
        self.tracking_threshold = 75
        self.frame_skip_count = 5
        self.current_frame_count = 0
        self.session_id = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.analyzer_backend = 'opencv'

    def _analyze_emotion(self, face_roi):
        try:
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(
                img_path=rgb_face,
                actions=['emotion'],
                detector_backend=self.analyzer_backend,
                enforce_detection=False,
                silent=True
            )
            if result and isinstance(result, list):
                emotions = result[0]['emotion']
                dominant = max(emotions, key=emotions.get)
                return dominant
            return 'neutral'
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return 'neutral'

    def start_session(self):
        with self.lock:
            if self.is_running:
                return False
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Webcam access failed")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            
            conn = get_db_connection()
            if conn:
                try:
                    with conn.cursor() as cursor:
                        # Fixed line below
                        cursor.execute(
                            "INSERT INTO sessions (user_id) VALUES (%s)",
                            (session.get('user_id'),)  # <-- Closing parenthesis added
                        )
                        self.session_id = cursor.lastrowid
                        conn.commit()
                        logger.info(f"New session started: {self.session_id}")
                except Exception as e:
                    logger.error(f"Session creation failed: {e}")
                    self.session_id = None
                finally:
                    conn.close()
            
            if not self.session_id:
                self.is_running = False
                return False
            
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.capture_thread.start()
            self.process_thread.start()
            return True

    def stop_session(self):
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            
            if self.capture_thread:
                self.capture_thread.join(timeout=2)
            if self.process_thread:
                self.process_thread.join(timeout=2)
            
            conn = get_db_connection()
            if conn and self.session_id:
                try:
                    with conn.cursor() as cursor:
                        # Insert emotion logs
                        for fid, (_, emotion, _) in self.face_tracker.items():
                            if emotion:
                                cursor.execute(
                                    "INSERT INTO emotion_logs (session_id, emotion) VALUES (%s, %s)",
                                    (self.session_id, emotion)
                                )
                        
                        # Update session stats
                        total_faces = self.emotion_summary['total_faces']
                        common_emotion = max(
                            self.emotion_summary['emotions'],
                            key=self.emotion_summary['emotions'].get,
                            default='neutral'
                        )
                        cursor.execute(
                            "UPDATE sessions SET end_time = NOW(), total_faces = %s, most_common_emotion = %s WHERE id = %s",
                            (total_faces, common_emotion, self.session_id)
                        )
                        
                        # Update user stats
                        cursor.execute('''
                            UPDATE users SET
                                total_sessions = total_sessions + 1,
                                total_faces_detected = total_faces_detected + %s,
                                most_common_emotion = (
                                    SELECT emotion FROM emotion_logs 
                                    WHERE session_id IN (
                                        SELECT id FROM sessions WHERE user_id = %s
                                    )
                                    GROUP BY emotion 
                                    ORDER BY COUNT(*) DESC 
                                    LIMIT 1
                                )
                            WHERE id = %s
                        ''', (total_faces, session.get('user_id'), session.get('user_id')))
                        
                        conn.commit()
                except Exception as e:
                    logger.error(f"Session cleanup failed: {e}")
                    conn.rollback()
                finally:
                    conn.close()
            
            self.face_tracker.clear()
            self.emotion_summary = {"total_faces": 0, "emotions": {}}
            self.session_id = None

    def _capture_frames(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    if not self.frame_queue.full():
                        small_frame = cv2.resize(frame, (320, 240))
                        self.frame_queue.put(small_frame)
                except:
                    pass
            time.sleep(0.01)

    def _process_frames(self):
        while self.is_running:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                small_frame = self.frame_queue.get()
                self.current_frame_count += 1
                process_emotion = (self.current_frame_count % self.frame_skip_count) == 0
                
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                
                frame = cv2.resize(small_frame, (640, 480))
                current_emotions = []
                new_tracker = {}
                
                for (x, y, w, h) in faces:
                    x *= 2; y *= 2; w *= 2; h *= 2
                    face_roi = frame[y:y+h, x:x+w]
                    
                    emotion = None
                    if process_emotion:
                        emotion = self._analyze_emotion(face_roi)
                    
                    centroid = (x + w//2, y + h//2)
                    closest_id = None
                    min_dist = float('inf')
                    
                    # Track existing faces
                    for fid, (old_cent, old_emotion, _) in self.face_tracker.items():
                        dist = ((centroid[0]-old_cent[0])**2 + (centroid[1]-old_cent[1])**2)**0.5
                        if dist < min_dist and dist < self.tracking_threshold:
                            min_dist = dist
                            closest_id = fid
                            if not process_emotion:
                                emotion = old_emotion
                    
                    fid = closest_id if closest_id else len(new_tracker)
                    new_tracker[fid] = (centroid, emotion, (x, y, w, h))
                    if emotion:
                        current_emotions.append(emotion)
                    
                    # Draw annotations
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    if emotion:
                        cv2.putText(frame, f"{emotion} ID:{fid}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.face_tracker = new_tracker
                if process_emotion:
                    emotion_counts = {e: current_emotions.count(e) for e in set(current_emotions)}
                    self.emotion_summary = {
                        "total_faces": len(faces),
                        "emotions": emotion_counts if emotion_counts else {"neutral": 0}
                    }
                
                self.processed_frame = frame
            except Exception as e:
                logger.error(f"Frame processing error: {e}")

    def generate_frames(self):
        while self.is_running:
            if self.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)
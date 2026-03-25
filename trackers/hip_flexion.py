import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class HipFlexionTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Parameters
        self.VISIBILITY_THRESHOLD = 0.5
        self.TORSO_TILT_THRESHOLD = 25.0
        self.KNEE_EXTENDED_THRESHOLD = 160
        self.HIP_EXTENDED_THRESHOLD = 170
        self.HIP_FLEXION_MIN = 130
        self.HIP_FLEXION_MAX = 140
        self.READY_FRAMES_REQUIRED = 8
        self.HOLD_PEAK_FRAMES = 4
        self.SMOOTH_ALPHA = 0.35
        self.MAX_HISTORY = 30

        # State
        self.counter = 0
        self.correct_reps = 0
        self.wrong_reps = 0
        self.stage = "INIT"
        self.feedback = ""
        self.ready_hold = 0
        self.peak_hold = 0
        self.hip_angle_smoothed = None
        self.hip_history = deque(maxlen=self.MAX_HISTORY)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def torso_tilt_deg(self, shoulder, hip):
        vec = np.array(shoulder) - np.array(hip)
        vertical = np.array([0.0, -1.0])
        cosine = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return abs(np.degrees(np.arccos(cosine)))

    def visible(self, landmark):
        return landmark and (landmark.visibility >= self.VISIBILITY_THRESHOLD)

    def reset_counters(self):
        self.counter = 0
        self.correct_reps = 0
        self.wrong_reps = 0
        self.stage = "INIT"
        self.feedback = "Reset Done"

    def get_stats(self):
        return {
            "total_reps": self.counter,
            "correct_reps": self.correct_reps,
            "wrong_reps": self.wrong_reps,
            "stage": self.stage,
            "feedback": self.feedback or "Get into starting posture"
        }

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image = frame.copy()

        self.feedback = ""
        active_side = "UNKNOWN"

        try:
            if not results.pose_landmarks:
                self.feedback = "No person detected"
                return image

            lm = results.pose_landmarks.landmark

            # Choose active side
            left_vis = np.mean([
                lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
                lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
                lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            ])
            right_vis = np.mean([
                lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
                lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            ])
            side = "LEFT" if left_vis > right_vis else "RIGHT"
            active_side = side

            # Get keypoints
            if side == "RIGHT":
                sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                hi = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                kn = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
                an = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            else:
                sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                hi = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                kn = lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
                an = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

            if not (self.visible(sh) and self.visible(hi) and self.visible(kn)):
                self.feedback = "Move fully into view"
                return image

            shoulder = [sh.x, sh.y]
            hip = [hi.x, hi.y]
            knee = [kn.x, kn.y]

            hip_angle = self.calculate_angle(shoulder, hip, knee)
            ankle = [an.x, an.y] if self.visible(an) else None
            knee_angle = self.calculate_angle(hip, knee, ankle) if ankle else 180

            if self.hip_angle_smoothed is None:
                self.hip_angle_smoothed = hip_angle
            else:
                self.hip_angle_smoothed = self.SMOOTH_ALPHA * hip_angle + (1 - self.SMOOTH_ALPHA) * self.hip_angle_smoothed

            torso_deg = self.torso_tilt_deg(shoulder, hip)
            posture_ok = (torso_deg <= self.TORSO_TILT_THRESHOLD) and (knee_angle >= self.KNEE_EXTENDED_THRESHOLD)

            if posture_ok:
                self.ready_hold += 1
            else:
                self.ready_hold = 0
            ready_to_count = self.ready_hold >= self.READY_FRAMES_REQUIRED

            # STATE MACHINE
            if self.stage == "INIT":
                self.feedback = "Get into starting posture"
                if ready_to_count:
                    self.stage = "READY"

            elif self.stage == "READY":
                self.feedback = "Start flexion"
                if self.hip_angle_smoothed < self.HIP_EXTENDED_THRESHOLD - 10:
                    self.stage = "FLEXING"

            elif self.stage == "FLEXING":
                if self.HIP_FLEXION_MIN <= self.hip_angle_smoothed <= self.HIP_FLEXION_MAX:
                    self.peak_hold += 1
                    self.feedback = f"Holding ({self.peak_hold}/{self.HOLD_PEAK_FRAMES})"
                    if self.peak_hold >= self.HOLD_PEAK_FRAMES:
                        self.counter += 1
                        if posture_ok:
                            self.correct_reps += 1
                            self.feedback = "✅ Good rep"
                        else:
                            self.wrong_reps += 1
                            self.feedback = "⚠️ Bad posture"
                        self.stage = "PEAK"
                else:
                    self.peak_hold = 0

            elif self.stage == "PEAK":
                if self.hip_angle_smoothed > (self.HIP_EXTENDED_THRESHOLD - 10):
                    self.stage = "READY"
                    self.feedback = "Extend back to start"

            # Draw skeleton
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                           self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

            # Draw colored leg lines
            hip_px, knee_px, ankle_px = (int(hi.x * w), int(hi.y * h)), (int(kn.x * w), int(kn.y * h)), (int(an.x * w), int(an.y * h))
            color = (0, 255, 0) if posture_ok else (0, 0, 255)
            cv2.line(image, hip_px, knee_px, color, 6)
            cv2.line(image, knee_px, ankle_px, color, 6)

            # Overlay text
            cv2.rectangle(image, (0, 0), (380, 120), (30, 30, 30), -1)
            cv2.putText(image, f'Side: {active_side}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Total: {self.counter}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f'Correct: {self.correct_reps}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f'Wrong: {self.wrong_reps}', (150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(image, self.feedback, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        except Exception as e:
            self.feedback = "Tracking error"
            cv2.putText(image, f"Error: {e}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return image
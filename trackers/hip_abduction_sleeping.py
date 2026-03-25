import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class HipAbductionSleepingTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Parameters
        self.VISIBILITY_THRESHOLD = 0.5
        self.HORIZONTAL_TORSO_TOLERANCE = 20.0
        self.KNEE_EXTENDED_THRESHOLD = 160.0
        self.HIP_EXTENDED_THRESHOLD = 170.0
        self.HIP_FLEXION_MIN = 135.0
        self.HIP_FLEXION_MAX = 145.0
        self.READY_FRAMES_REQUIRED = 10
        self.HOLD_PEAK_FRAMES = 5
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

    def torso_angle_from_vertical(self, shoulder, hip):
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
            "feedback": self.feedback or "Lie on your back with legs straight"
        }

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
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

            torso_angle = self.torso_angle_from_vertical(shoulder, hip)
            is_torso_horizontal = abs(torso_angle - 90) <= self.HORIZONTAL_TORSO_TOLERANCE
            is_leg_extended = knee_angle >= self.KNEE_EXTENDED_THRESHOLD
            is_hip_extended = hip_angle >= self.HIP_EXTENDED_THRESHOLD

            posture_ok = is_torso_horizontal and is_leg_extended and is_hip_extended

            if posture_ok:
                self.ready_hold += 1
            else:
                self.ready_hold = 0
            ready_to_count = self.ready_hold >= self.READY_FRAMES_REQUIRED

            # STATE MACHINE
            if self.stage == "INIT":
                if ready_to_count:
                    self.stage = "READY"
                    self.feedback = "Posture OK — raise your leg"
                else:
                    self.feedback = "Lie on your back with legs straight"

            elif self.stage == "READY":
                if self.hip_angle_smoothed < self.HIP_EXTENDED_THRESHOLD - 10:
                    self.stage = "RAISING"
                    self.feedback = "Raising..."
                    self.peak_hold = 0
                else:
                    self.feedback = "Ready — raise your leg"

            elif self.stage == "RAISING":
                if self.HIP_FLEXION_MIN <= self.hip_angle_smoothed <= self.HIP_FLEXION_MAX:
                    self.peak_hold += 1
                    self.feedback = f"Hold peak ({self.peak_hold}/{self.HOLD_PEAK_FRAMES})"
                    if self.peak_hold >= self.HOLD_PEAK_FRAMES:
                        if is_torso_horizontal and knee_angle >= self.KNEE_EXTENDED_THRESHOLD:
                            self.stage = "PEAK"
                            self.counter += 1
                            self.correct_reps += 1
                            self.feedback = "✅ Counted (good)"
                        else:
                            self.stage = "PEAK"
                            self.counter += 1
                            self.wrong_reps += 1
                            self.feedback = "⚠ Counted (posture issue)"
                else:
                    self.feedback = "Raise higher" if self.hip_angle_smoothed > self.HIP_FLEXION_MAX else "Lowering too soon"
                    self.peak_hold = 0

            elif self.stage == "PEAK":
                if self.hip_angle_smoothed > self.HIP_FLEXION_MAX:
                    self.stage = "LOWERING"
                    self.feedback = "Lowering..."
                else:
                    self.feedback = "Hold/lower slowly"

            elif self.stage == "LOWERING":
                if ready_to_count:
                    self.stage = "READY"
                    self.feedback = "Ready for next rep"
                else:
                    self.feedback = "Finish lowering your leg"

            # Draw skeleton with custom colors
            red_point_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
            white_line_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
            
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                         landmark_drawing_spec=red_point_spec,
                                         connection_drawing_spec=white_line_spec)

            # Draw colored leg lines
            if hi and kn and an and self.visible(hi) and self.visible(kn) and self.visible(an):
                hip_px = (int(hi.x * w), int(hi.y * h))
                knee_px = (int(kn.x * w), int(kn.y * h))
                ankle_px = (int(an.x * w), int(an.y * h))
                
                is_correct_posture = (is_torso_horizontal and 
                                    knee_angle >= self.KNEE_EXTENDED_THRESHOLD and 
                                    self.HIP_FLEXION_MIN <= self.hip_angle_smoothed <= self.HIP_FLEXION_MAX)
                
                color = (0, 255, 0) if is_correct_posture else (255, 255, 255)
                thickness = 8
                cv2.line(image, hip_px, knee_px, color, thickness)
                cv2.line(image, knee_px, ankle_px, color, thickness)

            # UI Overlays
            cv2.rectangle(image, (0, 0), (380, 140), (50, 50, 50), -1)
            cv2.putText(image, f'Side: {active_side}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.putText(image, f'Total: {self.counter}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f'Correct: {self.correct_reps}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f'Wrong: {self.wrong_reps}', (150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f'Stage: {self.stage}', (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Hip angle label
            if hi and self.visible(hi):
                hip_px = (int(hi.x * w), int(hi.y * h))
                angle_text = f'{int(self.hip_angle_smoothed) if self.hip_angle_smoothed is not None else "--"}°'
                cv2.putText(image, angle_text, hip_px, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Feedback text
            cv2.putText(image, self.feedback, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        except Exception as e:
            self.feedback = "Tracking error"
            cv2.putText(image, f"Error: {e}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return image
import cv2
import mediapipe as mp
from mediapipe import solutions as mp_solutions
import numpy as np
from collections import deque

class HipAbductionStandingTracker:
    def __init__(self):
        self.mp_pose = mp_solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Parameters
        self.VISIBILITY_THRESHOLD = 0.5
        self.TORSO_TILT_THRESHOLD = 15.0
        self.KNEE_STRAIGHT_THRESHOLD = 165
        self.HIP_ABDUCTION_MIN = 25.0
        self.HIP_ABDUCTION_MAX = 50.0
        self.READY_FRAMES_REQUIRED = 10
        self.HOLD_PEAK_FRAMES = 5
        self.SMOOTH_ALPHA = 0.3
        
        # State
        self.counter = 0
        self.correct_reps = 0
        self.wrong_reps = 0
        self.stage = "INIT"
        self.feedback = ""
        self.ready_hold = 0
        self.peak_hold = 0
        self.left_hip_angle_hist = deque(maxlen=5)
        self.right_hip_angle_hist = deque(maxlen=5)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    def angle_between_vectors(self, v1, v2):
        v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

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
            "feedback": self.feedback or "Stand straight to begin"
        }

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        self.feedback = ""
        active_side = "NONE"

        try:
            if not results.pose_landmarks:
                self.feedback = "No person detected"
                return image

            lm = results.pose_landmarks.landmark

            # Key landmarks
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            l_knee = lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            r_knee = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            l_ank = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            r_ank = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Visibility check
            core_visible = (self.visible(l_sh) and self.visible(r_sh) and 
                          self.visible(l_hip) and self.visible(r_hip))
            left_leg_visible = self.visible(l_knee) and self.visible(l_ank)
            right_leg_visible = self.visible(r_knee) and self.visible(r_ank)

            if not core_visible:
                self.feedback = "Move fully into view (torso)"
                return image

            # Angle calculations
            sh_mid = np.array([(l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2])
            hip_mid = np.array([(l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2])
            
            # Torso tilt
            shoulder_vec = np.array([r_sh.x - l_sh.x, r_sh.y - l_sh.y])
            horizontal = np.array([1, 0])
            torso_tilt = self.angle_between_vectors(shoulder_vec, horizontal)
            if torso_tilt > 90:
                torso_tilt = 180 - torso_tilt
                
            # Torso vertical vector
            torso_vertical_vec = hip_mid - sh_mid
            
            # Left leg angles
            left_hip_angle = 0
            left_knee_angle = 180
            if left_leg_visible:
                left_leg_vec = np.array([l_knee.x - l_hip.x, l_knee.y - l_hip.y])
                left_hip_angle = self.angle_between_vectors(torso_vertical_vec, left_leg_vec)
                left_knee_angle = self.calculate_angle([l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ank.x, l_ank.y])
            
            # Right leg angles
            right_hip_angle = 0
            right_knee_angle = 180
            if right_leg_visible:
                right_leg_vec = np.array([r_knee.x - r_hip.x, r_knee.y - r_hip.y])
                right_hip_angle = self.angle_between_vectors(torso_vertical_vec, right_leg_vec)
                right_knee_angle = self.calculate_angle([r_hip.x, r_hip.y], [r_knee.x, r_knee.y], [r_ank.x, r_ank.y])
                
            # Smoothing
            self.left_hip_angle_hist.append(left_hip_angle)
            self.right_hip_angle_hist.append(right_hip_angle)
            left_hip_angle_smoothed = np.mean(self.left_hip_angle_hist)
            right_hip_angle_smoothed = np.mean(self.right_hip_angle_hist)
            
            # Determine active side
            if left_hip_angle_smoothed > right_hip_angle_smoothed + 5:
                active_side = "LEFT"
                hip_angle = left_hip_angle_smoothed
                knee_angle = left_knee_angle
            elif right_hip_angle_smoothed > left_hip_angle_smoothed + 5:
                active_side = "RIGHT"
                hip_angle = right_hip_angle_smoothed
                knee_angle = right_knee_angle
            else:
                active_side = "NONE"
                hip_angle = 0
                knee_angle = 180
                
            # State machine
            posture_ok = (torso_tilt < self.TORSO_TILT_THRESHOLD) and (left_hip_angle < 10 and right_hip_angle < 10)

            if self.stage == "INIT":
                if posture_ok:
                    self.ready_hold += 1
                    self.feedback = f"Hold starting pose ({self.ready_hold}/{self.READY_FRAMES_REQUIRED})"
                    if self.ready_hold >= self.READY_FRAMES_REQUIRED:
                        self.stage = "READY"
                else:
                    self.ready_hold = 0
                    self.feedback = "Stand straight to begin"

            elif self.stage == "READY":
                self.feedback = "Ready: Lift a leg sideways"
                if hip_angle > 15 and active_side != "NONE":
                    self.stage = "ABDUCTING"
            
            elif self.stage == "ABDUCTING":
                self.feedback = "Lifting..."
                if self.HIP_ABDUCTION_MIN <= hip_angle <= self.HIP_ABDUCTION_MAX:
                    self.peak_hold += 1
                    self.feedback = f"Hold peak ({self.peak_hold}/{self.HOLD_PEAK_FRAMES})"
                    if self.peak_hold >= self.HOLD_PEAK_FRAMES:
                        self.stage = "PEAK"
                        self.counter += 1
                        # Check form at the peak
                        if torso_tilt < self.TORSO_TILT_THRESHOLD and knee_angle > self.KNEE_STRAIGHT_THRESHOLD:
                            self.correct_reps += 1
                            self.feedback = "✅ Counted (Good)"
                        else:
                            self.wrong_reps += 1
                            self.feedback = "⚠ Counted (Check form)"
                elif hip_angle > self.HIP_ABDUCTION_MAX:
                    self.feedback = "Too high!"
                    self.peak_hold = 0
                else:
                    self.peak_hold = 0
            
            elif self.stage == "PEAK":
                self.feedback = "Return to start"
                if hip_angle < 15:
                    self.stage = "RETURNING"
            
            elif self.stage == "RETURNING":
                self.feedback = "Returning..."
                if hip_angle < 10 and torso_tilt < self.TORSO_TILT_THRESHOLD:
                    self.stage = "READY"
                    self.peak_hold = 0
                    self.feedback = "Ready for next rep"

            # Determine if form is good for visual feedback
            form_is_good = False
            if self.stage == "PEAK" or self.stage == "ABDUCTING":
                if (active_side == "LEFT" and 
                    self.HIP_ABDUCTION_MIN <= left_hip_angle_smoothed <= self.HIP_ABDUCTION_MAX and 
                    torso_tilt < self.TORSO_TILT_THRESHOLD and 
                    left_knee_angle > self.KNEE_STRAIGHT_THRESHOLD):
                    form_is_good = True
                elif (active_side == "RIGHT" and 
                      self.HIP_ABDUCTION_MIN <= right_hip_angle_smoothed <= self.HIP_ABDUCTION_MAX and 
                      torso_tilt < self.TORSO_TILT_THRESHOLD and 
                      right_knee_angle > self.KNEE_STRAIGHT_THRESHOLD):
                    form_is_good = True

            # Draw skeleton with custom colors
            red_point_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
            white_line_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
            
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                         landmark_drawing_spec=red_point_spec,
                                         connection_drawing_spec=white_line_spec)

            # UI Overlays
            cv2.rectangle(image, (0, 0), (380, 140), (50, 50, 50), -1)
            cv2.putText(image, f'Side: {active_side}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.putText(image, f'Total: {self.counter}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f'Correct: {self.correct_reps}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f'Wrong: {self.wrong_reps}', (150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f'Stage: {self.stage}', (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Draw active leg with color-coded lines
            line_color = (0, 255, 0) if form_is_good else (0, 0, 255)
            
            if active_side == "LEFT" and left_leg_visible:
                hip_point = (int(l_hip.x*w), int(l_hip.y*h))
                knee_point = (int(l_knee.x*w), int(l_knee.y*h))
                ankle_point = (int(l_ank.x*w), int(l_ank.y*h))
                
                cv2.line(image, hip_point, knee_point, line_color, 6)
                cv2.line(image, knee_point, ankle_point, line_color, 6)
                
                cv2.putText(image, f"{int(left_hip_angle_smoothed)}°", 
                           (int(l_hip.x*w)+10, int(l_hip.y*h)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                           
            elif active_side == "RIGHT" and right_leg_visible:
                hip_point = (int(r_hip.x*w), int(r_hip.y*h))
                knee_point = (int(r_knee.x*w), int(r_knee.y*h))
                ankle_point = (int(r_ank.x*w), int(r_ank.y*h))
                
                cv2.line(image, hip_point, knee_point, line_color, 6)
                cv2.line(image, knee_point, ankle_point, line_color, 6)
                
                cv2.putText(image, f"{int(right_hip_angle_smoothed)}°", 
                           (int(r_hip.x*w)-50, int(r_hip.y*h)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Feedback text
            cv2.putText(image, self.feedback, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        except Exception as e:
            self.feedback = "Tracking error"
            cv2.putText(image, f"Error: {e}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return image
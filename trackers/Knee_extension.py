import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions as mp_solutions


class KneeExtensionTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp_solutions.pose
        self.pose = self.mp_pose.Pose()
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # thresholds
        self.KNEE_UP_THRESHOLD = 145
        self.KNEE_DOWN_THRESHOLD = 100
        self.BACK_MIN = 90
        
        # state variables
        self.counter = 0
        self.correct_reps = 0
        self.wrong_reps = 0
        self.stage = 'down'
        self.feedback = []
        self.last_up_knee = None
        self.last_up_hip = None

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    def process_frame(self, frame):
        image = cv2.flip(frame, 1)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        self.feedback = []

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # choose visible side
            left_vis = lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility
            right_vis = lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            use_left = left_vis >= right_vis

            if use_left:
                hip_idx = self.mp_pose.PoseLandmark.LEFT_HIP.value
                knee_idx = self.mp_pose.PoseLandmark.LEFT_KNEE.value
                ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value
                shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
                side_text = "Left"
            else:
                hip_idx = self.mp_pose.PoseLandmark.RIGHT_HIP.value
                knee_idx = self.mp_pose.PoseLandmark.RIGHT_KNEE.value
                ankle_idx = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
                shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                side_text = "Right"

            # normalized positions
            hip_n = [lm[hip_idx].x, lm[hip_idx].y]
            knee_n = [lm[knee_idx].x, lm[knee_idx].y]
            ankle_n = [lm[ankle_idx].x, lm[ankle_idx].y]
            shoulder_n = [lm[shoulder_idx].x, lm[shoulder_idx].y]

            # angles
            knee_angle = self.calculate_angle(hip_n, knee_n, ankle_n)
            hip_angle = self.calculate_angle(shoulder_n, hip_n, knee_n)

            # pixel positions
            def to_px(npt):
                return (int(npt[0]*w), int(npt[1]*h))
            hip_px = to_px(hip_n)
            knee_px = to_px(knee_n)
            ankle_px = to_px(ankle_n)
            shoulder_px = to_px(shoulder_n)

            # rep detection
            if knee_angle > self.KNEE_UP_THRESHOLD and self.stage == 'down':
                self.stage = 'up'
                self.last_up_knee = knee_angle
                self.last_up_hip = hip_angle
                self.feedback = ["Hold for 2 sec and then lower slowly."]
            elif knee_angle < self.KNEE_DOWN_THRESHOLD and self.stage == 'up':
                self.stage = 'down'
                self.counter += 1
                if self.last_up_hip >= self.BACK_MIN:
                    self.correct_reps += 1
                    self.feedback = ["Correct rep."]
                else:
                    self.wrong_reps += 1
                    self.feedback = ["Keep back upright next time."]
                self.last_up_knee, self.last_up_hip = None, None
            else:
                if knee_angle < self.KNEE_UP_THRESHOLD and self.stage == 'down':
                    self.feedback.append("Lift your leg higher.")
                if hip_angle < self.BACK_MIN:
                    self.feedback.append("Try to keep your back upright.")

            # drawing skeleton
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

            # leg color: green when at correct up posture
            leg_green = (knee_angle >= self.KNEE_UP_THRESHOLD and hip_angle >= self.BACK_MIN)
            leg_color = (0,255,0) if leg_green else (0,0,255)

            # draw hip–knee–ankle
            cv2.line(image, hip_px, knee_px, leg_color, 6)
            cv2.line(image, knee_px, ankle_px, leg_color, 6)
            for pt in [hip_px, knee_px, ankle_px]:
                cv2.circle(image, pt, 8, leg_color, -1)

            # angle display near joints
            cv2.putText(image, f"{int(knee_angle)}", (knee_px[0]+10, knee_px[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image, f"{int(hip_angle)}", (hip_px[0]+10, hip_px[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # UI Text
            cv2.putText(image, f"{side_text} Side", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(image, f"Reps: {self.counter}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(image, f"Correct: {self.correct_reps}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # cv2.putText(image, f"Wrong: {self.wrong_reps}", (220, 90),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            y0 = 130
            for msg in self.feedback:
                cv2.putText(image, msg, (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                y0 += 30

        return image

    def get_stats(self):
        return {
            'total_reps': self.counter,
            'correct_reps': self.correct_reps,
            # 'wrong_reps': self.wrong_reps,
            'feedback': self.feedback[0] if self.feedback else "Ready for knee extension exercise"
        }

    def reset_counters(self):
        self.counter = 0
        self.correct_reps = 0
        self.wrong_reps = 0
        self.stage = 'down'
        self.feedback = []
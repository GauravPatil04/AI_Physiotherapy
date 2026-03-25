import cv2
import mediapipe as mp
from mediapipe import solutions as mp_solutions
import numpy as np

class SquatsTracker:
    def __init__(self):
        self.mp_pose = mp_solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # More flexible parameters
        self.SQUAT_DEPTH_MIN = 70    # More flexible knee angle range for squat
        self.SQUAT_DEPTH_MAX = 120
        self.STAND_THRESHOLD = 150   # More flexible stand position
        self.HIP_ANGLE_MIN = 60      # More flexible hip angle
        self.HIP_ANGLE_MAX = 130
        self.ANKLE_ANGLE_MIN = 60    # More flexible ankle angle
        self.ANKLE_ANGLE_MAX = 120
        self.STANCE_RATIO_MIN = 0.7  # More flexible stance
        self.STANCE_RATIO_MAX = 1.8
        
        # Tracking variables
        self.counter = 0
        self.stage = "up"
        self.correct_reps = 0
        self.wrong_reps = 0
        self.feedback = []
        self.is_correct_posture = False

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360-angle
        return angle

    def check_posture_correctness(self, knee_angle_l, knee_angle_r, hip_angle_l, hip_angle_r, 
                                 ankle_angle_l, ankle_angle_r, stance_ratio):
        """Check if current posture is correct"""
        conditions = [
            self.SQUAT_DEPTH_MIN <= knee_angle_l <= self.SQUAT_DEPTH_MAX,
            self.SQUAT_DEPTH_MIN <= knee_angle_r <= self.SQUAT_DEPTH_MAX,
            self.HIP_ANGLE_MIN <= hip_angle_l <= self.HIP_ANGLE_MAX,
            self.HIP_ANGLE_MIN <= hip_angle_r <= self.HIP_ANGLE_MAX,
            self.ANKLE_ANGLE_MIN <= ankle_angle_l <= self.ANKLE_ANGLE_MAX,
            self.ANKLE_ANGLE_MIN <= ankle_angle_r <= self.ANKLE_ANGLE_MAX,
            self.STANCE_RATIO_MIN <= stance_ratio <= self.STANCE_RATIO_MAX
        ]
        
        return all(conditions)

    def get_feedback_messages(self, knee_angle_l, knee_angle_r, hip_angle_l, hip_angle_r,
                             ankle_angle_l, ankle_angle_r, stance_ratio):
        """Generate specific feedback messages"""
        feedback = []
        
        if not (self.SQUAT_DEPTH_MIN <= knee_angle_l <= self.SQUAT_DEPTH_MAX):
            if knee_angle_l > self.SQUAT_DEPTH_MAX:
                feedback.append("Squat deeper")
            else:
                feedback.append("Don't squat too deep")
                
        if not (self.HIP_ANGLE_MIN <= hip_angle_l <= self.HIP_ANGLE_MAX):
            if hip_angle_l > self.HIP_ANGLE_MAX:
                feedback.append("Keep back straighter")
            else:
                feedback.append("Lean forward slightly")
                
        if not (self.ANKLE_ANGLE_MIN <= ankle_angle_l <= self.ANKLE_ANGLE_MAX):
            feedback.append("Keep heels on ground")
            
        if not (self.STANCE_RATIO_MIN <= stance_ratio <= self.STANCE_RATIO_MAX):
            if stance_ratio < self.STANCE_RATIO_MIN:
                feedback.append("Widen your stance")
            else:
                feedback.append("Bring feet closer")
                
        return feedback

    def process_frame(self, frame):
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self.feedback = []
        self.is_correct_posture = False

        try:
            landmarks = results.pose_landmarks.landmark

            # Key points
            hip_l = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_l = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            hip_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            shoulder_l = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Calculate angles
            knee_angle_l = self.calculate_angle(hip_l, knee_l, ankle_l)
            knee_angle_r = self.calculate_angle(hip_r, knee_r, ankle_r)
            hip_angle_l = self.calculate_angle(shoulder_l, hip_l, knee_l)
            hip_angle_r = self.calculate_angle(shoulder_r, hip_r, knee_r)
            ankle_angle_l = self.calculate_angle(knee_l, ankle_l, [ankle_l[0]+0.01, ankle_l[1]])  
            ankle_angle_r = self.calculate_angle(knee_r, ankle_r, [ankle_r[0]+0.01, ankle_r[1]])

            # Leg stance
            leg_distance = abs(ankle_r[0] - ankle_l[0])
            shoulder_distance = abs(shoulder_r[0] - shoulder_l[0])
            stance_ratio = leg_distance / shoulder_distance if shoulder_distance > 0 else 1

            # Check posture correctness
            self.is_correct_posture = self.check_posture_correctness(
                knee_angle_l, knee_angle_r, hip_angle_l, hip_angle_r,
                ankle_angle_l, ankle_angle_r, stance_ratio
            )

            # Detect squat stages with hysteresis
            avg_knee_angle = (knee_angle_l + knee_angle_r) / 2
            
            if self.stage == "up" and avg_knee_angle < self.SQUAT_DEPTH_MAX:
                self.stage = "down"
            elif self.stage == "down" and avg_knee_angle > self.STAND_THRESHOLD:
                self.stage = "up"
                self.counter += 1
                
                # Count correct/wrong reps
                if self.is_correct_posture:
                    self.correct_reps += 1
                    self.feedback = ["✅ Perfect squat!"]
                else:
                    self.wrong_reps += 1
                    self.feedback = self.get_feedback_messages(
                        knee_angle_l, knee_angle_r, hip_angle_l, hip_angle_r,
                        ankle_angle_l, ankle_angle_r, stance_ratio
                    )

            # Define colors based on posture
            landmark_color = (0, 0, 255)  # Red points
            connection_color = (0, 255, 0) if self.is_correct_posture else (255, 255, 255)  # Green/White lines

            # Custom drawing specifications
            landmark_drawing_spec = self.mp_drawing.DrawingSpec(
                color=landmark_color, thickness=2, circle_radius=3
            )
            connection_drawing_spec = self.mp_drawing.DrawingSpec(
                color=connection_color, thickness=3
            )

            # Draw landmarks with custom colors
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

            # Display angles and information
            cv2.putText(image, f"Knee L: {int(knee_angle_l)}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Knee R: {int(knee_angle_r)}", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Hip L: {int(hip_angle_l)}", (50, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"Stage: {self.stage.upper()}", (50, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Rep counts with better styling
            cv2.rectangle(image, (5, 5), (300, 80), (0, 0, 0), -1)
            cv2.putText(image, f"Reps: {self.counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Correct: {self.correct_reps}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f"Wrong: {self.wrong_reps}", (150, 55),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Posture indicator
            posture_status = "GOOD POSTURE" if self.is_correct_posture else "ADJUST POSTURE"
            status_color = (0, 255, 0) if self.is_correct_posture else (0, 0, 255)
            cv2.putText(image, posture_status, (image.shape[1] - 250, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Show feedback messages
            y0 = 220
            for i, msg in enumerate(self.feedback):
                color = (0, 255, 0) if "Perfect" in msg else (0, 0, 255)
                cv2.putText(image, msg, (50, y0 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        except Exception as e:
            print(f"Error in squats tracking: {e}")

        return image

    def get_stats(self):
        return {
            'total_reps': self.counter,
            'correct_reps': self.correct_reps,
            'wrong_reps': self.wrong_reps,
            'stage': self.stage,
            'feedback': self.feedback[0] if self.feedback else "Ready for squats exercise",
            'is_correct_posture': self.is_correct_posture
        }

    def reset_counters(self):
        self.counter = 0
        self.correct_reps = 0
        self.wrong_reps = 0
        self.stage = "up"
        self.feedback = []
        self.is_correct_posture = False

    def update_parameters(self, **kwargs):
        """Allow dynamic parameter updates"""
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
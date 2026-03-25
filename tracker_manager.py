from trackers.squats import SquatsTracker
from trackers.Knee_extension import KneeExtensionTracker
from trackers.hip_flexion import HipFlexionTracker
from trackers.hip_abduction_sleeping import HipAbductionSleepingTracker
from trackers.hip_abduction_standing_side import HipAbductionStandingTracker

trackers = {
    "squats": SquatsTracker(),
    "knee_extension": KneeExtensionTracker(),
    "hip_flexion": HipFlexionTracker(),
    "hip_abduction_sleeping": HipAbductionSleepingTracker(),
    "hip_abduction_standing": HipAbductionStandingTracker()
}

current_exercise = "squats"


def get_tracker():
    return trackers[current_exercise]


def get_current_exercise_name() -> str:
    return current_exercise


def set_exercise(name: str):
    global current_exercise

    if name not in trackers:
        return False

    current_exercise = name
    trackers[name].reset_counters()
    return True
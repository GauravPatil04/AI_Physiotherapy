from pydantic import BaseModel

class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class SessionCreate(BaseModel):
    exercise_name: str
    total_reps: int
    correct_reps: int
    wrong_reps: int
    feedback: str


class FeedbackCreate(BaseModel):
    rating: int
    comments: str
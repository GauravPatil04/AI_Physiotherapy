from datetime import date

from sqlalchemy import Column, Integer, String, ForeignKey, Date, UniqueConstraint
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    google_id = Column(String)
    role = Column(String)


class ExerciseSession(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("user_id", "exercise_name", "session_date", name="uq_sessions_user_exercise_date"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise_name = Column(String)
    session_date = Column(Date, default=date.today, index=True)
    total_reps = Column(Integer)
    correct_reps = Column(Integer)
    wrong_reps = Column(Integer)
    feedback = Column(String)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    rating = Column(Integer)
    comments = Column(String)
    question_1 = Column(String)
    response_1 = Column(String)
    question_2 = Column(String)
    response_2 = Column(String)
    question_3 = Column(String)
    response_3 = Column(String)
    question_4 = Column(String)
    response_4 = Column(String)
    question_5 = Column(String)
    response_5 = Column(String)
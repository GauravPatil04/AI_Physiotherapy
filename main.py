from fastapi import FastAPI, Depends, Request,Form
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import text, func
from sqlalchemy.orm import Session
from datetime import date
import cv2
import os
import base64
import numpy as np
from pydantic import BaseModel

from database import SessionLocal, engine
import models
from tracker_manager import get_tracker, set_exercise, get_current_exercise_name
from auth import router as auth_router
from config import get_required_env

models.Base.metadata.create_all(bind=engine)


def ensure_sessions_schema() -> None:
    # Keep existing deployments compatible by adding missing date/index objects.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_date DATE DEFAULT CURRENT_DATE"))
        conn.execute(text("UPDATE sessions SET session_date = CURRENT_DATE WHERE session_date IS NULL"))
        conn.execute(
            text(
                """
                WITH dedup AS (
                    SELECT
                        MIN(id) AS keep_id,
                        user_id,
                        exercise_name,
                        session_date,
                        SUM(COALESCE(total_reps, 0)) AS total_reps_sum,
                        SUM(COALESCE(correct_reps, 0)) AS correct_reps_sum,
                        SUM(COALESCE(wrong_reps, 0)) AS wrong_reps_sum,
                        (ARRAY_AGG(feedback ORDER BY id DESC))[1] AS latest_feedback
                    FROM sessions
                    GROUP BY user_id, exercise_name, session_date
                    HAVING COUNT(*) > 1
                )
                UPDATE sessions s
                SET
                    total_reps = d.total_reps_sum,
                    correct_reps = d.correct_reps_sum,
                    wrong_reps = d.wrong_reps_sum,
                    feedback = d.latest_feedback
                FROM dedup d
                WHERE s.id = d.keep_id
                """
            )
        )
        conn.execute(
            text(
                """
                WITH dedup AS (
                    SELECT
                        MIN(id) AS keep_id,
                        user_id,
                        exercise_name,
                        session_date
                    FROM sessions
                    GROUP BY user_id, exercise_name, session_date
                    HAVING COUNT(*) > 1
                )
                DELETE FROM sessions s
                USING dedup d
                WHERE s.user_id = d.user_id
                  AND s.exercise_name = d.exercise_name
                  AND s.session_date = d.session_date
                  AND s.id <> d.keep_id
                """
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_sessions_user_exercise_date_idx "
                "ON sessions (user_id, exercise_name, session_date)"
            )
        )


def ensure_feedback_schema() -> None:
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS question_1 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS response_1 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS question_2 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS response_2 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS question_3 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS response_3 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS question_4 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS response_4 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS question_5 TEXT"))
        conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS response_5 TEXT"))


ensure_sessions_schema()
ensure_feedback_schema()

app = FastAPI(title="AI Physiotherapy System")

app.add_middleware(
    SessionMiddleware,
    secret_key=get_required_env("SESSION_SECRET_KEY")
)

app.include_router(auth_router)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

camera = None
is_running = False
live_sync_state: dict[tuple[int, str], dict[str, int]] = {}


class FramePayload(BaseModel):
    image: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_or_open_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def get_current_user(request: Request, db: Session) -> models.User | None:
    user_id_cookie = request.cookies.get("user_id")
    if not user_id_cookie:
        return None

    try:
        user_id = int(user_id_cookie)
    except ValueError:
        return None

    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_name_or_default(request: Request, db: Session) -> str:
    user = get_current_user(request, db)
    if not user or not getattr(user, "name", None):
        return "User"
    return str(user.name)


@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/login_page", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")


@app.get("/signup_page", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse(request=request, name="signup.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"user_name": get_user_name_or_default(request, db)},
    )


@app.get("/exercise_page", response_class=HTMLResponse)
def exercise_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        request=request,
        name="exercise.html",
        context={"user_name": get_user_name_or_default(request, db)},
    )


@app.get("/start")
def start():
    global is_running
    is_running = True
    return {"status": "started"}


@app.get("/stop")
def stop():
    global is_running, camera
    is_running = False
    if camera is not None and camera.isOpened():
        camera.release()
    return {"status": "stopped"}


@app.get("/exercise/{name}")
def change_exercise(name: str):
    if not set_exercise(name):
        return {"message": "Exercise not found"}
    return {"message": "Exercise changed"}


def _get_or_create_today_session(db: Session, user_id: int, exercise_name: str) -> models.ExerciseSession:
    today = date.today()
    session_row = (
        db.query(models.ExerciseSession)
        .filter(
            models.ExerciseSession.user_id == user_id,
            models.ExerciseSession.exercise_name == exercise_name,
            models.ExerciseSession.session_date == today,
        )
        .first()
    )

    if session_row:
        return session_row

    session_row = models.ExerciseSession(
        user_id=user_id,
        exercise_name=exercise_name,
        session_date=today,
        total_reps=0,
        correct_reps=0,
        wrong_reps=0,
        feedback="",
    )
    db.add(session_row)
    db.flush()
    return session_row


def _sync_live_stats_for_user(db: Session, user_id: int, exercise_name: str, live_stats: dict) -> tuple[int, int, int]:
    state_key = (user_id, exercise_name)
    session_row = _get_or_create_today_session(db, user_id, exercise_name)

    current_total = int(getattr(session_row, "total_reps", 0) or 0)
    current_correct = int(getattr(session_row, "correct_reps", 0) or 0)
    current_wrong = int(getattr(session_row, "wrong_reps", 0) or 0)
    current_feedback = str(getattr(session_row, "feedback", "") or "")

    if state_key not in live_sync_state:
        live_sync_state[state_key] = {
            "base_total_reps": current_total,
            "base_correct_reps": current_correct,
            "base_wrong_reps": current_wrong,
        }

    state = live_sync_state[state_key]
    latest_total = state["base_total_reps"] + int(live_stats.get("total_reps", 0) or 0)
    latest_correct = state["base_correct_reps"] + int(live_stats.get("correct_reps", 0) or 0)
    latest_wrong = state["base_wrong_reps"] + int(live_stats.get("wrong_reps", 0) or 0)
    latest_feedback = str(live_stats.get("feedback", "") or "")

    if (
        current_total != latest_total
        or current_correct != latest_correct
        or current_wrong != latest_wrong
        or current_feedback != latest_feedback
    ):
        setattr(session_row, "total_reps", latest_total)
        setattr(session_row, "correct_reps", latest_correct)
        setattr(session_row, "wrong_reps", latest_wrong)
        setattr(session_row, "feedback", latest_feedback)
        db.commit()

    totals = (
        db.query(
            func.coalesce(func.sum(models.ExerciseSession.total_reps), 0),
            func.coalesce(func.sum(models.ExerciseSession.correct_reps), 0),
            func.coalesce(func.sum(models.ExerciseSession.wrong_reps), 0),
        )
        .filter(
            models.ExerciseSession.user_id == user_id,
            models.ExerciseSession.exercise_name == exercise_name,
        )
        .one()
    )
    return int(totals[0]), int(totals[1]), int(totals[2])


@app.get("/stats")
def get_stats():
    tracker = get_tracker()
    stats = tracker.get_stats()
    stats["exercise_name"] = get_current_exercise_name()
    return stats


@app.post("/process_frame")
def process_frame(payload: FramePayload):
    tracker = get_tracker()

    if not is_running:
        return {"status": "stopped", "stats": tracker.get_stats()}

    image_data = payload.image
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        frame_bytes = base64.b64decode(image_data)
    except Exception:
        return {"status": "error", "message": "Invalid image data"}

    np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "error", "message": "Could not decode image"}

    processed_frame = tracker.process_frame(frame)
    if processed_frame is None:
        processed_frame = frame

    ok, buffer = cv2.imencode('.jpg', processed_frame)
    if not ok:
        return {"status": "error", "message": "Could not encode processed frame", "stats": tracker.get_stats()}

    image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return {
        "status": "ok",
        "stats": tracker.get_stats(),
        "image": image_b64,
    }


@app.post("/save/{user_id}")
def save_session(user_id: int, db: Session = Depends(get_db), session_date: date | None = None):
    tracker = get_tracker()
    stats = tracker.get_stats()
    exercise_name = get_current_exercise_name()
    total_reps, correct_reps, wrong_reps = _sync_live_stats_for_user(
        db=db,
        user_id=user_id,
        exercise_name=exercise_name,
        live_stats=stats,
    )

    return {
        "message": "Saved",
        "exercise_name": exercise_name,
        "session_date": str(session_date or date.today()),
        "total_reps": total_reps,
        "correct_reps": correct_reps,
        "wrong_reps": wrong_reps,
    }


@app.post("/save")
def save_current_user_session(request: Request, db: Session = Depends(get_db)):
    user_id_cookie = request.cookies.get("user_id")
    if user_id_cookie is None:
        return {"message": "Not logged in"}

    try:
        user_id = int(user_id_cookie)
    except ValueError:
        return {"message": "Invalid user"}

    requested_date = request.query_params.get("date")
    if requested_date:
        try:
            target_date = date.fromisoformat(requested_date)
        except ValueError:
            return {"message": "Invalid date"}
    else:
        target_date = date.today()

    return save_session(user_id=user_id, db=db, session_date=target_date)


@app.get("/history", response_class=HTMLResponse)
def patient_history(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return templates.TemplateResponse(request=request, name="login.html")

    sessions = (
        db.query(models.ExerciseSession)
        .filter(
            models.ExerciseSession.user_id == user.id,
        )
        .order_by(models.ExerciseSession.session_date.desc(), models.ExerciseSession.exercise_name.asc())
        .all()
    )

    total_sessions = len(sessions)
    active_days = len({str(session.session_date) for session in sessions if session.session_date is not None})

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "sessions": sessions,
            "user_name": str(user.name or "User"),
            "total_sessions": total_sessions,
            "active_days": active_days,
        },
    )


@app.get("/feedback_page", response_class=HTMLResponse)
def feedback_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        request=request,
        name="feedback.html",
        context={"user_name": get_user_name_or_default(request, db)},
    )


@app.post("/submit_feedback")
def submit_feedback(
    request: Request,
    q1: str = Form(...),
    q2: str = Form(...),
    q3: str = Form(...),
    q4: str = Form(...),
    q5: str = Form(...),
    db: Session = Depends(get_db)
):
    user = get_current_user(request, db)
    user_id = user.id if user else None

    question_1 = "How would you rate your overall experience? (1-5)"
    question_2 = "Was the exercise tracking accurate?"
    question_3 = "Was the interface easy to use?"
    question_4 = "Did you face any issues?"
    question_5 = "Suggestions for improvement"

    feedback = models.Feedback(
        user_id=user_id,
        rating=int(q1),
        comments=f"{q2} | {q3} | {q4} | {q5}",
        question_1=question_1,
        response_1=q1,
        question_2=question_2,
        response_2=q2,
        question_3=question_3,
        response_3=q3,
        question_4=question_4,
        response_4=q4,
        question_5=question_5,
        response_5=q5,
    )

    db.add(feedback)
    db.commit()

    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/history_data")
def get_history_data(request: Request, db: Session = Depends(get_db), selected_date: date | None = None):
    user_id_cookie = request.cookies.get("user_id")
    if user_id_cookie is None:
        return {"message": "Not logged in", "history": []}

    try:
        user_id = int(user_id_cookie)
    except ValueError:
        return {"message": "Invalid user", "history": []}

    target_date = selected_date or date.today()
    rows = (
        db.query(models.ExerciseSession)
        .filter(
            models.ExerciseSession.user_id == user_id,
            models.ExerciseSession.session_date == target_date,
        )
        .order_by(models.ExerciseSession.exercise_name.asc())
        .all()
    )

    history = [
        {
            "exercise_name": row.exercise_name,
            "session_date": str(row.session_date),
            "total_reps": row.total_reps or 0,
            "correct_reps": row.correct_reps or 0,
            "wrong_reps": row.wrong_reps or 0,
            "feedback": row.feedback or "",
        }
        for row in rows
    ]

    return {"message": "ok", "history": history, "session_date": str(target_date)}


def generate_frames():
    global is_running
    cam = _get_or_open_camera()

    while True:
        success, frame = cam.read()
        if not success:
            break

        if is_running:
            tracker = get_tracker()
            frame = tracker.process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
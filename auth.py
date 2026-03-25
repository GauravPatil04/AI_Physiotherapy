from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from database import SessionLocal
import models
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from authlib.integrations.starlette_client import OAuth
from config import get_required_env

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -----------------------
# DATABASE SESSION
# -----------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------
# PASSWORD HASHING
# -----------------------
def hash_password(password: str) -> str:
    password_bytes = password.encode('utf-8')[:72]
    return pwd_context.hash(password_bytes)


def verify_password(plain: str, hashed: object) -> bool:
    if not hashed:
        return False

    plain_bytes = plain.encode('utf-8')[:72]
    hashed_value = hashed.decode('utf-8', errors='ignore') if isinstance(hashed, bytes) else str(hashed)
    try:
        return pwd_context.verify(plain_bytes, hashed_value)
    except (ValueError, TypeError, UnknownHashError):
        return False


# -----------------------
# GOOGLE OAUTH SETUP
# -----------------------
oauth = OAuth()

oauth.register(
    name='google',
    client_id=get_required_env("GOOGLE_CLIENT_ID"),
    client_secret=get_required_env("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)


# -----------------------
# SIGNUP
# -----------------------
@router.post("/signup")
def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    existing = db.query(models.User).filter(models.User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        name=name,
        email=email,
        password=hash_password(password),
        role="patient"
    )

    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login_page", status_code=303)


# -----------------------
# LOGIN
# -----------------------
@router.post("/login")
def login(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    db_user = db.query(models.User).filter(models.User.email == email).first()

    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid email")

    if not verify_password(password, db_user.password):
        raise HTTPException(status_code=400, detail="Invalid password")

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="user_id", value=str(db_user.id), samesite="lax")
    return response


# -----------------------
# GOOGLE LOGIN
# -----------------------
@router.get("/login/google")
async def login_google(request: Request):
    redirect_uri = request.url_for("google_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


# -----------------------
# GOOGLE CALLBACK
# -----------------------
@router.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    if not user_info:
        user_info = await oauth.google.parse_id_token(request, token)

    if not user_info:
        raise HTTPException(status_code=400, detail="Unable to fetch Google user info")

    email = user_info.get('email')
    name = user_info.get('name', 'Google User')
    google_id = user_info.get('sub')

    if not email or not google_id:
        raise HTTPException(status_code=400, detail="Incomplete Google account data")

    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        new_user = models.User(
            name=name,
            email=email,
            google_id=google_id,
            role="patient"
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        user = new_user

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="user_id", value=str(user.id), samesite="lax")
    return response


# -----------------------
# LOGOUT
# -----------------------
@router.get("/logout")
def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("user_id")
    return response
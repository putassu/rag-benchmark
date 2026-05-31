import os
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
import database as db
import config

# Настройки безопасности (в продакшене вынести в config)
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-for-rag-pilot")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # Неделя

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

app = FastAPI(title="RAG Annotation Tool")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(config.DOCS_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/docs", StaticFiles(directory=config.DOCS_DIR), name="docs")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_db():
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()

# --- Auth Utilities ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = session.query(db.User).filter(db.User.username == username).first()
    if user is None: raise credentials_exception
    return user

# --- Init Test Users ---
@app.on_event("startup")
def create_initial_users():
    session = db.SessionLocal()
    if not session.query(db.User).first():
        # Создаем пользователей при первом запуске
        users = [
            {"username": "sasha", "password": "123", "is_admin": True},
            {"username": "analyst2", "password": "123", "is_admin": False},
        ]
        for u in users:
            db_user = db.User(username=u["username"], hashed_password=get_password_hash(u["password"]), is_admin=u["is_admin"])
            session.add(db_user)
        session.commit()
    session.close()

# --- Endpoints ---
class QAPairCreate(BaseModel):
    document_id: int
    question: str
    key_phrase: str
    relevant_chunks: List[Dict[str, int]] # e.g. [{"chunk_id": 1, "relevance": 2}]

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/api/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_db)):
    user = session.query(db.User).filter(db.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Неверный логин или пароль")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id}

@app.get("/api/documents")
def get_documents(current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    docs = session.query(db.Document).filter(db.Document.assignee_id == current_user.id).all()
    res = []
    for d in docs:
        qa_count = session.query(db.QAPair).filter(db.QAPair.document_id == d.id).count()
        res.append({
            "id": d.id, "filename": d.filename, "qa_count": qa_count, 
            "status": "Готово" if d.is_completed else "В процессе"
        })
    return res

@app.post("/api/document/{doc_id}/complete")
def complete_document(doc_id: int, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    doc = session.query(db.Document).filter(db.Document.id == doc_id, db.Document.assignee_id == current_user.id).first()
    if not doc: raise HTTPException(status_code=404)
    doc.is_completed = not doc.is_completed # Toggle
    session.commit()
    return {"status": "success", "is_completed": doc.is_completed}

@app.get("/api/document/{doc_id}/chunks")
def get_chunks(doc_id: int, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    chunks = session.query(db.Chunk).filter(db.Chunk.document_id == doc_id).order_by(db.Chunk.start_index).all()
    return [{"id": c.id, "text": c.text, "start_index": c.start_index} for c in chunks]

@app.post("/api/qa")
def save_qa(qa: QAPairCreate, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    if not qa.relevant_chunks:
        raise HTTPException(status_code=400, detail="Необходимо выбрать хотя бы один чанк!")
    db_qa = db.QAPair(**qa.model_dump())
    session.add(db_qa)
    session.commit()
    return {"status": "success"}

@app.get("/api/document/{doc_id}/qa")
def get_doc_qa(doc_id: int, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    qas = session.query(db.QAPair).filter(db.QAPair.document_id == doc_id).all()
    return qas
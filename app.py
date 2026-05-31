import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import database as db
import config

app = FastAPI(title="RAG Annotation Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает доступ с любого хоста
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Убедимся, что директории существуют, чтобы FastAPI не упал при старте
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

class QAPairCreate(BaseModel):
    document_id: int
    question: str
    chunk_id: int
    key_phrase: str

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.get("/api/analysts")
def get_analysts():
    return config.ANALYSTS

@app.get("/api/documents/{analyst_id}")
def get_documents(analyst_id: int, session: Session = Depends(get_db)):
    docs = session.query(db.Document).filter(db.Document.assignee_id == analyst_id).all()
    res = []
    for d in docs:
        qa_count = session.query(db.QAPair).filter(db.QAPair.document_id == d.id).count()
        res.append({
            "id": d.id, "filename": d.filename, "qa_count": qa_count, 
            "status": "Готово" if qa_count >= 2 else "В процессе"
        })
    return res

@app.get("/api/document/{doc_id}/chunks")
def get_chunks(doc_id: int, session: Session = Depends(get_db)):
    chunks = session.query(db.Chunk).filter(db.Chunk.document_id == doc_id).order_by(db.Chunk.start_index).all()
    return [{"id": c.id, "text": c.text, "start_index": c.start_index} for c in chunks]

@app.post("/api/qa")
def save_qa(qa: QAPairCreate, session: Session = Depends(get_db)):
    count = session.query(db.QAPair).filter(db.QAPair.document_id == qa.document_id).count()
    if count >= 2:
        raise HTTPException(status_code=400, detail="Уже добавлено 2 вопроса к этому документу!")
        
    # Использование model_dump() вместо dict() для Pydantic v2
    db_qa = db.QAPair(**qa.model_dump())
    session.add(db_qa)
    session.commit()
    return {"status": "success"}
    
@app.get("/api/document/{doc_id}/qa")
def get_doc_qa(doc_id: int, session: Session = Depends(get_db)):
    qas = session.query(db.QAPair).filter(db.QAPair.document_id == doc_id).all()
    return qas
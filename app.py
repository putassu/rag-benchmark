import os
import math
import requests
import mimetypes
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any
import bcrypt
from jose import JWTError, jwt
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from fastapi import BackgroundTasks

import database as db
import config
from config import logger  # Импортируем логгер из твоего конфига

# ==========================================
# 1. НАСТРОЙКИ СИСТЕМЫ И БЕЗОПАСНОСТЬ
# ==========================================
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-for-rag-pilot")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # Неделя

qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
COLLECTION_NAME = "rag_benchmark"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

app = FastAPI(title="RAG Annotation & Benchmark Platform")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(config.DOCS_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)
# Папки для раздачи статики (UI и скачивание документов)
app.mount("/docs", StaticFiles(directory=config.DOCS_DIR), name="docs")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================
# 2. СХЕМЫ ДАННЫХ (PYDANTIC)
# ==========================================
class ChunkRelevance(BaseModel):
    chunk_id: str
    relevance: int
    key_phrase: str = "" 

class QAPairCreate(BaseModel):
    document_id: int
    question: str
    relevant_chunks: List[ChunkRelevance]

class QAPairUpdate(BaseModel):
    question: str
    relevant_chunks: List[ChunkRelevance]

# ==========================================
# 3. УТИЛИТЫ БД И АВТОРИЗАЦИИ
# ==========================================
def get_db():
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )

def get_password_hash(password: str) -> str:
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed_bytes.decode('utf-8')

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

@app.on_event("startup")
def create_initial_users():
    session = db.SessionLocal()
    if not session.query(db.User).first():
        users = [
            {"username": "sasha", "password": "123", "is_admin": True},
            {"username": "analyst2", "password": "123", "is_admin": False},
        ]
        for u in users:
            db_user = db.User(username=u["username"], hashed_password=get_password_hash(u["password"]), is_admin=u["is_admin"])
            session.add(db_user)
        session.commit()
    session.close()

# ==========================================
# 4. БАЗОВЫЕ ЭНДПОИНТЫ РАЗМЕТКИ (LEGACY + CRUD)
# ==========================================
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
    doc.is_completed = not doc.is_completed 
    session.commit()
    return {"status": "success", "is_completed": doc.is_completed}

@app.get("/api/document/{doc_id}/chunks")
def get_chunks(doc_id: int, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    doc = session.query(db.Document).filter(db.Document.id == doc_id, db.Document.assignee_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=403, detail="Доступ запрещен или документ не найден")
        
    try:
        records, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            ),
            limit=2000, 
            with_payload=True,
            with_vectors=False 
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обращения к Qdrant: {str(e)}")

    chunks = []
    for record in records:
        chunks.append({
            "id": record.id,
            "text": record.payload.get("text", ""),
            "start_index": record.payload.get("start_index", 0)
        })
        
    chunks.sort(key=lambda x: x["start_index"])
    return chunks

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

@app.put("/api/qa/{qa_id}")
def update_qa(qa_id: int, qa_update: QAPairUpdate, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    qa = session.query(db.QAPair).filter(db.QAPair.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A не найден")
        
    doc = session.query(db.Document).filter(db.Document.id == qa.document_id, db.Document.assignee_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=403, detail="Доступ запрещен")
        
    qa.question = qa_update.question
    qa.relevant_chunks = [c.model_dump() for c in qa_update.relevant_chunks]
    session.commit()
    return {"status": "success"}

@app.delete("/api/qa/{qa_id}")
def delete_qa(qa_id: int, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    qa = session.query(db.QAPair).filter(db.QAPair.id == qa_id).first()
    if not qa:
        raise HTTPException(status_code=404, detail="Q&A не найден")
    
    doc = session.query(db.Document).filter(db.Document.id == qa.document_id, db.Document.assignee_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=403, detail="Доступ запрещен")
        
    session.delete(qa)
    session.commit()
    return {"status": "success"}

@app.delete("/api/document/{doc_id}/reset")
def reset_document_qa(doc_id: int, current_user: db.User = Depends(get_current_user), session: Session = Depends(get_db)):
    doc = session.query(db.Document).filter(db.Document.id == doc_id, db.Document.assignee_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=403, detail="Доступ запрещен или документ не найден")
    
    deleted_count = session.query(db.QAPair).filter(db.QAPair.document_id == doc_id).delete()
    doc.is_completed = False
    
    session.commit()
    return {"status": "success", "deleted_pairs": deleted_count, "message": "Разметка документа полностью очищена"}

@app.get("/api/admin/export-db")
def export_database(current_user: db.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Доступ только для администраторов")
    
    db_path = "data/dataset.db" 
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Файл базы данных не найден")
        
    return FileResponse(
        path=db_path, 
        filename=f"rag_dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.db",
        media_type="application/octet-stream"
    )

# ==========================================
# 5. НОВЫЙ БЛОК: ЗАГРУЗКА ДОКУМЕНТОВ БАТЧЕМ
# ==========================================
@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: db.User = Depends(get_current_user),
    session: Session = Depends(get_db)
):
    results = []
    
    for file in files:
        try:
            # 1. Сохраняем файл на хост (в папку /app/docs, которая слинкована с сервером)
            file_path = os.path.join(config.DOCS_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # 2. Отправляем в Open WebUI
            mime_type, _ = mimetypes.guess_type(file.filename)
            mime_type = mime_type or "application/octet-stream"
            
            headers = {"Authorization": f"Bearer {config.OPENWEB_API_KEY}"}
            with open(file_path, "rb") as f:
                res = requests.post(
                    f"{config.OPENWEB_URL}/api/v1/files/?process=true&internal=false",
                    headers=headers,
                    files={"file": (file.filename, f, mime_type)},
                    verify=False
                )
            
            res.raise_for_status()
            ow_data = res.json()
            
            # 3. Сохраняем документ в нашу БД с назначением на текущего пользователя
            new_doc = db.Document(
                filename=file.filename,
                openweb_id=ow_data["id"],
                collection_name=ow_data["meta"]["collection_name"],
                status="В процессе",
                assignee_id=current_user.id
            )
            session.add(new_doc)
            session.commit()
            
            results.append({"filename": file.filename, "status": "success"})
            
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "error", "detail": str(e)})

    return {"results": results}

# ==========================================
# 6. АСИНХРОННЫЙ РАСЧЕТ И КЭШИРОВАНИЕ БЕНЧМАРКОВ
# ==========================================
def calculate_idcg(gt_scores: List[int], k: int) -> float:
    sorted_gt = sorted(gt_scores, reverse=True)[:k]
    return sum(score / math.log2(i + 2) for i, score in enumerate(sorted_gt))

def calculate_dcg(retrieved_scores: List[int], k: int) -> float:
    return sum(score / math.log2(i + 2) for i, score in enumerate(retrieved_scores[:k]))


def heavy_benchmark_task(result_id: int):
    """Тяжелая функция расчета, которая выполняется в фоновом потоке часами"""
    # В фоновых задачах нужно открывать собственную сессию БД!
    session = db.SessionLocal()
    try:
        logger.info(f"[Бенчмарк #{result_id}] Фоновый расчет запущен.")
        
        qa_pairs = session.query(db.QAPair).all()
        if not qa_pairs:
            raise Exception("База QA пар пуста.")

        headers = {
            "Authorization": f"Bearer {config.OPENWEB_API_KEY}",
            "Content-Type": "application/json"
        }

        metrics_summary = {
            "total_queries": 0, "skipped": 0,
            "recall_3_sum": 0.0, "recall_5_sum": 0.0, "ndcg_5_sum": 0.0
        }
        detailed_report = []

        # ------------------------------------------------------------
        # Основной цикл расчета (может идти долго)
        # ------------------------------------------------------------
        for qa in qa_pairs:
            if not qa.relevant_chunks:
                metrics_summary["skipped"] += 1
                continue

            doc = session.query(db.Document).filter_by(id=qa.document_id).first()
            if not doc or not doc.collection_name:
                metrics_summary["skipped"] += 1
                continue

            gt_map = {chunk["chunk_id"]: chunk["relevance"] for chunk in qa.relevant_chunks}
            gt_scores_only = list(gt_map.values())

            try:
                scroll_res, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc.id))]),
                    limit=10000, with_payload=True, with_vectors=False
                )
                start_index_to_uuid = { point.payload.get("start_index"): point.id for point in scroll_res }
            except Exception as e:
                logger.error(f"Ошибка Qdrant в фоне: {e}")
                metrics_summary["skipped"] += 1
                continue

            payload = {
                "collection_names": [doc.collection_name], "query": qa.question,
                "k": 5, "k_reranker": 0, "r": 0, "hybrid": True, "hybrid_bm25_weight": 0
            }

            try:
                res = requests.post(f"{config.OPENWEB_URL}/api/v1/retrieval/query/collection", headers=headers, json=payload, verify=False)
                res.raise_for_status()
                retrieval_data = res.json()
            except Exception as e:
                logger.error(f"Сбой OpenWebUI для QA {qa.id} в фоне: {e}")
                metrics_summary["skipped"] += 1
                continue

            retrieved_metadatas = retrieval_data.get("metadatas", [[]])[0]
            if not retrieved_metadatas:
                detailed_report.append({"qa_id": qa.id, "question": qa.question, "recall_3": 0.0, "recall_5": 0.0, "ndcg_5": 0.0})
                metrics_summary["total_queries"] += 1
                continue

            retrieved_scores_top5 = []
            for meta in retrieved_metadatas[:5]:
                start_idx = meta.get("start_index")
                mapped_uuid = start_index_to_uuid.get(start_idx)
                score = gt_map.get(mapped_uuid, 0) if mapped_uuid else 0
                retrieved_scores_top5.append(score)

            relevant_gt_count = len([s for s in gt_scores_only if s >= 1])
            if relevant_gt_count == 0:
                metrics_summary["skipped"] += 1
                continue
                
            retrieved_relevant_top3 = len([s for s in retrieved_scores_top5[:3] if s >= 1])
            retrieved_relevant_top5 = len([s for s in retrieved_scores_top5[:5] if s >= 1])

            recall_3 = min(1.0, retrieved_relevant_top3 / relevant_gt_count)
            recall_5 = min(1.0, retrieved_relevant_top5 / relevant_gt_count)

            idcg_5 = calculate_idcg(gt_scores_only, 5)
            ndcg_5 = 0.0 if idcg_5 == 0 else calculate_dcg(retrieved_scores_top5, 5) / idcg_5

            metrics_summary["total_queries"] += 1
            metrics_summary["recall_3_sum"] += recall_3
            metrics_summary["recall_5_sum"] += recall_5
            metrics_summary["ndcg_5_sum"] += ndcg_5
            
            detailed_report.append({
                "qa_id": qa.id, "question": qa.question,
                "recall_3": round(recall_3, 3), "recall_5": round(recall_5, 3), "ndcg_5": round(ndcg_5, 3)
            })

        # ------------------------------------------------------------
        # Сохранение результатов в базу по окончании
        # ------------------------------------------------------------
        db_result = session.query(db.BenchmarkResult).filter_by(id=result_id).first()
        if db_result:
            t = metrics_summary["total_queries"]
            if t > 0:
                db_result.total_queries = t
                db_result.skipped = metrics_summary["skipped"]
                db_result.avg_recall_3 = round(metrics_summary["recall_3_sum"] / t, 4)
                db_result.avg_recall_5 = round(metrics_summary["recall_5_sum"] / t, 4)
                db_result.avg_ndcg_5 = round(metrics_summary["ndcg_5_sum"] / t, 4)
                db_result.details = detailed_report
                db_result.status = "completed"
            else:
                db_result.status = "failed"
            session.commit()
            logger.info(f"[Бенчмарк #{result_id}] Расчет успешно завершен и сохранен в БД.")

    except Exception as main_ext:
        logger.error(f"[Бенчмарк #{result_id}] КРИТИЧЕСКАЯ ОШИБКА: {main_ext}")
        db_result = session.query(db.BenchmarkResult).filter_by(id=result_id).first()
        if db_result:
            db_result.status = "failed"
            session.commit()
    finally:
        session.close()


@app.post("/api/benchmark/run")
def trigger_benchmark(
    background_tasks: BackgroundTasks,
    current_user: db.User = Depends(get_current_user), 
    session: Session = Depends(get_db)
):
    # Проверяем, не запущен ли уже какой-то расчет прямо сейчас
    active_run = session.query(db.BenchmarkResult).filter_by(status="running").first()
    if active_run:
        return {"status": "running", "message": "Расчет уже выполняется в фоновом режиме", "run_id": active_run.id}

    # 1. Создаем пустую запись в БД со статусом "running"
    new_run = db.BenchmarkResult(status="running")
    session.add(new_run)
    session.commit()
    session.refresh(new_run)

    # 2. Отправляем тяжелую задачу в пул FastAPI BackgroundTasks
    background_tasks.add_task(heavy_benchmark_task, new_run.id)

    # 3. Мгновенно возвращаем ответ пользователю
    return {"status": "running", "message": "Расчет успешно запущен в фоновом режиме", "run_id": new_run.id}


@app.get("/api/benchmark/latest")
def get_latest_benchmark_results(
    current_user: db.User = Depends(get_current_user), 
    session: Session = Depends(get_db)
):
    """Эндпоинт, который вызывается при рендеринге страницы тестов. Ничего не считает, просто тянет данные из кэша БД"""
    latest_run = session.query(db.BenchmarkResult).order_by(db.BenchmarkResult.id.desc()).first()
    if not latest_run:
        return {"status": "empty", "message": "Бенчмарки еще ни разу не запускались"}
        
    return {
        "status": latest_run.status,
        "created_at": latest_run.created_at.strftime("%d.%m.%Y %H:%M:%S"),
        "metrics": {
            "total_queries": latest_run.total_queries,
            "skipped": latest_run.skipped,
            "avg_recall_3": latest_run.avg_recall_3 or 0.0,
            "avg_recall_5": latest_run.avg_recall_5 or 0.0,
            "avg_ndcg_5": latest_run.avg_ndcg_5 or 0.0
        } if latest_run.status == "completed" else None
    }

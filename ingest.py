import os
import itertools
import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from config import logger, OPENWEB_URL, OPENWEB_API_KEY, QDRANT_URL, QDRANT_API_KEY, DOCS_DIR
from database import SessionLocal, Document, User

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "rag_benchmark"

def init_qdrant():
    try:
        qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Коллекция {COLLECTION_NAME} уже существует.")
    except Exception:
        logger.info("Создаем коллекцию в Qdrant...")
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=1, distance=qmodels.Distance.COSINE)
        )

def upload_to_openweb(filepath):
    logger.debug(f"Загрузка {filepath} в Open WebUI")
    url = f"{OPENWEB_URL}/api/v1/files/?process=true&internal=false"
    headers = {"Authorization": f"Bearer {OPENWEB_API_KEY}"}
    
    with open(filepath, "rb") as f:
        response = requests.post(
            url, headers=headers,
            files={"file": (os.path.basename(filepath), f, "application/pdf")},
            verify=False
        )
    response.raise_for_status()
    data = response.json()
    return data["id"], data["meta"]["collection_name"]

def get_chunks_from_openweb(collection_name):
    logger.debug(f"Получение чанков для коллекции {collection_name}")
    url = f"{OPENWEB_URL}/api/v1/retrieval/query/collection"
    headers = {"Authorization": f"Bearer {OPENWEB_API_KEY}", "Content-Type": "application/json"}
    
    payload = {
        "collection_names": [collection_name],
        "query": "a", 
        "k": 2000,  # Увеличил на случай очень больших PDF
        "k_reranker": 0, "r": 0, "hybrid": True, "hybrid_bm25_weight": 0
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)
    response.raise_for_status()
    data = response.json()
    
    if not data.get("documents") or not data["documents"][0]:
        return []
        
    texts = data["documents"][0]
    metas = data["metadatas"][0]
    
    chunks = [{"text": t, "meta": m} for t, m in zip(texts, metas)]
    chunks.sort(key=lambda x: x["meta"].get("start_index", 0))
    return chunks

def test_chunks_fidelity(chunks, filename):
    """Отказоустойчивый тест корректности чанков"""
    if not chunks:
        logger.error(f"[{filename}] Ошибка: чанки не вернулись")
        return False
        
    valid = True
    for i, chunk in enumerate(chunks):
        if "start_index" not in chunk["meta"]:
            logger.error(f"[{filename}] В метадате чанка {i} нет start_index")
            valid = False
        if len(chunk["text"]) < 5:
            logger.warning(f"[{filename}] Чанк {i} подозрительно короткий (менее 5 символов)")

    start_indices = [c["meta"].get("start_index", 0) for c in chunks]
    if start_indices != sorted(start_indices):
        logger.error(f"[{filename}] Ошибка сортировки чанков")
        valid = False
        
    if valid:
        logger.debug(f"[{filename}] Тест пройден: {len(chunks)} корректных чанков.")
    return valid

def ingest_all():
    init_qdrant()
    db = SessionLocal()
    
    # 1. Забираем реальных пользователей из БД
    users = db.query(User).all()
    if not users:
        logger.error("В базе нет пользователей! Сначала запустите app.py (uvicorn), чтобы отработал lifespan.")
        db.close()
        return
        
    os.makedirs(DOCS_DIR, exist_ok=True)
    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    
    users_cycle = itertools.cycle(users)
    
    for filename in files:
        filepath = os.path.join(DOCS_DIR, filename)
        logger.info(f"Обработка файла: {filename}")
        
        try:
            openweb_id, collection_name = upload_to_openweb(filepath)
            chunks = get_chunks_from_openweb(collection_name)
            
            if not test_chunks_fidelity(chunks, filename):
                continue
                
            assignee = next(users_cycle)
            
            # 2. Запись документа в SQLite (создаем задачу на разметку)
            db_doc = Document(
                filename=filename,
                openweb_id=openweb_id,
                collection_name=collection_name,
                assignee_id=assignee.id
            )
            db.add(db_doc)
            db.commit()
            db.refresh(db_doc)
            
            # 3. Отправка чанков ТОЛЬКО в Qdrant
            points = []
            for chunk in chunks:
                chunk_id = uuid.uuid4().hex # Строковый ID для Qdrant
                points.append(qmodels.PointStruct(
                    id=chunk_id, 
                    vector=[0.0],
                    payload={
                        "doc_id": db_doc.id, # Привязка к документу в SQLite
                        "openweb_id": openweb_id, 
                        "text": chunk["text"], 
                        "start_index": chunk["meta"]["start_index"]
                    }
                ))
                
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Успешно! Документ '{filename}' назначен на юзера '{assignee.username}'. Чанков в Qdrant: {len(chunks)}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {filename}: {str(e)}")
            db.rollback()
            
    db.close()

if __name__ == "__main__":
    ingest_all()
import os
import itertools
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from config import logger, OPENWEB_URL, OPENWEB_API_KEY, QDRANT_URL, QDRANT_API_KEY, DOCS_DIR, ANALYSTS
from database import SessionLocal, Document, Chunk

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "rag_benchmark"

def init_qdrant():
    try:
        qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Коллекция {COLLECTION_NAME} уже существует.")
    except Exception:
        logger.info("Создаем коллекцию в Qdrant (fake vectors size=1)...")
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
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
    
    # Строго поровну распределяем файлы
    analysts_cycle = itertools.cycle(ANALYSTS)
    
    for filename in files:
        filepath = os.path.join(DOCS_DIR, filename)
        logger.info(f"Обработка файла: {filename}")
        
        try:
            openweb_id, collection_name = upload_to_openweb(filepath)
            chunks = get_chunks_from_openweb(collection_name)
            
            if not test_chunks_fidelity(chunks, filename):
                logger.warning(f"Файл {filename} пропущен из-за ошибок чанкирования.")
                continue
                
            assignee = next(analysts_cycle)
            
            db_doc = Document(
                filename=filename,
                openweb_id=openweb_id,
                collection_name=collection_name,
                assignee_id=assignee["id"]
            )
            db.add(db_doc)
            db.commit()
            db.refresh(db_doc)
            
            points = []
            for idx, chunk in enumerate(chunks):
                db_chunk = Chunk(
                    document_id=db_doc.id,
                    text=chunk["text"],
                    start_index=chunk["meta"]["start_index"]
                )
                db.add(db_chunk)
                
                points.append(qmodels.PointStruct(
                    id=db_doc.id * 10000 + idx, 
                    vector=[0.0],
                    payload={"doc_id": openweb_id, "text": chunk["text"], "start_index": chunk["meta"]["start_index"]}
                ))
                
            db.commit()
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Успешно! Документ {filename} назначен на: {assignee['name']}. Чанков: {len(chunks)}")
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке {filename}: {str(e)}")
            db.rollback()
            
    db.close()

if __name__ == "__main__":
    ingest_all()
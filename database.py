from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

engine = create_engine("sqlite:////app/server_data/dataset.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_admin = Column(Boolean, default=False)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    openweb_id = Column(String, unique=True, index=True)
    collection_name = Column(String)
    assignee_id = Column(Integer, ForeignKey("users.id"))
    is_completed = Column(Boolean, default=False)
    
    # Связь с чанками удалена! Чанки теперь только в Qdrant.
    qa_pairs = relationship("QAPair", back_populates="document", cascade="all, delete-orphan")

class QAPair(Base):
    __tablename__ = "qa_pairs"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    question = Column(Text)
    
    # Оставляем старую колонку для обратной совместимости со старой разметкой
    key_phrase = Column(Text, nullable=True) 
    
    # А новые ключевые фразы теперь будут лежать внутри JSON-массива чанков:
    # [{"chunk_id": "...", "relevance": 2, "key_phrases": ["фраза 1", "фраза 2"]}]
    relevant_chunks = Column(JSON) 
    
    document = relationship("Document", back_populates="qa_pairs")

Base.metadata.create_all(bind=engine)

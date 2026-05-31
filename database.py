from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import config

engine = create_engine("sqlite:///dataset.db", connect_args={"check_same_thread": False})
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
    is_completed = Column(Boolean, default=False) # Статус готовности документа
    
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    qa_pairs = relationship("QAPair", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(Text)
    start_index = Column(Integer)
    
    document = relationship("Document", back_populates="chunks")

class QAPair(Base):
    __tablename__ = "qa_pairs"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    question = Column(Text)
    key_phrase = Column(Text) 
    
    # Храним список [{"chunk_id": 1, "relevance": 2}, {"chunk_id": 5, "relevance": 1}]
    relevant_chunks = Column(JSON) 
    
    document = relationship("Document", back_populates="qa_pairs")

Base.metadata.create_all(bind=engine)
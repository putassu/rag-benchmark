from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import config

engine = create_engine("sqlite:///dataset.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    openweb_id = Column(String, unique=True, index=True)
    collection_name = Column(String)
    assignee_id = Column(Integer, index=True) # Привязка к аналитику
    
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
    chunk_id = Column(Integer, ForeignKey("chunks.id"))
    key_phrase = Column(Text) 
    
    document = relationship("Document", back_populates="qa_pairs")

Base.metadata.create_all(bind=engine)
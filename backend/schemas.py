"""
Pydantic models for API requests, responses, and database schema.
"""
from typing import List
from pydantic import BaseModel, Field
from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

# SQLAlchemy base for database models
Base = declarative_base()


class APIRequest(BaseModel):
    """
    Pydantic model for incoming API requests.
    
    Attributes:
        documents: Raw document content as a string
        questions: List of questions to be answered based on the documents
    """
    documents: str = Field(..., description="Document content to process")
    questions: List[str] = Field(..., min_items=1, description="List of questions to answer")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": "This is a sample document content...",
                "questions": [
                    "What is the main topic of this document?",
                    "What are the key findings?"
                ]
            }
        }


class APIResponse(BaseModel):
    """
    Pydantic model for API responses.
    
    Attributes:
        answers: List of answers corresponding to the input questions
    """
    answers: List[str] = Field(..., description="Answers to the submitted questions")

    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "The main topic is artificial intelligence.",
                    "Key findings include improved accuracy and reduced processing time."
                ]
            }
        }


class DocumentCache(Base):
    """
    SQLAlchemy model for PostgreSQL document cache table.
    
    This model stores document embeddings in PostgreSQL using JSONB for efficient
    storage and querying of vector data.
    
    Attributes:
        doc_url_hash: Primary key - hash of the document URL for unique identification
        embeddings: JSONB field containing document embeddings and metadata
    """
    __tablename__ = "document_cache"
    
    doc_url_hash: str = Column(
        String, 
        primary_key=True, 
        nullable=False,
        comment="SHA-256 hash of document URL"
    )
    embeddings = Column(
        JSONB,
        nullable=False,
        comment="Document embeddings and associated metadata stored as JSONB"
    )

    def __repr__(self) -> str:
        return f"<DocumentCache(doc_url_hash='{self.doc_url_hash}')>"
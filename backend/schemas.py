"""
Simplified Pydantic models for API requests and responses.
REMOVED: SQLAlchemy database models (no PostgreSQL dependency)
KEPT: APIRequest and APIResponse models
"""

from typing import List
from pydantic import BaseModel, Field


class APIRequest(BaseModel):
    """
    Pydantic model for incoming API requests.
    
    Attributes:
        documents: Document URL to process
        questions: List of questions to be answered based on the documents
    """
    documents: str = Field(..., description="Document URL to process")
    questions: List[str] = Field(..., min_items=1, description="List of questions to answer")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
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
                    "The main topic is artificial intelligence in healthcare.",
                    "Key findings include improved diagnostic accuracy and reduced processing time."
                ]
            }
        }
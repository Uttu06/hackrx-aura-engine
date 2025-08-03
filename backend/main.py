"""
Simplified FastAPI application for RAG pipeline
REMOVED: PostgreSQL caching, parallel processing, visual triage, database complexity
KEPT: Bearer token authentication, Basic FastAPI structure, Simple request/response
"""

import asyncio
import os
import traceback
from typing import List

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from .core_logic import process_document, get_answer, get_system_info
from .schemas import APIRequest, APIResponse
from .model_manager import model_manager

# Load environment variables
load_dotenv()

# Environment configuration
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")
if not HACKRX_BEARER_TOKEN:
    raise ValueError("HACKRX_BEARER_TOKEN environment variable is required")

# Security setup
security = HTTPBearer()

# FastAPI app initialization
app = FastAPI(
    title="Simplified RAG Pipeline API",
    description="Simple, reliable document processing and Q&A API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify Bearer token authentication."""
    if credentials.credentials != HACKRX_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        print("✅ Simplified RAG API started successfully")
        print(f"🔐 Bearer token authentication enabled")
        print(f"🧠 System info: {get_system_info()}")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    model_manager.cleanup()
    print("👋 FastAPI application shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Simplified RAG Pipeline API is running",
        "status": "healthy",
        "version": "1.0.0",
        "philosophy": "Simple systems that work reliably"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        memory_info = model_manager.get_memory_usage()
        
        return {
            "status": "healthy",
            "models": "ready",
            "authentication": "enabled",
            "memory": memory_info,
            "system_info": get_system_info()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/hackrx/run", response_model=APIResponse)
async def run_rag_pipeline(
    request: APIRequest,
    authenticated: bool = Depends(verify_token)
):
    """
    Simplified RAG pipeline endpoint.
    
    REMOVED: PostgreSQL caching, parallel processing, visual triage
    ADDED: Sequential processing, direct document processing, simple error handling
    
    Args:
        request: APIRequest containing documents and questions
        authenticated: Authentication status (dependency)
        
    Returns:
        APIResponse with answers to all questions
    """
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        print("🚀 Starting simplified RAG pipeline")
        print(f"📄 Document: {request.documents}")
        print(f"❓ Questions: {len(request.questions)}")
        
        # Step 1: Process document (no caching - fresh processing each time)
        print("📊 Processing document (no cache - fresh processing)")
        vector_store = await process_document(request.documents)
        
        if not vector_store:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        # Step 2: Process questions sequentially (no parallel processing)
        print(f"🔄 Processing {len(request.questions)} questions sequentially")
        answers = []
        
        for i, question in enumerate(request.questions, 1):
            try:
                print(f"📝 Processing question {i}/{len(request.questions)}: {question[:50]}...")
                answer = await get_answer(question, vector_store)
                answers.append(answer)
                print(f"✅ Question {i} completed")
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        # Step 3: Return response
        print("🏆 Pipeline completed successfully")
        return APIResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in RAG pipeline: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/memory")
async def get_memory_info(authenticated: bool = Depends(verify_token)):
    """Get current memory usage information."""
    try:
        memory_info = model_manager.get_memory_usage()
        return {
            "memory_usage": memory_info,
            "system_info": get_system_info()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting memory info: {str(e)}")


@app.post("/cleanup")
async def cleanup_models(authenticated: bool = Depends(verify_token)):
    """Manually cleanup models to free GPU memory."""
    try:
        model_manager.cleanup()
        return {
            "message": "Models cleaned up successfully",
            "memory_usage": model_manager.get_memory_usage()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
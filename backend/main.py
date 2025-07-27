"""
FastAPI application for RAG pipeline with PostgreSQL caching and parallel Q&A processing.
"""
import traceback
from dotenv import load_dotenv
load_dotenv()
import asyncio
import hashlib
import json
import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
import uvicorn

from .core_logic import process_document, get_answer, create_faiss_retriever, initialize_models
from .schemas import APIRequest, APIResponse, DocumentCache, Base

# Environment configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Convert sync DATABASE_URL to async if needed
if DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg2://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
else:
    ASYNC_DATABASE_URL = DATABASE_URL

# Database setup
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# FastAPI app initialization
app = FastAPI(
    title="RAG Pipeline API",
    description="Document processing and Q&A API with PostgreSQL caching",
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


async def get_db_session() -> AsyncSession:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def generate_doc_hash(doc_url: str) -> str:
    """Generate SHA-256 hash for document URL."""
    return hashlib.sha256(doc_url.encode('utf-8')).hexdigest()


async def get_cached_embeddings(
    doc_hash: str, 
    db: AsyncSession
) -> Optional[Tuple[List[str], List[List[float]]]]:
    """
    Retrieve cached embeddings from PostgreSQL.
    
    Args:
        doc_hash: SHA-256 hash of document URL
        db: Database session
        
    Returns:
        Tuple of (chunk_texts, embeddings) if found, None otherwise
    """
    try:
        stmt = select(DocumentCache).where(DocumentCache.doc_url_hash == doc_hash)
        result = await db.execute(stmt)
        cache_entry = result.scalar_one_or_none()
        
        if cache_entry:
            cached_data = cache_entry.embeddings
            chunk_texts = cached_data.get("chunk_texts", [])
            embeddings = cached_data.get("embeddings", [])
            
            if chunk_texts and embeddings:
                return chunk_texts, embeddings
                
        return None
        
    except Exception as e:
        print(f"Error retrieving cached embeddings: {e}")
        return None


async def save_embeddings_to_cache(
    doc_hash: str,
    chunk_texts: List[str],
    embeddings: List[List[float]],
    db: AsyncSession
) -> None:
    """
    Save embeddings to PostgreSQL cache.
    
    Args:
        doc_hash: SHA-256 hash of document URL
        chunk_texts: List of text chunks
        embeddings: List of embedding vectors
        db: Database session
    """
    try:
        # Prepare data for JSONB storage
        cache_data = {
            "chunk_texts": chunk_texts,
            "embeddings": embeddings,
            "chunk_count": len(chunk_texts),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0
        }
        
        # Create new cache entry
        cache_entry = DocumentCache(
            doc_url_hash=doc_hash,
            embeddings=cache_data
        )
        
        db.add(cache_entry)
        await db.commit()
        
        print(f"Cached embeddings for document hash: {doc_hash}")
        
    except Exception as e:
        await db.rollback()
        print(f"Error saving embeddings to cache: {e}")
        raise


async def get_or_create_embeddings(
    doc_url: str,
    db: AsyncSession
) -> Tuple[List[str], List[List[float]]]:
    """
    Get embeddings from cache or create new ones.
    
    Args:
        doc_url: Document URL to process
        db: Database session
        
    Returns:
        Tuple of (chunk_texts, embeddings)
    """
    doc_hash = generate_doc_hash(doc_url)
    
    # Try to get from cache first
    cached_result = await get_cached_embeddings(doc_hash, db)
    if cached_result:
        print(f"Using cached embeddings for document: {doc_url}")
        return cached_result
    
    # Not in cache, process document
    print(f"Processing new document: {doc_url}")
    chunk_texts, embeddings = await process_document(doc_url)
    
    # Save to cache for future use
    await save_embeddings_to_cache(doc_hash, chunk_texts, embeddings, db)
    
    return chunk_texts, embeddings


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        # Create database tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize AI models
        initialize_models()
        
        print("FastAPI application started successfully")
        print(f"Database connected: {ASYNC_DATABASE_URL}")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    await async_engine.dispose()
    print("FastAPI application shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "RAG Pipeline API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        # Test database connection
        async with AsyncSessionLocal() as session:
            await session.execute(select(1))
        
        return {
            "status": "healthy",
            "database": "connected",
            "models": "initialized"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/hackrx/run", response_model=APIResponse)
async def run_rag_pipeline(
    request: APIRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Main RAG pipeline endpoint with caching and parallel Q&A processing.
    
    Args:
        request: APIRequest containing documents and questions
        db: Database session
        
    Returns:
        APIResponse with answers to all questions
    """
    try:
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Step 1: Get or create embeddings (with caching)
        chunk_texts, embeddings = await get_or_create_embeddings(request.documents, db)
        
        if not chunk_texts or not embeddings:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        # Step 2: Create in-memory FAISS index
        retriever = create_faiss_retriever(chunk_texts, embeddings)
        
        # Step 3: Process all questions in parallel using asyncio.gather
        print(f"Processing {len(request.questions)} questions in parallel")
        
        import asyncio # Make sure asyncio is imported

        answer_tasks = [
            asyncio.to_thread(get_answer, question, retriever) 
            for question in request.questions
        ]
        answers = await asyncio.gather(*answer_tasks, return_exceptions=True)
        
        # Step 4: Handle any exceptions in parallel processing
        processed_answers = []
       
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                print(f"Error processing question {i}: {answer}")
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        # Step 5: Return response
        return APIResponse(answers=processed_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/cache/stats")
async def get_cache_stats(db: AsyncSession = Depends(get_db_session)):
    """Get cache statistics."""
    try:
        from sqlalchemy import func
        
        # Get total cached documents
        stmt = select(func.count(DocumentCache.doc_url_hash))
        result = await db.execute(stmt)
        total_cached = result.scalar()
        
        return {
            "total_cached_documents": total_cached,
            "cache_table": "document_cache"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")


@app.delete("/cache/clear")
async def clear_document_cache(db: AsyncSession = Depends(get_db_session)):
    """
    DANGER: Deletes all entries from the document_cache table.
    Used for forcing re-processing of documents after a model change.
    """
    try:
        from sqlalchemy import text
        
        stmt = text(f"DELETE FROM {DocumentCache.__tablename__}")
        await db.execute(stmt)
        await db.commit()
        
        message = f"Successfully cleared all entries from the '{DocumentCache.__tablename__}' table."
        print(f"INFO: {message}")
        return {"message": message}
        
    except Exception as e:
        await db.rollback()
        print(f"ERROR: Could not clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
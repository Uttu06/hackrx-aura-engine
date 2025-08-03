"""
Simplified Production-Ready RAG Architecture
Philosophy: "Boring Technology That Works"

REMOVED:
- Parent-child chunking complexity
- BM25 indexing
- CrossEncoder re-ranking
- Multi-stage retrieval
- Complex context assembly
- PostgreSQL dependencies

KEPT:
- Document processing
- Simple chunking (512 chars)
- FAISS semantic search
- Direct answer generation
- Sequential model loading
"""

import asyncio
import tempfile
import os
import time
import gc
from typing import List, Optional, Tuple
import httpx
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from unstructured.partition.pdf import partition_pdf

from .model_manager import model_manager

# Simplified Configuration
CHUNK_SIZE = 512           # Simple chunk size
CHUNK_OVERLAP = 64         # Minimal overlap
MAX_CONTEXT_LENGTH = 2000  # Reduced context for memory
HTTP_TIMEOUT = 30.0
RETRIEVAL_K = 5           # Fewer chunks for stability

# Simple Direct Prompt (no complex quality validation)
SIMPLE_PROMPT = PromptTemplate.from_template(
    """Answer this question based on the provided context. Be direct and specific.

Context:
{context}

Question: {question}

Answer:"""
)

print("🚀 Simplified RAG Architecture Initialized")
print("🎯 Target: Reliable operation under 4GB VRAM")
print("💡 Philosophy: Simple systems that work beat complex systems that crash")


async def download_document(doc_url: str) -> bytes:
    """Download document with basic error handling."""
    start_time = time.time()
    print(f"📥 Downloading: {doc_url}")
    
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(doc_url)
            response.raise_for_status()
            content = response.content
        
        download_time = time.time() - start_time
        print(f"✅ Downloaded in {download_time:.2f}s ({len(content)} bytes)")
        return content
        
    except Exception as e:
        raise Exception(f"Download failed: {e}")


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Simple text extraction from PDF."""
    start_time = time.time()
    print("📄 Extracting text from PDF...")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name
        
        try:
            # Use basic fast strategy instead of hi_res
            elements = partition_pdf(
                filename=temp_file_path,
                strategy="fast",  # Much faster than hi_res
                extract_images_in_pdf=False
            )
            
            # Extract text from all elements
            text_content = "\n\n".join([str(element) for element in elements if str(element).strip()])
            
            extraction_time = time.time() - start_time
            print(f"✅ Extracted {len(text_content)} chars in {extraction_time:.2f}s")
            
            return text_content
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")


def create_simple_chunks(text: str) -> List[Document]:
    """Create simple recursive chunks - no parent-child complexity."""
    start_time = time.time()
    print(f"✂️ Creating simple chunks (size: {CHUNK_SIZE})")
    
    try:
        # Simple recursive character splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split text into chunks
        chunk_texts = text_splitter.split_text(text)
        
        # Convert to Documents
        documents = [
            Document(
                page_content=chunk_text,
                metadata={"chunk_id": i}
            )
            for i, chunk_text in enumerate(chunk_texts)
            if chunk_text.strip()  # Skip empty chunks
        ]
        
        processing_time = time.time() - start_time
        print(f"✅ Created {len(documents)} chunks in {processing_time:.2f}s")
        
        return documents
        
    except Exception as e:
        raise Exception(f"Chunking failed: {e}")


async def create_embeddings_and_faiss(documents: List[Document]) -> FAISS:
    """Create embeddings and FAISS index with memory management."""
    start_time = time.time()
    print(f"🔮 Creating embeddings for {len(documents)} chunks")
    
    try:
        # Load embedder
        embedder = model_manager.load_embedder()
        
        # Extract texts
        texts = [doc.page_content for doc in documents]
        
        # Create FAISS index from texts
        print("🏗️ Building FAISS index...")
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embedder
        )
        
        processing_time = time.time() - start_time
        print(f"✅ FAISS index created in {processing_time:.2f}s")
        
        return vector_store
        
    except Exception as e:
        raise Exception(f"FAISS creation failed: {e}")


def simple_retrieval(question: str, vector_store: FAISS) -> str:
    """Simple semantic retrieval - no BM25, no re-ranking."""
    try:
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )
        
        # Get relevant documents
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Simple context assembly
        context_parts = [doc.page_content for doc in relevant_docs]
        context = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH] + "..."
        
        print(f"📖 Retrieved {len(relevant_docs)} chunks ({len(context)} chars)")
        return context
        
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return "Context retrieval failed."


async def generate_answer(question: str, context: str) -> str:
    """Simple answer generation with direct prompt."""
    start_time = time.time()
    print("✨ Generating answer...")
    
    try:
        # Load LLM
        llm = model_manager.load_llm()
        
        # Create simple chain
        chain = SIMPLE_PROMPT | llm | StrOutputParser()
        
        # Generate answer
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        generation_time = time.time() - start_time
        print(f"✅ Answer generated in {generation_time:.2f}s")
        
        return answer.strip()
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return f"Answer generation failed: {str(e)}"


async def process_document(doc_url: str) -> FAISS:
    """
    Simplified document processing pipeline.
    
    OLD: Document → Parent Chunks → Child Chunks → BM25 + Semantic + CrossEncoder → Complex Assembly
    NEW: Document → Simple Chunks → Semantic Search → Direct Generation
    
    Returns:
        FAISS vector store ready for querying
    """
    pipeline_start = time.time()
    print(f"🚀 Processing document: {doc_url}")
    print("=" * 50)
    
    try:
        # Step 1: Download document
        print("📥 STEP 1: Download")
        pdf_content = await download_document(doc_url)
        
        # Step 2: Extract text
        print("📄 STEP 2: Text extraction")
        text_content = extract_text_from_pdf(pdf_content)
        
        if not text_content.strip():
            raise Exception("No text extracted from document")
        
        # Step 3: Create simple chunks
        print("✂️ STEP 3: Simple chunking")
        documents = create_simple_chunks(text_content)
        
        if not documents:
            raise Exception("No chunks created")
        
        # Step 4: Create FAISS index
        print("🔮 STEP 4: FAISS index creation")
        vector_store = await create_embeddings_and_faiss(documents)
        
        total_time = time.time() - pipeline_start
        print("=" * 50)
        print("🏆 Document Processing Complete!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Chunks: {len(documents)}")
        print(f"💾 Memory: {model_manager.get_memory_usage()}")
        
        return vector_store
        
    except Exception as e:
        pipeline_time = time.time() - pipeline_start
        print(f"❌ Processing failed after {pipeline_time:.2f}s: {e}")
        raise Exception(f"Document processing failed: {e}")


async def get_answer(question: str, vector_store: FAISS) -> str:
    """
    Simple answer generation pipeline.
    
    Args:
        question: User's question
        vector_store: FAISS vector store
        
    Returns:
        Direct answer string
    """
    start_time = time.time()
    print(f"❓ Question: {question[:50]}...")
    
    try:
        # Simple retrieval
        context = simple_retrieval(question, vector_store)
        
        # Generate answer
        answer = await generate_answer(question, context)
        
        total_time = time.time() - start_time
        print(f"✅ Answer completed in {total_time:.2f}s")
        
        return answer
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return f"Sorry, I couldn't generate an answer: {str(e)}"


def get_system_info():
    """Get system information."""
    return {
        "architecture": "simplified_production_rag",
        "version": "1.0.0",
        "description": "Boring technology that works reliably",
        "embedding_model": "znbang/bge:small-en-v1.5-f16",
        "llm_model": "qwen2:1.5b",
        "target_memory": "<2.1GB VRAM",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_k": RETRIEVAL_K,
        "philosophy": "Simple systems that work beat complex systems that crash",
        "optimizations": [
            "sequential_model_loading",
            "simple_chunking",
            "direct_retrieval",
            "memory_management",
            "error_handling"
        ]
    }
"""
Core RAG pipeline logic using self-hosted Hugging Face models.
Supports dual-model setup for local testing and production deployment.
"""
import asyncio
import tempfile
import os
from typing import List
from urllib.parse import urlparse

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Configuration constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_DOCUMENTS = 3
HTTP_TIMEOUT = 30.0

# Model configuration constants
LOCAL_LLM_MODEL_NAME = "microsoft/phi-2"  # A small, fast model for CPU
PRODUCTION_LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # The powerful model for GPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting separators in order of preference
TEXT_SEPARATORS = ["\n\n", "\n", " ", ""]

# Global text splitter
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=TEXT_SEPARATORS
)

# Global prompt templates for different models
PHI2_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Instruct: You are a helpful assistant that answers questions based only on the provided context. 
Be direct and concise in your responses.

Context: {context}

Question: {question}

Output: Based only on the provided context:"""
)

MISTRAL_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """<s>[INST] You are a helpful assistant that answers questions based only on the provided context. 
Be direct and concise in your responses.

Context: {context}

Question: {question}

Answer based only on the provided context: [/INST]"""
)


def load_llm_pipeline(use_production_model: bool) -> HuggingFacePipeline:
    """
    Load LLM pipeline based on environment configuration.
    
    Args:
        use_production_model: If True, loads production model with GPU optimization.
                            If False, loads local model for CPU development.
    
    Returns:
        HuggingFacePipeline: LangChain-wrapped LLM pipeline ready for use.
    """
    if use_production_model:
        return _load_production_model()
    else:
        return _load_local_model()


def _load_production_model() -> HuggingFacePipeline:
    """
    Load production model (Mistral-7B) with GPU optimization and quantization.
    
    Returns:
        HuggingFacePipeline: Production-ready LLM pipeline
    """
    print(f"🏭 Loading production model: {PRODUCTION_LLM_MODEL_NAME}")
    
    # Check if CUDA is available for production model
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available. Production model will run slowly on CPU.")
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            PRODUCTION_LLM_MODEL_NAME,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with quantization and auto device mapping
        model = AutoModelForCausalLM.from_pretrained(
            PRODUCTION_LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Create production pipeline with optimized parameters
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Wrap in LangChain HuggingFacePipeline
        langchain_llm = HuggingFacePipeline(
            pipeline=llm_pipeline,
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 512,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        )
        
        print("✅ Production model loaded successfully with 4-bit quantization")
        return langchain_llm
        
    except Exception as e:
        print(f"❌ Error loading production model: {e}")
        raise


def _load_local_model() -> HuggingFacePipeline:
    """
    Load local development model (Phi-2) optimized for CPU usage.
    
    Returns:
        HuggingFacePipeline: Local development LLM pipeline
    """
    print(f"🏠 Loading local development model: {LOCAL_LLM_MODEL_NAME}")
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_LLM_MODEL_NAME,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model for CPU usage
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL_NAME,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        
        # Create local pipeline with conservative parameters for CPU
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,  # Reduced for faster CPU inference
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Wrap in LangChain HuggingFacePipeline
        langchain_llm = HuggingFacePipeline(
            pipeline=llm_pipeline,
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 300,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        )
        
        print("✅ Local development model loaded successfully on CPU")
        return langchain_llm
        
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        raise


# Global model initialization
print("🚀 Initializing AI models...")

# Initialize embedding model (always runs on CPU for efficiency)
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load production model only if an environment variable is set
IS_PRODUCTION = os.getenv("APP_ENV") == "production"
print(f"ℹ️  INFO: Loading model for {'Production' if IS_PRODUCTION else 'Local Development'} environment.")

# Load appropriate LLM based on environment
LLM = load_llm_pipeline(use_production_model=IS_PRODUCTION)

# Set appropriate prompt template based on model
PROMPT_TEMPLATE = MISTRAL_PROMPT_TEMPLATE if IS_PRODUCTION else PHI2_PROMPT_TEMPLATE

print(f"✅ Model initialization complete!")
print(f"📊 Environment: {'Production' if IS_PRODUCTION else 'Local Development'}")
print(f"📊 LLM Model: {PRODUCTION_LLM_MODEL_NAME if IS_PRODUCTION else LOCAL_LLM_MODEL_NAME}")
print(f"📊 Embedding Model: {EMBEDDING_MODEL_NAME}")


async def process_document(doc_url: str) -> tuple[List[str], List[List[float]]]:
    """
    Download a document from URL and convert it to text chunks and embeddings.
    
    Args:
        doc_url: URL string pointing to the document to process
        
    Returns:
        Tuple of (chunk_texts, embeddings) where:
        - chunk_texts: List of text chunks from the document
        - embeddings: List of corresponding embedding vectors
        
    Raises:
        httpx.HTTPStatusError: If document download fails
        Exception: If document processing or embedding generation fails
    """
    try:
        # Download document content asynchronously
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(doc_url)
            response.raise_for_status()
            content = response.content
        
        # Create temporary file to work with UnstructuredFileLoader
        parsed_url = urlparse(doc_url)
        file_extension = os.path.splitext(parsed_url.path)[1] or '.txt'
        
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Load and parse document using UnstructuredFileLoader asynchronously
            loader = UnstructuredFileLoader(temp_file_path)
            documents = await loader.aload()
            
            # Split documents into chunks using global text splitter
            chunks = TEXT_SPLITTER.split_documents(documents)
            
            # Extract text content from chunks
            chunk_texts = [chunk.page_content for chunk in chunks]
            
            # Generate embeddings using synchronous method in thread pool
            embeddings = await asyncio.to_thread(
                EMBEDDING_MODEL.embed_documents, 
                chunk_texts
            )
            
            return chunk_texts, embeddings
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except httpx.HTTPStatusError as e:
        raise Exception(f"Failed to download document from {doc_url}: {e}")
    except Exception as e:
        raise Exception(f"Error processing document: {e}")


def get_answer(question: str, retriever) -> str:
    """
    Generates an answer to a question using FAISS retriever and local Hugging Face LLM.
    NOTE: This is a SYNCHRONOUS, CPU/GPU-bound function.
    It should be run in a thread pool to avoid blocking the main async event loop.
    """
    try:
        # Retrieve top K relevant documents (use the synchronous method)
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Combine document content for context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create and invoke LangChain chain with appropriate model and prompt
        chain = PROMPT_TEMPLATE | LLM | StrOutputParser()
        
        # Generate answer with the retrieved context (use the synchronous method)
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return answer.strip()
        
    except Exception as e:
        # We add a print statement here to see the actual error in the server log
        print(f"ERROR inside get_answer for question '{question}': {e}")
        raise Exception(f"Error generating answer: {e}")


def create_faiss_retriever(chunk_texts: List[str], embeddings: List[List[float]], k: int = TOP_K_DOCUMENTS):
    """
    Create a FAISS retriever from pre-computed embeddings and texts.
    
    Args:
        chunk_texts: List of text chunks
        embeddings: List of pre-computed embedding vectors corresponding to chunk_texts
        k: Number of documents to retrieve (default: TOP_K_DOCUMENTS)
        
    Returns:
        FAISS retriever object
    """
    from langchain_community.vectorstores import FAISS
    import numpy as np
    
    # Convert embeddings to numpy array for FAISS
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    
    # Create FAISS vector store from pre-computed embeddings using global model
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(chunk_texts, embeddings)),
        embedding=EMBEDDING_MODEL
    )
    
    # Create retriever with specified k value
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever


def initialize_models():
    """
    Explicit model initialization function for compatibility.
    Models are already initialized at module level.
    """
    print("ℹ️  Models are already initialized at module level")
    print(f"📊 Current environment: {'Production' if IS_PRODUCTION else 'Local Development'}")


def get_model_info():
    """
    Get information about loaded models for debugging and monitoring.
    
    Returns:
        Dictionary with model information
    """
    return {
        "environment": "production" if IS_PRODUCTION else "local_development",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": PRODUCTION_LLM_MODEL_NAME if IS_PRODUCTION else LOCAL_LLM_MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() and IS_PRODUCTION else "cpu",
        "models_loaded": {
            "embedding": EMBEDDING_MODEL is not None,
            "llm": LLM is not None,
            "prompt_template": PROMPT_TEMPLATE is not None
        },
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_documents": TOP_K_DOCUMENTS
        }
    }


# Memory management utilities
def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")


def get_memory_usage():
    """Get current memory usage information."""
    if torch.cuda.is_available() and IS_PRODUCTION:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return {
            "gpu_allocated_gb": round(allocated, 2),
            "gpu_cached_gb": round(cached, 2),
            "device": "cuda",
            "environment": "production"
        }
    else:
        return {
            "device": "cpu", 
            "environment": "local_development" if not IS_PRODUCTION else "production_cpu",
            "note": "CPU mode - no GPU memory tracking"
        }
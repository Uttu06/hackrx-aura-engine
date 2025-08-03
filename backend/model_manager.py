"""
Simple Model Manager for Memory-Constrained GPU
Handles sequential loading of models to stay within 4GB VRAM limit
"""

import torch
import gc
from typing import Optional
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings


class SimpleModelManager:
    """
    Memory-first model manager for RTX 3050 4GB constraints.
    
    Philosophy: Never load both models simultaneously.
    Strategy: Load -> Use -> Unload -> Load next model
    """
    
    def __init__(self):
        self.embedder: Optional[OllamaEmbeddings] = None
        self.llm: Optional[ChatOllama] = None
        self.current_loaded = None  # 'embedder' or 'llm'
        
        print("🧠 SimpleModelManager initialized")
        print("💾 Target: <2.1GB VRAM usage")
    
    def _clear_gpu_memory(self):
        """Aggressively clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def load_embedder(self):
        """Load BGE-small embedder (350MB)."""
        if self.current_loaded == 'embedder' and self.embedder is not None:
            return self.embedder
        
        print("🔮 Loading embedder (BGE-small)...")
        
        # Unload LLM if loaded
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        self._clear_gpu_memory()
        
        # Load lightweight embedder with minimal parameters
        self.embedder = OllamaEmbeddings(
            model="znbang/bge:small-en-v1.5-f16"
        )
        
        self.current_loaded = 'embedder'
        print("✅ Embedder loaded (350MB)")
        return self.embedder
    
    def load_llm(self):
        """Load Qwen2-1.5B LLM (1.8GB)."""
        if self.current_loaded == 'llm' and self.llm is not None:
            return self.llm
        
        print("🤖 Loading LLM (Qwen2-1.5B)...")
        
        # Unload embedder if loaded
        if self.embedder is not None:
            del self.embedder
            self.embedder = None
        
        self._clear_gpu_memory()
        
        # Load lightweight LLM with minimal parameters
        self.llm = ChatOllama(
            model="qwen2:1.5b",
            temperature=0.1,
            num_predict=512,
            num_ctx=2048,
        )
        
        self.current_loaded = 'llm'
        print("✅ LLM loaded (1.8GB)")
        return self.llm
    
    def get_memory_usage(self):
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            return {
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "current_model": self.current_loaded
            }
        return {"status": "no_gpu"}
    
    def cleanup(self):
        """Full cleanup of all models."""
        print("🧹 Cleaning up all models...")
        
        if self.embedder is not None:
            del self.embedder
            self.embedder = None
        
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        self.current_loaded = None
        self._clear_gpu_memory()
        print("✅ All models unloaded")


# Global model manager instance
model_manager = SimpleModelManager()
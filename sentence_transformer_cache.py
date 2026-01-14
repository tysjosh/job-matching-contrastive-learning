"""
Embedding cache system for Sentence Transformers to avoid recomputing embeddings.
Uses text content hash as cache keys and supports persistence to disk.
"""

import os
import pickle
import hashlib
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import logging


class EmbeddingCache:
    """
    LRU cache for storing computed embeddings with optional disk persistence.
    """
    
    def __init__(self, max_size: int = 10000, cache_dir: str = ".sentence_transformer_cache", 
                 persist: bool = True):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache in memory
            cache_dir: Directory for persistent cache storage
            persist: Whether to persist cache to disk
        """
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.persist = persist
        
        # In-memory LRU cache
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._total_requests = 0
        
        # Setup cache directory
        if self.persist:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_persistent_cache()
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute SHA-256 hash of text content for cache key."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _load_persistent_cache(self) -> None:
        """Load cache from disk if it exists."""
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        
        if os.path.exists(cache_file):
            try:
                import builtins
                with builtins.open(cache_file, 'rb') as f:
                    persistent_cache = pickle.load(f)
                
                # Load up to max_size entries
                loaded_count = 0
                for key, embedding in persistent_cache.items():
                    if loaded_count >= self.max_size:
                        break
                    self._cache[key] = embedding
                    loaded_count += 1
                
                logging.info(f"Loaded {loaded_count} embeddings from persistent cache")
                
            except Exception as e:
                logging.warning(f"Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self) -> None:
        """Save current cache to disk."""
        if not self.persist:
            return
        
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        
        try:
            import builtins
            with builtins.open(cache_file, 'wb') as f:
                pickle.dump(dict(self._cache), f)
            
            logging.info(f"Saved {len(self._cache)} embeddings to persistent cache")
            
        except Exception as e:
            logging.warning(f"Failed to save persistent cache: {e}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache if available.
        
        Args:
            text: Input text to look up
            
        Returns:
            Cached embedding if found, None otherwise
        """
        self._total_requests += 1
        
        if not text or not text.strip():
            return None
        
        text_hash = self._compute_text_hash(text.strip())
        
        if text_hash in self._cache:
            # Move to end (most recently used)
            embedding = self._cache.pop(text_hash)
            self._cache[text_hash] = embedding
            self._hits += 1
            return embedding.copy()  # Return copy to prevent modification
        
        self._misses += 1
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Input text (used to generate cache key)
            embedding: Computed embedding to cache
        """
        if not text or not text.strip():
            return
        
        text_hash = self._compute_text_hash(text.strip())
        
        # Remove oldest entries if cache is full
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove oldest (FIFO)
        
        # Store embedding (make a copy to prevent external modification)
        self._cache[text_hash] = embedding.copy()
    
    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[np.ndarray]], List[str]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (cached_embeddings, uncached_texts)
            - cached_embeddings: List with embeddings or None for each text
            - uncached_texts: List of texts that need to be computed
        """
        cached_embeddings = []
        uncached_texts = []
        
        for text in texts:
            embedding = self.get(text)
            cached_embeddings.append(embedding)
            
            if embedding is None:
                uncached_texts.append(text)
        
        return cached_embeddings, uncached_texts
    
    def put_batch(self, texts: List[str], embeddings: List[np.ndarray]) -> None:
        """
        Store a batch of embeddings in cache.
        
        Args:
            texts: List of input texts
            embeddings: List of computed embeddings
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._total_requests = 0
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        hit_rate = self._hits / self._total_requests if self._total_requests > 0 else 0.0
        
        # Calculate memory usage (approximate)
        memory_usage = 0
        for embedding in self._cache.values():
            memory_usage += embedding.nbytes
        
        return {
            'total_requests': self._total_requests,
            'cache_hits': self._hits,
            'cache_misses': self._misses,
            'hit_rate': hit_rate,
            'cached_items': len(self._cache),
            'max_size': self.max_size,
            'memory_usage_bytes': memory_usage,
            'memory_usage_mb': memory_usage / (1024 * 1024)
        }
    
    def save_to_disk(self) -> None:
        """Manually save cache to disk."""
        self._save_persistent_cache()
    
    def __del__(self):
        """Save cache when object is destroyed."""
        if hasattr(self, 'persist') and self.persist:
            self._save_persistent_cache()


class BatchEmbeddingProcessor:
    """
    Utility class for efficient batch processing of embeddings with caching.
    """
    
    def __init__(self, model, cache: EmbeddingCache, batch_size: int = 32):
        """
        Initialize batch processor.
        
        Args:
            model: SentenceTransformer model
            cache: EmbeddingCache instance
            batch_size: Batch size for processing
        """
        self.model = model
        self.cache = cache
        self.batch_size = batch_size
    
    def encode_with_cache(self, texts: List[str], show_progress_bar: bool = True) -> List[np.ndarray]:
        """
        Encode texts with caching support.
        
        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress bar
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [text.strip() if text else "" for text in texts]
        
        # Get cached embeddings
        cached_embeddings, uncached_texts = self.cache.get_batch(cleaned_texts)
        
        # Compute embeddings for uncached texts
        if uncached_texts:
            logging.info(f"Computing embeddings for {len(uncached_texts)} uncached texts")
            
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            
            # Cache new embeddings
            self.cache.put_batch(uncached_texts, new_embeddings)
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings
        result_embeddings = []
        new_embedding_idx = 0
        
        for i, text in enumerate(cleaned_texts):
            if cached_embeddings[i] is not None:
                result_embeddings.append(cached_embeddings[i])
            else:
                result_embeddings.append(new_embeddings[new_embedding_idx])
                new_embedding_idx += 1
        
        return result_embeddings
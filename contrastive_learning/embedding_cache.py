"""
Efficient Embedding Cache for Contrastive Learning Training

This module implements a global embedding cache that eliminates redundant encoding
of the same content across batches, providing significant speedup for training.
"""

import torch
import logging
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Global embedding cache that stores computed embeddings to avoid redundant encoding.

    Features:
    - Cross-batch caching: Same content encoded once across all batches
    - Batch-efficient encoding: Encode multiple new items in single forward pass
    - Memory management: Optional cache size limits and cleanup
    - Statistics tracking: Monitor cache hit rates and performance
    """

    def __init__(self,
                 max_cache_size: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 enable_stats: bool = True):
        """
        Initialize the embedding cache.

        Args:
            max_cache_size: Maximum number of embeddings to cache (None = unlimited)
            device: Device to store embeddings on
            enable_stats: Whether to track cache statistics
        """
        self.cache: Dict[str, torch.Tensor] = {}
        self.content_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = max_cache_size
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_stats = enable_stats

        # Statistics tracking
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_encodings': 0,
            'batch_encodings': 0,
            'time_saved_seconds': 0.0,
            'memory_usage_mb': 0.0
        }

        # Access tracking for LRU eviction
        self.access_order: List[str] = []
        self.access_count: Dict[str, int] = defaultdict(int)

        logger.info(
            f"EmbeddingCache initialized with max_size={max_cache_size}, device={self.device}")

    def get_content_key(self, content: Dict[str, Any]) -> str:
        """
        Generate a consistent, deterministic key for content.

        Args:
            content: Content dictionary (resume or job)

        Returns:
            str: Unique content key for caching
        """
        try:
            # Create a normalized representation for consistent hashing
            normalized_content = self._normalize_content_for_hashing(content)
            content_str = json.dumps(
                normalized_content, sort_keys=True, default=str)

            # Use SHA-256 for better collision resistance than hash()
            return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]
        except (TypeError, ValueError) as e:
            logger.warning(f"Error creating content key: {e}")
            # Fallback to string representation
            content_str = str(sorted(content.items()))
            return hashlib.md5(content_str.encode('utf-8')).hexdigest()[:16]

    def _normalize_content_for_hashing(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize content for consistent hashing by removing non-essential fields.

        Args:
            content: Original content dictionary

        Returns:
            Dict: Normalized content for hashing
        """
        # Create a copy and remove metadata that doesn't affect embeddings
        normalized = {}

        # Essential fields that affect text encoding
        essential_fields = {
            'experience', 'role', 'experience_level', 'skills', 'keywords',
            'title', 'jobtitle', 'description', 'jobdescription'
        }

        for key, value in content.items():
            if key in essential_fields:
                if isinstance(value, (dict, list)):
                    # Recursively normalize nested structures
                    normalized[key] = self._normalize_nested_structure(value)
                else:
                    normalized[key] = value

        return normalized

    def _normalize_nested_structure(self, obj: Any) -> Any:
        """Recursively normalize nested dictionaries and lists."""
        if isinstance(obj, dict):
            return {k: self._normalize_nested_structure(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_nested_structure(item) for item in obj]
        else:
            return obj

    def get_embeddings_batch(self,
                             content_items: List[Tuple[Dict[str, Any], str]],
                             encoder_func) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for a batch of content items, using cache when possible.

        Args:
            content_items: List of (content_dict, content_type) tuples
            encoder_func: Function to encode new content: func(content_list) -> List[torch.Tensor]

        Returns:
            Dict mapping content keys to embeddings
        """
        start_time = time.time()

        # Step 1: Identify cached vs new content
        cached_embeddings = {}
        new_content_items = []
        new_content_keys = []

        for content_data, content_type in content_items:
            content_key = self.get_content_key(content_data)

            if content_key in self.cache:
                # Cache hit - detach and clone to avoid computational graph reuse
                cached_embeddings[content_key] = self.cache[content_key].detach(
                ).clone()
                self._record_cache_access(content_key, hit=True)
            else:
                # Cache miss - need to encode
                new_content_items.append((content_data, content_type))
                new_content_keys.append(content_key)
                self._record_cache_access(content_key, hit=False)

        # Step 2: Batch encode new content
        new_embeddings = {}
        if new_content_items:
            try:
                # Encode all new content in a single batch operation
                encoded_embeddings = encoder_func(new_content_items)

                # Store new embeddings in cache
                for i, content_key in enumerate(new_content_keys):
                    embedding = encoded_embeddings[i].to(self.device)
                    new_embeddings[content_key] = embedding
                    self._add_to_cache(
                        content_key, embedding, new_content_items[i])

                if self.enable_stats:
                    self.stats['batch_encodings'] += 1
                    self.stats['total_encodings'] += len(new_content_items)

            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                # Fallback to individual encoding
                for i, (content_data, content_type) in enumerate(new_content_items):
                    try:
                        embedding = encoder_func(
                            [(content_data, content_type)])[0]
                        content_key = new_content_keys[i]
                        embedding = embedding.to(self.device)
                        new_embeddings[content_key] = embedding
                        self._add_to_cache(
                            content_key, embedding, (content_data, content_type))
                    except Exception as individual_error:
                        logger.error(
                            f"Individual encoding failed: {individual_error}")
                        continue

        # Step 3: Combine cached and new embeddings
        all_embeddings = {**cached_embeddings, **new_embeddings}

        # Update statistics
        if self.enable_stats:
            encoding_time = time.time() - start_time
            if cached_embeddings:
                # Estimate time saved by caching
                cache_ratio = len(cached_embeddings) / len(content_items)
                estimated_time_saved = encoding_time * cache_ratio / \
                    (1 - cache_ratio) if cache_ratio < 1 else encoding_time
                self.stats['time_saved_seconds'] += estimated_time_saved

        return all_embeddings

    def _add_to_cache(self,
                      content_key: str,
                      embedding: torch.Tensor,
                      content_item: Tuple[Dict[str, Any], str]):
        """
        Add embedding to cache with optional size management.

        Args:
            content_key: Unique content identifier
            embedding: Computed embedding tensor
            content_item: Original content data for metadata
        """
        # Ensure embedding is on correct device and detach from computational graph
        embedding = embedding.to(self.device).detach()

        # Check cache size limit
        if self.max_cache_size and len(self.cache) >= self.max_cache_size:
            self._evict_lru_items(1)

        # Add to cache
        self.cache[content_key] = embedding
        self.content_metadata[content_key] = {
            'content_type': content_item[1],
            'added_time': time.time(),
            'access_count': 1
        }

        # Update access tracking
        self.access_order.append(content_key)
        self.access_count[content_key] = 1

        # Update memory usage stats
        if self.enable_stats:
            self._update_memory_stats()

    def _record_cache_access(self, content_key: str, hit: bool):
        """Record cache access for statistics and LRU tracking."""
        if self.enable_stats:
            if hit:
                self.stats['cache_hits'] += 1
                # Update access tracking for LRU
                if content_key in self.access_order:
                    self.access_order.remove(content_key)
                self.access_order.append(content_key)
                self.access_count[content_key] += 1
            else:
                self.stats['cache_misses'] += 1

    def _evict_lru_items(self, num_items: int):
        """Evict least recently used items from cache."""
        for _ in range(min(num_items, len(self.cache))):
            if self.access_order:
                lru_key = self.access_order.pop(0)
                if lru_key in self.cache:
                    del self.cache[lru_key]
                    del self.content_metadata[lru_key]
                    del self.access_count[lru_key]

    def _update_memory_stats(self):
        """Update memory usage statistics."""
        if not self.enable_stats:
            return

        total_memory = 0
        for embedding in self.cache.values():
            # Estimate memory: num_elements * bytes_per_float
            total_memory += embedding.numel() * 4  # 4 bytes per float32

        self.stats['memory_usage_mb'] = total_memory / (1024 * 1024)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.enable_stats:
            return {"stats_disabled": True}

        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / \
            total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'total_encodings': self.stats['total_encodings'],
            'batch_encodings': self.stats['batch_encodings'],
            'time_saved_seconds': self.stats['time_saved_seconds'],
            'memory_usage_mb': self.stats['memory_usage_mb'],
            'avg_encodings_per_batch': self.stats['total_encodings'] / max(1, self.stats['batch_encodings'])
        }

    def clear_cache(self):
        """Clear all cached embeddings and reset statistics."""
        self.cache.clear()
        self.content_metadata.clear()
        self.access_order.clear()
        self.access_count.clear()

        if self.enable_stats:
            self.stats = {
                'cache_hits': 0,
                'cache_misses': 0,
                'total_encodings': 0,
                'batch_encodings': 0,
                'time_saved_seconds': 0.0,
                'memory_usage_mb': 0.0
            }

        logger.info("Embedding cache cleared")

    def preload_embeddings(self,
                           content_items: List[Tuple[Dict[str, Any], str]],
                           encoder_func,
                           batch_size: int = 32):
        """
        Preload embeddings for a large set of content items.

        Args:
            content_items: List of (content_dict, content_type) tuples
            encoder_func: Function to encode content
            batch_size: Batch size for encoding
        """
        logger.info(f"Preloading embeddings for {len(content_items)} items...")

        # Process in batches
        for i in range(0, len(content_items), batch_size):
            batch = content_items[i:i + batch_size]
            self.get_embeddings_batch(batch, encoder_func)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(
                    f"Preloaded {i + len(batch)}/{len(content_items)} items")

        stats = self.get_cache_stats()
        logger.info(f"Preloading complete. Cache size: {stats['cache_size']}, "
                    f"Memory usage: {stats['memory_usage_mb']:.1f} MB")


class BatchEfficientEncoder:
    """
    Wrapper that makes the encoding function batch-efficient for the cache.
    Supports both text-only and text+structured feature encoding.
    """

    def __init__(self, text_encoder, model, device, 
                 use_structured_features: bool = False,
                 feature_extractor = None):
        """
        Initialize batch encoder.

        Args:
            text_encoder: SentenceTransformer model
            model: ContrastiveDualEncoder model  
            device: Device for computation
            use_structured_features: Whether to extract and use structured features
            feature_extractor: StructuredFeatureExtractor instance (required if use_structured_features=True)
        """
        self.text_encoder = text_encoder
        self.model = model
        self.device = device
        self.use_structured_features = use_structured_features
        self.feature_extractor = feature_extractor
        
        if use_structured_features and feature_extractor is None:
            logger.warning("use_structured_features=True but no feature_extractor provided. "
                          "Falling back to text-only encoding.")

    def encode_batch(self, content_items: List[Tuple[Dict[str, Any], str]]) -> List[torch.Tensor]:
        """
        Efficiently encode a batch of content items using frozen SentenceTransformer approach.
        Supports both text-only and text+structured feature encoding.

        Args:
            content_items: List of (content_dict, content_type) tuples

        Returns:
            List of embedding tensors
        """
        if not content_items:
            return []

        try:
            # Step 1: Convert all content to text
            texts = []
            for content_data, content_type in content_items:
                text = self._content_to_text(content_data, content_type)
                texts.append(text)

            # Step 2: Batch encode with frozen SentenceTransformer (research-grade approach)
            with torch.no_grad():
                # Get base embeddings from frozen SentenceTransformer
                text_embeddings = self.text_encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                    normalize_embeddings=False  # Projection head handles normalization
                )

            # Step 3: Clone tensors to make them trainable (fix for frozen encoder)
            if text_embeddings.dim() == 1:
                text_embeddings = text_embeddings.unsqueeze(0)

            # Clone to create regular tensors from frozen encoder output
            # Don't require gradients since the text encoder is frozen
            text_embeddings = text_embeddings.clone().detach().requires_grad_(False)

            # Step 4: Extract structured features if enabled
            if self.use_structured_features and self.feature_extractor is not None:
                # Extract structured features for the batch
                experience_levels = []
                numerical_features = []
                
                for content_data, content_type in content_items:
                    features = self.feature_extractor.extract_features(content_data, content_type)
                    
                    # Split into experience level (one-hot) and numerical
                    exp_onehot = features[:10]  # First 10 are one-hot
                    numerical = features[10:]   # Rest are numerical
                    
                    # Convert one-hot to index
                    exp_idx = exp_onehot.argmax().item()
                    experience_levels.append(exp_idx)
                    numerical_features.append(numerical)
                
                exp_tensor = torch.tensor(experience_levels, dtype=torch.long, device=self.device)
                num_tensor = torch.stack(numerical_features).to(self.device)
                
                # Forward pass through enhanced model with structured features
                final_embeddings = self.model(
                    text_embeddings,
                    experience_level_idx=exp_tensor,
                    numerical_features=num_tensor
                )
            else:
                # Forward pass through minimal projection model (text only)
                final_embeddings = self.model(text_embeddings)

            # Convert to list of individual tensors
            all_embeddings = [final_embeddings[i]
                              for i in range(final_embeddings.size(0))]

            return all_embeddings

        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            # Fallback to individual encoding
            return [self._encode_single(content_data, content_type)
                    for content_data, content_type in content_items]

    def _content_to_text(self, content: Dict[str, Any], content_type: str) -> str:
        """Convert content dictionary to text representation.

        Prioritizes the most discriminative fields first to fit within
        the SentenceTransformer's token window. Skills come before experience
        to guarantee they are always encoded.
        """
        text_parts = []

        if content_type == 'resume':
            # 1. Role and experience level FIRST (most discriminative metadata)
            role = content.get('role', '')
            exp_level = content.get('experience_level', '')
            if role and exp_level:
                text_parts.append(f"Position: {exp_level} {role}")
            elif role:
                text_parts.append(f"Position: {role}")

            # 2. Skills SECOND (key signal for matching, must always be included)
            if 'skills' in content:
                skill_names = []
                for skill in content['skills']:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            skill_names.append(skill_name)
                    elif isinstance(skill, str):
                        skill_names.append(skill)
                if skill_names:
                    text_parts.append(f"Skills: {', '.join(skill_names)}")

            # 3. Experience text LAST (truncated, fills remaining token budget)
            if 'experience' in content:
                experience = content['experience']
                experience_text = ''
                if isinstance(experience, list) and len(experience) > 0:
                    if isinstance(experience[0], dict):
                        desc_field = experience[0].get('description', '')
                        if isinstance(desc_field, list) and len(desc_field) > 0:
                            if isinstance(desc_field[0], dict):
                                experience_text = desc_field[0].get('description', '')
                            else:
                                experience_text = str(desc_field[0])
                        elif isinstance(desc_field, str):
                            experience_text = desc_field
                        else:
                            experience_text = str(desc_field) if desc_field else ''
                    else:
                        experience_text = str(experience[0])
                elif isinstance(experience, str):
                    experience_text = experience

                if experience_text:
                    # Truncate to ~800 chars to fit within 512-token limit
                    if len(experience_text) > 800:
                        experience_text = experience_text[:800] + "..."
                    text_parts.append(f"Profile: {experience_text}")

        elif content_type == 'job':
            # 1. Title FIRST
            title = content.get('title', content.get('jobtitle', ''))
            if title:
                text_parts.append(f"Position: {title}")

            # 2. Job skills SECOND (key signal for matching, must always be included)
            skills = content.get('skills', [])
            if skills:
                skill_strings = []
                for skill in skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', skill.get('skill', ''))
                        if skill_name:
                            skill_strings.append(skill_name)
                    elif isinstance(skill, str):
                        # Skip malformed skills (sentence fragments)
                        if len(skill) <= 40 and '.' not in skill:
                            skill_strings.append(skill)
                        elif len(skill) <= 20:
                            skill_strings.append(skill)
                if skill_strings:
                    text_parts.append(
                        f"Required Skills: {', '.join(skill_strings)}")

            # 3. Job description LAST (fills remaining token budget)
            description = content.get(
                'description', content.get('jobdescription', ''))
            if description:
                if isinstance(description, dict):
                    desc_text = description.get('original', '')
                elif isinstance(description, str):
                    desc_text = description
                else:
                    desc_text = str(description)

                if desc_text:
                    if len(desc_text) > 800:
                        desc_text = desc_text[:800] + "..."
                    text_parts.append(f"Description: {desc_text}")

        return " [SEP] ".join(text_parts) if text_parts else str(content)

    def _encode_single(self, content_data: Dict[str, Any], content_type: str) -> torch.Tensor:
        """Encode a single content item as fallback using frozen encoder approach."""
        try:
            text = self._content_to_text(content_data, content_type)

            # Use frozen SentenceTransformer (research-grade approach)
            with torch.no_grad():
                text_embedding = self.text_encoder.encode(
                    text,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                    normalize_embeddings=False  # Projection head handles normalization
                )

            # Clone to create trainable tensor from frozen encoder output
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)

            # Create a fresh tensor with gradients for each use
            text_embedding = text_embedding.clone().detach().requires_grad_(False)

            # Apply model with or without structured features
            if self.use_structured_features and self.feature_extractor is not None:
                features = self.feature_extractor.extract_features(content_data, content_type)
                exp_onehot = features[:10]
                numerical = features[10:]
                
                exp_idx = torch.tensor([exp_onehot.argmax().item()], dtype=torch.long, device=self.device)
                num_tensor = numerical.unsqueeze(0).to(self.device)
                
                final_embedding = self.model(
                    text_embedding,
                    experience_level_idx=exp_idx,
                    numerical_features=num_tensor
                ).squeeze(0)
            else:
                final_embedding = self.model(text_embedding).squeeze(0)

            return final_embedding

        except Exception as e:
            logger.error(f"Single encoding failed: {e}")
            # Return zero tensor as last resort (use correct dimension)
            output_dim = getattr(
                self.model, 'get_embedding_dim', lambda: 128)()
            return torch.zeros(output_dim, device=self.device)

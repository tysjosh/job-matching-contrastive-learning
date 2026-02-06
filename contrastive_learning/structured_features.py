"""
Structured Feature Extraction for Career-Aware Contrastive Learning.

This module extracts explicit structured features from resumes and jobs
to complement text embeddings. Key features:
1. Experience level encoding (one-hot or learned embedding)
2. Skill proficiency aggregation
3. Career domain indicators

These features help the model distinguish between similar roles at different levels
(e.g., mid vs senior software engineer) which text embeddings alone struggle with.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Experience level mapping (normalized to 0-1 scale)
EXPERIENCE_LEVELS = {
    'entry': 0,
    'junior': 1,
    'mid': 2,
    'mid-level': 2,
    'senior': 3,
    'lead': 4,
    'principal': 5,
    'staff': 5,
    'director': 6,
    'vp': 7,
    'executive': 8,
    'c-level': 9
}

# Skill proficiency mapping (supports both traditional and career-level naming)
SKILL_PROFICIENCY = {
    # Traditional proficiency levels
    'beginner': 0.2,
    'basic': 0.3,
    'intermediate': 0.5,
    'advanced': 0.7,
    'expert': 0.9,
    'master': 1.0,
    # Career-level naming (from augmentation)
    'entry': 0.2,
    'junior': 0.3,
    'mid': 0.5,
    'mid-level': 0.5,
    'senior': 0.7,
    'lead': 0.8,
    'principal': 0.85,
    'staff': 0.85,
    'director': 0.9,
    'vp': 0.95,
    'executive': 0.95,
    'c-level': 1.0
}


class StructuredFeatureExtractor:
    """
    Extracts structured numerical features from resume and job content.
    
    Features extracted:
    1. Experience level (one-hot encoding, 10 levels)
    2. Average skill proficiency (0-1 scale)
    3. Skill count (normalized)
    4. Years of experience (normalized)
    """
    
    def __init__(self, 
                 num_experience_levels: int = 10,
                 max_skills: int = 50,
                 max_years: int = 30):
        """
        Initialize the feature extractor.
        
        Args:
            num_experience_levels: Number of experience level categories
            max_skills: Maximum skill count for normalization
            max_years: Maximum years of experience for normalization
        """
        self.num_experience_levels = num_experience_levels
        self.max_skills = max_skills
        self.max_years = max_years
        
        # Total feature dimension
        self.feature_dim = (
            num_experience_levels +  # One-hot experience level
            1 +  # Average skill proficiency
            1 +  # Normalized skill count
            1    # Normalized years of experience
        )
        
        logger.info(f"StructuredFeatureExtractor initialized with {self.feature_dim} features")
    
    def extract_features(self, content: Dict[str, Any], content_type: str) -> torch.Tensor:
        """
        Extract structured features from content.
        
        Args:
            content: Resume or job content dictionary
            content_type: 'resume' or 'job'
            
        Returns:
            Tensor of structured features [feature_dim]
        """
        features = []
        
        # 1. Experience level one-hot encoding
        exp_level_onehot = self._extract_experience_level(content, content_type)
        features.append(exp_level_onehot)
        
        # 2. Average skill proficiency
        avg_proficiency = self._extract_skill_proficiency(content)
        features.append(torch.tensor([avg_proficiency], dtype=torch.float32))
        
        # 3. Normalized skill count
        skill_count = self._extract_skill_count(content)
        features.append(torch.tensor([skill_count], dtype=torch.float32))
        
        # 4. Normalized years of experience
        years_exp = self._extract_years_experience(content)
        features.append(torch.tensor([years_exp], dtype=torch.float32))
        
        return torch.cat(features)
    
    def _extract_experience_level(self, content: Dict[str, Any], content_type: str) -> torch.Tensor:
        """Extract experience level as one-hot encoding."""
        onehot = torch.zeros(self.num_experience_levels, dtype=torch.float32)
        
        # Get experience level from content
        exp_level = None
        
        if content_type == 'resume':
            exp_level = content.get('experience_level', '')
        elif content_type == 'job':
            # Jobs might have experience level in title or requirements
            title = content.get('title', content.get('jobtitle', '')).lower()
            
            # Try to infer from title
            for level_name, level_idx in EXPERIENCE_LEVELS.items():
                if level_name in title:
                    exp_level = level_name
                    break
            
            # Also check explicit field
            if not exp_level:
                exp_level = content.get('experience_level', '')
        
        # Map to index
        if exp_level:
            exp_level_lower = exp_level.lower().strip()
            level_idx = EXPERIENCE_LEVELS.get(exp_level_lower, 2)  # Default to mid
            level_idx = min(level_idx, self.num_experience_levels - 1)
            onehot[level_idx] = 1.0
        else:
            # Default to mid-level if unknown
            onehot[2] = 1.0
        
        return onehot
    
    def _extract_skill_proficiency(self, content: Dict[str, Any]) -> float:
        """Extract average skill proficiency (0-1 scale)."""
        skills = content.get('skills', [])
        
        if not skills:
            return 0.5  # Default to intermediate
        
        proficiencies = []
        
        for skill in skills:
            if isinstance(skill, dict):
                level = skill.get('level', '').lower()
                proficiency = SKILL_PROFICIENCY.get(level, 0.5)
                proficiencies.append(proficiency)
            elif isinstance(skill, str):
                # No proficiency info, assume intermediate
                proficiencies.append(0.5)
        
        if proficiencies:
            return sum(proficiencies) / len(proficiencies)
        return 0.5
    
    def _extract_skill_count(self, content: Dict[str, Any]) -> float:
        """Extract normalized skill count (0-1 scale)."""
        skills = content.get('skills', [])
        count = len(skills) if isinstance(skills, list) else 0
        return min(count / self.max_skills, 1.0)
    
    def _extract_years_experience(self, content: Dict[str, Any]) -> float:
        """Extract normalized years of experience (0-1 scale)."""
        # Try to get years from various fields
        years = content.get('years_experience', 0)
        
        if not years:
            # Try to infer from experience level
            exp_level = content.get('experience_level', '').lower()
            level_to_years = {
                'entry': 0,
                'junior': 1,
                'mid': 3,
                'mid-level': 3,
                'senior': 6,
                'lead': 8,
                'principal': 10,
                'staff': 10,
                'director': 12,
                'vp': 15,
                'executive': 18,
                'c-level': 20
            }
            years = level_to_years.get(exp_level, 3)
        
        return min(years / self.max_years, 1.0)
    
    def get_feature_dim(self) -> int:
        """Get the total feature dimension."""
        return self.feature_dim


class StructuredFeatureEncoder(nn.Module):
    """
    Neural network module that encodes structured features.
    
    Can be used to:
    1. Learn embeddings for categorical features (experience level)
    2. Transform numerical features
    3. Combine with text embeddings
    """
    
    def __init__(self,
                 num_experience_levels: int = 10,
                 experience_embed_dim: int = 16,
                 numerical_features: int = 3,
                 output_dim: int = 32):
        """
        Initialize the structured feature encoder.
        
        Args:
            num_experience_levels: Number of experience level categories
            experience_embed_dim: Dimension of experience level embedding
            numerical_features: Number of numerical features (proficiency, count, years)
            output_dim: Output dimension of encoded features
        """
        super().__init__()
        
        # Learnable experience level embedding
        self.experience_embedding = nn.Embedding(
            num_experience_levels, 
            experience_embed_dim
        )
        
        # MLP for numerical features
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Combine experience embedding and numerical features
        combined_dim = experience_embed_dim + 16
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU()
        )
        
        self.output_dim = output_dim
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, 
                experience_level_idx: torch.Tensor,
                numerical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the structured feature encoder.
        
        Args:
            experience_level_idx: Experience level indices [batch_size]
            numerical_features: Numerical features [batch_size, 3]
            
        Returns:
            Encoded structured features [batch_size, output_dim]
        """
        # Get experience level embedding
        exp_embed = self.experience_embedding(experience_level_idx)
        
        # Encode numerical features
        num_encoded = self.numerical_encoder(numerical_features)
        
        # Combine
        combined = torch.cat([exp_embed, num_encoded], dim=-1)
        
        return self.combiner(combined)
    
    def get_output_dim(self) -> int:
        """Get the output dimension."""
        return self.output_dim


class EnhancedContrastiveModel(nn.Module):
    """
    Enhanced contrastive model that combines text embeddings with structured features.
    
    Architecture:
    1. Text embeddings from frozen SentenceTransformer (384 dims)
    2. Structured features (experience level, skill proficiency, etc.)
    3. Combined projection head
    """
    
    def __init__(self,
                 text_embed_dim: int = 384,
                 structured_feature_dim: int = 32,
                 projection_dim: int = 128,
                 dropout: float = 0.1,
                 use_structured_features: bool = True):
        """
        Initialize the enhanced contrastive model.
        
        Args:
            text_embed_dim: Dimension of text embeddings from SentenceTransformer
            structured_feature_dim: Dimension of encoded structured features
            projection_dim: Final projection dimension
            dropout: Dropout rate
            use_structured_features: Whether to use structured features
        """
        super().__init__()
        
        self.use_structured_features = use_structured_features
        self.text_embed_dim = text_embed_dim
        self.structured_feature_dim = structured_feature_dim if use_structured_features else 0
        
        # Structured feature encoder (if enabled)
        if use_structured_features:
            self.structured_encoder = StructuredFeatureEncoder(
                num_experience_levels=10,
                experience_embed_dim=16,
                numerical_features=3,
                output_dim=structured_feature_dim
            )
        else:
            self.structured_encoder = None
        
        # Combined input dimension
        combined_dim = text_embed_dim + self.structured_feature_dim
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(combined_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        
        self.projection_dim = projection_dim
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"EnhancedContrastiveModel initialized: "
                   f"text_dim={text_embed_dim}, structured_dim={self.structured_feature_dim}, "
                   f"projection_dim={projection_dim}, use_structured={use_structured_features}")
    
    def _initialize_weights(self):
        """Initialize projection head weights."""
        for module in self.projection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                text_embeddings: torch.Tensor,
                experience_level_idx: Optional[torch.Tensor] = None,
                numerical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the enhanced model.
        
        Args:
            text_embeddings: Text embeddings from SentenceTransformer [batch_size, text_embed_dim]
            experience_level_idx: Experience level indices [batch_size] (optional)
            numerical_features: Numerical features [batch_size, 3] (optional)
            
        Returns:
            Normalized embeddings for contrastive learning [batch_size, projection_dim]
        """
        if self.use_structured_features and experience_level_idx is not None and numerical_features is not None:
            # Encode structured features
            structured_encoded = self.structured_encoder(
                experience_level_idx, 
                numerical_features
            )
            
            # Concatenate text and structured features
            combined = torch.cat([text_embeddings, structured_encoded], dim=-1)
        else:
            # Text only
            combined = text_embeddings
        
        # Project
        projected = self.projection_head(combined)
        
        # L2 normalize
        return torch.nn.functional.normalize(projected, p=2, dim=-1)
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.projection_dim


def extract_structured_features_batch(
    content_items: List[Tuple[Dict[str, Any], str]],
    feature_extractor: StructuredFeatureExtractor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract structured features for a batch of content items.
    
    Args:
        content_items: List of (content_dict, content_type) tuples
        feature_extractor: StructuredFeatureExtractor instance
        device: Device to put tensors on
        
    Returns:
        Tuple of (experience_level_indices, numerical_features)
    """
    experience_levels = []
    numerical_features = []
    
    for content, content_type in content_items:
        features = feature_extractor.extract_features(content, content_type)
        
        # Split into experience level (one-hot) and numerical
        exp_onehot = features[:10]  # First 10 are one-hot
        numerical = features[10:]   # Rest are numerical
        
        # Convert one-hot to index
        exp_idx = exp_onehot.argmax().item()
        experience_levels.append(exp_idx)
        numerical_features.append(numerical)
    
    exp_tensor = torch.tensor(experience_levels, dtype=torch.long, device=device)
    num_tensor = torch.stack(numerical_features).to(device)
    
    return exp_tensor, num_tensor

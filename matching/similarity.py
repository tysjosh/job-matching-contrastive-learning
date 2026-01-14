import numpy as np

class CareerAwareSimilarityComputer:
    """
    COMPLETE SIMILARITY SCORING COMPONENT
    Computes career-distance weighted similarities for ranking
    """
    def __init__(self, career_graph, distance_weights=None):
        self.career_graph = career_graph
        
        # Distance weights from training objective (same as used in loss function)
        self.distance_weights = distance_weights or {
            0: 1.0,    # Same career level
            1: 0.8,    # Adjacent level (1 hop)
            2: 0.6,    # 2 hops apart  
            3: 0.4,    # 3 hops apart
            999: 0.2   # Cross-domain or very distant
        }
    
    def compute_similarity(self, resume_embedding: np.ndarray, job_embedding: np.ndarray,
                          resume_metadata: dict, job_metadata: dict) -> float:
        """
        MAIN METHOD - Compute career-aware similarity score for ranking
        
        This is the core scoring function that determines ranking order
        """
        
        # Step 1: Base cosine similarity from embeddings
        base_similarity = self._cosine_similarity(resume_embedding, job_embedding)
        
        # Step 2: Career distance computation
        career_distance = self._compute_career_distance(
            resume_metadata['career_level'], 
            job_metadata['career_level']
        )
        
        # Step 3: Apply distance weighting (consistent with training)
        distance_weight = self.distance_weights.get(career_distance, 0.2)
        
        # Step 4: Final weighted similarity (THIS IS THE RANKING SCORE)
        final_similarity = base_similarity * distance_weight
        
        return final_similarity
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors"""
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm)
        
        # Ensure similarity is in valid range [-1, 1]
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return float(similarity)
    
    def _compute_career_distance(self, level1: str, level2: str) -> int:
        """Compute distance in career progression graph"""
        
        try:
            # Use career graph to find shortest path distance
            distance = self.career_graph.shortest_path_distance(level1, level2)
            return distance
        except:
            # Fallback for unknown career levels or disconnected graph components
            if level1 == level2:
                return 0
            else:
                return 999  # Large distance for unknown relationships
    
    def compute_batch_similarities(self, resume_embeddings: np.ndarray, 
                                 job_embeddings: np.ndarray,
                                 resume_metadata_list: list[dict],
                                 job_metadata_list: list[dict]) -> np.ndarray:
        """
        Batch computation for efficiency when scoring many pairs
        """
        
        similarities = np.zeros((len(resume_embeddings), len(job_embeddings)))
        
        for i, (resume_emb, resume_meta) in enumerate(zip(resume_embeddings, resume_metadata_list)):
            for j, (job_emb, job_meta) in enumerate(zip(job_embeddings, job_metadata_list)):
                similarities[i, j] = self.compute_similarity(
                    resume_emb, job_emb, resume_meta, job_meta
                )
        
        return similarities
    
    def get_distance_weight(self, career_distance: int) -> float:
        """Get the distance weight for a given career distance"""
        return self.distance_weights.get(career_distance, 0.2)
    
    def update_distance_weights(self, new_weights: dict[int, float]):
        """Update distance weights (e.g., after retraining)"""
        self.distance_weights.update(new_weights)
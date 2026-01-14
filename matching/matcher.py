import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import faiss

from .similarity import CareerAwareSimilarityComputer
from .tech_similarity import TechAwareSimilarityComputer
from .explanation import ExplanationTextGenerator
from .utils import SkillExtractor, CareerLevelClassifier
from .tech_utils import TechSkillExtractor, TechCareerLevelClassifier

@dataclass
class MatchResult:
    item_id: str
    similarity_score: float
    career_distance: int
    explanation: Dict

class ExplainableJobMatcher:
    """
    COMPLETE CORE MATCHING SYSTEM - Orchestrates all components
    """
    def __init__(self, resume_encoder, job_encoder, career_graph, config: dict, 
                 resume_index=None, job_index=None, use_tech_specialized=False):
        # Component 1: Trained encoders
        self.resume_encoder = resume_encoder
        self.job_encoder = job_encoder
        self.career_graph = career_graph
        self.config = config
        self.use_tech_specialized = use_tech_specialized
        
        # Component 2: Similarity computer (tech-aware or generic)
        if use_tech_specialized:
            self.similarity_computer = TechAwareSimilarityComputer(career_graph)
        else:
            self.similarity_computer = CareerAwareSimilarityComputer(career_graph)
        
        # Component 3: Explanation generator
        self.explanation_generator = ExplanationTextGenerator(career_graph)
        
        # Pre-computed indices for efficiency
        self.resume_index = resume_index  # FAISS index for resume embeddings
        self.job_index = job_index       # FAISS index for job embeddings
        
        # Metadata storage
        self.resume_metadata = {}  # {resume_id: {text, skills, level, etc}}
        self.job_metadata = {}     # {job_id: {text, requirements, level, etc}}
        
        # Supporting processors (tech-specialized or generic)
        if use_tech_specialized:
            self.skill_extractor = TechSkillExtractor()
            self.career_classifier = TechCareerLevelClassifier(career_graph, config)
        else:
            self.skill_extractor = SkillExtractor()
            self.career_classifier = CareerLevelClassifier(career_graph, config)
        
        # Index mappings for FAISS lookups
        self.resume_id_to_index = {}  # {resume_id: faiss_index}
        self.job_id_to_index = {}     # {job_id: faiss_index}
        self.index_to_resume_id = {}  # {faiss_index: resume_id}
        self.index_to_job_id = {}     # {faiss_index: job_id}
    
    def match_resume_to_jobs(self, resume: str, top_k: int = 10) -> List[MatchResult]:
        """
        MAIN ENTRY POINT: Given a new resume, find top-k matching jobs with explanations
        """
        # Step 1: Process input resume
        resume_data = self._process_resume(resume)
        resume_embedding = self.resume_encoder.encode(resume)
        
        # Step 2: Retrieve candidate jobs using FAISS
        candidate_jobs = self._retrieve_candidate_jobs(resume_embedding, top_k * 2)
        
        # Step 3: Compute career-aware similarities for all candidates
        ranked_matches = []
        
        for candidate in candidate_jobs:
            job_id = candidate['job_id']
            job_data = self.job_metadata[job_id]
            job_embedding = self._get_job_embedding(job_id)
            
            # >>> CALLS SIMILARITY COMPUTER <<<
            similarity_score = self.similarity_computer.compute_similarity(
                resume_embedding=resume_embedding,
                job_embedding=job_embedding,
                resume_metadata=resume_data,
                job_metadata=job_data
            )
            
            career_distance = self.career_graph.distance(
                resume_data['career_level'], job_data['career_level']
            )
            
            ranked_matches.append({
                'job_id': job_id,
                'similarity_score': similarity_score,  # This is the ranking score
                'career_distance': career_distance,
                'job_data': job_data
            })
        
        # Step 4: Sort by career-aware similarity score
        ranked_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Step 5: Generate explanations for top-k matches
        results = []
        for match in ranked_matches[:top_k]:
            
            # >>> CALLS EXPLANATION GENERATOR <<<
            explanation = self.explanation_generator.generate_explanation(
                resume_data, 
                match['job_data'], 
                match['similarity_score'],
                audience='candidate'
            )
            
            results.append(MatchResult(
                item_id=match['job_id'],
                similarity_score=match['similarity_score'],
                career_distance=match['career_distance'],
                explanation=explanation
            ))
        
        return results
    
    def match_job_to_resumes(self, job_description: str, top_k: int = 10) -> List[MatchResult]:
        """
        MAIN ENTRY POINT: Given a new job posting, find top-k matching resumes with explanations
        """
        # Step 1: Process input job
        job_data = self._process_job(job_description)
        job_embedding = self.job_encoder.encode(job_description)
        
        # Step 2: Retrieve candidate resumes using FAISS
        candidate_resumes = self._retrieve_candidate_resumes(job_embedding, top_k * 2)
        
        # Step 3: Compute career-aware similarities
        ranked_matches = []
        
        for candidate in candidate_resumes:
            resume_id = candidate['resume_id']
            resume_data = self.resume_metadata[resume_id]
            resume_embedding = self._get_resume_embedding(resume_id)
            
            # >>> CALLS SIMILARITY COMPUTER <<<
            similarity_score = self.similarity_computer.compute_similarity(
                resume_embedding=resume_embedding,
                job_embedding=job_embedding,
                resume_metadata=resume_data,
                job_metadata=job_data
            )
            
            career_distance = self.career_graph.distance(
                resume_data['career_level'], job_data['career_level']
            )
            
            ranked_matches.append({
                'resume_id': resume_id,
                'similarity_score': similarity_score,
                'career_distance': career_distance,
                'resume_data': resume_data
            })
        
        # Step 4: Sort by career-aware similarity
        ranked_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Step 5: Generate explanations for top-k matches
        results = []
        for match in ranked_matches[:top_k]:
            
            # >>> CALLS EXPLANATION GENERATOR <<<
            explanation = self.explanation_generator.generate_explanation(
                match['resume_data'],
                job_data,
                match['similarity_score'],
                audience='recruiter'
            )
            
            results.append(MatchResult(
                item_id=match['resume_id'],
                similarity_score=match['similarity_score'],
                career_distance=match['career_distance'],
                explanation=explanation
            ))
        
        return results
    
    # ==== INPUT PROCESSING METHODS ====
    
    def _process_resume(self, resume: str) -> Dict:
        """Extract structured information from resume text"""
        if self.use_tech_specialized:
            # Tech-specialized processing with additional metadata
            extracted_skills = self.skill_extractor.extract_tech_skills(resume)
            
            # Extract skills from each category
            all_skills = []
            programming_languages = []
            frameworks = []
            databases = []
            cloud_platforms = []
            
            for category, skills in extracted_skills.items():
                for skill in skills:
                    all_skills.append(skill.skill)
                    if category == 'programming_languages':
                        programming_languages.append(skill.skill)
                    elif category in ['web_frameworks']:
                        frameworks.append(skill.skill)
                    elif category == 'databases':
                        databases.append(skill.skill)
                    elif category == 'cloud_platforms':
                        cloud_platforms.append(skill.skill)
            
            return {
                'text': resume,
                'skills': all_skills,
                'programming_languages': programming_languages,
                'frameworks': frameworks,
                'databases': databases,
                'cloud_platforms': cloud_platforms,
                'career_level': self.career_classifier.classify_tech_career_level(resume),
            }
        else:
            # Generic processing
            return {
                'text': resume,
                'skills': self.skill_extractor.extract_skills(resume),
                'career_level': self.career_classifier.infer_level(resume),
            }
    
    def _process_job(self, job_description: str) -> Dict:
        """Extract structured information from job posting"""
        if self.use_tech_specialized:
            # Tech-specialized processing with additional metadata
            extracted_skills = self.skill_extractor.extract_tech_skills(job_description)
            
            # Extract skills from each category (for both required and preferred)
            all_skills = []
            programming_languages = []
            frameworks = []
            databases = []
            cloud_platforms = []
            
            for category, skills in extracted_skills.items():
                for skill in skills:
                    all_skills.append(skill.skill)
                    if category == 'programming_languages':
                        programming_languages.append(skill.skill)
                    elif category in ['web_frameworks']:
                        frameworks.append(skill.skill)
                    elif category == 'databases':
                        databases.append(skill.skill)
                    elif category == 'cloud_platforms':
                        cloud_platforms.append(skill.skill)
            
            # For simplicity, treat all extracted skills as required
            # In a real implementation, you might parse "required" vs "preferred" sections
            return {
                'text': job_description,
                'required_skills': all_skills,
                'required_languages': programming_languages,
                'required_frameworks': frameworks,
                'preferred_skills': [],  # Could be enhanced to parse preferred vs required
                'preferred_languages': [],
                'preferred_frameworks': [],
                'career_level': self.career_classifier.classify_tech_career_level(job_description),
            }
        else:
            # Generic processing
            return {
                'text': job_description,
                'required_skills': self.skill_extractor.extract_required_skills(job_description),
                'preferred_skills': self.skill_extractor.extract_preferred_skills(job_description),
                'career_level': self.career_classifier.infer_level(job_description),
            }
    
    # ==== CANDIDATE RETRIEVAL METHODS ====
    
    def _retrieve_candidate_jobs(self, resume_embedding: np.ndarray, k: int) -> List[Dict]:
        """Retrieve candidate jobs using FAISS index"""
        similarities, indices = self.job_index.search(
            resume_embedding.reshape(1, -1), k
        )
        
        candidates = []
        for sim, idx in zip(similarities[0], indices[0]):
            job_id = self.index_to_job_id[idx]
            candidates.append({
                'job_id': job_id,
                'base_similarity': float(sim),
                'job_data': self.job_metadata[job_id]
            })
        
        return candidates
    
    def _retrieve_candidate_resumes(self, job_embedding: np.ndarray, k: int) -> List[Dict]:
        """Retrieve candidate resumes using FAISS index"""
        similarities, indices = self.resume_index.search(
            job_embedding.reshape(1, -1), k
        )
        
        candidates = []
        for sim, idx in zip(similarities[0], indices[0]):
            resume_id = self.index_to_resume_id[idx]
            candidates.append({
                'resume_id': resume_id,
                'base_similarity': float(sim),
                'resume_data': self.resume_metadata[resume_id]
            })
        
        return candidates
    
    # ==== EMBEDDING RETRIEVAL METHODS ====
    
    def _get_job_embedding(self, job_id: str) -> np.ndarray:
        """Get pre-computed job embedding from storage"""
        if job_id not in self.job_id_to_index:
            raise ValueError(f"Job ID {job_id} not found in stored embeddings. Call add_job() first.")
        
        if self.job_index is None:
            raise RuntimeError("Job index not initialized. No jobs have been added to the system.")
        
        # Get the FAISS index for this job
        faiss_idx = self.job_id_to_index[job_id]
        
        # Validate FAISS index bounds
        if faiss_idx >= self.job_index.ntotal:
            raise IndexError(f"FAISS index {faiss_idx} out of bounds for job {job_id}")
        
        try:
            # Retrieve the embedding directly from the FAISS index
            embedding = self.job_index.reconstruct(faiss_idx)
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve embedding for job {job_id}: {str(e)}")
    
    def _get_resume_embedding(self, resume_id: str) -> np.ndarray:
        """Get pre-computed resume embedding from storage"""
        if resume_id not in self.resume_id_to_index:
            raise ValueError(f"Resume ID {resume_id} not found in stored embeddings. Call add_resume() first.")
        
        if self.resume_index is None:
            raise RuntimeError("Resume index not initialized. No resumes have been added to the system.")
        
        # Get the FAISS index for this resume
        faiss_idx = self.resume_id_to_index[resume_id]
        
        # Validate FAISS index bounds
        if faiss_idx >= self.resume_index.ntotal:
            raise IndexError(f"FAISS index {faiss_idx} out of bounds for resume {resume_id}")
        
        try:
            # Retrieve the embedding directly from the FAISS index
            embedding = self.resume_index.reconstruct(faiss_idx)
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve embedding for resume {resume_id}: {str(e)}")
    
    # ==== EMBEDDING STORAGE/POPULATION METHODS ====
    
    def add_job(self, job_id: str, job_text: str, job_metadata: Dict = None) -> None:
        """Add a new job to the system with pre-computed embedding storage"""
        # Step 1: Process and store metadata
        processed_job = self._process_job(job_text)
        if job_metadata:
            processed_job.update(job_metadata)
        self.job_metadata[job_id] = processed_job
        
        # Step 2: Compute and store embedding
        job_embedding = self.job_encoder.encode(job_text)
        
        # Step 3: Add to FAISS index
        if self.job_index is None:
            # Initialize FAISS index if it doesn't exist
            embedding_dim = job_embedding.shape[0]
            self.job_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        
        # Get the next available index
        next_faiss_idx = self.job_index.ntotal
        
        # Add embedding to FAISS index
        self.job_index.add(job_embedding.reshape(1, -1))
        
        # Update mappings
        self.job_id_to_index[job_id] = next_faiss_idx
        self.index_to_job_id[next_faiss_idx] = job_id
    
    def add_resume(self, resume_id: str, resume_text: str, resume_metadata: Dict = None) -> None:
        """Add a new resume to the system with pre-computed embedding storage"""
        # Step 1: Process and store metadata
        processed_resume = self._process_resume(resume_text)
        if resume_metadata:
            processed_resume.update(resume_metadata)
        self.resume_metadata[resume_id] = processed_resume
        
        # Step 2: Compute and store embedding
        resume_embedding = self.resume_encoder.encode(resume_text)
        
        # Step 3: Add to FAISS index
        if self.resume_index is None:
            # Initialize FAISS index if it doesn't exist
            embedding_dim = resume_embedding.shape[0]
            self.resume_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        
        # Get the next available index
        next_faiss_idx = self.resume_index.ntotal
        
        # Add embedding to FAISS index
        self.resume_index.add(resume_embedding.reshape(1, -1))
        
        # Update mappings
        self.resume_id_to_index[resume_id] = next_faiss_idx
        self.index_to_resume_id[next_faiss_idx] = resume_id
    
    def batch_add_jobs(self, jobs: List[Tuple[str, str, Optional[Dict]]]) -> None:
        """Efficiently add multiple jobs in batch"""
        job_embeddings = []
        job_ids = []
        
        for job_id, job_text, job_metadata in jobs:
            # Process and store metadata
            processed_job = self._process_job(job_text)
            if job_metadata:
                processed_job.update(job_metadata)
            self.job_metadata[job_id] = processed_job
            
            # Compute embedding
            job_embedding = self.job_encoder.encode(job_text)
            job_embeddings.append(job_embedding)
            job_ids.append(job_id)
        
        # Batch add to FAISS
        if job_embeddings:
            embeddings_matrix = np.vstack(job_embeddings)
            
            if self.job_index is None:
                embedding_dim = embeddings_matrix.shape[1]
                self.job_index = faiss.IndexFlatIP(embedding_dim)
            
            start_idx = self.job_index.ntotal
            self.job_index.add(embeddings_matrix)
            
            # Update mappings
            for i, job_id in enumerate(job_ids):
                faiss_idx = start_idx + i
                self.job_id_to_index[job_id] = faiss_idx
                self.index_to_job_id[faiss_idx] = job_id
    
    def batch_add_resumes(self, resumes: List[Tuple[str, str, Optional[Dict]]]) -> None:
        """Efficiently add multiple resumes in batch"""
        resume_embeddings = []
        resume_ids = []
        
        for resume_id, resume_text, resume_metadata in resumes:
            # Process and store metadata
            processed_resume = self._process_resume(resume_text)
            if resume_metadata:
                processed_resume.update(resume_metadata)
            self.resume_metadata[resume_id] = processed_resume
            
            # Compute embedding
            resume_embedding = self.resume_encoder.encode(resume_text)
            resume_embeddings.append(resume_embedding)
            resume_ids.append(resume_id)
        
        # Batch add to FAISS
        if resume_embeddings:
            embeddings_matrix = np.vstack(resume_embeddings)
            
            if self.resume_index is None:
                embedding_dim = embeddings_matrix.shape[1]
                self.resume_index = faiss.IndexFlatIP(embedding_dim)
            
            start_idx = self.resume_index.ntotal
            self.resume_index.add(embeddings_matrix)
            
            # Update mappings
            for i, resume_id in enumerate(resume_ids):
                faiss_idx = start_idx + i
                self.resume_id_to_index[resume_id] = faiss_idx
                self.index_to_resume_id[faiss_idx] = resume_id
    
    def get_stored_job_ids(self) -> List[str]:
        """Get list of all stored job IDs"""
        return list(self.job_metadata.keys())
    
    def get_stored_resume_ids(self) -> List[str]:
        """Get list of all stored resume IDs"""
        return list(self.resume_metadata.keys())
    
    def remove_job(self, job_id: str) -> None:
        """Remove a job from the system (note: FAISS doesn't support efficient removal)"""
        if job_id in self.job_metadata:
            del self.job_metadata[job_id]
        if job_id in self.job_id_to_index:
            faiss_idx = self.job_id_to_index[job_id]
            del self.job_id_to_index[job_id]
            if faiss_idx in self.index_to_job_id:
                del self.index_to_job_id[faiss_idx]
        # Note: FAISS index itself is not modified as it doesn't support efficient removal
        print(f"Warning: Job {job_id} metadata removed, but embedding remains in FAISS index")
    
    def remove_resume(self, resume_id: str) -> None:
        """Remove a resume from the system (note: FAISS doesn't support efficient removal)"""
        if resume_id in self.resume_metadata:
            del self.resume_metadata[resume_id]
        if resume_id in self.resume_id_to_index:
            faiss_idx = self.resume_id_to_index[resume_id]
            del self.resume_id_to_index[resume_id]
            if faiss_idx in self.index_to_resume_id:
                del self.index_to_resume_id[faiss_idx]
        # Note: FAISS index itself is not modified as it doesn't support efficient removal
        print(f"Warning: Resume {resume_id} metadata removed, but embedding remains in FAISS index")
    
    # ==== SYSTEM VALIDATION AND UTILITY METHODS ====
    
    def validate_system_integrity(self) -> Dict[str, bool]:
        """Validate the integrity of the matching system"""
        results = {
            'job_mappings_consistent': True,
            'resume_mappings_consistent': True,
            'job_metadata_complete': True,
            'resume_metadata_complete': True,
            'faiss_indices_initialized': True
        }
        
        # Check job mappings consistency
        try:
            if self.job_index is not None:
                for job_id, faiss_idx in self.job_id_to_index.items():
                    if faiss_idx not in self.index_to_job_id:
                        results['job_mappings_consistent'] = False
                        break
                    if self.index_to_job_id[faiss_idx] != job_id:
                        results['job_mappings_consistent'] = False
                        break
                    if faiss_idx >= self.job_index.ntotal:
                        results['job_mappings_consistent'] = False
                        break
        except Exception:
            results['job_mappings_consistent'] = False
        
        # Check resume mappings consistency
        try:
            if self.resume_index is not None:
                for resume_id, faiss_idx in self.resume_id_to_index.items():
                    if faiss_idx not in self.index_to_resume_id:
                        results['resume_mappings_consistent'] = False
                        break
                    if self.index_to_resume_id[faiss_idx] != resume_id:
                        results['resume_mappings_consistent'] = False
                        break
                    if faiss_idx >= self.resume_index.ntotal:
                        results['resume_mappings_consistent'] = False
                        break
        except Exception:
            results['resume_mappings_consistent'] = False
        
        # Check metadata completeness
        for job_id in self.job_id_to_index.keys():
            if job_id not in self.job_metadata:
                results['job_metadata_complete'] = False
                break
        
        for resume_id in self.resume_id_to_index.keys():
            if resume_id not in self.resume_metadata:
                results['resume_metadata_complete'] = False
                break
        
        # Check FAISS indices initialization
        if len(self.job_id_to_index) > 0 and self.job_index is None:
            results['faiss_indices_initialized'] = False
        if len(self.resume_id_to_index) > 0 and self.resume_index is None:
            results['faiss_indices_initialized'] = False
        
        return results
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the current system state"""
        return {
            'total_jobs': len(self.job_metadata),
            'total_resumes': len(self.resume_metadata),
            'job_embeddings_stored': self.job_index.ntotal if self.job_index else 0,
            'resume_embeddings_stored': self.resume_index.ntotal if self.resume_index else 0,
            'job_index_type': type(self.job_index).__name__ if self.job_index else None,
            'resume_index_type': type(self.resume_index).__name__ if self.resume_index else None,
            'embedding_dimension_jobs': self.job_index.d if self.job_index else None,
            'embedding_dimension_resumes': self.resume_index.d if self.resume_index else None
        }
    
    def has_embeddings_for_id(self, item_id: str, item_type: str = 'auto') -> bool:
        """Check if embeddings exist for a given ID"""
        if item_type == 'auto':
            # Auto-detect based on which mappings contain the ID
            if item_id in self.job_id_to_index:
                item_type = 'job'
            elif item_id in self.resume_id_to_index:
                item_type = 'resume'
            else:
                return False
        
        if item_type == 'job':
            return (item_id in self.job_id_to_index and 
                   self.job_index is not None and 
                   self.job_id_to_index[item_id] < self.job_index.ntotal)
        elif item_type == 'resume':
            return (item_id in self.resume_id_to_index and 
                   self.resume_index is not None and 
                   self.resume_id_to_index[item_id] < self.resume_index.ntotal)
        else:
            raise ValueError(f"Invalid item_type: {item_type}. Must be 'job', 'resume', or 'auto'")

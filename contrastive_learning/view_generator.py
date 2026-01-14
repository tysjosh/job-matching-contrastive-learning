"""
View Generator for integrating with existing view augmentation system.
Creates multiple views of resumes and jobs for contrastive learning training.
"""

import logging
from typing import List, Dict, Any, Tuple
from itertools import product

ResumeViewAugmentation = None
JobViewAugmentation = None

try:
    from preprocess.resume_view_augmentation import ResumeViewAugmentation
    logging.info("ResumeViewAugmentation imported successfully")
except (ImportError, SyntaxError) as e:
    logging.warning(f"Could not import ResumeViewAugmentation: {e}")
    ResumeViewAugmentation = None

try:
    from preprocess.job_view_augmentation import JobViewAugmentation
    logging.info("JobViewAugmentation imported successfully")
except (ImportError, SyntaxError) as e:
    logging.warning(f"Could not import JobViewAugmentation: {e}")
    JobViewAugmentation = None


class ViewGenerator:
    """
    Integrates with existing view augmentation system to generate multiple views
    of resumes and jobs for contrastive learning training.
    """
    
    def __init__(self, enable_augmentation: bool = True, max_resume_views: int = 5, 
                 max_job_views: int = 5, fallback_on_failure: bool = True):
        """
        Initialize ViewGenerator with augmentation classes.
        
        Args:
            enable_augmentation: Whether to enable view augmentation
            max_resume_views: Maximum number of resume views to generate
            max_job_views: Maximum number of job views to generate
            fallback_on_failure: Whether to fallback to original data on augmentation failure
        """
        self.enable_augmentation = enable_augmentation
        self.max_resume_views = max_resume_views
        self.max_job_views = max_job_views
        self.fallback_on_failure = fallback_on_failure
        self.resume_augmenter = None
        self.job_augmenter = None
        
        if self.enable_augmentation:
            try:
                if ResumeViewAugmentation is not None:
                    self.resume_augmenter = ResumeViewAugmentation()
                    logging.info("ResumeViewAugmentation initialized successfully")
                else:
                    logging.warning("ResumeViewAugmentation not available")
                    
                if JobViewAugmentation is not None:
                    self.job_augmenter = JobViewAugmentation()
                    logging.info("JobViewAugmentation initialized successfully")
                else:
                    logging.warning("JobViewAugmentation not available")
                
                # Disable augmentation if neither augmenter is available
                if self.resume_augmenter is None and self.job_augmenter is None:
                    logging.warning("No augmentation classes available, disabling augmentation")
                    self.enable_augmentation = False
                    
            except Exception as e:
                logging.error(f"Error initializing augmentation classes: {e}")
                self.enable_augmentation = False
        
        logging.info(f"ViewGenerator initialized with augmentation {'enabled' if self.enable_augmentation else 'disabled'}")
    
    @classmethod
    def from_config(cls, config) -> 'ViewGenerator':
        """
        Create ViewGenerator from TrainingConfig.
        
        Args:
            config: TrainingConfig object or dict with configuration
            
        Returns:
            ViewGenerator instance configured according to the config
        """
        if hasattr(config, 'use_view_augmentation'):
            # TrainingConfig object
            return cls(
                enable_augmentation=config.use_view_augmentation,
                max_resume_views=config.max_resume_views,
                max_job_views=config.max_job_views,
                fallback_on_failure=config.fallback_on_augmentation_failure
            )
        elif isinstance(config, dict):
            # Dictionary config
            return cls(
                enable_augmentation=config.get('use_view_augmentation', True),
                max_resume_views=config.get('max_resume_views', 5),
                max_job_views=config.get('max_job_views', 5),
                fallback_on_failure=config.get('fallback_on_augmentation_failure', True)
            )
        else:
            raise ValueError("Config must be a TrainingConfig object or dictionary")
    
    def generate_resume_views(self, resume: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple views of a resume.
        
        Args:
            resume: Original resume data
            
        Returns:
            List of resume views including original + augmented versions
        """
        if not self.enable_augmentation or self.resume_augmenter is None:
            return [resume]
        
        try:
            views = self.resume_augmenter.generate_resume_views(resume)
            
            # Limit number of views according to configuration
            if len(views) > self.max_resume_views:
                # Keep original (first) and sample from the rest
                original = views[0]
                augmented = views[1:]
                import random
                sampled_augmented = random.sample(augmented, min(len(augmented), self.max_resume_views - 1))
                views = [original] + sampled_augmented
            
            logging.debug(f"Generated {len(views)} resume views (limited to {self.max_resume_views})")
            return views
        except Exception as e:
            logging.error(f"Error generating resume views: {e}")
            # Fallback to original data if configured to do so
            if self.fallback_on_failure:
                return [resume]
            else:
                raise
    
    def generate_job_views(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple views of a job description.
        
        Args:
            job: Original job data
            
        Returns:
            List of job views including original + augmented versions
        """
        if not self.enable_augmentation or self.job_augmenter is None:
            return [job]
        
        try:
            views = self.job_augmenter.generate_job_views(job)
            
            # Limit number of views according to configuration
            if len(views) > self.max_job_views:
                # Keep original (first) and sample from the rest
                original = views[0]
                augmented = views[1:]
                import random
                sampled_augmented = random.sample(augmented, min(len(augmented), self.max_job_views - 1))
                views = [original] + sampled_augmented
            
            logging.debug(f"Generated {len(views)} job views (limited to {self.max_job_views})")
            return views
        except Exception as e:
            logging.error(f"Error generating job views: {e}")
            # Fallback to original data if configured to do so
            if self.fallback_on_failure:
                return [job]
            else:
                raise
    
    def get_view_combinations(self, resume_views: List[Dict[str, Any]], 
                            job_views: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generate all combinations of resume views and job views.
        
        Args:
            resume_views: List of resume views
            job_views: List of job views
            
        Returns:
            List of (resume_view, job_view) tuples representing all combinations
        """
        try:
            combinations = list(product(resume_views, job_views))
            logging.debug(f"Generated {len(combinations)} view combinations from {len(resume_views)} resume views and {len(job_views)} job views")
            return combinations
        except Exception as e:
            logging.error(f"Error generating view combinations: {e}")
            # Fallback to original combination
            return [(resume_views[0] if resume_views else {}, job_views[0] if job_views else {})]
    
    def generate_sample_views(self, resume: Dict[str, Any], job: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generate all view combinations for a resume-job pair.
        
        Args:
            resume: Original resume data
            job: Original job data
            
        Returns:
            List of (resume_view, job_view) tuples
        """
        try:
            resume_views = self.generate_resume_views(resume)
            job_views = self.generate_job_views(job)
            combinations = self.get_view_combinations(resume_views, job_views)
            
            logging.debug(f"Generated {len(combinations)} total view combinations for sample")
            return combinations
            
        except Exception as e:
            logging.error(f"Error generating sample views: {e}")
            # Fallback to original data
            return [(resume, job)]
    
    def compute_view_metadata(self, original_resume: Dict[str, Any], 
                            original_job: Dict[str, Any],
                            resume_view: Dict[str, Any], 
                            job_view: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute metadata about the view transformation.
        
        Args:
            original_resume: Original resume data
            original_job: Original job data
            resume_view: Resume view
            job_view: Job view
            
        Returns:
            Dictionary containing view metadata
        """
        metadata = {
            'is_original_resume': resume_view is original_resume,
            'is_original_job': job_view is original_job,
            'resume_view_type': 'original' if resume_view is original_resume else 'augmented',
            'job_view_type': 'original' if job_view is original_job else 'augmented'
        }
        
        # Add career distance if job augmenter is available
        if self.job_augmenter is not None and hasattr(self.job_augmenter, 'compute_job_career_distances'):
            try:
                if job_view is not original_job:
                    distances = self.job_augmenter.compute_job_career_distances(original_job, [job_view])
                    metadata['job_career_distance'] = distances.get('view_1', 0.0)
                else:
                    metadata['job_career_distance'] = 0.0
            except Exception as e:
                logging.warning(f"Could not compute job career distance: {e}")
                metadata['job_career_distance'] = 0.0
        
        return metadata
    
    def is_augmentation_available(self) -> bool:
        """
        Check if view augmentation is available and enabled.
        
        Returns:
            True if augmentation is available and enabled
        """
        return (self.enable_augmentation and 
                self.resume_augmenter is not None and 
                self.job_augmenter is not None)
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the augmentation system.
        
        Returns:
            Dictionary containing augmentation statistics
        """
        return {
            'augmentation_enabled': self.enable_augmentation,
            'resume_augmenter_available': self.resume_augmenter is not None,
            'job_augmenter_available': self.job_augmenter is not None,
            'fully_functional': self.is_augmentation_available()
        }
    
    def disable_augmentation(self) -> None:
        """Disable view augmentation."""
        self.enable_augmentation = False
        logging.info("View augmentation disabled")
    
    def enable_augmentation_if_available(self) -> None:
        """Enable view augmentation if augmenters are available."""
        if self.resume_augmenter is not None and self.job_augmenter is not None:
            self.enable_augmentation = True
            logging.info("View augmentation enabled")
        else:
            logging.warning("Cannot enable augmentation - augmenters not available")
    
    def update_config(self, enable_augmentation: bool = None, max_resume_views: int = None,
                     max_job_views: int = None, fallback_on_failure: bool = None) -> None:
        """
        Update ViewGenerator configuration.
        
        Args:
            enable_augmentation: Whether to enable view augmentation
            max_resume_views: Maximum number of resume views to generate
            max_job_views: Maximum number of job views to generate
            fallback_on_failure: Whether to fallback to original data on failure
        """
        if enable_augmentation is not None:
            if enable_augmentation and not self.is_augmentation_available():
                logging.warning("Cannot enable augmentation - augmenters not available")
            else:
                self.enable_augmentation = enable_augmentation
                
        if max_resume_views is not None:
            if max_resume_views <= 0:
                raise ValueError("max_resume_views must be positive")
            self.max_resume_views = max_resume_views
            
        if max_job_views is not None:
            if max_job_views <= 0:
                raise ValueError("max_job_views must be positive")
            self.max_job_views = max_job_views
            
        if fallback_on_failure is not None:
            self.fallback_on_failure = fallback_on_failure
            
        logging.info(f"ViewGenerator config updated: augmentation={self.enable_augmentation}, "
                    f"max_resume_views={self.max_resume_views}, max_job_views={self.max_job_views}, "
                    f"fallback={self.fallback_on_failure}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current ViewGenerator configuration.
        
        Returns:
            Dictionary containing current configuration
        """
        return {
            'enable_augmentation': self.enable_augmentation,
            'max_resume_views': self.max_resume_views,
            'max_job_views': self.max_job_views,
            'fallback_on_failure': self.fallback_on_failure
        }
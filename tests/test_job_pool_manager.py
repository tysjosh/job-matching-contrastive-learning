"""Tests for JobPoolManager."""

import pytest
import json
from unittest.mock import patch, MagicMock

from augmentation.job_pool_manager import JobPoolManager

class TestJobPoolManager:
    """Wraps all tests for JobPoolManager in a class to be compatible with conftest.py"""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        dataset_content = [
            {"job_applicant_id": 1, "resume": {"role": "Senior Software Engineer"}, "job": {"title": "Senior Software Engineer", "description": {"original": "Develop backend systems"}, "experience_level": "senior", "skills": ["Python", "Django"]}},
            {"job_applicant_id": 2, "resume": {"role": "Junior Data Scientist"}, "job": {"title": "Junior Data Scientist", "description": {"original": "Analyze data"}, "experience_level": "entry", "skills": ["Python", "Pandas"]}},
            {"job_applicant_id": 3, "resume": {"role": "Marketing Manager"}, "job": {"title": "Marketing Manager", "description": {"original": "Run campaigns"}, "experience_level": "mid", "skills": ["SEO", "SEM"]}},
            {"job_applicant_id": 4, "resume": {"role": "Senior Software Engineer"}, "job": {"title": "Senior Java Developer", "description": {"original": "Develop enterprise apps"}, "experience_level": "senior", "skills": ["Java", "Spring"]}}
        ]
        dataset_file = tmp_path / "dataset.jsonl"
        with open(dataset_file, 'w') as f:
            for record in dataset_content:
                f.write(json.dumps(record) + '\n')
        return str(dataset_file)

    @pytest.fixture
    def mock_career_graph(self):
        return {
            'occupations': {
                'Senior Software Engineer': {},
                'Junior Data Scientist': {},
                'Marketing Manager': {},
                'Senior Java Developer': {}
            }
        }

    @pytest.fixture
    def mock_esco_components(self):
        """Mock all external ESCO-related dependencies."""
        # Patch where the objects are defined, not where they are used.
        with patch('augmentation.progression_constraints.ESCODomainLoader') as mock_domain_loader, \
             patch('augmentation.progression_constraints.ESCODataLoader') as mock_data_loader, \
             patch('augmentation.progression_constraints.ESCOSkillNormalizer') as mock_skill_normalizer:
            
            # Mock Domain Loader
            domain_loader_instance = mock_domain_loader.return_value
            def get_domain(text):
                if 'software' in text.lower() or 'java' in text.lower():
                    return 'technology'
                if 'data' in text.lower():
                    return 'data_science'
                if 'marketing' in text.lower():
                    return 'marketing'
                return 'other'
            domain_loader_instance.get_domain_for_text.side_effect = get_domain
            domain_loader_instance.load_career_domains.return_value = {'technology', 'data_science', 'marketing'}

            # Mock other loaders
            mock_data_loader.return_value.load_occupation_skill_relations.return_value = {}
            mock_skill_normalizer.return_value.normalize_skill = lambda x: x.lower()

            yield {
                'domain_loader': mock_domain_loader,
                'data_loader': mock_data_loader,
                'skill_normalizer': mock_skill_normalizer
            }

    @pytest.fixture
    def job_pool_manager(self, mock_dataset, mock_career_graph, mock_esco_components):
        """Fixture to create a JobPoolManager instance with mocked dependencies."""
        return JobPoolManager(
            dataset_path=mock_dataset,
            esco_config_file='dummy_config.json',
            career_graph=mock_career_graph
        )

    # --- Test Cases ---

    def test_initialization_and_job_loading(self, job_pool_manager):
        """Test that the manager initializes and loads jobs correctly."""
        assert job_pool_manager is not None
        assert len(job_pool_manager.all_jobs) == 4
        assert job_pool_manager.jobs_by_domain['technology'][0]['title'] == 'Senior Software Engineer'
        assert len(job_pool_manager.jobs_by_domain['technology']) == 2

    def test_extract_occupation(self, job_pool_manager):
        """Test the occupation extraction logic."""
        job = {"title": "Senior Software Engineer", "description": {"original": "Develop backend systems"}}
        occupation = job_pool_manager._extract_job_occupation_esco(job)
        assert occupation == 'Senior Software Engineer'

    def test_select_cross_domain_job(self, job_pool_manager):
        """Test selecting a job from a different domain."""
        original_job = {"title": "Senior Software Engineer", "description": {"original": "Develop backend systems"}}
        
        # Run multiple times to account for randomness
        for _ in range(10):
            cross_domain_job = job_pool_manager.select_cross_domain_job(original_job)
            assert cross_domain_job is not None
            original_domain = job_pool_manager._extract_job_domain_esco(original_job)
            new_domain = job_pool_manager._extract_job_domain_esco(cross_domain_job)
            assert new_domain != original_domain
            assert new_domain in ['data_science', 'marketing']

    def test_select_skill_mismatch_job(self, job_pool_manager):
        """Test selecting a job with minimal skill overlap."""
        original_job = {"title": "Senior Software Engineer", "skills": ["Python", "Django"]}
        
        # Mock the skill extraction to control the test
        def mock_skill_extraction(job):
            return job.get('skills', [])
        
        job_pool_manager._extract_job_skills_esco = MagicMock(side_effect=mock_skill_extraction)

        mismatch_job = job_pool_manager.select_skill_mismatch_job(original_job)
        
        # The job with the least overlap is the Marketing Manager or Java Developer
        assert mismatch_job['title'] in ['Marketing Manager', 'Senior Java Developer']


    # --- Tests for the New `find_matching_job` Method ---

    def test_find_matching_job_success(self, job_pool_manager):
        """Test finding a job that exists."""
        # We need to manually run _organize_jobs_with_esco to populate jobs_by_occupation
        job_pool_manager._organize_jobs_with_esco()

        occupation = 'Senior Software Engineer'
        level = 'senior'
        
        matching_job = job_pool_manager.find_matching_job(occupation, level)
        
        assert matching_job is not None
        assert job_pool_manager._extract_job_occupation_esco(matching_job) == occupation
        assert matching_job.get('experience_level') == level
        assert matching_job['title'] == 'Senior Software Engineer'

    def test_find_matching_job_level_mismatch(self, job_pool_manager):
        """Test finding a job where occupation exists but level does not."""
        job_pool_manager._organize_jobs_with_esco()

        occupation = 'Senior Software Engineer'
        level = 'entry'  # This level does not exist for this occupation in mock data
        
        matching_job = job_pool_manager.find_matching_job(occupation, level)
        
        assert matching_job is None

    def test_find_matching_job_occupation_not_found(self, job_pool_manager):
        """Test finding a job for an occupation that does not exist."""
        job_pool_manager._organize_jobs_with_esco()

        occupation = 'Blockchain Developer'
        level = 'senior'
        
        matching_job = job_pool_manager.find_matching_job(occupation, level)
        
        assert matching_job is None

    def test_find_matching_job_empty_pool(self, job_pool_manager):
        """Test behavior when the jobs_by_occupation pool is empty."""
        job_pool_manager.jobs_by_occupation = {}
        
        matching_job = job_pool_manager.find_matching_job('Senior Software Engineer', 'senior')
        
        assert matching_job is None

    # --- Test Orchestrator Integration (Conceptual) ---
    @patch('augmentation.job_pool_manager.JobPoolManager')
    def test_orchestrator_calls_find_matching_job(self, MockJobPoolManager, mock_career_graph):
        """Check if the orchestrator correctly uses the new JPM method."""
        from augmentation.dataset_augmentation_orchestrator import DatasetAugmentationOrchestrator

        # Setup mocks
        mock_jpm_instance = MockJobPoolManager.return_value
        orchestrator = DatasetAugmentationOrchestrator(esco_skills_hierarchy={}, career_graph=mock_career_graph)
        orchestrator.job_pool_manager = mock_jpm_instance

        # A "fixable" mismatch (same occupation, different level)
        original_record = {
            'label': 0,
            'job_applicant_id': 100,
            'resume': {'role': 'Software Engineer', 'experience_level': 'mid'},
            'job': {'title': 'Senior Software Engineer', 'experience_level': 'senior'}
        }

        # Mock the JPM methods
        mock_jpm_instance._extract_job_occupation_esco.return_value = 'Software Engineer'
        mock_jpm_instance.find_matching_job.return_value = {'title': 'Mid-Level Software Engineer', 'experience_level': 'mid'}

        # Execute
        orchestrator._create_role_match_correction_positive(original_record, '100')

        # Assert
        mock_jpm_instance.find_matching_job.assert_called_once_with(occupation='Software Engineer', level='mid')

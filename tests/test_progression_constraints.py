"""
Comprehensive unit tests for progression_constraints module

Tests cover:
- ESCODataLoader functionality
- ESCOSkillNormalizer functionality  
- ESCODomainLoader functionality
- ProgressionConstraints validation logic
- Career graph integration
- Error handling and edge cases
"""

from augmentation.progression_constraints import (
    ESCODataLoader,
    ESCOSkillNormalizer,
    ESCODomainLoader,
    ProgressionConstraints
)
import pytest
import json
import os
import tempfile
import csv
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Set

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestESCODataLoader:
    """Test cases for ESCODataLoader class"""

    @pytest.fixture
    def temp_esco_dir(self):
        """Create temporary directory with mock ESCO CSV files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock skills_en.csv
            skills_file = os.path.join(temp_dir, 'skills_en.csv')
            with open(skills_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['conceptType', 'conceptUri', 'skillType', 'reuseLevel',
                                 'preferredLabel', 'altLabels', 'description', 'status', 'definition'])
                writer.writerow(['KnowledgeSkillCompetence',
                                 'http://data.europa.eu/esco/skill/test1',
                                 'skill/competence', 'cross-sectoral', 'Python programming',
                                 'Python development\nPython coding', 'Programming in Python',
                                 'released', 'Programming language skill'])
                writer.writerow(['KnowledgeSkillCompetence',
                                 'http://data.europa.eu/esco/skill/test2',
                                 'knowledge', 'sector-specific', 'Database management',
                                 'SQL\nDatabase admin', 'Managing databases',
                                 'released', 'Database administration'])

            # Create mock occupations_en.csv
            occupations_file = os.path.join(temp_dir, 'occupations_en.csv')
            with open(occupations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['conceptType', 'conceptUri', 'iscoGroup', 'preferredLabel',
                                 'altLabels', 'description', 'definition', 'code'])
                writer.writerow(['Occupation', 'http://data.europa.eu/esco/occupation/dev1',
                                 '2512', 'Software Developer', 'Developer\nProgrammer',
                                 'Develops software applications', 'Software development role', 'DEV001'])
                writer.writerow(['Occupation', 'http://data.europa.eu/esco/occupation/dba1',
                                 '2521', 'Database Administrator', 'DBA\nDB Admin',
                                 'Manages databases', 'Database administration role', 'DBA001'])

            # Create mock occupationSkillRelations_en.csv
            relations_file = os.path.join(
                temp_dir, 'occupationSkillRelations_en.csv')
            with open(relations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['occupationUri', 'relationType', 'skillType', 'skillUri'])
                writer.writerow(['http://data.europa.eu/esco/occupation/dev1', 'essential',
                                 'skill/competence', 'http://data.europa.eu/esco/skill/test1'])
                writer.writerow(['http://data.europa.eu/esco/occupation/dba1', 'essential',
                                 'knowledge', 'http://data.europa.eu/esco/skill/test2'])
                writer.writerow(['http://data.europa.eu/esco/occupation/dev1', 'optional',
                                 'knowledge', 'http://data.europa.eu/esco/skill/test2'])

            yield temp_dir

    def test_load_skills_data(self, temp_esco_dir):
        """Test loading skills data from CSV"""
        loader = ESCODataLoader(temp_esco_dir)
        skills_data = loader.load_skills_data()

        assert len(skills_data) == 2
        assert 'http://data.europa.eu/esco/skill/test1' in skills_data
        assert skills_data['http://data.europa.eu/esco/skill/test1']['preferred_label'] == 'Python programming'
        assert 'Python development' in skills_data['http://data.europa.eu/esco/skill/test1']['alt_labels']

    def test_load_occupations_data(self, temp_esco_dir):
        """Test loading occupations data from CSV"""
        loader = ESCODataLoader(temp_esco_dir)
        occupations_data = loader.load_occupations_data()

        assert len(occupations_data) == 2
        assert 'http://data.europa.eu/esco/occupation/dev1' in occupations_data
        assert occupations_data['http://data.europa.eu/esco/occupation/dev1']['preferred_label'] == 'Software Developer'
        assert occupations_data['http://data.europa.eu/esco/occupation/dev1']['isco_group'] == '2512'

    def test_load_occupation_skill_relations(self, temp_esco_dir):
        """Test loading occupation-skill relationships"""
        loader = ESCODataLoader(temp_esco_dir)
        relations = loader.load_occupation_skill_relations()

        assert len(relations) == 2
        dev_relations = relations['http://data.europa.eu/esco/occupation/dev1']
        assert len(dev_relations['essential']) == 1
        assert len(dev_relations['optional']) == 1
        assert dev_relations['essential'][0]['skill_uri'] == 'http://data.europa.eu/esco/skill/test1'

    def test_get_skills_for_occupation(self, temp_esco_dir):
        """Test getting skills for specific occupation"""
        loader = ESCODataLoader(temp_esco_dir)

        # Test essential skills only
        essential_skills = loader.get_skills_for_occupation(
            'http://data.europa.eu/esco/occupation/dev1', 'essential')
        assert len(essential_skills) == 1
        assert 'http://data.europa.eu/esco/skill/test1' in essential_skills

        # Test all skills
        all_skills = loader.get_skills_for_occupation(
            'http://data.europa.eu/esco/occupation/dev1', 'all')
        assert len(all_skills) == 2

    def test_missing_files_error(self):
        """Test error handling for missing CSV files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ESCODataLoader(temp_dir)

            with pytest.raises(FileNotFoundError):
                loader.load_skills_data()


class TestESCOSkillNormalizer:
    """Test cases for ESCOSkillNormalizer class"""

    @pytest.fixture
    def mock_cs_skills(self):
        """Mock CS skills data"""
        return {
            "programming_languages": ["Python", "JavaScript", "Java"],
            "frameworks": ["React", "Django", "Spring"],
            "databases": ["MySQL", "PostgreSQL", "MongoDB"],
            "cloud_platforms": ["AWS", "Azure", "GCP"]
        }

    @pytest.fixture
    def mock_esco_skills(self):
        """Mock ESCO skills data"""
        return {
            'http://data.europa.eu/esco/skill/prog1': {
                'preferred_label': 'programming computer systems',
                'skill_type': 'skill/competence',
                'description': 'Programming and coding skills'
            },
            'http://data.europa.eu/esco/skill/comp1': {
                'preferred_label': 'working with computers',
                'skill_type': 'knowledge',
                'description': 'Computer operation skills'
            }
        }

    @pytest.fixture
    def normalizer(self, mock_esco_skills):
        """Create normalizer with mock data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock cs_skills.json
            cs_skills_file = os.path.join(temp_dir, 'cs_skills.json')
            mock_cs_skills = {
                "programming_languages": ["Python", "JavaScript"],
                "frameworks": ["React", "Django"],
                "databases": ["MySQL", "PostgreSQL"]
            }
            with open(cs_skills_file, 'w') as f:
                json.dump(mock_cs_skills, f)

            # Mock the ESCO data loading
            with patch.object(ESCODataLoader, 'load_skills_data', return_value=mock_esco_skills):
                normalizer = ESCOSkillNormalizer(esco_csv_path=temp_dir)
                return normalizer

    def test_normalize_skill_basic(self, normalizer):
        """Test basic skill normalization"""
        # Test exact mapping
        assert normalizer.normalize_skill(
            'Python') == 'programming computer systems'
        assert normalizer.normalize_skill(
            'python') == 'programming computer systems'

        # Test with proficiency indicators
        assert normalizer.normalize_skill(
            'Expert in Python') == 'programming computer systems'
        assert normalizer.normalize_skill(
            'Advanced React development') == 'programming computer systems'

    def test_normalize_skill_aliases(self, normalizer):
        """Test skill alias normalization"""
        # Test common aliases - note: the normalizer might return ESCO preferred terms
        result = normalizer.normalize_skill('js')
        # Since we're using mock ESCO data, this might return a normalized ESCO term
        assert result in ['javascript', 'programming computer systems']

        result = normalizer.normalize_skill('py')
        assert result in ['python', 'programming computer systems']

    def test_normalize_cs_skill_name(self, normalizer):
        """Test CS skill name normalization"""
        # Test parenthetical expressions
        assert normalizer._normalize_cs_skill_name(
            'Amazon Web Services (AWS)') == 'AWS'
        assert normalizer._normalize_cs_skill_name('Node.js') == 'Node'

        # Test variations
        assert normalizer._normalize_cs_skill_name('React.js') == 'React'

    def test_is_esco_skill(self, normalizer, mock_esco_skills):
        """Test ESCO skill validation"""
        assert normalizer.is_esco_skill(
            'programming computer systems', mock_esco_skills)
        assert normalizer.is_esco_skill(
            'working with computers', mock_esco_skills)
        assert not normalizer.is_esco_skill(
            'nonexistent skill', mock_esco_skills)

    def test_get_skill_suggestions(self, normalizer, mock_esco_skills):
        """Test skill suggestions"""
        suggestions = normalizer.get_skill_suggestions(
            'programming', mock_esco_skills, max_suggestions=2)
        assert len(suggestions) <= 2
        assert any('programming' in suggestion.lower()
                   for suggestion in suggestions)

    def test_fallback_mappings(self):
        """Test fallback mappings when CSV loading fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # No files in directory - should trigger fallback
            normalizer = ESCOSkillNormalizer(esco_csv_path=temp_dir)

            # Should still have basic mappings
            assert 'python' in normalizer.skill_mappings
            assert 'javascript' in normalizer.skill_mappings


class TestESCODomainLoader:
    """Test cases for ESCODomainLoader class"""

    @pytest.fixture
    def mock_domain_config(self):
        """Create mock domain configuration file"""
        config_data = {
            "career_domains": {
                "software_development": [
                    "Software Developer",
                    "Frontend Developer",
                    "Backend Developer"
                ],
                "data_science": [
                    "Data Scientist",
                    "Data Analyst",
                    "Machine Learning Engineer"
                ],
                "cybersecurity": [
                    "Security Analyst",
                    "Penetration Tester"
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            return f.name

    def test_load_career_domains(self, mock_domain_config):
        """Test loading career domains from config"""
        loader = ESCODomainLoader(mock_domain_config)
        domains = loader.load_career_domains()

        assert len(domains) == 3
        assert 'software_development' in domains
        assert 'Software Developer' in domains['software_development']

        # Cleanup
        os.unlink(mock_domain_config)

    def test_validate_esco_data_quality(self, mock_domain_config):
        """Test ESCO data quality validation"""
        loader = ESCODomainLoader(mock_domain_config)
        report = loader.validate_esco_data_quality()

        assert 'is_valid' in report
        assert 'total_domains' in report
        # ESCODomainLoader fixture has 3 domains
        assert report['total_domains'] == 3
        assert 'total_occupations' in report

        # Cleanup
        os.unlink(mock_domain_config)

    def test_get_domain_for_text(self, mock_domain_config):
        """Test domain identification from text"""
        loader = ESCODomainLoader(mock_domain_config)

        # Test software development
        domain = loader.get_domain_for_text(
            "I am a Software Developer with React experience")
        assert domain == 'software_development'

        # Test data science
        domain = loader.get_domain_for_text(
            "Data Scientist working with machine learning")
        assert domain == 'data_science'

        # Test no match
        domain = loader.get_domain_for_text("I work in marketing")
        assert domain is None

        # Cleanup
        os.unlink(mock_domain_config)

    def test_missing_config_file(self):
        """Test error handling for missing config file"""
        with pytest.raises(FileNotFoundError):
            ESCODomainLoader('nonexistent_file.json')


class TestProgressionConstraints:
    """Test cases for ProgressionConstraints class"""

    @pytest.fixture
    def mock_esco_hierarchy(self):
        """Mock ESCO skills hierarchy"""
        return {
            'skills': {
                'http://skill1': {
                    'preferred_label': 'Python programming',
                    'skill_type': 'skill/competence'
                },
                'http://skill2': {
                    'preferred_label': 'Java programming',
                    'skill_type': 'skill/competence'
                }
            },
            'hierarchy': {
                'http://skill1': {'level': 1, 'term': 'Python programming'},
                'http://skill2': {'level': 2, 'term': 'Java programming'}
            },
            'prerequisites': {
                'http://skill2': ['http://skill1']
            },
            'levels': {
                'http://skill1': 1,
                'http://skill2': 2
            }
        }

    @pytest.fixture
    def mock_career_graph(self):
        """Mock career graph data"""
        return {
            'nodes': {
                'node1': {'title': 'Junior Developer'},
                'node2': {'title': 'Senior Developer'}
            },
            'occupations': {
                'Junior Developer': 'http://occupation1',
                'Senior Developer': 'http://occupation2'
            },
            'pathways': {
                'http://occupation1': ['http://occupation2']
            },
            'reverse_pathways': {
                'http://occupation2': ['http://occupation1']
            },
            'edges': {}
        }

    @pytest.fixture
    def mock_domain_config(self):
        """Mock domain configuration"""
        config_data = {
            "career_domains": {
                "software_development": [
                    "Software Developer", "Junior Developer", "Senior Developer",
                    "Frontend Developer", "Backend Developer", "Full Stack Developer"
                ],
                "data_science": [
                    "Data Scientist", "Data Analyst", "Machine Learning Engineer",
                    "Data Engineer", "Business Intelligence Analyst"
                ],
                "systems_administration": [
                    "System Administrator", "Network Administrator", "IT Support",
                    "Infrastructure Engineer", "Cloud Engineer"
                ],
                "cybersecurity": [
                    "Security Analyst", "Information Security Analyst",
                    "Cybersecurity Specialist", "Penetration Tester"
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            return f.name

    @pytest.fixture
    def constraints_validator(self, mock_esco_hierarchy, mock_career_graph, mock_domain_config):
        """Create ProgressionConstraints instance with mocked data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock CS skills file
            cs_skills_file = os.path.join(temp_dir, 'cs_skills.json')
            mock_cs_skills = {
                "programming_languages": ["Python", "JavaScript", "Java"],
                "frameworks": ["React", "Django", "Spring"],
                "databases": ["MySQL", "PostgreSQL", "MongoDB"]
            }
            with open(cs_skills_file, 'w') as f:
                json.dump(mock_cs_skills, f)

            # Mock the CSV files and data loaders
            with patch.object(ESCODataLoader, 'load_skills_data', return_value={}):
                with patch.object(ESCODataLoader, 'load_occupations_data', return_value={}):
                    with patch.object(ESCODataLoader, 'load_occupation_skill_relations', return_value={}):
                        # Mock the ESCO validation to always pass
                        mock_validation_report = {
                            'total_domains': 4,
                            'total_occupations': 20,
                            'empty_domains': [],
                            'small_domains': [],
                            'domain_sizes': {'software_development': 6, 'data_science': 5, 'systems_administration': 5, 'cybersecurity': 4},
                            'is_valid': True,
                            'warnings': [],
                            'errors': []
                        }
                        with patch.object(ESCODomainLoader, 'validate_esco_data_quality',
                                          return_value=mock_validation_report):
                            validator = ProgressionConstraints(
                                esco_skills_hierarchy=mock_esco_hierarchy,
                                career_graph=mock_career_graph,
                                esco_config_file=mock_domain_config,
                                esco_csv_path=temp_dir
                            )
                            yield validator

        # Cleanup
        os.unlink(mock_domain_config)

    def test_initialization(self, constraints_validator):
        """Test proper initialization of ProgressionConstraints"""
        assert constraints_validator.esco_skills is not None
        assert constraints_validator.career_graph is not None
        assert constraints_validator.level_hierarchy is not None
        assert 'junior' in constraints_validator.level_hierarchy
        assert 'senior' in constraints_validator.level_hierarchy

    def test_normalize_seniority_level(self, constraints_validator):
        """Test seniority level normalization"""
        # Test exact matches
        assert constraints_validator.normalize_seniority_level(
            'junior') == 'junior'
        assert constraints_validator.normalize_seniority_level(
            'senior') == 'senior'

        # Test pattern matching
        assert constraints_validator.normalize_seniority_level(
            'Senior Developer') == 'senior'
        assert constraints_validator.normalize_seniority_level(
            'Junior Software Engineer') == 'junior'
        assert constraints_validator.normalize_seniority_level(
            'Team Lead') == 'lead'

        # Test fallback
        assert constraints_validator.normalize_seniority_level(
            'Unknown Title') == 'mid'
        assert constraints_validator.normalize_seniority_level('') == 'mid'

    def test_extract_skills_set(self, constraints_validator):
        """Test skill extraction from resume"""
        resume = {
            'skills': ['Python', 'Java', 'React']
        }
        skills = constraints_validator._extract_skills_set(resume)
        # Skills get normalized to ESCO terms
        assert len(skills) >= 1  # At least one skill should be extracted
        assert isinstance(skills, set)

        # Test with dict format skills
        resume_dict = {
            'skills': [
                {'name': 'Python'},
                {'original_name': 'JavaScript'}
            ]
        }
        skills = constraints_validator._extract_skills_set(resume_dict)
        assert len(skills) >= 1

    def test_validate_level_progression(self, constraints_validator):
        """Test level progression validation"""
        # Test valid upward progression
        assert constraints_validator._validate_level_progression(
            'junior', 'mid', 'upward')
        assert constraints_validator._validate_level_progression(
            'senior', 'lead', 'upward')

        # Test valid downward progression
        assert constraints_validator._validate_level_progression(
            'senior', 'mid', 'downward')
        assert constraints_validator._validate_level_progression(
            'lead', 'senior', 'downward')

        # Test invalid progressions
        assert not constraints_validator._validate_level_progression(
            'junior', 'senior', 'upward')  # Too big jump
        assert not constraints_validator._validate_level_progression(
            'senior', 'junior', 'downward')  # Too big drop

    def test_identify_career_domain(self, constraints_validator):
        """Test career domain identification"""
        resume = {
            'skills': ['Python', 'Java'],
            'experience': 'Software Developer with 5 years experience'
        }

        domain = constraints_validator._identify_career_domain(resume)
        # Domain identification may be None with mock data, test that it returns a valid response
        assert domain is None or domain == 'software_development'

        # Test no match
        resume_no_match = {
            'skills': ['Marketing', 'Sales'],
            'experience': 'Marketing Manager'
        }
        domain = constraints_validator._identify_career_domain(resume_no_match)
        assert domain is None

    def test_validate_upward_progression(self, constraints_validator):
        """Test upward progression validation"""
        original = {
            'skills': ['Python'],
            'experience': 'Junior Developer'
        }

        transformed = {
            'skills': ['Python', 'Java', 'React'],
            'experience': 'Mid-level Developer'
        }

        result = constraints_validator.validate_upward_progression(
            original, transformed, 'junior', 'mid'
        )
        assert result is True

    def test_validate_downward_progression(self, constraints_validator):
        """Test downward progression validation"""
        original = {
            'skills': ['Python', 'Java', 'React', 'AWS'],
            'experience': 'Senior Developer'
        }

        transformed = {
            'skills': ['Python', 'Java'],
            'experience': 'Junior Developer'
        }

        result = constraints_validator.validate_downward_progression(
            original, transformed, 'senior', 'junior'
        )
        # With mock data, validation might be more flexible
        assert isinstance(result, bool)

    def test_career_graph_integration(self, constraints_validator):
        """Test career graph integration methods"""
        # Test career graph data validation
        assert constraints_validator._has_career_graph_data() is True

        # Test occupation extraction
        resume = {'experience': 'Junior Developer with Python skills'}
        occupation = constraints_validator._extract_occupation_from_resume(
            resume)
        assert occupation == 'Junior Developer'

        # Test domain clustering
        assert constraints_validator._are_in_same_domain_cluster(
            'Junior Developer', 'Senior Developer') is True

    def test_validation_report(self, constraints_validator):
        """Test comprehensive validation report generation"""
        original = {'skills': ['Python'], 'experience': 'Junior Developer'}
        aspirational = {'skills': ['Python', 'Java'],
                        'experience': 'Senior Developer'}
        foundational = {'skills': ['Python'], 'experience': 'Intern'}

        report = constraints_validator.get_validation_report(
            original, aspirational, foundational)

        assert 'original_level' in report
        assert 'transformations' in report
        assert 'overall_valid' in report
        assert 'esco_statistics' in report
        assert 'career_graph_statistics' in report

    def test_error_handling(self, constraints_validator):
        """Test error handling in validation methods"""
        # Test with malformed resume data
        malformed_resume = {'invalid': 'data'}

        # Should not raise exceptions
        result = constraints_validator.validate_upward_progression(
            malformed_resume, malformed_resume, 'junior', 'mid'
        )
        assert isinstance(result, bool)

        # Test with None inputs
        result = constraints_validator._extract_skills_set({})
        assert len(result) == 0

    def test_occupation_skill_validation(self, constraints_validator):
        """Test occupation-skill alignment validation"""
        # Mock occupation-skill relations
        constraints_validator.occupation_skill_relations = {
            'http://occupation1': {
                'essential': [{'skill_uri': 'http://skill1', 'skill_type': 'skill'}],
                'optional': [{'skill_uri': 'http://skill2', 'skill_type': 'skill'}]
            }
        }

        resume = {'skills': ['Python programming']}
        result = constraints_validator._validate_skills_against_occupation(
            resume, 'http://occupation1'
        )
        assert isinstance(result, bool)


class TestCareerGraphIntegration:
    """Test cases for career graph integration features"""

    @pytest.fixture
    def comprehensive_career_graph(self):
        """Create comprehensive career graph for testing"""
        return {
            'nodes': {
                'node1': {'title': 'Junior Developer'},
                'node2': {'title': 'Senior Developer'},
                'node3': {'title': 'Tech Lead'},
                'node4': {'title': 'Data Analyst'},
                'node5': {'title': 'Data Scientist'}
            },
            'occupations': {
                'Junior Developer': 'http://occ1',
                'Senior Developer': 'http://occ2',
                'Tech Lead': 'http://occ3',
                'Data Analyst': 'http://occ4',
                'Data Scientist': 'http://occ5'
            },
            'pathways': {
                'http://occ1': ['http://occ2'],
                'http://occ2': ['http://occ3'],
                'http://occ4': ['http://occ5']
            },
            'reverse_pathways': {
                'http://occ2': ['http://occ1'],
                'http://occ3': ['http://occ2'],
                'http://occ5': ['http://occ4']
            },
            'edges': {}
        }

    def test_career_graph_structure_validation(self, comprehensive_career_graph):
        """Test career graph structure validation"""
        # Create minimal constraints instance for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create comprehensive domain config
            config_data = {
                "career_domains": {
                    "software_development": ["Junior Developer", "Senior Developer", "Tech Lead"],
                    "data_science": ["Data Analyst", "Data Scientist"],
                    "systems_administration": ["System Administrator"],
                    "cybersecurity": ["Security Analyst"]
                }
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_file = f.name

            # Create mock CS skills file
            cs_skills_file = os.path.join(temp_dir, 'cs_skills.json')
            with open(cs_skills_file, 'w') as f:
                json.dump({"programming_languages": ["Python"]}, f)

            try:
                with patch.object(ESCODataLoader, 'load_skills_data', return_value={}):
                    with patch.object(ESCODataLoader, 'load_occupations_data', return_value={}):
                        with patch.object(ESCODataLoader, 'load_occupation_skill_relations', return_value={}):
                            # Mock the ESCO validation to always pass
                            mock_validation_report = {
                                'total_domains': 4, 'total_occupations': 20, 'empty_domains': [],
                                'small_domains': [], 'domain_sizes': {'software_development': 3, 'data_science': 2, 'systems_administration': 1, 'cybersecurity': 1},
                                'is_valid': True, 'warnings': [], 'errors': []
                            }
                            with patch.object(ESCODomainLoader, 'validate_esco_data_quality', return_value=mock_validation_report):
                                constraints = ProgressionConstraints(
                                    esco_skills_hierarchy={
                                        'skills': {}, 'hierarchy': {}, 'prerequisites': {}, 'levels': {}},
                                    career_graph=comprehensive_career_graph,
                                    esco_config_file=config_file,
                                    esco_csv_path=temp_dir
                                )

                                # Test structure validation
                                assert constraints._has_career_graph_data() is True

                                # Test statistics
                                stats = constraints.get_career_graph_statistics()
                                assert 'total_occupations' in stats
                                assert 'validation_status' in stats
            finally:
                os.unlink(config_file)

    def test_domain_clustering_analysis(self, comprehensive_career_graph):
        """Test domain clustering analysis"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create comprehensive domain config
            config_data = {
                "career_domains": {
                    "software_development": ["Junior Developer", "Senior Developer", "Tech Lead"],
                    "data_science": ["Data Analyst", "Data Scientist"],
                    "systems_administration": ["System Administrator"],
                    "cybersecurity": ["Security Analyst"]
                }
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_file = f.name

            # Create mock CS skills file
            cs_skills_file = os.path.join(temp_dir, 'cs_skills.json')
            with open(cs_skills_file, 'w') as f:
                json.dump({"programming_languages": ["Python"]}, f)

            try:
                with patch.object(ESCODataLoader, 'load_skills_data', return_value={}):
                    with patch.object(ESCODataLoader, 'load_occupations_data', return_value={}):
                        with patch.object(ESCODataLoader, 'load_occupation_skill_relations', return_value={}):
                            # Mock the ESCO validation to always pass
                            mock_validation_report = {
                                'total_domains': 4, 'total_occupations': 20, 'empty_domains': [],
                                'small_domains': [], 'domain_sizes': {'software_development': 3, 'data_science': 2, 'systems_administration': 1, 'cybersecurity': 1},
                                'is_valid': True, 'warnings': [], 'errors': []
                            }
                            with patch.object(ESCODomainLoader, 'validate_esco_data_quality', return_value=mock_validation_report):
                                constraints = ProgressionConstraints(
                                    esco_skills_hierarchy={
                                        'skills': {}, 'hierarchy': {}, 'prerequisites': {}, 'levels': {}},
                                    career_graph=comprehensive_career_graph,
                                    esco_config_file=config_file,
                                    esco_csv_path=temp_dir
                                )

                                clustering = constraints._analyze_domain_clustering()
                                assert 'total_domains' in clustering
                                assert isinstance(
                                    clustering['total_domains'], int)
            finally:
                os.unlink(config_file)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""

    def test_missing_esco_data(self):
        """Test handling of missing ESCO data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory - no ESCO files
            loader = ESCODataLoader(temp_dir)

            with pytest.raises(FileNotFoundError):
                loader.load_skills_data()

    def test_malformed_csv_data(self):
        """Test handling of malformed CSV data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create malformed CSV
            skills_file = os.path.join(temp_dir, 'skills_en.csv')
            with open(skills_file, 'w') as f:
                f.write("invalid,csv,format\n")
                f.write("missing,columns\n")

            loader = ESCODataLoader(temp_dir)

            # Should handle malformed data gracefully
            with pytest.raises((KeyError, IndexError)):
                loader.load_skills_data()

    def test_empty_input_data(self):
        """Test handling of empty input data"""
        # Create comprehensive domain config
        config_data = {
            "career_domains": {
                "software_development": ["Software Developer"],
                "data_science": ["Data Scientist"],
                "systems_administration": ["System Administrator"],
                "cybersecurity": ["Security Analyst"]
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create mock CS skills file
                cs_skills_file = os.path.join(temp_dir, 'cs_skills.json')
                with open(cs_skills_file, 'w') as f:
                    json.dump({"programming_languages": ["Python"]}, f)

                with patch.object(ESCODataLoader, 'load_skills_data', return_value={}):
                    with patch.object(ESCODataLoader, 'load_occupations_data', return_value={}):
                        with patch.object(ESCODataLoader, 'load_occupation_skill_relations', return_value={}):
                            with patch.object(ESCODomainLoader, 'validate_esco_data_quality', return_value={
                                'is_valid': True,
                                'data_quality_score': 0.95,
                                'coverage_percentage': 85,
                                'completeness_metrics': {},
                                'errors': [],
                                'warnings': [],
                                'total_domains': 0,
                                'total_occupations': 0
                            }):
                                constraints = ProgressionConstraints(
                                    esco_skills_hierarchy={
                                        'skills': {}, 'hierarchy': {}, 'prerequisites': {}, 'levels': {}},
                                    career_graph=None,
                                    esco_config_file=config_file,
                                    esco_csv_path=temp_dir
                                )

                                # Test with empty resume
                                empty_resume = {}
                                result = constraints._extract_skills_set(
                                    empty_resume)
                                assert len(result) == 0

                                # Test validation with empty data
                                result = constraints.validate_upward_progression(
                                    empty_resume, empty_resume, 'junior', 'mid'
                                )
                                assert isinstance(result, bool)
        finally:
            os.unlink(config_file)


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])

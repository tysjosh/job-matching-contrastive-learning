"""
Pytest configuration and shared fixtures for progression_constraints tests
"""

import pytest
import os
import sys
import tempfile
import json
import csv
from unittest.mock import patch, MagicMock

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def mock_esco_data_dir():
    """Create a temporary directory with complete mock ESCO data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create comprehensive mock skills_en.csv
        skills_file = os.path.join(temp_dir, 'skills_en.csv')
        with open(skills_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['conceptType', 'conceptUri', 'skillType', 'reuseLevel',
                             'preferredLabel', 'altLabels', 'description', 'status', 'definition'])

            # Programming skills
            writer.writerow(['KnowledgeSkillCompetence',
                             'http://data.europa.eu/esco/skill/S1.1.1',
                             'skill/competence', 'cross-sectoral', 'programming computer systems',
                             'Python programming\nJava programming\nJavaScript programming',
                             'Programming and developing computer applications',
                             'released', 'Skill in programming languages'])

            # Database skills
            writer.writerow(['KnowledgeSkillCompetence',
                             'http://data.europa.eu/esco/skill/S1.2.1',
                             'knowledge', 'sector-specific', 'database management',
                             'SQL\nMySQL\nPostgreSQL\nMongoDB',
                             'Managing and maintaining databases',
                             'released', 'Database administration and management'])

            # Web development
            writer.writerow(['KnowledgeSkillCompetence',
                             'http://data.europa.eu/esco/skill/S1.3.1',
                             'skill/competence', 'cross-sectoral', 'web development',
                             'HTML\nCSS\nReact\nAngular\nVue.js',
                             'Developing web applications and websites',
                             'released', 'Web application development'])

            # Cloud platforms
            writer.writerow(['KnowledgeSkillCompetence',
                             'http://data.europa.eu/esco/skill/S1.4.1',
                             'skill/competence', 'sector-specific', 'cloud computing',
                             'AWS\nAzure\nGoogle Cloud\nDocker\nKubernetes',
                             'Working with cloud platforms and services',
                             'released', 'Cloud computing and deployment'])

        # Create comprehensive mock occupations_en.csv
        occupations_file = os.path.join(temp_dir, 'occupations_en.csv')
        with open(occupations_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['conceptType', 'conceptUri', 'iscoGroup', 'preferredLabel',
                             'altLabels', 'description', 'definition', 'code'])

            # Software development occupations
            writer.writerow(['Occupation', 'http://data.europa.eu/esco/occupation/O1.1.1',
                             '2512', 'software developer',
                             'Software Engineer\nDeveloper\nProgrammer\nSoftware Programmer',
                             'Develops and maintains software applications using programming languages',
                             'Professional who designs, codes, tests and maintains software', 'SD001'])

            writer.writerow(['Occupation', 'http://data.europa.eu/esco/occupation/O1.1.2',
                             '2512', 'web developer',
                             'Web Designer\nFrontend Developer\nBackend Developer\nFull Stack Developer',
                             'Develops websites and web applications using web technologies',
                             'Professional specializing in web-based software development', 'WD001'])

            # Data science occupations
            writer.writerow(['Occupation', 'http://data.europa.eu/esco/occupation/O1.2.1',
                             '2521', 'data scientist',
                             'Data Analyst\nMachine Learning Engineer\nData Engineer',
                             'Analyzes large datasets to extract insights using statistical methods',
                             'Professional who uses data science techniques for analysis', 'DS001'])

            # Database occupations
            writer.writerow(['Occupation', 'http://data.europa.eu/esco/occupation/O1.3.1',
                             '2521', 'database administrator',
                             'DBA\nDatabase Manager\nDatabase Engineer',
                             'Manages and maintains database systems and ensures data integrity',
                             'Professional responsible for database management and optimization', 'DBA001'])

            # Leadership roles
            writer.writerow(['Occupation', 'http://data.europa.eu/esco/occupation/O1.4.1',
                             '1330', 'software development manager',
                             'Tech Lead\nEngineering Manager\nDevelopment Lead\nTeam Lead',
                             'Manages software development teams and oversees project delivery',
                             'Management role overseeing software development processes', 'SDM001'])

        # Create comprehensive mock occupationSkillRelations_en.csv
        relations_file = os.path.join(
            temp_dir, 'occupationSkillRelations_en.csv')
        with open(relations_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['occupationUri', 'relationType', 'skillType', 'skillUri'])

            # Software developer skills
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.1.1', 'essential',
                             'skill/competence', 'http://data.europa.eu/esco/skill/S1.1.1'])
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.1.1', 'optional',
                             'knowledge', 'http://data.europa.eu/esco/skill/S1.2.1'])

            # Web developer skills
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.1.2', 'essential',
                             'skill/competence', 'http://data.europa.eu/esco/skill/S1.3.1'])
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.1.2', 'essential',
                             'skill/competence', 'http://data.europa.eu/esco/skill/S1.1.1'])

            # Data scientist skills
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.2.1', 'essential',
                             'knowledge', 'http://data.europa.eu/esco/skill/S1.1.1'])
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.2.1', 'essential',
                             'knowledge', 'http://data.europa.eu/esco/skill/S1.2.1'])

            # Database administrator skills
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.3.1', 'essential',
                             'knowledge', 'http://data.europa.eu/esco/skill/S1.2.1'])

            # Manager skills (includes programming plus management)
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.4.1', 'optional',
                             'skill/competence', 'http://data.europa.eu/esco/skill/S1.1.1'])
            writer.writerow(['http://data.europa.eu/esco/occupation/O1.4.1', 'essential',
                             'skill/competence', 'http://data.europa.eu/esco/skill/S1.4.1'])

        yield temp_dir


@pytest.fixture
def mock_cs_skills_file():
    """Create mock CS skills JSON file"""
    cs_skills_data = {
        "programming_languages": [
            "Python", "JavaScript", "Java", "C++", "C#", "TypeScript", "Go", "Rust", "PHP", "Ruby"
        ],
        "frameworks": [
            "React", "Angular", "Vue.js", "Django", "Flask", "Spring", "Express.js", "Laravel", "Rails"
        ],
        "databases": [
            "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "SQLite", "Oracle", "SQL Server"
        ],
        "cloud_platforms": [
            "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform", "Ansible"
        ],
        "tools": [
            "Git", "Jenkins", "Docker", "VS Code", "IntelliJ", "Postman", "Jira", "Slack"
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(cs_skills_data, f)
        return f.name


@pytest.fixture
def mock_domain_config_file():
    """Create mock career domain configuration file"""
    domain_config = {
        "career_domains": {
            "software_development": [
                "Software Developer", "Software Engineer", "Programmer", "Software Programmer",
                "Web Developer", "Frontend Developer", "Backend Developer", "Full Stack Developer",
                "Mobile Developer", "Game Developer"
            ],
            "data_science": [
                "Data Scientist", "Data Analyst", "Machine Learning Engineer", "Data Engineer",
                "Business Intelligence Analyst", "Research Scientist", "Statistician"
            ],
            "cybersecurity": [
                "Security Analyst", "Information Security Analyst", "Cybersecurity Specialist",
                "Penetration Tester", "Security Engineer", "Security Consultant"
            ],
            "database_administration": [
                "Database Administrator", "DBA", "Database Manager", "Database Engineer",
                "Database Analyst", "Data Warehouse Administrator"
            ],
            "it_management": [
                "Software Development Manager", "Tech Lead", "Engineering Manager",
                "Development Lead", "Team Lead", "CTO", "VP Engineering"
            ],
            "devops": [
                "DevOps Engineer", "Site Reliability Engineer", "Infrastructure Engineer",
                "Cloud Engineer", "Platform Engineer", "Release Engineer"
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(domain_config, f)
        return f.name


@pytest.fixture
def mock_career_graph():
    """Create comprehensive mock career graph"""
    return {
        'nodes': {
            'node1': {'title': 'Junior Developer', 'level': 1, 'domain': 'software_development'},
            'node2': {'title': 'Software Developer', 'level': 2, 'domain': 'software_development'},
            'node3': {'title': 'Senior Developer', 'level': 3, 'domain': 'software_development'},
            'node4': {'title': 'Tech Lead', 'level': 4, 'domain': 'software_development'},
            'node5': {'title': 'Engineering Manager', 'level': 5, 'domain': 'it_management'},
            'node6': {'title': 'Data Analyst', 'level': 2, 'domain': 'data_science'},
            'node7': {'title': 'Data Scientist', 'level': 3, 'domain': 'data_science'},
            'node8': {'title': 'Senior Data Scientist', 'level': 4, 'domain': 'data_science'},
            'node9': {'title': 'Database Administrator', 'level': 2, 'domain': 'database_administration'},
            'node10': {'title': 'DevOps Engineer', 'level': 3, 'domain': 'devops'}
        },
        'occupations': {
            'Junior Developer': 'http://data.europa.eu/esco/occupation/O1.1.0',
            'Software Developer': 'http://data.europa.eu/esco/occupation/O1.1.1',
            'Senior Developer': 'http://data.europa.eu/esco/occupation/O1.1.3',
            'Tech Lead': 'http://data.europa.eu/esco/occupation/O1.4.1',
            'Engineering Manager': 'http://data.europa.eu/esco/occupation/O1.4.2',
            'Data Analyst': 'http://data.europa.eu/esco/occupation/O1.2.0',
            'Data Scientist': 'http://data.europa.eu/esco/occupation/O1.2.1',
            'Senior Data Scientist': 'http://data.europa.eu/esco/occupation/O1.2.3',
            'Database Administrator': 'http://data.europa.eu/esco/occupation/O1.3.1',
            'DevOps Engineer': 'http://data.europa.eu/esco/occupation/O1.5.1'
        },
        'pathways': {
            'http://data.europa.eu/esco/occupation/O1.1.0': ['http://data.europa.eu/esco/occupation/O1.1.1'],
            'http://data.europa.eu/esco/occupation/O1.1.1': ['http://data.europa.eu/esco/occupation/O1.1.3'],
            'http://data.europa.eu/esco/occupation/O1.1.3': ['http://data.europa.eu/esco/occupation/O1.4.1'],
            'http://data.europa.eu/esco/occupation/O1.4.1': ['http://data.europa.eu/esco/occupation/O1.4.2'],
            'http://data.europa.eu/esco/occupation/O1.2.0': ['http://data.europa.eu/esco/occupation/O1.2.1'],
            'http://data.europa.eu/esco/occupation/O1.2.1': ['http://data.europa.eu/esco/occupation/O1.2.3']
        },
        'reverse_pathways': {
            'http://data.europa.eu/esco/occupation/O1.1.1': ['http://data.europa.eu/esco/occupation/O1.1.0'],
            'http://data.europa.eu/esco/occupation/O1.1.3': ['http://data.europa.eu/esco/occupation/O1.1.1'],
            'http://data.europa.eu/esco/occupation/O1.4.1': ['http://data.europa.eu/esco/occupation/O1.1.3'],
            'http://data.europa.eu/esco/occupation/O1.4.2': ['http://data.europa.eu/esco/occupation/O1.4.1'],
            'http://data.europa.eu/esco/occupation/O1.2.1': ['http://data.europa.eu/esco/occupation/O1.2.0'],
            'http://data.europa.eu/esco/occupation/O1.2.3': ['http://data.europa.eu/esco/occupation/O1.2.1']
        },
        'edges': {}
    }


@pytest.fixture
def mock_esco_skills_hierarchy():
    """Create mock ESCO skills hierarchy"""
    return {
        'skills': {
            'http://data.europa.eu/esco/skill/S1.1.1': {
                'preferred_label': 'programming computer systems',
                'skill_type': 'skill/competence',
                'description': 'Programming and developing computer applications'
            },
            'http://data.europa.eu/esco/skill/S1.2.1': {
                'preferred_label': 'database management',
                'skill_type': 'knowledge',
                'description': 'Managing and maintaining databases'
            },
            'http://data.europa.eu/esco/skill/S1.3.1': {
                'preferred_label': 'web development',
                'skill_type': 'skill/competence',
                'description': 'Developing web applications and websites'
            },
            'http://data.europa.eu/esco/skill/S1.4.1': {
                'preferred_label': 'cloud computing',
                'skill_type': 'skill/competence',
                'description': 'Working with cloud platforms and services'
            }
        },
        'hierarchy': {
            'http://data.europa.eu/esco/skill/S1.1.1': {'level': 1, 'term': 'programming computer systems'},
            'http://data.europa.eu/esco/skill/S1.2.1': {'level': 2, 'term': 'database management'},
            'http://data.europa.eu/esco/skill/S1.3.1': {'level': 2, 'term': 'web development'},
            'http://data.europa.eu/esco/skill/S1.4.1': {'level': 3, 'term': 'cloud computing'}
        },
        'prerequisites': {
            'http://data.europa.eu/esco/skill/S1.2.1': ['http://data.europa.eu/esco/skill/S1.1.1'],
            'http://data.europa.eu/esco/skill/S1.3.1': ['http://data.europa.eu/esco/skill/S1.1.1'],
            'http://data.europa.eu/esco/skill/S1.4.1': ['http://data.europa.eu/esco/skill/S1.1.1', 'http://data.europa.eu/esco/skill/S1.3.1']
        },
        'levels': {
            'http://data.europa.eu/esco/skill/S1.1.1': 1,
            'http://data.europa.eu/esco/skill/S1.2.1': 2,
            'http://data.europa.eu/esco/skill/S1.3.1': 2,
            'http://data.europa.eu/esco/skill/S1.4.1': 3
        }
    }


@pytest.fixture
def sample_resumes():
    """Create sample resume data for testing"""
    return {
        'junior_developer': {
            'skills': ['Python', 'Git', 'HTML', 'CSS'],
            'experience': 'Junior Software Developer with 1 year experience in Python development',
            'education': 'Bachelor of Computer Science',
            'years_experience': 1
        },
        'mid_developer': {
            'skills': ['Python', 'JavaScript', 'React', 'Django', 'PostgreSQL', 'Git'],
            'experience': 'Software Developer with 3 years experience building web applications',
            'education': 'Bachelor of Computer Science',
            'years_experience': 3
        },
        'senior_developer': {
            'skills': ['Python', 'JavaScript', 'React', 'Django', 'PostgreSQL', 'AWS', 'Docker', 'Kubernetes'],
            'experience': 'Senior Software Developer with 7 years experience leading development teams',
            'education': 'Master of Computer Science',
            'years_experience': 7
        },
        'data_scientist': {
            'skills': ['Python', 'R', 'TensorFlow', 'Pandas', 'NumPy', 'SQL', 'Machine Learning'],
            'experience': 'Data Scientist with 4 years experience in machine learning and analytics',
            'education': 'PhD in Statistics',
            'years_experience': 4
        },
        'dba': {
            'skills': ['SQL', 'Oracle', 'MySQL', 'PostgreSQL', 'Database Tuning', 'Backup Recovery'],
            'experience': 'Database Administrator with 5 years experience managing enterprise databases',
            'education': 'Bachelor of Information Systems',
            'years_experience': 5
        }
    }


# Mock cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files(request):
    """Automatically cleanup temporary files after tests"""
    temp_files = []

    def add_temp_file(filepath):
        temp_files.append(filepath)

    request.instance.add_temp_file = add_temp_file if hasattr(
        request, 'instance') else lambda x: None

    yield

    # Cleanup
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except Exception:
            pass  # Ignore cleanup errors

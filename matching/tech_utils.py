"""
Tech-specialized utilities for CS/IT/Tech role matching
Optimized for the 4-step matching pipeline with tech-specific enhancements
"""

import re
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import logging

@dataclass
class TechSkillMatch:
    skill: str
    category: str
    confidence: float
    match_type: str  # 'exact', 'pattern', 'alias'

@dataclass
class TechStackSummary:
    primary_languages: List[str]
    frameworks: List[str]
    databases: List[str]
    cloud_platforms: List[str]
    specialization: str
    experience_indicators: Dict[str, int]

class TechSkillExtractor:
    """Specialized skill extractor for CS/IT/Tech roles only"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Tech-specific skill hierarchies and patterns
        self.tech_skill_patterns = self._load_comprehensive_tech_patterns()
        self.framework_aliases = self._load_framework_aliases()
        self.tech_stack_categories = self._define_tech_categories()
        
        # Pre-compiled regex for performance
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in patterns]
            for category, patterns in self.tech_skill_patterns.items()
        }
        
        # Skill confidence weights
        self.confidence_weights = {
            'exact_match': 1.0,
            'pattern_match': 0.9,
            'alias_match': 0.8,
            'context_bonus': 0.1
        }
    
    def _load_comprehensive_tech_patterns(self) -> Dict[str, List[str]]:
        """Comprehensive tech skill patterns for CS/IT roles"""
        return {
            'programming_languages': [
                r'\b(?:python|java|javascript|typescript|c\+\+|c#|go|rust|scala|kotlin|swift|ruby|php|perl|r\b|matlab)\b',
                r'\bjs\b', r'\bts\b', r'\bc\+\+\b', r'\bc#\b', r'\bjava\b(?!\s*script)',
                r'\b(?:objective-c|visual\s?basic|fortran|cobol|assembly|bash|powershell)\b'
            ],
            'web_frameworks': [
                r'\b(?:react|angular|vue|svelte|next\.?js|nuxt\.?js|gatsby)\b',
                r'\b(?:express|fastapi|django|flask|spring|laravel|rails|asp\.net)\b',
                r'\b(?:node\.?js|nest\.?js|ember\.?js|backbone\.?js)\b',
                r'\b(?:bootstrap|tailwind|material-ui|ant\s?design)\b'
            ],
            'databases': [
                r'\b(?:postgresql|mysql|mongodb|redis|elasticsearch|cassandra|dynamodb|influxdb)\b',
                r'\b(?:postgres|sql\s?server|oracle|sqlite|neo4j|couchdb|mariadb)\b',
                r'\b(?:nosql|sql|rdbms|graph\s?database|time\s?series)\b'
            ],
            'cloud_platforms': [
                r'\b(?:aws|amazon\s?web\s?services|azure|microsoft\s?azure|gcp|google\s?cloud)\b',
                r'\b(?:docker|kubernetes|k8s|terraform|ansible|jenkins|gitlab)\b',
                r'\b(?:serverless|lambda|cloud\s?functions|fargate|ecs|eks)\b',
                r'\b(?:s3|ec2|rds|cloudformation|cloud\s?watch)\b'
            ],
            'devops_tools': [
                r'\b(?:jenkins|gitlab\s?ci|github\s?actions|circleci|travis\s?ci|bamboo)\b',
                r'\b(?:git|svn|mercurial|perforce|bitbucket)\b',
                r'\b(?:nginx|apache|haproxy|load\s?balancer|cdn|cloudflare)\b',
                r'\b(?:prometheus|grafana|elk\s?stack|datadog|new\s?relic)\b'
            ],
            'data_science': [
                r'\b(?:pandas|numpy|scikit-learn|tensorflow|pytorch|keras|xgboost)\b',
                r'\b(?:jupyter|anaconda|matplotlib|seaborn|plotly|tableau)\b',
                r'\b(?:machine\s?learning|deep\s?learning|ai|nlp|computer\s?vision|mlops)\b',
                r'\b(?:spark|hadoop|kafka|airflow|dbt|snowflake)\b'
            ],
            'mobile_development': [
                r'\b(?:react\s?native|flutter|xamarin|ionic|cordova)\b',
                r'\b(?:ios|android|swift|kotlin|objective-c|java)\b',
                r'\b(?:xcode|android\s?studio|app\s?store|play\s?store)\b'
            ],
            'testing_tools': [
                r'\b(?:jest|pytest|junit|selenium|cypress|playwright|mocha)\b',
                r'\b(?:unit\s?testing|integration\s?testing|e2e|tdd|bdd)\b',
                r'\b(?:postman|insomnia|swagger|api\s?testing)\b'
            ],
            'methodologies': [
                r'\b(?:agile|scrum|kanban|lean|waterfall|devops|ci\/cd)\b',
                r'\b(?:microservices|monolith|soa|event\s?driven|rest|graphql)\b',
                r'\b(?:design\s?patterns|solid\s?principles|clean\s?code|refactoring)\b'
            ]
        }
    
    def _load_framework_aliases(self) -> Dict[str, str]:
        """Map common aliases to standard framework names"""
        return {
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'py': 'Python',
            'c++': 'C++',
            'c#': 'C#',
            'node': 'Node.js',
            'vue.js': 'Vue.js',
            'react.js': 'React',
            'k8s': 'Kubernetes',
            'tf': 'TensorFlow',
            'sklearn': 'Scikit-learn',
            'postgres': 'PostgreSQL',
            'mongo': 'MongoDB',
            'aws': 'Amazon Web Services',
            'gcp': 'Google Cloud Platform'
        }
    
    def _define_tech_categories(self) -> Dict[str, float]:
        """Define importance weights for different tech skill categories"""
        return {
            'programming_languages': 0.3,
            'web_frameworks': 0.25,
            'databases': 0.15,
            'cloud_platforms': 0.15,
            'devops_tools': 0.1,
            'data_science': 0.2,
            'mobile_development': 0.15,
            'testing_tools': 0.1,
            'methodologies': 0.1
        }
    
    def extract_tech_skills(self, text: str) -> Dict[str, List[TechSkillMatch]]:
        """Extract categorized tech skills with confidence scoring"""
        text_lower = text.lower()
        extracted_skills = {}
        
        for category, patterns in self.compiled_patterns.items():
            category_skills = []
            
            for pattern in patterns:
                matches = pattern.findall(text_lower)
                for match in matches:
                    # Normalize the skill name
                    normalized_skill = self._normalize_tech_skill(match)
                    if normalized_skill:
                        confidence = self._calculate_skill_confidence(
                            normalized_skill, text_lower, category
                        )
                        
                        skill_match = TechSkillMatch(
                            skill=normalized_skill,
                            category=category,
                            confidence=confidence,
                            match_type='pattern'
                        )
                        category_skills.append(skill_match)
            
            # Remove duplicates and sort by confidence
            unique_skills = self._deduplicate_skills(category_skills)
            extracted_skills[category] = sorted(
                unique_skills, 
                key=lambda x: x.confidence, 
                reverse=True
            )
        
        return extracted_skills
    
    def _normalize_tech_skill(self, skill: str) -> Optional[str]:
        """Normalize tech skill names using aliases and proper casing"""
        skill_lower = skill.lower().strip()
        
        # Check aliases first
        if skill_lower in self.framework_aliases:
            return self.framework_aliases[skill_lower]
        
        # Apply proper casing for known technologies
        tech_casing = {
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'python': 'Python',
            'java': 'Java',
            'react': 'React',
            'angular': 'Angular',
            'vue': 'Vue.js',
            'node.js': 'Node.js',
            'express': 'Express.js',
            'django': 'Django',
            'flask': 'Flask',
            'postgresql': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'mysql': 'MySQL',
            'redis': 'Redis',
            'docker': 'Docker',
            'kubernetes': 'Kubernetes',
            'tensorflow': 'TensorFlow',
            'pytorch': 'PyTorch',
            'aws': 'Amazon Web Services',
            'azure': 'Microsoft Azure',
            'gcp': 'Google Cloud Platform'
        }
        
        return tech_casing.get(skill_lower, skill.title())
    
    def _calculate_skill_confidence(self, skill: str, text: str, category: str) -> float:
        """Calculate confidence score for extracted skill"""
        base_confidence = 0.8
        
        # Context bonuses
        context_bonus = 0.0
        
        # Check for experience mentions
        if re.search(rf'{re.escape(skill.lower())}.*?(\d+)[\s\-]*years?', text):
            context_bonus += 0.1
        
        # Check for proficiency mentions
        proficiency_terms = ['expert', 'advanced', 'proficient', 'experienced']
        for term in proficiency_terms:
            if term in text:
                if skill.lower() in text[max(0, text.find(term)-50):text.find(term)+50]:
                    context_bonus += 0.1
                    break
        
        # Category-based confidence adjustment
        category_weight = self.tech_stack_categories.get(category, 1.0)
        
        return min(base_confidence + context_bonus * category_weight, 1.0)
    
    def _deduplicate_skills(self, skills: List[TechSkillMatch]) -> List[TechSkillMatch]:
        """Remove duplicate skills, keeping the highest confidence match"""
        skill_map = {}
        for skill in skills:
            if skill.skill not in skill_map or skill.confidence > skill_map[skill.skill].confidence:
                skill_map[skill.skill] = skill
        
        return list(skill_map.values())
    
    def summarize_tech_stack(self, skills: Dict[str, List[TechSkillMatch]]) -> TechStackSummary:
        """Create a summary of the candidate's tech stack"""
        
        # Extract top skills from each category
        primary_languages = [s.skill for s in skills.get('programming_languages', [])[:3]]
        frameworks = [s.skill for s in skills.get('web_frameworks', [])[:3]]
        databases = [s.skill for s in skills.get('databases', [])[:2]]
        cloud_platforms = [s.skill for s in skills.get('cloud_platforms', [])[:2]]
        
        # Determine specialization
        specialization = self._detect_specialization(skills)
        
        # Extract experience indicators
        experience_indicators = self._extract_experience_indicators(skills)
        
        return TechStackSummary(
            primary_languages=primary_languages,
            frameworks=frameworks,
            databases=databases,
            cloud_platforms=cloud_platforms,
            specialization=specialization,
            experience_indicators=experience_indicators
        )
    
    def _detect_specialization(self, skills: Dict[str, List[TechSkillMatch]]) -> str:
        """Detect the primary tech specialization"""
        frontend_skills = ['React', 'Angular', 'Vue.js', 'JavaScript', 'TypeScript']
        backend_skills = ['Django', 'Flask', 'Spring', 'Express.js', 'FastAPI']
        data_skills = ['Python', 'TensorFlow', 'PyTorch', 'Pandas', 'Scikit-learn']
        mobile_skills = ['React Native', 'Flutter', 'Swift', 'Kotlin', 'iOS', 'Android']
        devops_skills = ['Docker', 'Kubernetes', 'AWS', 'Jenkins', 'Terraform']
        
        all_skills = []
        for skill_list in skills.values():
            all_skills.extend([s.skill for s in skill_list])
        
        specializations = {
            'Frontend Developer': len(set(all_skills) & set(frontend_skills)),
            'Backend Developer': len(set(all_skills) & set(backend_skills)),
            'Data Scientist': len(set(all_skills) & set(data_skills)),
            'Mobile Developer': len(set(all_skills) & set(mobile_skills)),
            'DevOps Engineer': len(set(all_skills) & set(devops_skills)),
            'Full-Stack Developer': (len(set(all_skills) & set(frontend_skills)) > 0 and 
                                   len(set(all_skills) & set(backend_skills)) > 0)
        }
        
        # Convert boolean to int for Full-Stack Developer
        specializations['Full-Stack Developer'] = int(specializations['Full-Stack Developer'])
        
        return max(specializations.items(), key=lambda x: x[1])[0]
    
    def _extract_experience_indicators(self, skills: Dict[str, List[TechSkillMatch]]) -> Dict[str, int]:
        """Extract experience indicators from skills"""
        indicators = {
            'total_skills': sum(len(skill_list) for skill_list in skills.values()),
            'high_confidence_skills': sum(1 for skill_list in skills.values() 
                                        for skill in skill_list if skill.confidence > 0.9),
            'diverse_categories': len([cat for cat, skill_list in skills.items() if skill_list])
        }
        
        return indicators


class TechCareerLevelClassifier:
    """Specialized career classifier for tech roles with precise level mapping"""
    
    def __init__(self, career_graph, config: Dict = None):
        self.career_graph = career_graph
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.tech_career_levels = self._define_tech_career_levels()
        self.seniority_indicators = self._define_seniority_indicators()
    
    def _define_tech_career_levels(self) -> Dict[str, Dict]:
        """Define precise tech career progression levels"""
        return {
            'intern': {
                'keywords': ['intern', 'trainee', 'apprentice', 'co-op', 'student'],
                'years_range': (0, 1),
                'graph_level': 'Junior Developer',
                'complexity_threshold': 0
            },
            'junior': {
                'keywords': ['junior', 'entry', 'graduate', 'associate', 'level 1', 'jr'],
                'years_range': (0, 3),
                'graph_level': 'Junior Developer',
                'complexity_threshold': 2
            },
            'mid': {
                'keywords': ['developer', 'engineer', 'software engineer', 'level 2', 'level 3'],
                'years_range': (2, 6),
                'graph_level': 'Software Developer',
                'complexity_threshold': 4
            },
            'senior': {
                'keywords': ['senior', 'sr.', 'sr', 'level 4', 'level 5'],
                'years_range': (4, 10),
                'graph_level': 'Senior Developer',
                'complexity_threshold': 6
            },
            'staff': {
                'keywords': ['staff', 'level 6', 'level 7'],
                'years_range': (7, 15),
                'graph_level': 'Staff Engineer',
                'complexity_threshold': 8
            },
            'principal': {
                'keywords': ['principal', 'architect', 'distinguished', 'level 8', 'level 9'],
                'years_range': (10, float('inf')),
                'graph_level': 'Principal Engineer',
                'complexity_threshold': 10
            },
            'lead': {
                'keywords': ['lead', 'team lead', 'tech lead', 'technical lead', 'engineering lead'],
                'years_range': (5, float('inf')),
                'graph_level': 'Tech Lead',
                'complexity_threshold': 7
            },
            'manager': {
                'keywords': ['manager', 'engineering manager', 'team manager', 'development manager'],
                'years_range': (6, float('inf')),
                'graph_level': 'Engineering Manager',
                'complexity_threshold': 6
            }
        }
    
    def _define_seniority_indicators(self) -> Dict[str, List[str]]:
        """Define indicators of technical seniority and complexity"""
        return {
            'architecture': ['architecture', 'design patterns', 'system design', 'scalability', 'microservices', 'distributed systems'],
            'leadership': ['mentoring', 'code review', 'technical guidance', 'team lead', 'project lead'],
            'advanced_tech': ['kubernetes', 'terraform', 'aws certified', 'cloud architect', 'machine learning', 'ai'],
            'project_scale': ['enterprise', 'large-scale', 'high-traffic', 'production systems', 'critical systems'],
            'methodologies': ['agile coach', 'scrum master', 'technical strategy', 'roadmap', 'cross-functional']
        }
    
    def classify_tech_career_level(self, text: str) -> str:
        """Classify career level specifically for tech roles"""
        text_lower = text.lower()
        
        # Step 1: Direct role title matching (highest priority)
        for level, config in self.tech_career_levels.items():
            for keyword in config['keywords']:
                if keyword in text_lower:
                    return config['graph_level']
        
        # Step 2: Years of experience mapping
        years = self._extract_years_experience(text_lower)
        if years is not None:
            level_by_years = self._map_years_to_tech_level(years)
            if level_by_years:
                return level_by_years
        
        # Step 3: Technology complexity analysis
        complexity_score = self._calculate_complexity_score(text_lower)
        return self._map_complexity_to_level(complexity_score)
    
    def _extract_years_experience(self, text: str) -> Optional[int]:
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)[\s\-]*(?:to|\+)?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\+?\s*years?\s+(?:experience|exp)',
            r'(?:experience|exp).*?(\d+)[\s\-]*years?',
            r'(\d+)[\s\-]*years?\s+(?:in|with|of)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        
        return None
    
    def _map_years_to_tech_level(self, years: int) -> Optional[str]:
        """Map years of experience to tech career levels"""
        for level, config in self.tech_career_levels.items():
            min_years, max_years = config['years_range']
            if min_years <= years < max_years:
                return config['graph_level']
        
        return None
    
    def _calculate_complexity_score(self, text: str) -> int:
        """Calculate technical complexity and seniority score"""
        complexity_score = 0
        
        for category, indicators in self.seniority_indicators.items():
            category_score = sum(1 for indicator in indicators if indicator in text)
            
            # Weight different categories
            weights = {
                'architecture': 3,
                'leadership': 2,
                'advanced_tech': 2,
                'project_scale': 2,
                'methodologies': 1
            }
            
            complexity_score += category_score * weights.get(category, 1)
        
        return complexity_score
    
    def _map_complexity_to_level(self, complexity_score: int) -> str:
        """Map complexity score to career level"""
        if complexity_score >= 12:
            return 'Principal Engineer'
        elif complexity_score >= 8:
            return 'Staff Engineer'
        elif complexity_score >= 5:
            return 'Senior Developer'
        elif complexity_score >= 2:
            return 'Software Developer'
        else:
            return 'Junior Developer'
    
    def get_career_level_confidence(self, text: str, predicted_level: str) -> float:
        """Get confidence score for the predicted career level"""
        text_lower = text.lower()
        
        # Check for explicit mentions
        for level, config in self.tech_career_levels.items():
            if config['graph_level'] == predicted_level:
                explicit_mentions = sum(1 for keyword in config['keywords'] if keyword in text_lower)
                if explicit_mentions > 0:
                    return 0.9
        
        # Check for years alignment
        years = self._extract_years_experience(text_lower)
        if years:
            expected_level = self._map_years_to_tech_level(years)
            if expected_level == predicted_level:
                return 0.8
        
        # Check for complexity alignment
        complexity_score = self._calculate_complexity_score(text_lower)
        expected_level = self._map_complexity_to_level(complexity_score)
        if expected_level == predicted_level:
            return 0.7
        
        return 0.6  # Default confidence

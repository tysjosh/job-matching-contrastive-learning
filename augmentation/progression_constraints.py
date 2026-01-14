"""
Progression Constraints: Enforces realistic career progression rules

This validator ensures transformations respect career progression logic,
skill hierarchies, and domain boundaries using ESCO and career graph data.

"""

import csv
import json
import logging
import os
import re
from typing import Dict, Any, Optional, Set, List

logger = logging.getLogger(__name__)


class ESCODataLoader:
    """Enhanced ESCO data loader with occupation-skill relationships"""

    def __init__(self, esco_csv_path: str = "dataset/esco/"):
        """Initialize ESCO data loader"""
        self.esco_csv_path = esco_csv_path
        self._skills_data = None
        self._occupations_data = None
        self._occupation_skill_relations = None
        self._skill_hierarchy = None

    def load_skills_data(self) -> Dict[str, Dict]:
        """Load ESCO skills data from CSV"""
        if self._skills_data is None:
            skills_file = os.path.join(self.esco_csv_path, "skills_en.csv")
            self._skills_data = {}

            if not os.path.exists(skills_file):
                raise FileNotFoundError(
                    f"ESCO skills file not found: {skills_file}")

            with open(skills_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uri = row['conceptUri']
                    self._skills_data[uri] = {
                        'preferred_label': row['preferredLabel'],
                        'skill_type': row['skillType'],
                        'reuse_level': row['reuseLevel'],
                        'description': row.get('description', ''),
                        'alt_labels': row.get('altLabels', '').split('\n') if row.get('altLabels') else [],
                        'concept_type': row.get('conceptType', ''),
                        'status': row.get('status', ''),
                        'definition': row.get('definition', '')
                    }

        return self._skills_data

    def load_occupations_data(self) -> Dict[str, Dict]:
        """Load ESCO occupations data from CSV"""
        if self._occupations_data is None:
            occupations_file = os.path.join(
                self.esco_csv_path, "occupations_en.csv")
            self._occupations_data = {}

            if not os.path.exists(occupations_file):
                raise FileNotFoundError(
                    f"ESCO occupations file not found: {occupations_file}")

            with open(occupations_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uri = row['conceptUri']
                    self._occupations_data[uri] = {
                        'preferred_label': row['preferredLabel'],
                        'isco_group': row.get('iscoGroup', ''),
                        'alt_labels': row.get('altLabels', '').split('\n') if row.get('altLabels') else [],
                        'description': row.get('description', ''),
                        'definition': row.get('definition', ''),
                        'code': row.get('code', '')
                    }

        return self._occupations_data

    def load_occupation_skill_relations(self) -> Dict[str, List[Dict]]:
        """Load occupation-skill relationships from CSV"""
        if self._occupation_skill_relations is None:
            relations_file = os.path.join(
                self.esco_csv_path, "occupationSkillRelations_en.csv")
            self._occupation_skill_relations = {}

            if not os.path.exists(relations_file):
                raise FileNotFoundError(
                    f"ESCO occupation-skill relations file not found: {relations_file}")

            with open(relations_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    occupation_uri = row['occupationUri']
                    if occupation_uri not in self._occupation_skill_relations:
                        self._occupation_skill_relations[occupation_uri] = {
                            'essential': [],
                            'optional': []
                        }

                    skill_info = {
                        'skill_uri': row['skillUri'],
                        'skill_type': row['skillType']
                    }

                    relation_type = row['relationType']
                    if relation_type == 'essential':
                        self._occupation_skill_relations[occupation_uri]['essential'].append(
                            skill_info)
                    elif relation_type == 'optional':
                        self._occupation_skill_relations[occupation_uri]['optional'].append(
                            skill_info)

        return self._occupation_skill_relations

    def get_skills_for_occupation(self, occupation_uri: str, relation_type: str = 'all') -> List[str]:
        """Get skills required for a specific occupation"""
        relations = self.load_occupation_skill_relations()
        if occupation_uri not in relations:
            return []

        skills = []
        if relation_type in ['all', 'essential']:
            skills.extend([skill['skill_uri']
                          for skill in relations[occupation_uri]['essential']])
        if relation_type in ['all', 'optional']:
            skills.extend([skill['skill_uri']
                          for skill in relations[occupation_uri]['optional']])

        return skills


class ESCOSkillNormalizer:
    """Normalizes common technology skills to ESCO skill taxonomy"""

    def __init__(self, esco_skills_hierarchy: Dict = None, esco_csv_path: str = "dataset/esco/"):
        """Initialize skill normalization mappings from ESCO CSV data"""
        self.esco_skills_hierarchy = esco_skills_hierarchy
        self.esco_csv_path = esco_csv_path
        self.esco_data_loader = ESCODataLoader(esco_csv_path)
        self._load_skill_mappings()

    def _load_skill_mappings(self):
        """Load skill normalization mappings from ESCO CSV files"""
        try:
            if self.esco_skills_hierarchy:
                self.esco_skills_data = self.esco_skills_hierarchy.get(
                    'skills', {})
                self.skill_hierarchy = self.esco_skills_hierarchy.get(
                    'hierarchy', {})
            else:
                # Load ESCO skills data from CSV files using the new data loader
                self.esco_skills_data = self.esco_data_loader.load_skills_data()
                self.skill_hierarchy = self._load_esco_hierarchy_from_csv()

            # Build skill mappings from ESCO data
            try:
                self.skill_mappings = self._build_skill_mappings()
            except Exception as e:
                logger.error(
                    f"Failed to build skill mappings from ESCO data: {e}")
                self._load_fallback_mappings()
                return

            # Build prerequisite relationships from hierarchy
            try:
                self.skill_prerequisites = self._build_prerequisite_mappings()
            except Exception as e:
                logger.error(f"Failed to build skill prerequisites: {e}")
                self.skill_prerequisites = {}

            # Build skill levels from hierarchy
            try:
                self.skill_levels = self._build_skill_level_mappings()
            except Exception as e:
                logger.error(f"Failed to build skill levels: {e}")
                self.skill_levels = {}

            logger.info(
                f"Loaded {len(self.esco_skills_data)} ESCO skills with {len(self.skill_prerequisites)} prerequisite relationships")

        except Exception as e:
            logger.error(f"Failed to load ESCO data from CSV: {e}")
            self._load_fallback_mappings()

    def _load_esco_hierarchy_from_csv(self) -> Dict[str, Dict]:
        """Load ESCO skill hierarchy from skillsHierarchy_en.csv"""
        hierarchy_file = os.path.join(
            self.esco_csv_path, "skillsHierarchy_en.csv")
        hierarchy_data = {}

        if not os.path.exists(hierarchy_file):
            raise FileNotFoundError(
                f"ESCO hierarchy file not found: {hierarchy_file}")

        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build hierarchy structure for each level
                for level in range(4):  # ESCO has 4 levels (0-3)
                    uri_key = f'Level {level} URI'
                    term_key = f'Level {level} preferred term'

                    if row.get(uri_key) and row.get(term_key):
                        uri = row[uri_key]
                        if uri not in hierarchy_data:
                            hierarchy_data[uri] = {
                                'level': level,
                                'term': row[term_key],
                                'description': row.get('Description', ''),
                                'children': [],
                                'parent': None
                            }

                        # Link parent-child relationships
                        if level > 0:
                            parent_uri = row.get(f'Level {level-1} URI')
                            if parent_uri and parent_uri in hierarchy_data:
                                hierarchy_data[uri]['parent'] = parent_uri
                                if uri not in hierarchy_data[parent_uri]['children']:
                                    hierarchy_data[parent_uri]['children'].append(
                                        uri)

        return hierarchy_data

    def _build_skill_mappings(self) -> Dict[str, str]:
        """Build common technology skill mappings to ESCO terms using CS skills database"""
        mappings = {}

        # Load CS skills database
        cs_skills = self._load_cs_skills_database()

        # Find programming and computer-related skills in ESCO
        programming_skills = []
        computer_skills = []

        for uri, skill_data in self.esco_skills_data.items():
            # Safely get preferred label and description, handling float/NaN values
            preferred_label = skill_data.get('preferred_label', '')
            description = skill_data.get('description', '')

            # Convert to string and handle NaN/float values
            label_lower = str(preferred_label).lower() if preferred_label and str(
                preferred_label) != 'nan' else ''
            description_lower = str(description).lower(
            ) if description and str(description) != 'nan' else ''

            if 'programming' in label_lower or 'programming' in description_lower:
                programming_skills.append(
                    (uri, str(preferred_label) if preferred_label else ''))
            elif 'computer' in label_lower or 'computer' in description_lower:
                computer_skills.append(
                    (uri, str(preferred_label) if preferred_label else ''))

        # Map common technology terms to appropriate ESCO skills
        programming_esco_term = "programming computer systems"
        computer_esco_term = "working with computers"
        sysadmin_esco_term = "setting up and protecting computer systems"

        # Build mappings from CS skills database
        if cs_skills:
            # Programming languages and frameworks
            programming_categories = ['programming_languages', 'frameworks', 'web_development_frameworks',
                                      'mobile_development_frameworks', 'aiml_frameworks', 'backend_frameworks',
                                      'frontend_frameworks', 'game_development_tools', 'testing_frameworks']
            for category in programming_categories:
                if category in cs_skills:
                    for skill in cs_skills[category]:
                        if skill:
                            # Handle AWS variations and other special cases
                            skill_normalized = self._normalize_cs_skill_name(
                                skill)
                            mappings[skill_normalized.lower()
                                     ] = programming_esco_term

            # Database skills
            database_categories = ['databases', 'database_technologies']
            for category in database_categories:
                if category in cs_skills:
                    for skill in cs_skills[category]:
                        if skill:
                            skill_normalized = self._normalize_cs_skill_name(
                                skill)
                            mappings[skill_normalized.lower()
                                     ] = computer_esco_term

            # Cloud and DevOps skills
            cloud_devops_categories = ['cloud_platforms', 'devops_tools', 'cloud_platforms__devops_tools',
                                       'containerization', 'configuration_management', 'ci_cd_tools',
                                       'monitoring_logging', 'infrastructure_tools']
            for category in cloud_devops_categories:
                if category in cs_skills:
                    for skill in cs_skills[category]:
                        if skill:
                            skill_normalized = self._normalize_cs_skill_name(
                                skill)
                            mappings[skill_normalized.lower()
                                     ] = sysadmin_esco_term

            # General computer skills
            computer_categories = ['data_analysis_tools', 'data_science__analytics_tools',
                                   'project_management_tools', 'collaboration_tools', 'soft_skills',
                                   'non_technical_skills', 'version_control', 'version_control_systems',
                                   'ides', 'ides_and_code_editors', 'networking', 'networking_protocols',
                                   'operating_systems', 'productivity_tools']
            for category in computer_categories:
                if category in cs_skills:
                    for skill in cs_skills[category]:
                        if skill:
                            skill_normalized = self._normalize_cs_skill_name(
                                skill)
                            mappings[skill_normalized.lower()
                                     ] = computer_esco_term

        # Skill aliases (enhanced from CS skills data)
        self.skill_aliases = self._build_skill_aliases(cs_skills)

        return mappings

    def _normalize_cs_skill_name(self, skill_name: str) -> str:
        """Normalize CS skill names to handle variations and parenthetical expressions"""
        if not skill_name:
            return skill_name

        # Handle parenthetical variations like "Amazon Web Services (AWS)" -> "AWS"
        if '(' in skill_name and ')' in skill_name:
            # Extract abbreviation from parentheses
            paren_content = skill_name[skill_name.find(
                '(')+1:skill_name.find(')')]
            if paren_content and len(paren_content) <= 5:  # Likely an abbreviation
                return paren_content

        # Handle other common variations
        skill_variations = {
            'Amazon Web Services': 'AWS',
            'Google Cloud Platform': 'GCP',
            'Microsoft Azure': 'Azure',
            'Node.js': 'Node',
            'React.js': 'React',
            'Vue.js': 'Vue',
            'Angular.js': 'Angular'
        }

        for original, normalized in skill_variations.items():
            if skill_name == original:
                return normalized

        return skill_name

    def _load_cs_skills_database(self) -> Dict:
        """Load CS skills database from JSON file"""
        # CS skills file is in dataset/ directory, not esco/ subdirectory
        dataset_dir = os.path.dirname(self.esco_csv_path) if self.esco_csv_path.endswith(
            '/') else os.path.dirname(self.esco_csv_path + '/')
        if not dataset_dir or dataset_dir == 'dataset/esco':
            dataset_dir = 'dataset'
        cs_skills_file = os.path.join(dataset_dir, "cs_skills.json")

        try:
            with open(cs_skills_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"CS skills file not found at {cs_skills_file}")
            return {}
        except Exception as e:
            logger.error(f"Error loading CS skills: {e}")
            return {}

    def _build_skill_aliases(self, cs_skills: Dict) -> Dict[str, str]:
        """Build skill aliases from CS skills data and common variations"""
        aliases = {
            'js': 'javascript',
            'ts': 'typescript',
            'reactjs': 'react',
            'nodejs': 'node.js',
            'k8s': 'kubernetes',
            'ml': 'machine learning',
            'ai': 'artificial intelligence'
        }

        # Add aliases from CS skills data
        if cs_skills:
            # Programming language aliases
            programming_languages = cs_skills.get('programming_languages', [])
            for lang in programming_languages:
                if lang == "JavaScript":
                    aliases.update(
                        {'js': 'javascript', 'javascript': 'javascript'})
                elif lang == "TypeScript":
                    aliases.update(
                        {'ts': 'typescript', 'typescript': 'typescript'})
                elif lang == "Python":
                    aliases.update({'py': 'python', 'python': 'python'})
                elif lang == "C++":
                    aliases.update(
                        {'cpp': 'c++', 'c++': 'c++', 'cplusplus': 'c++'})
                elif lang == "C#":
                    aliases.update({'csharp': 'c#', 'c#': 'c#'})

            # Framework aliases
            frameworks = cs_skills.get('frameworks', [])
            for framework in frameworks:
                if framework == "React":
                    aliases.update({'reactjs': 'react', 'react': 'react'})
                elif framework == "Vue.js":
                    aliases.update({'vue': 'vue.js', 'vuejs': 'vue.js'})
                elif framework == "Angular":
                    aliases.update(
                        {'angular': 'angular', 'angularjs': 'angular'})

        return aliases

    def _build_prerequisite_mappings(self) -> Dict[str, List[str]]:
        """Build prerequisite relationships from ESCO hierarchy"""
        prerequisites = {}

        for uri, skill_data in self.skill_hierarchy.items():
            level = skill_data['level']
            parent_uri = skill_data.get('parent')

            # Skills at level 1+ require their parent as prerequisite
            if level > 0 and parent_uri:
                prerequisites[uri] = [parent_uri]

                # Also include grandparent for level 2+ skills
                if level > 1 and parent_uri in self.skill_hierarchy:
                    grandparent_uri = self.skill_hierarchy[parent_uri].get(
                        'parent')
                    if grandparent_uri:
                        prerequisites[uri].append(grandparent_uri)

        return prerequisites

    def _build_skill_level_mappings(self) -> Dict[str, int]:
        """Build skill level mappings from ESCO hierarchy"""
        levels = {}

        for uri, skill_data in self.skill_hierarchy.items():
            levels[uri] = skill_data['level']

        return levels

    def _load_fallback_mappings(self):
        """Fallback to basic hardcoded mappings if CSV loading fails"""
        logger.warning("Using fallback hardcoded skill mappings")

        # Basic fallback mappings
        self.skill_mappings = {
            'python': 'programming computer systems',
            'javascript': 'programming computer systems',
            'java': 'programming computer systems',
            'react': 'programming computer systems',
            'sql': 'working with computers',
            'aws': 'setting up and protecting computer systems'
        }

        self.skill_aliases = {
            'js': 'javascript',
            'py': 'python'
        }

        self.skill_prerequisites = {}
        self.skill_levels = {}
        self.esco_skills_data = {}
        self.skill_hierarchy = {}

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a skill term to ESCO format

        Args:
            skill: Raw skill term

        Returns:
            Normalized skill term
        """
        if not skill:
            return skill

        skill_lower = skill.lower().strip()
        original_skill_lower = skill_lower

        # Remove common prefixes/suffixes
        skill_lower = re.sub(
            r'^(expert in|advanced|proficient in|skilled in|experienced with)\s+', '', skill_lower)
        skill_lower = re.sub(
            r'\s+(programming|development|framework|library|database)$', '', skill_lower)

        # Apply CS skill name normalization first
        skill_normalized = self._normalize_cs_skill_name(skill)
        skill_normalized_lower = skill_normalized.lower().strip()

        # Check aliases first (both original and normalized versions)
        if original_skill_lower in self.skill_aliases:
            skill_lower = self.skill_aliases[original_skill_lower]
        elif skill_lower in self.skill_aliases:
            skill_lower = self.skill_aliases[skill_lower]
        elif skill_normalized_lower in self.skill_aliases:
            skill_lower = self.skill_aliases[skill_normalized_lower]

        # Check direct mappings (try normalized version first, then original)
        if skill_normalized_lower in self.skill_mappings:
            return self.skill_mappings[skill_normalized_lower]
        elif skill_lower in self.skill_mappings:
            return self.skill_mappings[skill_lower]

        # Check partial matches and common variations
        for common_term, esco_term in self.skill_mappings.items():
            # Exact match after normalization
            if skill_lower == common_term or skill_normalized_lower == common_term:
                return esco_term
            # Partial matches for compound terms
            elif len(common_term.split()) > 1 and common_term in skill_lower:
                return esco_term
            elif len(skill_lower.split()) > 1 and skill_lower in common_term:
                return esco_term
            # Handle common variations like "reactjs" -> "react"
            elif common_term in skill_lower.replace('js', '').replace('.js', ''):
                return esco_term

        # Final attempt: normalize to base form and check mappings again
        base_skill = skill_lower.replace('js', '').replace('.js', '').strip()
        if base_skill in self.skill_mappings:
            return self.skill_mappings[base_skill]

        # Return the ESCO normalized form if we found a mapping, otherwise return cleaned skill
        return skill_lower

    def is_esco_skill(self, skill: str, esco_skills: Dict) -> bool:
        """
        Check if a skill exists in ESCO taxonomy

        Args:
            skill: Skill to check
            esco_skills: ESCO skills dictionary

        Returns:
            True if skill exists in ESCO
        """
        if not esco_skills:
            return False

        skill_lower = skill.lower().strip()

        # Check direct match in ESCO skills
        for skill_id, skill_info in esco_skills.items():
            if isinstance(skill_info, dict):
                # Check 'preferred_label' field (used in ESCO skills data)
                preferred_label = skill_info.get('preferred_label', '')
                esco_label = str(preferred_label).lower(
                ) if preferred_label else ''

                # Also check 'term' field (used in ESCO hierarchy)
                term = skill_info.get('term', '')
                esco_term = str(term).lower() if term else ''

                # Safely handle description
                description = skill_info.get('description', '')
                esco_desc = str(description).lower() if description else ''

                if skill_lower == esco_label or skill_lower == esco_term or skill_lower in esco_desc:
                    return True
            elif isinstance(skill_info, str) and skill_lower == skill_info.lower():
                return True

        return False

    def get_skill_suggestions(self, skill: str, esco_skills: Dict, max_suggestions: int = 3) -> List[str]:
        """
        Get ESCO skill suggestions for a given skill

        Args:
            skill: Skill to find suggestions for
            esco_skills: ESCO skills dictionary
            max_suggestions: Maximum number of suggestions

        Returns:
            List of suggested ESCO skills
        """
        if not esco_skills:
            return []

        skill_lower = skill.lower().strip()
        suggestions = []

        # Find partial matches in ESCO skills
        for skill_id, skill_info in esco_skills.items():
            if len(suggestions) >= max_suggestions:
                break

            if isinstance(skill_info, dict):
                preferred_label = skill_info.get('preferred_label', '')
                esco_label = str(preferred_label) if preferred_label else ''
                if esco_label and (skill_lower in esco_label.lower() or any(word in esco_label.lower() for word in skill_lower.split())):
                    suggestions.append(esco_label)

        return suggestions


class ESCODomainLoader:
    """Loads career domains from ESCO data instead of hardcoded values"""

    def __init__(self, esco_config_file: str = None):
        """
        Initialize ESCO domain loader

        Args:
            esco_config_file: Path to ESCO configuration JSON file (required)
        """
        self.esco_config_file = esco_config_file or 'esco_it_career_domains_refined.json'
        self._career_domains = None

        # Validate that ESCO config file exists at initialization
        if not os.path.exists(self.esco_config_file):
            # Try relative to this file's directory
            config_path = os.path.join(
                os.path.dirname(__file__), self.esco_config_file)
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"ESCO configuration file not found: {self.esco_config_file}. "
                    f"This file is required for career domain classification. "
                    f"Please ensure the ESCO data has been properly extracted and configured."
                )

    def load_career_domains(self) -> Dict[str, List[str]]:
        """Load career domains from ESCO configuration"""
        if self._career_domains is None:
            try:
                import json
                import os

                # Try to load from current directory first
                config_path = self.esco_config_file
                if not os.path.exists(config_path):
                    # Try relative to this file's directory
                    config_path = os.path.join(
                        os.path.dirname(__file__), self.esco_config_file)

                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    self._career_domains = config['career_domains']
                    logger.info(
                        f"Loaded {len(self._career_domains)} ESCO career domains")
                else:
                    logger.error(f"ESCO config file not found: {config_path}")
                    self._get_fallback_domains()  # This will raise an error

            except Exception as e:
                logger.error(f"Failed to load ESCO domains: {e}")
                self._get_fallback_domains()  # This will raise an error

        return self._career_domains

    def validate_esco_data_quality(self) -> Dict[str, Any]:
        """Validate the quality and completeness of loaded ESCO data"""
        domains = self.load_career_domains()

        validation_report = {
            'total_domains': len(domains),
            'total_occupations': sum(len(occupations) for occupations in domains.values()),
            'empty_domains': [],
            'small_domains': [],
            'domain_sizes': {},
            'is_valid': True,
            'warnings': [],
            'errors': []
        }

        for domain, occupations in domains.items():
            domain_size = len(occupations)
            validation_report['domain_sizes'][domain] = domain_size

            if domain_size == 0:
                validation_report['empty_domains'].append(domain)
                validation_report['errors'].append(
                    f"Domain '{domain}' has no occupations")
                validation_report['is_valid'] = False
            elif domain_size < 5:
                validation_report['small_domains'].append(domain)
                validation_report['warnings'].append(
                    f"Domain '{domain}' has only {domain_size} occupations")

        # Check for minimum required domains
        required_domains = {'software_development', 'data_science',
                            'cybersecurity', 'systems_administration'}
        missing_domains = required_domains - set(domains.keys())

        if missing_domains:
            validation_report['errors'].extend(
                [f"Missing required domain: {domain}" for domain in missing_domains])
            validation_report['is_valid'] = False

        return validation_report

    def _get_fallback_domains(self) -> Dict[str, List[str]]:
        """No fallback domains - ESCO data is required"""
        logger.error(
            "ESCO career domains are required but not available. System cannot function without ESCO data.")
        raise RuntimeError(
            "ESCO career domains are required for career progression validation. "
            "Please ensure the ESCO configuration file is available and properly formatted."
        )

    def get_domain_for_text(self, text: str) -> Optional[str]:
        """
        Identify the most likely career domain for given text

        Args:
            text: Text to analyze (resume content, job description, etc.)

        Returns:
            Most likely career domain or None
        """
        domains = self.load_career_domains()
        text_lower = text.lower()

        domain_scores = {}
        for domain, occupations in domains.items():
            score = 0
            for occupation in occupations:
                if occupation.lower() in text_lower:
                    score += 1
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return None


class ProgressionConstraints:
    """
    Enforces career progression constraints to ensure realistic transformations.

    Implements the three key guardrails:
    1. Skill Progression Constraints (ESCO hierarchy)
    2. Experience Coherence (level-appropriate changes)
    3. Domain Boundaries (career field consistency)
    """

    def __init__(self,
                 esco_skills_hierarchy: Dict,
                 career_graph: Any = None,
                 esco_config_file: str = None,
                 esco_csv_path: str = "dataset/esco/"):
        """
        Initialize progression constraints validator.

        Args:
            esco_skills_hierarchy: ESCO skills hierarchy dictionary
            career_graph: Career graph for domain boundary enforcement (optional)
            esco_config_file: Path to ESCO career domains configuration file (optional)
            esco_csv_path: Path to ESCO CSV files directory
        """
        self.esco_skills_hierarchy = esco_skills_hierarchy
        self.career_graph = career_graph
        self.esco_csv_path = esco_csv_path
        self.esco_domain_loader = ESCODomainLoader(esco_config_file)
        self.skill_normalizer = ESCOSkillNormalizer(
            esco_skills_hierarchy=self.esco_skills_hierarchy,
            esco_csv_path=esco_csv_path
        )

        # Initialize the ESCO data loader for occupation-skill relationships
        self.esco_data_loader = ESCODataLoader(esco_csv_path)

        # Load ESCO skills hierarchy from the normalizer
        self.esco_skills = {
            'skills': self.skill_normalizer.esco_skills_data,
            'hierarchy': self.skill_normalizer.skill_hierarchy,
            'prerequisites': self.skill_normalizer.skill_prerequisites,
            'levels': self.skill_normalizer.skill_levels
        }

        # Load occupation data for enhanced validation
        try:
            self.occupations_data = self.esco_data_loader.load_occupations_data()
            self.occupation_skill_relations = self.esco_data_loader.load_occupation_skill_relations()
            logger.info(
                f"Loaded {len(self.occupations_data)} ESCO occupations with skill relationships")
        except Exception as e:
            logger.warning(f"Could not load occupation data: {e}")
            self.occupations_data = {}
            self.occupation_skill_relations = {}

        # Validate ESCO data quality on initialization
        validation_report = self.esco_domain_loader.validate_esco_data_quality()
        if not validation_report['is_valid']:
            error_msg = f"ESCO data validation failed: {'; '.join(validation_report['errors'])}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if validation_report['warnings']:
            for warning in validation_report['warnings']:
                logger.warning(f"ESCO data quality warning: {warning}")

        logger.info(f"ESCO data validation passed: {validation_report['total_domains']} domains, "
                    f"{validation_report['total_occupations']} total occupations")

        # Validate skill normalization coverage
        self._validate_skill_normalization_coverage()

        self._load_progression_rules()

    def _validate_skill_normalization_coverage(self):
        """Validate that skill normalization covers common technology skills"""
        if not self.esco_skills:
            logger.warning(
                "No ESCO skills available for normalization validation")
            return

        # Test normalization of common skills
        test_skills = ['Python', 'JavaScript', 'React',
                       'Node.js', 'MongoDB', 'AWS', 'Docker']
        normalization_stats = {
            'total_tested': len(test_skills),
            'successfully_normalized': 0,
            'found_in_esco': 0,
            'missing_from_esco': []
        }

        for skill in test_skills:
            normalized = self.skill_normalizer.normalize_skill(skill)
            if normalized != skill:
                normalization_stats['successfully_normalized'] += 1

            if self.skill_normalizer.is_esco_skill(normalized, self.esco_skills.get('skills', {})):
                normalization_stats['found_in_esco'] += 1
            else:
                normalization_stats['missing_from_esco'].append(skill)

        coverage_ratio = normalization_stats['found_in_esco'] / \
            normalization_stats['total_tested']
        logger.info(f"Skill normalization coverage: {coverage_ratio:.1%} "
                    f"({normalization_stats['found_in_esco']}/{normalization_stats['total_tested']})")

        if normalization_stats['missing_from_esco']:
            logger.warning(
                f"Skills not found in ESCO after normalization: {normalization_stats['missing_from_esco']}")

    def _load_progression_rules(self):
        """Load career progression rules and constraints"""

        # Unified experience level progression mapping (ESCO + Hardcoded)
        self.level_hierarchy = {
            # Entry levels
            'intern': 0,
            'trainee': 0,
            'entry': 0,
            'junior': 0,
            'assistant': 0,

            # Mid levels
            'associate': 1,
            'mid': 1,
            'specialist': 1,

            # Senior levels
            'senior': 2,
            'experienced': 2,
            'advanced': 2,

            # Leadership levels
            'lead': 3,
            'staff': 3,
            'team_lead': 3,

            # Management levels
            'manager': 4,
            'senior_manager': 4,
            'principal': 4,

            # Executive levels
            'director': 5,
            'head': 5,
            'executive': 5,

            # C-level
            'chief': 6,
            'vp': 6,
            'cto': 6,
            'cio': 6
        }

        # ESCO seniority pattern mappings to unified levels
        self.esco_to_unified_mapping = {
            # ESCO patterns -> Unified level
            'intern': 'intern',
            'trainee': 'trainee',
            'junior': 'junior',
            'assistant': 'assistant',
            'associate': 'associate',
            'specialist': 'specialist',
            'senior': 'senior',
            'lead': 'lead',
            'staff': 'staff',
            'manager': 'manager',
            'director': 'director',
            'head': 'head',
            'chief': 'chief',
            'principal': 'principal',
            'cto': 'chief',  # Map CTO to chief level
            'cio': 'chief',  # Map CIO to chief level
            'vp': 'vp',

            # Common variations
            'entry level': 'entry',
            'entry-level': 'entry',
            'mid level': 'mid',
            'mid-level': 'mid',
            'team lead': 'lead',
            'team leader': 'lead',
            'senior manager': 'manager',
            'vice president': 'vp',
            'chief technology officer': 'chief',
            'chief information officer': 'chief',
            'executive director': 'executive'
        }

        # Level-appropriate responsibilities (much more inclusive - allow realistic skill overlap)
        self.level_responsibilities = {
            'entry': [
                'learning', 'assisting', 'implementing', 'testing',
                'documenting', 'debugging', 'following procedures', 'coding',
                'developing', 'collaborating'
            ],
            'junior': [
                'learning', 'assisting', 'implementing', 'testing',
                'documenting', 'debugging', 'following procedures', 'coding',
                'developing', 'collaborating', 'optimizing', 'reviewing'
            ],
            'mid': [
                'developing', 'designing', 'collaborating', 'reviewing',
                'optimizing', 'mentoring juniors', 'project planning', 'testing',
                'learning', 'implementing', 'team leadership', 'innovation',
                'architecting', 'leading', 'mentoring', 'technical decisions',
                'code reviews', 'best practices', 'process improvement'
            ],
            'senior': [
                'architecting', 'leading', 'designing systems', 'mentoring',
                'technical decisions', 'code reviews', 'best practices',
                'developing', 'designing', 'collaborating', 'reviewing',
                'optimizing', 'testing', 'learning', 'implementing',
                'team leadership', 'innovation', 'strategic planning'
            ],
            'lead': [
                'strategic planning', 'team leadership', 'cross-team collaboration',
                'technical vision', 'architecture decisions', 'process improvement',
                'architecting', 'leading', 'designing systems', 'mentoring',
                'optimizing', 'testing', 'learning', 'innovation',
                'organizational impact', 'technical strategy'
            ],
            'principal': [
                'organizational impact', 'technical strategy', 'innovation',
                'industry expertise', 'thought leadership', 'technology roadmap',
                'strategic planning', 'team leadership', 'architecture decisions',
                'architecting', 'leading', 'mentoring', 'process improvement',
                'optimizing', 'testing'
            ]
        }

        # Domain-specific career paths (loaded from ESCO)
        self.career_domains = self.esco_domain_loader.load_career_domains()

    def validate_upward_progression(self,
                                    original: Dict[str, Any],
                                    transformed: Dict[str, Any],
                                    current_level: str,
                                    target_level: str) -> bool:
        """
        Validate upward career progression transformation.

        Args:
            original: Original resume
            transformed: Upward-transformed resume
            current_level: Current experience level
            target_level: Target senior level

        Returns:
            bool: True if progression is valid
        """
        try:
            # Validate level progression is logical
            if not self._validate_level_progression(current_level, target_level, 'upward'):
                return False

            # Validate education-aware progression constraints
            if not self._validate_education_progression(original, current_level, target_level, 'upward'):
                return False

            # Validate skill progression constraints
            if not self._validate_skill_progression(original, transformed, 'upward'):
                return False

            # Validate responsibility appropriateness
            if not self._validate_responsibility_level(transformed, target_level):
                return False

            # Validate domain consistency
            if not self._validate_domain_consistency(original, transformed):
                return False

            return True

        except Exception as e:
            logger.error(f"Upward progression validation error: {e}")
            return False

    def validate_downward_progression(self,
                                      original: Dict[str, Any],
                                      transformed: Dict[str, Any],
                                      current_level: str,
                                      target_level: str) -> bool:
        """
        Validate downward career progression transformation.

        Args:
            original: Original resume
            transformed: Downward-transformed resume
            current_level: Current experience level
            target_level: Target junior level

        Returns:
            bool: True if progression is valid
        """
        try:
            # Validate level progression is logical
            if not self._validate_level_progression(current_level, target_level, 'downward'):
                return False

            # Validate education-aware progression constraints
            if not self._validate_education_progression(original, current_level, target_level, 'downward'):
                return False

            # Validate skill progression constraints
            if not self._validate_skill_progression(original, transformed, 'downward'):
                return False

            # Validate responsibility appropriateness
            if not self._validate_responsibility_level(transformed, target_level):
                return False

            # Validate domain consistency
            if not self._validate_domain_consistency(original, transformed):
                return False

            return True

        except Exception as e:
            logger.error(f"Downward progression validation error: {e}")
            return False

    def _validate_level_progression(self,
                                    current_level: str,
                                    target_level: str,
                                    direction: str) -> bool:
        """Validate that level progression is logical"""
        current_rank = self.level_hierarchy.get(current_level, -1)
        target_rank = self.level_hierarchy.get(target_level, -1)

        if current_rank == -1 or target_rank == -1:
            logger.debug(f"Invalid levels: {current_level} -> {target_level}")
            return False

        if direction == 'upward':
            # Can only go up one level
            return target_rank == current_rank + 1 or target_rank == current_rank
        elif direction == 'downward':
            # Can only go down one level
            return target_rank == current_rank - 1 or target_rank == current_rank

        return False

    def _validate_education_progression(self,
                                        resume: Dict[str, Any],
                                        current_level: str,
                                        target_level: str,
                                        direction: str) -> bool:
        """
        Validate career progression considering education context.

        Args:
            resume: Resume data containing education field
            current_level: Current experience level
            target_level: Target experience level 
            direction: 'upward' or 'downward'

        Returns:
            bool: True if progression aligns with education background
        """
        try:
            # Extract education level if available
            education_level = resume.get('education', '').strip()
            if not education_level:
                return True  # No education constraint if not available

            # Education level hierarchy (higher number = more advanced)
            # Note: Certifications are excluded - they're preserved as metadata only
            education_hierarchy = {
                'high school': 1,
                'associate': 2,
                'associates': 2,
                'bachelors': 3,
                'bachelor': 3,
                'masters': 4,
                'master': 4,
                'phd': 5,
                'doctorate': 5,
                'doctoral': 5
            }

            # Experience level to typical education requirements
            level_education_requirements = {
                'entry': 1,     # High school acceptable
                'junior': 2,    # Associate preferred
                'mid': 3,       # Bachelor's typically required
                'senior': 3,    # Bachelor's required, Master's preferred
                'lead': 4,      # Master's typically required
                'principal': 4  # Master's or PhD typically required
            }

            # Get numeric values for comparison
            education_rank = 0
            education_lower = education_level.lower()
            for edu_level, rank in education_hierarchy.items():
                if edu_level in education_lower:
                    education_rank = max(education_rank, rank)
                    break

            if education_rank == 0:
                return True  # Unknown education level, allow progression

            # Get target level education requirement
            target_education_req = level_education_requirements.get(
                target_level, 3)
            current_education_req = level_education_requirements.get(
                current_level, 3)

            if direction == 'upward':
                # For upward progression, check if education supports target level
                # Allow some flexibility - education can be 1 level below requirement
                flexibility_margin = 1
                return education_rank >= (target_education_req - flexibility_margin)

            elif direction == 'downward':
                # For downward progression, education constraints are more relaxed
                # Someone with higher education can work at lower levels
                return True

            return True

        except Exception as e:
            logger.debug(f"Education progression validation error: {e}")
            return True  # Allow on error

    def _validate_skills_against_occupation(self, resume: Dict[str, Any], occupation_uri: str = None) -> bool:
        """Validate that resume skills align with occupation requirements from ESCO"""
        if not self.occupation_skill_relations or not occupation_uri:
            return True  # Skip validation if data not available

        try:
            # Get resume skills
            resume_skills = self._extract_skills_set(resume)
            if not resume_skills:
                return True  # No skills to validate

            # Get required skills for occupation
            essential_skills = self.esco_data_loader.get_skills_for_occupation(
                occupation_uri, 'essential')
            optional_skills = self.esco_data_loader.get_skills_for_occupation(
                occupation_uri, 'optional')

            if not essential_skills and not optional_skills:
                return True  # No requirements to validate against

            # Convert skill URIs to skill terms for comparison
            essential_skill_terms = set()
            for skill_uri in essential_skills:
                skill_term = self._get_skill_term_by_uri(skill_uri)
                if skill_term:
                    essential_skill_terms.add(skill_term.lower())

            # Check if resume has sufficient essential skills
            matching_essential_skills = 0
            for resume_skill in resume_skills:
                normalized_skill = self.skill_normalizer.normalize_skill(
                    resume_skill)
                if normalized_skill.lower() in essential_skill_terms:
                    matching_essential_skills += 1

            # Require at least 30% of essential skills to be present
            if essential_skill_terms:
                coverage_ratio = matching_essential_skills / \
                    len(essential_skill_terms)
                return coverage_ratio >= 0.3

            return True

        except Exception as e:
            logger.debug(f"Occupation skill validation error: {e}")
            return True  # Allow on error

    def _identify_occupation_from_resume(self, resume: Dict[str, Any]) -> Optional[str]:
        """Identify the most likely occupation URI from resume content"""
        if not self.occupations_data:
            return None

        try:
            # Extract text from resume
            resume_text = str(resume).lower()

            # Score occupations based on text similarity
            best_occupation = None
            best_score = 0

            for occupation_uri, occupation_data in self.occupations_data.items():
                score = 0

                # Check preferred label
                preferred_label = occupation_data.get(
                    'preferred_label', '').lower()
                if preferred_label in resume_text:
                    # Weight preferred labels higher
                    score += len(preferred_label) * 2

                # Check alternative labels
                for alt_label in occupation_data.get('alt_labels', []):
                    if alt_label.lower() in resume_text:
                        score += len(alt_label)

                # Check description keywords
                description = occupation_data.get('description', '').lower()
                common_words = set(description.split()) & set(
                    resume_text.split())
                score += len(common_words)

                if score > best_score:
                    best_score = score
                    best_occupation = occupation_uri

            # Threshold to avoid weak matches
            return best_occupation if best_score > 10 else None

        except Exception as e:
            logger.debug(f"Occupation identification error: {e}")
            return None

    def _validate_skill_progression(self,
                                    original: Dict,
                                    transformed: Dict,
                                    direction: str) -> bool:
        """Validate skill progression follows ESCO hierarchy and career graph"""

        # Extract skills from both resumes
        original_skills = self._extract_skills_set(original)
        transformed_skills = self._extract_skills_set(transformed)

        # First validate with career graph if available
        if self.career_graph and self._has_career_graph_data():
            if not self._validate_skills_with_career_graph(original_skills, transformed_skills, direction):
                return False

        # Enhanced validation with occupation-skill relationships
        if self.occupation_skill_relations:
            # Try to identify occupation from transformed resume
            occupation_uri = self._identify_occupation_from_resume(transformed)
            if occupation_uri:
                if not self._validate_skills_against_occupation(transformed, occupation_uri):
                    logger.debug(
                        f"Skills don't align with identified occupation: {occupation_uri}")
                    return False

        # Then validate with ESCO hierarchy
        new_skills = transformed_skills - original_skills

        if direction == 'upward':
            # For upward transformation, new skills must have prerequisites
            for new_skill in new_skills:
                if not self._has_skill_prerequisites(new_skill, original_skills):
                    logger.debug(f"Skill {new_skill} lacks prerequisites")
                    return False

        elif direction == 'downward':
            # For downward transformation, advanced skills should be removed/simplified
            removed_skills = original_skills - transformed_skills
            # This is acceptable for downward transformation

        return True

    def _validate_responsibility_level(self, resume: Dict, target_level: str) -> bool:
        """Validate that responsibilities match the target experience level"""

        # Extract responsibility text
        resp_text = self._extract_responsibility_text(resume).lower()

        # Check for level-appropriate language
        appropriate_responsibilities = self.level_responsibilities.get(
            target_level, [])

        # Count appropriate responsibility indicators
        appropriate_count = sum(1 for resp in appropriate_responsibilities
                                if resp in resp_text)

        # More permissive validation: Check if target level has appropriate responsibilities
        # Rather than rejecting based on what other levels have
        target_rank = self.level_hierarchy.get(target_level, 0)

        # Only reject if we find responsibilities that are clearly too advanced
        # (from levels 2+ ranks higher) and the target level doesn't have them
        for level, responsibilities in self.level_responsibilities.items():
            level_rank = self.level_hierarchy.get(level, 0)

            # Only flag responsibilities that are much too advanced (2+ levels higher)
            # and not appropriate for the target level
            if level_rank > target_rank + 1:
                for resp in responsibilities:
                    if resp in resp_text and resp not in appropriate_responsibilities:
                        logger.debug(
                            f"Responsibility '{resp}' too advanced for level {target_level}")
                        return False

        # Accept if we found any appropriate responsibilities or no problematic ones
        return True

    def _validate_domain_consistency(self, original: Dict, transformed: Dict) -> bool:
        """Validate that career domain remains consistent using career graph"""

        # First try career graph validation if available
        if self.career_graph and self._has_career_graph_data():
            return self._validate_domain_boundaries_with_graph(original, transformed)

        # Fallback to ESCO domain validation
        original_domain = self._identify_career_domain(original)
        transformed_domain = self._identify_career_domain(transformed)

        # Domain should remain consistent
        if original_domain and transformed_domain:
            return original_domain == transformed_domain

        return True  # Allow if domain can't be determined

    def normalize_seniority_level(self, level_text: str) -> str:
        """
        Normalize seniority level using ESCO patterns and unified mapping

        Args:
            level_text: Raw seniority level text from job titles or descriptions

        Returns:
            Normalized unified level name
        """
        if not level_text:
            return 'mid'  # Default fallback

        level_lower = level_text.lower().strip()

        # Direct mapping first
        if level_lower in self.esco_to_unified_mapping:
            return self.esco_to_unified_mapping[level_lower]

        # Pattern matching for complex titles (order matters - most specific first)
        seniority_patterns = {
            'chief': ['chief technology officer', 'chief information officer', 'cto', 'cio', 'chief'],
            'vp': ['vice president', 'vice-president', 'vp of', 'vp'],
            'head': ['head of', 'department head', 'head'],
            'director': ['executive director', 'senior director', 'engineering director', 'director'],
            'manager': ['engineering manager', 'team manager', 'senior manager', 'manager', 'mgr'],
            'principal': ['principal engineer', 'principal architect', 'principal scientist', 'principal'],
            'lead': ['technical lead', 'team lead', 'tech lead', 'lead'],
            'senior': ['senior engineer', 'senior developer', 'senior', 'sr'],
            'specialist': ['data specialist', 'technical specialist', 'specialist'],
            'trainee': ['graduate trainee', 'management trainee', 'trainee'],
            'intern': ['internship', 'intern'],
            'entry': ['entry level', 'entry-level', 'new grad', 'recent grad', 'entry'],
            'junior': ['junior developer', 'junior engineer', 'junior', 'jr'],
            'associate': ['associate'],
            'assistant': ['assistant'],
            'mid': ['developer', 'engineer', 'analyst', 'consultant'],
            'experienced': ['experienced', 'advanced', 'expert'],
            'staff': ['staff engineer', 'staff']
        }

        # Find best match
        for unified_level, patterns in seniority_patterns.items():
            for pattern in patterns:
                if pattern in level_lower:
                    return unified_level

        # Fallback to mid-level if no pattern matches - reduce logging frequency
        # Only log when it's not a common case (avoid spam for 'mid')
        if level_text != 'mid':
            logger.debug(
                f"No seniority pattern match found for: {level_text}, defaulting to 'mid'")
        return 'mid'

    def _extract_skills_set(self, resume: Dict[str, Any]) -> Set[str]:
        """Extract and normalize set of skills from resume"""
        skills_set = set()

        if 'skills' in resume:
            skills = resume['skills']
            if isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, str):
                        # Normalize skill before adding to set
                        normalized_skill = self.skill_normalizer.normalize_skill(
                            skill)
                        skills_set.add(normalized_skill.lower().strip())
                    elif isinstance(skill, dict):
                        if 'name' in skill:
                            normalized_skill = self.skill_normalizer.normalize_skill(
                                skill['name'])
                            skills_set.add(normalized_skill.lower().strip())
                        elif 'original_name' in skill:
                            normalized_skill = self.skill_normalizer.normalize_skill(
                                skill['original_name'])
                            skills_set.add(normalized_skill.lower().strip())

        return skills_set

    def _has_skill_prerequisites(self, skill: str, existing_skills: Set[str]) -> bool:
        """Check if a skill's prerequisites are met using ESCO hierarchy"""
        # Normalize the skill first
        normalized_skill = self.skill_normalizer.normalize_skill(skill)
        skill_lower = normalized_skill.lower().strip()

        # Remove proficiency indicators for prerequisite checking
        skill_lower = re.sub(r'^(expert in|advanced|learning|developing skills in)\s+',
                             '', skill_lower)

        # Find skill URI in ESCO hierarchy by matching skill terms
        skill_uri = self._find_skill_uri_by_term(skill_lower)
        if not skill_uri:
            # Check if the normalized skill exists in ESCO
            if self.skill_normalizer.is_esco_skill(normalized_skill, self.esco_skills.get('skills', {})):
                logger.debug(
                    f"Skill {normalized_skill} found in ESCO but no URI mapping")
                return True
            else:
                logger.debug(
                    f"Skill {skill} -> {normalized_skill} not found in ESCO, allowing")
                return True  # Allow skills not in ESCO

        # Check if skill has prerequisites in ESCO hierarchy
        if not self._has_esco_prerequisites(skill_uri):
            return True  # No prerequisites required

        # Get required prerequisites from ESCO
        required_prerequisites = self._get_esco_prerequisites(skill_uri)

        # Normalize existing skills for comparison
        normalized_existing_skills = set()
        for existing_skill in existing_skills:
            normalized_existing = self.skill_normalizer.normalize_skill(
                existing_skill)
            normalized_existing_skills.add(normalized_existing.lower())

        # Check if any prerequisite is present in existing skills
        for prerequisite_uri in required_prerequisites:
            prerequisite_term = self._get_skill_term_by_uri(prerequisite_uri)
            if prerequisite_term:
                prerequisite_normalized = self.skill_normalizer.normalize_skill(
                    prerequisite_term)
                if prerequisite_normalized.lower() in normalized_existing_skills:
                    return True

        logger.debug(
            f"Skill {skill} -> {normalized_skill} missing ESCO prerequisites")
        return False

    def _extract_responsibility_text(self, resume: Dict[str, Any]) -> str:
        """Extract responsibility text from resume"""
        text_parts = []

        if 'experience' in resume:
            exp = resume['experience']
            if isinstance(exp, str):
                text_parts.append(exp)
            elif isinstance(exp, list):
                for item in exp:
                    if isinstance(item, dict) and 'responsibilities' in item:
                        resp = item['responsibilities']
                        if isinstance(resp, str):
                            text_parts.append(resp)
                        elif isinstance(resp, dict):
                            text_parts.extend(str(v) for v in resp.values())

        return ' '.join(text_parts)

    def _identify_career_domain(self, resume: Dict[str, Any]) -> Optional[str]:
        """Identify the primary career domain from resume"""
        text = (self._extract_responsibility_text(resume) + ' ' +
                str(resume.get('skills', []))).lower()

        # Score each domain based on keyword presence
        domain_scores = {}
        for domain, keywords in self.career_domains.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                domain_scores[domain] = score

        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return None

    def get_validation_report(self,
                              original: Dict[str, Any],
                              aspirational: Dict[str, Any],
                              foundational: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for all transformations.

        Args:
            original: Original resume
            aspirational: Senior-level view
            foundational: Junior-level view

        Returns:
            Dict: Comprehensive validation report
        """
        report = {
            'timestamp': None,  # Would add timestamp in real implementation
            'original_level': self._estimate_experience_level(original),
            'transformations': {},
            'overall_valid': False,
            'recommendations': []
        }

        # Validate upward transformation
        upward_valid, upward_report = self._validate_transformation_detailed(
            original, aspirational, 'upward')
        report['transformations']['aspirational'] = upward_report

        # Validate downward transformation
        downward_valid, downward_report = self._validate_transformation_detailed(
            original, foundational, 'downward')
        report['transformations']['foundational'] = downward_report

        # Overall validation
        report['overall_valid'] = upward_valid and downward_valid

        # Generate recommendations
        if not upward_valid:
            report['recommendations'].append(
                "Upward transformation needs refinement")
        if not downward_valid:
            report['recommendations'].append(
                "Downward transformation needs refinement")

        # Add ESCO statistics to report
        report['esco_statistics'] = self._get_esco_skill_statistics()

        # Add career graph statistics to report
        report['career_graph_statistics'] = self.get_career_graph_statistics()

        # Add ESCO domain quality validation to report
        report['esco_domain_validation'] = self.esco_domain_loader.validate_esco_data_quality()

        # Add skill normalization statistics to report
        report['skill_normalization_stats'] = self._get_skill_normalization_statistics(
            [original, aspirational, foundational]
        )

        return report

    def _get_esco_skill_statistics(self) -> Dict[str, Any]:
        """Get ESCO skill statistics for reporting"""
        try:
            return {
                'total_skills': len(self.esco_skills.get('skills', {})),
                'skill_hierarchy_levels': len(self.esco_skills.get('hierarchy', {})),
                'skills_with_prerequisites': len(self.esco_skills.get('prerequisites', {})),
                'total_occupations': len(self.occupations_data),
                'occupation_skill_relations': len(self.occupation_skill_relations),
                'data_quality': 'good' if self.esco_skills else 'limited'
            }
        except Exception as e:
            logger.warning(f"Could not generate ESCO statistics: {e}")
            return {
                'total_skills': 0,
                'skill_hierarchy_levels': 0,
                'skills_with_prerequisites': 0,
                'total_occupations': 0,
                'occupation_skill_relations': 0,
                'data_quality': 'unavailable'
            }

    def _validate_transformation_detailed(self,
                                          original: Dict,
                                          transformed: Dict,
                                          direction: str) -> tuple:
        """Detailed validation of a single transformation"""

        validation_results = {
            'direction': direction,
            'skill_progression_valid': True,
            'experience_coherence_valid': True,
            'domain_consistency_valid': True,
            'level_appropriate': True,
            'issues': []
        }

        try:
            # Validate skill progression
            if not self._validate_skill_progression_detailed(original, transformed, direction):
                validation_results['skill_progression_valid'] = False
                validation_results['issues'].append(
                    'Invalid skill progression')

            # Validate experience coherence
            if not self._validate_experience_coherence(original, transformed, direction):
                validation_results['experience_coherence_valid'] = False
                validation_results['issues'].append(
                    'Experience coherence violation')

            # Validate domain consistency (now includes career graph validation)
            if not self._validate_domain_consistency(original, transformed):
                validation_results['domain_consistency_valid'] = False
                validation_results['issues'].append(
                    'Domain consistency violation')

            # Additional career graph pathway validation if available
            if self.career_graph and self._has_career_graph_data():
                original_occupation = self._extract_occupation_from_resume(
                    original)
                transformed_occupation = self._extract_occupation_from_resume(
                    transformed)

                if original_occupation and transformed_occupation:
                    if not self._validate_career_pathway(original_occupation, transformed_occupation, direction):
                        validation_results['issues'].append(
                            f'Invalid career pathway: {original_occupation} -> {transformed_occupation}')

            # Overall validation
            overall_valid = all([
                validation_results['skill_progression_valid'],
                validation_results['experience_coherence_valid'],
                validation_results['domain_consistency_valid']
            ])

            return overall_valid, validation_results

        except Exception as e:
            logger.error(f"Detailed validation error: {e}")
            validation_results['issues'].append(f"Validation error: {e}")
            return False, validation_results

    def _validate_experience_coherence(self,
                                       original: Dict,
                                       transformed: Dict,
                                       direction: str) -> bool:
        """Validate experience coherence constraints"""

        # Extract experience level indicators
        original_level = self._estimate_experience_level(original)
        transformed_level = self._estimate_experience_level(transformed)

        original_rank = self.level_hierarchy.get(original_level, 1)
        transformed_rank = self.level_hierarchy.get(transformed_level, 1)

        if direction == 'upward':
            # Transformed should be same or higher level
            return transformed_rank >= original_rank
        elif direction == 'downward':
            # Transformed should be same or lower level
            return transformed_rank <= original_rank

        return True

    def _has_skill_prerequisites(self, skill: str, existing_skills: Set[str]) -> bool:
        """Check if skill prerequisites are met using ESCO hierarchy"""
        skill_lower = skill.lower().strip()

        # Remove proficiency indicators for prerequisite checking
        skill_lower = re.sub(r'^(expert in|advanced|learning|developing skills in)\s+',
                             '', skill_lower)

        # Find skill URI in ESCO hierarchy by matching skill terms
        skill_uri = self._find_skill_uri_by_term(skill_lower)
        if not skill_uri:
            return True  # No ESCO match found, allow skill

        # Check if skill has prerequisites in ESCO hierarchy
        if not self._has_esco_prerequisites(skill_uri):
            return True  # No prerequisites required

        # Get required prerequisites from ESCO
        required_prerequisites = self._get_esco_prerequisites(skill_uri)

        # Check if any prerequisite is present in existing skills
        for prerequisite_uri in required_prerequisites:
            prerequisite_term = self._get_skill_term_by_uri(prerequisite_uri)
            if prerequisite_term and any(prerequisite_term.lower() in existing_skill
                                         for existing_skill in existing_skills):
                return True

        logger.debug(f"Skill {skill} missing ESCO prerequisites")
        return False

    def _estimate_experience_level(self, resume: Dict[str, Any]) -> str:
        """Estimate experience level from resume content using unified ESCO system"""
        text = str(resume).lower()

        # Use unified level indicators that match ESCO patterns
        level_indicators = {
            'intern': ['intern', 'internship'],
            'trainee': ['trainee', 'graduate trainee'],
            'entry': ['entry', 'entry level', 'entry-level', 'new grad', 'recent grad'],
            'junior': ['junior', 'jr', 'assistant'],
            'associate': ['associate'],
            'mid': ['developer', 'engineer', 'analyst', 'specialist', 'consultant'],
            'senior': ['senior', 'sr', 'experienced', 'advanced', 'expert'],
            'lead': ['lead', 'team lead', 'tech lead', 'technical lead', 'staff'],
            'manager': ['manager', 'mgr', 'team manager', 'engineering manager'],
            'principal': ['principal', 'principal engineer', 'principal architect'],
            'director': ['director', 'senior director', 'engineering director'],
            'head': ['head', 'head of', 'department head'],
            'chief': ['chief', 'cto', 'cio', 'chief technology officer']
        }

        level_scores = {}
        for level, indicators in level_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > 0:
                level_scores[level] = score

        # Return level with highest score, default to 'mid'
        if level_scores:
            estimated_level = max(level_scores, key=level_scores.get)
            # Normalize using the new system
            return self.normalize_seniority_level(estimated_level)

        return 'mid'

    def _identify_career_domain(self, resume: Dict[str, Any]) -> Optional[str]:
        """Identify career domain from resume content"""
        text = str(resume).lower()

        domain_scores = {}
        for domain, keywords in self.career_domains.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return None

    def _find_skill_uri_by_term(self, skill_term: str) -> Optional[str]:
        """Find ESCO skill URI by matching skill term (optimized with caching)"""
        if not self.esco_skills or 'skills' not in self.esco_skills:
            return None

        # OPTIMIZATION: Initialize cache on first call
        if not hasattr(self, '_skill_uri_cache'):
            self._skill_uri_cache = {}
            self._skill_term_to_uri_index = {}

            # Pre-build index for fast lookups
            for skill_uri, skill_info in self.esco_skills['skills'].items():
                if isinstance(skill_info, dict):
                    # Index by preferred_label
                    preferred_label_raw = skill_info.get('preferred_label', '')
                    if isinstance(preferred_label_raw, str):
                        preferred_label = preferred_label_raw.lower().strip()
                        if preferred_label:
                            self._skill_term_to_uri_index[preferred_label] = skill_uri

                    # Also index by term if available
                    term_raw = skill_info.get('term', '')
                    if isinstance(term_raw, str):
                        term = term_raw.lower().strip()
                        if term and term not in self._skill_term_to_uri_index:
                            self._skill_term_to_uri_index[term] = skill_uri

                elif isinstance(skill_info, str):
                    skill_str = skill_info.lower().strip()
                    if skill_str:
                        self._skill_term_to_uri_index[skill_str] = skill_uri

            # Also index hierarchy terms
            if 'hierarchy' in self.esco_skills:
                for skill_uri, hierarchy_info in self.esco_skills['hierarchy'].items():
                    term = hierarchy_info.get('term', '').lower().strip()
                    if term and term not in self._skill_term_to_uri_index:
                        self._skill_term_to_uri_index[term] = skill_uri

        skill_term_lower = skill_term.lower().strip()

        # Check cache first
        if skill_term_lower in self._skill_uri_cache:
            return self._skill_uri_cache[skill_term_lower]

        # Check pre-built index (O(1) lookup)
        if skill_term_lower in self._skill_term_to_uri_index:
            result = self._skill_term_to_uri_index[skill_term_lower]
            self._skill_uri_cache[skill_term_lower] = result
            return result

        # Fallback: Check descriptions (slower, only if exact match not found)
        for skill_uri, skill_info in self.esco_skills['skills'].items():
            if isinstance(skill_info, dict):
                description_raw = skill_info.get('description', '')
                if isinstance(description_raw, str):
                    description = description_raw.lower()
                    if skill_term_lower in description:
                        self._skill_uri_cache[skill_term_lower] = skill_uri
                        return skill_uri

        # Cache negative result too
        self._skill_uri_cache[skill_term_lower] = None
        return None

    def _get_skill_normalization_statistics(self, resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get skill normalization statistics for given resumes"""
        stats = {
            'total_skills_processed': 0,
            'successfully_normalized': 0,
            'found_in_esco': 0,
            'missing_from_esco': []
        }

        all_skills = set()
        for resume in resumes:
            resume_skills = self._extract_skills_set(resume)
            all_skills.update(resume_skills)

        stats['total_skills_processed'] = len(all_skills)

        for skill in all_skills:
            normalized = self.skill_normalizer.normalize_skill(skill)
            if normalized != skill:
                stats['successfully_normalized'] += 1

            if self.skill_normalizer.is_esco_skill(normalized, self.esco_skills.get('skills', {})):
                stats['found_in_esco'] += 1
            else:
                stats['missing_from_esco'].append(skill)

        return stats

    def _get_skill_term_by_uri(self, skill_uri: str) -> Optional[str]:
        """Get skill term by URI from ESCO hierarchy"""
        if not self.esco_skills or 'skills' not in self.esco_skills:
            return None

        skill_info = self.esco_skills['skills'].get(skill_uri)
        if skill_info:
            return skill_info.get('term', '')

        return None

    def _has_esco_prerequisites(self, skill_uri: str) -> bool:
        """Check if skill has prerequisites in ESCO hierarchy"""
        if not self.esco_skills or 'prerequisites' not in self.esco_skills:
            return False

        return skill_uri in self.esco_skills['prerequisites']

    def _get_esco_prerequisites(self, skill_uri: str) -> List[str]:
        """Get prerequisite skill URIs from ESCO hierarchy"""
        if not self.esco_skills or 'prerequisites' not in self.esco_skills:
            return []

        return self.esco_skills['prerequisites'].get(skill_uri, [])

    def _get_skill_level(self, skill_uri: str) -> Optional[int]:
        """Get skill level from ESCO hierarchy (0-3, where 0 is most general)"""
        if not self.esco_skills or 'levels' not in self.esco_skills:
            return None

        return self.esco_skills['levels'].get(skill_uri)

    def _validate_skill_level_progression(self,
                                          original_skills: Set[str],
                                          transformed_skills: Set[str],
                                          direction: str) -> bool:
        """Validate skill level progression using ESCO hierarchy levels"""
        if not self.esco_skills:
            return True  # Skip validation if ESCO not available

        # Get skill levels for original and transformed skills
        original_levels = []
        transformed_levels = []

        for skill in original_skills:
            skill_uri = self._find_skill_uri_by_term(skill)
            if skill_uri:
                level = self._get_skill_level(skill_uri)
                if level is not None:
                    original_levels.append(level)

        for skill in transformed_skills:
            skill_uri = self._find_skill_uri_by_term(skill)
            if skill_uri:
                level = self._get_skill_level(skill_uri)
                if level is not None:
                    transformed_levels.append(level)

        if not original_levels or not transformed_levels:
            return True  # Skip if no ESCO levels found

        # Calculate average skill levels
        avg_original_level = sum(original_levels) / len(original_levels)
        avg_transformed_level = sum(
            transformed_levels) / len(transformed_levels)

        if direction == 'upward':
            # For upward transformation, average skill level should increase or stay same
            return avg_transformed_level >= avg_original_level
        elif direction == 'downward':
            # For downward transformation, average skill level should decrease or stay same
            return avg_transformed_level <= avg_original_level + 0.5  # Allow small tolerance

        return True

    # Career Graph Integration Methods

    def _has_career_graph_data(self) -> bool:
        """Check if career graph has required data structure"""
        if not self.career_graph:
            return False

        required_keys = ['nodes', 'pathways',
                         'reverse_pathways', 'occupations']
        if not all(key in self.career_graph for key in required_keys):
            logger.debug("Career graph missing required keys")
            return False

        # Validate data structure integrity
        return self._validate_career_graph_structure()

    def _validate_career_graph_structure(self) -> bool:
        """Validate the internal structure of the career graph"""
        try:
            occupations = self.career_graph.get('occupations', {})
            pathways = self.career_graph.get('pathways', {})
            reverse_pathways = self.career_graph.get('reverse_pathways', {})
            nodes = self.career_graph.get('nodes', {})

            if not occupations:
                logger.debug("Career graph has no occupations")
                return False

            # Check that occupation URIs are consistent
            occupation_uris = set(occupations.values())
            pathway_uris = set(pathways.keys())

            # At least some occupations should have pathways
            if len(pathway_uris & occupation_uris) == 0:
                logger.debug("No pathway data for any occupations")
                return False

            # Check pathway consistency
            for uri, paths in pathways.items():
                if not isinstance(paths, list):
                    logger.debug(f"Invalid pathway data structure for {uri}")
                    return False

                # Check that pathway targets exist
                invalid_targets = [
                    p for p in paths if p not in occupation_uris]
                if len(invalid_targets) > len(paths) * 0.5:  # Allow some inconsistency
                    logger.debug(f"Too many invalid pathway targets for {uri}")
                    return False

            # Check reverse pathway consistency
            for uri, reverse_paths in reverse_pathways.items():
                if not isinstance(reverse_paths, list):
                    logger.debug(
                        f"Invalid reverse pathway data structure for {uri}")
                    return False

            logger.debug("Career graph structure validation passed")
            return True

        except Exception as e:
            logger.debug(f"Career graph structure validation error: {e}")
            return False

    def _validate_domain_boundaries_with_graph(self, original: Dict, transformed: Dict) -> bool:
        """Use career graph to validate domain consistency"""
        try:
            # Extract occupations from resumes
            original_occupation = self._extract_occupation_from_resume(
                original)
            transformed_occupation = self._extract_occupation_from_resume(
                transformed)

            if not original_occupation or not transformed_occupation:
                # Fallback to ESCO domain validation if occupations can't be extracted
                original_domain = self._identify_career_domain(original)
                transformed_domain = self._identify_career_domain(transformed)
                if original_domain and transformed_domain:
                    return original_domain == transformed_domain
                return True

            # Check if occupations are in same domain cluster in career graph
            return self._are_in_same_domain_cluster(original_occupation, transformed_occupation)

        except Exception as e:
            logger.debug(f"Career graph domain validation error: {e}")
            original_domain = self._identify_career_domain(original)
            transformed_domain = self._identify_career_domain(transformed)
            if original_domain and transformed_domain:
                return original_domain == transformed_domain
            return True

    def _extract_occupation_from_resume(self, resume: Dict[str, Any]) -> Optional[str]:
        """Extract occupation title from resume that matches career graph nodes with enhanced matching"""
        if not self.career_graph or 'occupations' not in self.career_graph:
            return None

        # Extract text from resume with priority on job titles and experience
        resume_text = self._extract_occupation_relevant_text(resume).lower()

        if not resume_text.strip():
            return None

        # Enhanced matching with multiple strategies
        best_match = None
        best_score = 0

        for occupation_title in self.career_graph['occupations'].keys():
            occupation_lower = occupation_title.lower()

            # Strategy 1: Exact phrase matching (highest weight)
            if occupation_lower in resume_text:
                score = len(occupation_lower) * 3
                if score > best_score:
                    best_score = score
                    best_match = occupation_title
                    continue

            # Strategy 2: Word-level matching
            occupation_words = set(occupation_lower.split())
            resume_words = set(resume_text.split())
            common_words = occupation_words & resume_words

            if common_words:
                # Score based on percentage of occupation words found
                word_coverage = len(common_words) / len(occupation_words)
                if word_coverage >= 0.6:  # At least 60% word match
                    score = word_coverage * len(occupation_lower) * 2
                    if score > best_score:
                        best_score = score
                        best_match = occupation_title

            # Strategy 3: Partial matching for compound titles
            occupation_parts = [part.strip()
                                for part in occupation_lower.split(',')]
            for part in occupation_parts:
                if len(part) > 3 and part in resume_text:
                    score = len(part) * 1.5
                    if score > best_score:
                        best_score = score
                        best_match = occupation_title

        # Only return if we have a reasonable confidence score
        return best_match if best_score > 15 else None

    def _extract_occupation_relevant_text(self, resume: Dict[str, Any]) -> str:
        """Extract text most relevant for occupation identification"""
        relevant_text = []

        # Priority 1: Job titles from experience
        if 'experience' in resume:
            exp = resume['experience']
            if isinstance(exp, list):
                for item in exp:
                    if isinstance(item, dict):
                        # Job title has highest priority
                        if 'title' in item:
                            # Weight job titles heavily
                            relevant_text.append(str(item['title']) * 3)
                        elif 'position' in item:
                            relevant_text.append(str(item['position']) * 3)
                        elif 'role' in item:
                            relevant_text.append(str(item['role']) * 3)

                        # Company and responsibilities have lower weight
                        if 'company' in item:
                            relevant_text.append(str(item['company']))
                        if 'responsibilities' in item:
                            resp = item['responsibilities']
                            if isinstance(resp, str):
                                relevant_text.append(resp)
                            elif isinstance(resp, list):
                                relevant_text.extend(str(r) for r in resp)
            elif isinstance(exp, str):
                relevant_text.append(exp * 2)  # Weight experience text

        # Priority 2: Summary/objective
        if 'summary' in resume:
            relevant_text.append(str(resume['summary']) * 2)
        if 'objective' in resume:
            relevant_text.append(str(resume['objective']) * 2)

        # Priority 3: Education (for academic/research roles)
        if 'education' in resume:
            edu = resume['education']
            if isinstance(edu, list):
                for item in edu:
                    if isinstance(item, dict) and 'degree' in item:
                        relevant_text.append(str(item['degree']))
            elif isinstance(edu, str):
                relevant_text.append(edu)

        # Priority 4: Skills (lower weight, for specialized roles)
        if 'skills' in resume:
            skills = resume['skills']
            if isinstance(skills, list):
                # Limit to avoid noise
                skills_text = ' '.join(str(skill) for skill in skills[:10])
                relevant_text.append(skills_text)

        return ' '.join(relevant_text)

    def _are_in_same_domain_cluster(self, occupation1: str, occupation2: str) -> bool:
        """Check if two occupations are in the same domain cluster using career graph"""
        if not self.career_graph or 'pathways' not in self.career_graph:
            return True

        # Get occupation URIs
        occupation1_uri = self.career_graph['occupations'].get(occupation1)
        occupation2_uri = self.career_graph['occupations'].get(occupation2)

        if not occupation1_uri or not occupation2_uri:
            return True  # Allow if occupations not found in graph

        # Check if there's a direct pathway between occupations (either direction)
        pathways1 = self.career_graph['pathways'].get(occupation1_uri, [])
        pathways2 = self.career_graph['pathways'].get(occupation2_uri, [])

        # Same occupation
        if occupation1_uri == occupation2_uri:
            return True

        # Direct pathway exists
        if occupation2_uri in pathways1 or occupation1_uri in pathways2:
            return True

        # Check reverse pathways
        reverse_pathways1 = self.career_graph['reverse_pathways'].get(
            occupation1_uri, [])
        reverse_pathways2 = self.career_graph['reverse_pathways'].get(
            occupation2_uri, [])

        if occupation2_uri in reverse_pathways1 or occupation1_uri in reverse_pathways2:
            return True

        # Check for common connections (shared pathway destinations)
        common_targets = set(pathways1) & set(pathways2)
        common_sources = set(reverse_pathways1) & set(reverse_pathways2)

        # If they share common career progression targets or sources, they're likely in same domain
        return len(common_targets) > 0 or len(common_sources) > 0

    def _validate_skills_with_career_graph(self, original_skills: Set[str],
                                           transformed_skills: Set[str],
                                           direction: str) -> bool:
        """Use career graph to validate skill progression with ESCO integration"""
        try:
            if not self.career_graph or 'nodes' not in self.career_graph:
                return True

            # Get occupation URIs from career graph for skill analysis
            career_occupations = self.career_graph.get('occupations', {})
            if not career_occupations:
                return True

            # Enhanced validation using ESCO occupation-skill relationships
            new_skills = transformed_skills - original_skills
            removed_skills = original_skills - transformed_skills

            # Find relevant occupations in career graph that match skill sets
            relevant_occupations = self._find_occupations_for_skills(
                transformed_skills)

            if direction == 'upward':
                # For upward progression, validate against higher-level occupations
                return self._validate_upward_skill_progression_with_graph(
                    original_skills, transformed_skills, relevant_occupations)
            elif direction == 'downward':
                # For downward progression, validate against entry-level occupations
                return self._validate_downward_skill_progression_with_graph(
                    original_skills, transformed_skills, relevant_occupations)

            return True

        except Exception as e:
            logger.debug(f"Career graph skill validation error: {e}")
            return True  # Allow on error

    def _find_occupations_for_skills(self, skills: Set[str]) -> List[str]:
        """Find career graph occupations that match the given skill set (optimized with caching)"""
        if not self.occupation_skill_relations or not self.career_graph:
            return []

        # OPTIMIZATION: Create a cache key from the sorted skill set
        skills_key = tuple(sorted(skills))

        if not hasattr(self, '_occupation_for_skills_cache'):
            self._occupation_for_skills_cache = {}

        if skills_key in self._occupation_for_skills_cache:
            return self._occupation_for_skills_cache[skills_key]

        # OPTIMIZATION: Convert resume skills to URIs once, outside the loop
        resume_skill_uris = set()
        for skill in skills:
            skill_uri = self._find_skill_uri_by_term(skill)
            if skill_uri:
                resume_skill_uris.add(skill_uri)

        # Early exit if no valid skill URIs found
        if not resume_skill_uris:
            self._occupation_for_skills_cache[skills_key] = []
            return []

        occupation_scores = {}

        for occupation_uri, relations in self.occupation_skill_relations.items():
            # Check if this occupation is in the career graph
            occupation_title = None
            for title, uri in self.career_graph.get('occupations', {}).items():
                if uri == occupation_uri:
                    occupation_title = title
                    break

            if not occupation_title:
                continue

            # Score based on skill overlap
            essential_skills = set(skill['skill_uri']
                                   for skill in relations.get('essential', []))
            optional_skills = set(skill['skill_uri']
                                  for skill in relations.get('optional', []))

            # Calculate overlap scores (skills already converted to URIs above)
            essential_overlap = len(resume_skill_uris & essential_skills)
            optional_overlap = len(resume_skill_uris & optional_skills)

            # Weight essential skills more heavily
            total_score = essential_overlap * 2 + optional_overlap

            if total_score > 0:
                occupation_scores[occupation_title] = total_score

        # Return top matching occupations
        result = sorted(occupation_scores.keys(),
                        key=lambda x: occupation_scores[x], reverse=True)[:5]

        # Cache the result
        self._occupation_for_skills_cache[skills_key] = result
        return result

    def _validate_upward_skill_progression_with_graph(self,
                                                      original_skills: Set[str],
                                                      transformed_skills: Set[str],
                                                      relevant_occupations: List[str]) -> bool:
        """Validate upward skill progression using career graph pathways"""
        if not relevant_occupations or not self.career_graph:
            return True

        new_skills = transformed_skills - original_skills

        # Check if new skills align with career progression pathways
        for occupation in relevant_occupations:
            occupation_uri = self.career_graph.get(
                'occupations', {}).get(occupation)
            if not occupation_uri:
                continue

            # Get forward pathways (higher-level positions)
            pathways = self.career_graph.get(
                'pathways', {}).get(occupation_uri, [])

            for pathway_uri in pathways:
                # Check if skills align with pathway requirements
                if self._skills_align_with_occupation(new_skills, pathway_uri):
                    return True

        # Allow if we can't find specific alignment (graceful degradation)
        return len(new_skills) > 0  # At least some skill growth

    def _validate_downward_skill_progression_with_graph(self,
                                                        original_skills: Set[str],
                                                        transformed_skills: Set[str],
                                                        relevant_occupations: List[str]) -> bool:
        """Validate downward skill progression using career graph pathways"""
        if not relevant_occupations or not self.career_graph:
            return True

        removed_skills = original_skills - transformed_skills

        # Check if skill removal aligns with entry-level positions
        for occupation in relevant_occupations:
            occupation_uri = self.career_graph.get(
                'occupations', {}).get(occupation)
            if not occupation_uri:
                continue

            # Get reverse pathways (entry-level positions)
            reverse_pathways = self.career_graph.get(
                'reverse_pathways', {}).get(occupation_uri, [])

            for pathway_uri in reverse_pathways:
                # Check if remaining skills align with entry-level requirements
                if self._skills_align_with_occupation(transformed_skills, pathway_uri):
                    return True

        # Allow if we can't find specific alignment (graceful degradation)
        return True  # More permissive for downward progression

    def _skills_align_with_occupation(self, skills: Set[str], occupation_uri: str) -> bool:
        """Check if skills align with occupation requirements"""
        if not self.occupation_skill_relations or occupation_uri not in self.occupation_skill_relations:
            return True  # Allow if no data available

        relations = self.occupation_skill_relations[occupation_uri]
        essential_skills = set(skill['skill_uri']
                               for skill in relations.get('essential', []))

        if not essential_skills:
            return True  # No requirements to check

        # Convert skills to URIs
        skill_uris = set()
        for skill in skills:
            skill_uri = self._find_skill_uri_by_term(skill)
            if skill_uri:
                skill_uris.add(skill_uri)

        # Check if at least 25% of essential skills are covered
        overlap = len(skill_uris & essential_skills)
        coverage_ratio = overlap / \
            len(essential_skills) if essential_skills else 0

        return coverage_ratio >= 0.25

    def _validate_career_pathway(self, original_occupation: str, target_occupation: str,
                                 direction: str) -> bool:
        """Validate if career transition exists in career graph"""
        if not self.career_graph or 'pathways' not in self.career_graph:
            return True  # Skip if no career graph

        try:
            # Get occupation URIs
            original_uri = self.career_graph['occupations'].get(
                original_occupation)
            target_uri = self.career_graph['occupations'].get(
                target_occupation)

            if not original_uri or not target_uri:
                return True  # Allow if occupations not found

            if direction == 'upward':
                # Check if target is in forward pathways from original
                pathways = self.career_graph['pathways'].get(original_uri, [])
                return target_uri in pathways
            elif direction == 'downward':
                # Check if target is in reverse pathways from original
                reverse_pathways = self.career_graph['reverse_pathways'].get(
                    original_uri, [])
                return target_uri in reverse_pathways

            return True

        except Exception as e:
            logger.debug(f"Career pathway validation error: {e}")
            return True  # Allow on error

    def get_career_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded career graph"""
        if not self.career_graph:
            return {'error': 'No career graph loaded'}

        try:
            stats = {
                'total_occupations': len(self.career_graph.get('occupations', {})),
                'total_nodes': len(self.career_graph.get('nodes', {})),
                'total_edges': len(self.career_graph.get('edges', {})),
                'has_required_structure': self._has_career_graph_data(),
                'validation_status': 'valid' if self._validate_career_graph_structure() else 'invalid'
            }

            # Pathway statistics
            pathways = self.career_graph.get('pathways', {})
            if pathways:
                pathway_counts = [len(paths) for paths in pathways.values()]
                stats['occupations_with_pathways'] = len([
                    occ for occ, paths in pathways.items() if paths
                ])

                if pathway_counts:
                    stats['avg_pathways_per_occupation'] = sum(
                        pathway_counts) / len(pathway_counts)
                    stats['max_pathways_per_occupation'] = max(pathway_counts)
                    stats['min_pathways_per_occupation'] = min(pathway_counts)
                    stats['total_pathway_connections'] = sum(pathway_counts)

            # Reverse pathway statistics
            reverse_pathways = self.career_graph.get('reverse_pathways', {})
            if reverse_pathways:
                reverse_counts = [len(paths)
                                  for paths in reverse_pathways.values()]
                stats['occupations_with_reverse_pathways'] = len([
                    occ for occ, paths in reverse_pathways.items() if paths
                ])

                if reverse_counts:
                    stats['avg_reverse_pathways_per_occupation'] = sum(
                        reverse_counts) / len(reverse_counts)

            # Integration with ESCO data
            if self.occupation_skill_relations:
                # Count how many career graph occupations have ESCO skill data
                occupation_uris = set(
                    self.career_graph.get('occupations', {}).values())
                esco_occupation_uris = set(
                    self.occupation_skill_relations.keys())
                overlap = occupation_uris & esco_occupation_uris

                stats['esco_integration'] = {
                    'career_graph_occupations_with_esco_data': len(overlap),
                    'esco_coverage_percentage': (len(overlap) / len(occupation_uris) * 100) if occupation_uris else 0,
                    'total_esco_occupations': len(esco_occupation_uris)
                }

            # Domain clustering analysis
            if pathways:
                stats['domain_clustering'] = self._analyze_domain_clustering()

            return stats

        except Exception as e:
            logger.error(f"Error generating career graph statistics: {e}")
            return {'error': f'Failed to generate statistics: {e}'}

    def _analyze_domain_clustering(self) -> Dict[str, Any]:
        """Analyze domain clustering in the career graph"""
        try:
            pathways = self.career_graph.get('pathways', {})
            occupations = self.career_graph.get('occupations', {})

            # Build connectivity matrix
            uri_to_title = {uri: title for title, uri in occupations.items()}

            # Find strongly connected components (domains)
            domains = []
            visited = set()

            for occupation_uri in pathways.keys():
                if occupation_uri not in visited:
                    domain = self._find_connected_occupations(
                        occupation_uri, pathways, visited)
                    if len(domain) > 1:  # Only consider clusters with multiple occupations
                        domain_titles = [uri_to_title.get(
                            uri, uri) for uri in domain]
                        domains.append(domain_titles)

            return {
                'total_domains': len(domains),
                'largest_domain_size': max(len(d) for d in domains) if domains else 0,
                'average_domain_size': sum(len(d) for d in domains) / len(domains) if domains else 0,
                'isolated_occupations': len(pathways) - sum(len(d) for d in domains),
                'domain_distribution': [len(d) for d in domains]
            }

        except Exception as e:
            logger.debug(f"Domain clustering analysis error: {e}")
            return {'error': 'Analysis failed'}

    def _find_connected_occupations(self, start_uri: str, pathways: Dict, visited: set) -> List[str]:
        """Find all occupations connected to the starting occupation"""
        if start_uri in visited:
            return []

        connected = [start_uri]
        visited.add(start_uri)
        queue = [start_uri]

        while queue:
            current = queue.pop(0)

            # Forward connections
            for target in pathways.get(current, []):
                if target not in visited:
                    visited.add(target)
                    connected.append(target)
                    queue.append(target)

            # Backward connections
            for uri, targets in pathways.items():
                if current in targets and uri not in visited:
                    visited.add(uri)
                    connected.append(uri)
                    queue.append(uri)

        return connected

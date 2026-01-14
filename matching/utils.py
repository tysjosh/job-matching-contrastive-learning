
import re
import json


class SkillExtractor:
    """Extract skills from resume/job text using ESCO mappings and CS skills database"""

    def __init__(self, esco_skills_path="dataset/esco_skills.json", cs_skills_path="dataset/cs_skills.json"):
        self.esco_skills = self._load_esco_skills(esco_skills_path)
        self.cs_skills = self._load_cs_skills(cs_skills_path)
        # This is for a more advanced, regex-based implementation
        self.skill_patterns = self._load_skill_patterns()

    def _load_esco_skills(self, file_path: str) -> dict:
        """Load ESCO skill taxonomy from the processed JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(
                f"Warning: ESCO skills file not found at {file_path}. Skill extraction will be limited.")
            return {}

    def _load_cs_skills(self, file_path: str) -> dict:
        """Load comprehensive CS skills database from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(
                f"Warning: CS skills file not found at {file_path}. Using basic fallback skills.")
            return {}

    def _get_all_cs_skills(self) -> list[str]:
        """Get all skills from the CS skills database as a flat list."""
        all_skills = []
        if self.cs_skills:
            for category, skills in self.cs_skills.items():
                if isinstance(skills, list):
                    all_skills.extend(skills)
        return all_skills

    def extract_skills(self, text: str) -> list[str]:
        """Extract skills from text by matching against the ESCO skill list and CS skills database."""
        found_skills = set()
        text_lower = text.lower()

        # Pass 1: Match against the ESCO skills list
        if self.esco_skills:
            for skill_uri, skill_info in self.esco_skills.items():
                # Handle both 'preferredLabel' and 'preferred_label' key formats
                skill_label = skill_info.get(
                    'preferredLabel') or skill_info.get('preferred_label', '')
                if skill_label and re.search(r'\b' + re.escape(skill_label.lower()) + r'\b', text_lower):
                    found_skills.add(skill_label)

        # Pass 2: Match against the regex patterns for common skills and variations
        for skill_name, patterns in self.skill_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_skills.add(skill_name)

        # Pass 3: Match against comprehensive CS skills database
        cs_skills = self._get_all_cs_skills()
        for skill in cs_skills:
            if skill and re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.add(skill)

        # Fallback to a basic list only if no CS skills database is available
        if not found_skills and not self.cs_skills:
            basic_fallback_skills = ['Python', 'JavaScript', 'React', 'SQL', 'AWS', 'Docker',
                                     'Machine Learning', 'System Design', 'Leadership', 'Project Management']
            for skill in basic_fallback_skills:
                if skill.lower() in text_lower:
                    found_skills.add(skill)

        return list(found_skills)  # Return unique skills

    def extract_required_skills(self, job_text: str) -> list[str]:
        """Extract required skills from job description"""
        # This is a simplified implementation. A real-world version would
        # parse sections like "Requirements" or "Must-Haves".
        return self.extract_skills(job_text)

    def extract_preferred_skills(self, job_text: str) -> list[str]:
        """Extract preferred skills from job description by parsing specific sections."""
        preferred_skills = []
        # Define regex patterns for preferred skills sections
        preferred_section_patterns = [
            r"(preferred skills|nice to have|bonus points|additional skills|good to have)s?:\s*(.*?)(?:\n\n|Requirements:)",
            r"(preferred qualifications|desired skills):\s*(.*?)(?:\n\n|Responsibilities:)"
        ]

        job_text_lower = job_text.lower()
        for pattern in preferred_section_patterns:
            match = re.search(pattern, job_text_lower,
                              re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(2)
                # Use the main skill extractor on the identified section
                preferred_skills.extend(self.extract_skills(section_text))

        return list(set(preferred_skills))

    def _load_skill_patterns(self) -> dict:
        """Load a dictionary of regex patterns for skill extraction."""
        # Build patterns from CS skills data if available
        patterns = {}

        if self.cs_skills:
            # Add patterns for programming languages with common variations
            programming_languages = self.cs_skills.get(
                'programming_languages', [])
            for lang in programming_languages:
                lang_patterns = []
                if lang == "JavaScript":
                    lang_patterns = [r'\bjs\b', r'javascript']
                elif lang == "Python":
                    lang_patterns = [r'\bpython\b']
                elif lang == "TypeScript":
                    lang_patterns = [r'\bts\b', r'typescript']
                elif lang == "C++":
                    lang_patterns = [r'\bc\+\+\b', r'\bcpp\b']
                elif lang == "C#":
                    lang_patterns = [r'\bc#\b', r'\bc-sharp\b']
                else:
                    # Generic pattern for exact matches
                    lang_patterns = [r'\b' + re.escape(lang.lower()) + r'\b']

                if lang_patterns:
                    patterns[lang] = lang_patterns

            # Add patterns for frameworks with variations
            frameworks = self.cs_skills.get('frameworks', [])
            for framework in frameworks:
                framework_patterns = []
                if framework == "React":
                    framework_patterns = [r'react(?:\.js)?\b']
                elif framework == "Vue.js":
                    framework_patterns = [r'vue(?:\.js)?\b']
                elif framework == "Angular":
                    framework_patterns = [r'\bangular\b']
                else:
                    # Generic pattern
                    framework_patterns = [
                        r'\b' + re.escape(framework.lower()) + r'\b']

                if framework_patterns:
                    patterns[framework] = framework_patterns

            # Add patterns for cloud platforms
            cloud_platforms = self.cs_skills.get('cloud_platforms', [])
            for platform in cloud_platforms:
                platform_patterns = []
                if "AWS" in platform:
                    platform_patterns = [r'\baws\b', r'amazon web services']
                elif "Azure" in platform:
                    platform_patterns = [r'\bazure\b']
                elif "GCP" in platform or "Google Cloud" in platform:
                    platform_patterns = [r'\bgcp\b', r'google cloud platform']
                else:
                    # Generic pattern
                    platform_patterns = [
                        r'\b' + re.escape(platform.lower()) + r'\b']

                if platform_patterns:
                    patterns[platform] = platform_patterns

        # Fallback to basic hardcoded patterns if CS skills not available
        if not patterns:
            patterns = {
                "JavaScript": [r'\bjs\b', r'javascript'],
                "Python": [r'\bpython\b'],
                "React": [r'react(?:\.js)?\b'],
                "Node.js": [r'node(?:\.js)?\b'],
                "SQL": [r'\bsql\b'],
                "NoSQL": [r'\bnosql\b'],
                "PostgreSQL": [r'postgres(?:ql)?\b'],
                "MongoDB": [r'mongo(?:db)?\b'],
                "AWS": [r'\baws\b', 'amazon web services'],
                "Azure": [r'\bazure\b'],
                "GCP": [r'\bgcp\b', 'google cloud platform'],
                "Docker": [r'\bdocker\b'],
                "Kubernetes": [r'\bk8s\b', r'kubernetes'],
                "CI/CD": [r'\bci/cd\b', 'continuous integration', 'continuous delivery', 'continuous deployment'],
                "Git": [r'\bgit\b'],
                "Jira": [r'\bjira\b'],
                "Agile": [r'\bagile\b'],
                "Scrum": [r'\bscrum\b']
            }

        return patterns


class CareerLevelClassifier:
    """Classify career level from resume/job text"""

    def __init__(self, career_graph, config: dict = None):
        self.career_graph = career_graph
        self.level_indicators = self._load_level_indicators(config)

    def _load_level_indicators(self, config: dict) -> dict:
        """Load career level classification indicators from config or use defaults."""
        if config and 'career_level_classifier' in config:
            return config['career_level_classifier']['level_indicators']
        else:
            print(
                "Warning: Career level indicators not found in config. Using default values.")
            return {
                'junior': ['junior', 'entry', 'intern', 'graduate', 'associate'],
                'mid': ['developer', 'engineer', 'analyst'],
                'senior': ['senior', 'lead', 'principal', 'staff', 'architect']
            }

    def infer_level(self, text: str) -> str:
        """Infer career level from text content."""
        text_lower = text.lower()

        # Check for senior level indicators
        if any(indicator in text_lower for indicator in self.level_indicators.get('senior', [])):
            return 'Senior Developer'

        # Check for junior level indicators
        if any(indicator in text_lower for indicator in self.level_indicators.get('junior', [])):
            return 'Junior Developer'

        # Infer level based on years of experience
        years_match = re.search(r'(\d+)\s*years', text_lower)
        if years_match:
            try:
                years = int(years_match.group(1))
                if years >= 5:
                    return 'Senior Developer'
                elif years >= 2:
                    return 'Software Developer'
                else:
                    return 'Junior Developer'
            except (ValueError, IndexError):
                pass

        # Default to mid-level if no other indicators are found
        return 'Software Developer'

from typing import Dict, List

class ExplanationTextGenerator:
    """
    COMPONENT 3 - Complete implementation with all supporting methods
    """
    def __init__(self, career_graph):
        self.templates = self._load_explanation_templates()
        self.career_graph = career_graph
    
    def generate_explanation(self, resume_data, job_data, similarity_score, audience):
        """
        MAIN ENTRY POINT - Called by ExplainableJobMatcher
        """
        
        # Analyze components (detailed analysis methods from earlier)
        skill_analysis = self._analyze_skill_alignment(resume_data, job_data)
        career_analysis = self._analyze_career_alignment(resume_data, job_data)
        
        # Select and fill template (detailed template methods from earlier)
        template_key = self._select_template(skill_analysis, career_analysis, audience)
        explanation_text = self._generate_explanation_text(
            template_key, skill_analysis, career_analysis, similarity_score
        )
        
        return {
            'similarity_score': similarity_score,
            'skill_analysis': skill_analysis,
            'career_analysis': career_analysis,
            'explanation_text': explanation_text
        }
    
    # ==== DETAILED ANALYSIS METHODS (from earlier version) ====
    
    def _analyze_skill_alignment(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Analyze skill overlap between candidate and job requirements"""
        
        candidate_skills = set(resume_data['skills'])
        required_skills = set(job_data['required_skills'])
        preferred_skills = set(job_data.get('preferred_skills', []))
        
        # Calculate overlaps
        required_overlap = candidate_skills.intersection(required_skills)
        preferred_overlap = candidate_skills.intersection(preferred_skills)
        missing_required = required_skills - candidate_skills
        missing_preferred = preferred_skills - candidate_skills
        
        # Calculate coverage scores
        required_coverage = len(required_overlap) / len(required_skills) if required_skills else 1.0
        preferred_coverage = len(preferred_overlap) / len(preferred_skills) if preferred_skills else 1.0
        
        return {
            'required_skill_coverage': required_coverage,
            'preferred_skill_coverage': preferred_coverage,
            'matching_required_skills': list(required_overlap),
            'matching_preferred_skills': list(preferred_overlap),
            'missing_required_skills': list(missing_required),
            'missing_preferred_skills': list(missing_preferred)
        }
    
    def _analyze_career_alignment(self, resume_data: Dict, job_data: Dict) -> Dict:
        """Analyze career level alignment between candidate and role"""
        
        candidate_level = resume_data['career_level']
        target_level = job_data['career_level']
        career_distance = self._compute_career_distance(candidate_level, target_level)
        
        if career_distance == 0:
            alignment_type = "perfect_match"
            description = f"Candidate at {candidate_level} level matches {target_level} role requirements"
        elif career_distance == 1:
            alignment_type = "adjacent_level"
            description = f"Candidate at {candidate_level} level, one step from {target_level}"
        elif career_distance == 2:
            alignment_type = "stretch_opportunity"  
            description = f"Candidate at {candidate_level} level, two steps from {target_level}"
        else:
            alignment_type = "major_mismatch"
            description = f"Large career gap between {candidate_level} and {target_level}"
        
        return {
            'alignment_type': alignment_type,
            'career_distance': career_distance,
            'candidate_level': candidate_level,
            'target_level': target_level,
            'description': description
        }
    
    # ==== TEMPLATE METHODS (from earlier version) ====
    
    def _select_template(self, skill_analysis: Dict, career_analysis: Dict, audience: str) -> str:
        """Select template based on analysis results"""
        
        career_distance = career_analysis['career_distance']
        skill_coverage = skill_analysis['required_skill_coverage']
        
        if career_distance == 0 and skill_coverage >= 0.8:
            return f"excellent_match_{audience}"
        elif career_distance <= 1 and skill_coverage >= 0.6:
            return f"good_match_{audience}"
        elif skill_coverage < 0.5:
            return f"skill_gap_focus_{audience}"
        else:
            return f"general_assessment_{audience}"
    
    def _generate_explanation_text(self, template_key: str, skill_analysis: Dict, 
                                 career_analysis: Dict, similarity_score: float) -> str:
        """Generate explanation text using selected template"""
        
        # Extract metrics for template filling
        metrics = self._extract_metrics(skill_analysis, career_analysis, similarity_score)
        
        # Generate dynamic content sections  
        dynamic_content = self._generate_dynamic_content(skill_analysis, career_analysis)
        
        # Get and fill template
        template = self.templates[template_key]
        filled_template = template.format(**metrics, **dynamic_content)
        
        return filled_template
    
    def _extract_metrics(self, skill_analysis: Dict, career_analysis: Dict, 
                        similarity_score: float) -> Dict[str, str]:
        """Extract formatted metrics for template filling"""
        
        # Calculate skill fraction
        total_required = len(skill_analysis.get('matching_required_skills', [])) + \
                        len(skill_analysis.get('missing_required_skills', []))
        matched_count = len(skill_analysis.get('matching_required_skills', []))
        
        return {
            'skill_fraction': f"{matched_count}/{total_required}",
            'coverage_percent': f"{skill_analysis['required_skill_coverage']:.0%}",
            'career_distance': str(career_analysis['career_distance']),
            'candidate_level': career_analysis['candidate_level'],
            'target_level': career_analysis['target_level'], 
            'match_score': f"{similarity_score:.2f}"
        }
    
    def _generate_dynamic_content(self, skill_analysis: Dict, career_analysis: Dict) -> Dict[str, str]:
        """Generate dynamic text sections based on structured data"""
        
        dynamic_content = {}
        
        # Generate skill lists (KEEP - this is for match explanation)
        if skill_analysis.get('matching_required_skills'):
            skills_list = skill_analysis['matching_required_skills'][:3]
            dynamic_content['matching_skills'] = self._format_skill_list(skills_list)
        else:
            dynamic_content['matching_skills'] = "limited skill overlap"
        
        # Generate career progression text (KEEP - this explains match compatibility)
        dynamic_content['career_progression'] = self._generate_career_progression_text(
            career_analysis
        )
        
        # REMOVE career development recommendations - not needed for matching
        
        return dynamic_content
    
    def _format_skill_list(self, skills: List[str]) -> str:
        """Format skill lists with proper grammar"""
        if len(skills) == 1:
            return skills[0]
        elif len(skills) == 2:
            return f"{skills[0]} and {skills[1]}"
        else:
            return f"{', '.join(skills[:-1])}, and {skills[-1]}"
    
    def _generate_career_progression_text(self, career_analysis: Dict) -> str:
        """Generate career alignment explanation text"""
        
        distance = career_analysis['career_distance']
        candidate_level = career_analysis['candidate_level']
        target_level = career_analysis['target_level']
        
        if distance == 0:
            return f"at the same career level ({candidate_level})"
        elif distance == 1:
            return f"one level away from the target role (currently {candidate_level}, targeting {target_level})"
        else:
            return f"{distance} career levels away from the target role"
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load templates focused on match assessment only"""
        
        return {
            'excellent_match_candidate': '''
**Excellent Match** (Score: {match_score})

Your profile strongly aligns with this role. You demonstrate {skill_fraction} of the core requirements including {matching_skills}. You are {career_progression}, indicating strong compatibility for this position.
            '''.strip(),
            
            'good_match_candidate': '''
**Good Match** (Score: {match_score})  

You show {coverage_percent} alignment with the role requirements, demonstrating {matching_skills}. You are currently {career_progression}, representing reasonable compatibility with this position.
            '''.strip(),
            
            'skill_gap_focus_candidate': '''
**Moderate Match** (Score: {match_score})

You match {skill_fraction} of the requirements with {matching_skills}. You are {career_progression}. While there are skill gaps to consider, the match shows potential compatibility.
            '''.strip(),
            
            'excellent_match_recruiter': '''
**Strong Candidate** (Score: {match_score})

This candidate demonstrates excellent role alignment with {coverage_percent} requirement coverage, including {matching_skills}. The candidate is {career_progression}, indicating strong role-level compatibility.
            '''.strip(),
            
            'good_match_recruiter': '''
**Good Candidate** (Score: {match_score})

Candidate shows {coverage_percent} requirement alignment including {matching_skills}. Candidate is {career_progression}, representing reasonable role compatibility.
            '''.strip(),
            
            'skill_gap_focus_recruiter': '''
**Moderate Candidate** (Score: {match_score})

Candidate meets {skill_fraction} of role requirements with {matching_skills}. Currently {career_progression}. Shows potential compatibility despite some requirement gaps.
            '''.strip(),
        }

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
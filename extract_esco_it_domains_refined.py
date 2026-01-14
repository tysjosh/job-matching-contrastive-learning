#!/usr/bin/env python3
"""
Refined ESCO IT/CS Career Domains Extraction

This script provides more precise extraction of IT/CS occupations from ESCO data
by using stricter filtering and validation.
"""

import pandas as pd
import re
from collections import defaultdict
from typing import Dict, List, Set
import json

def load_esco_it_occupations_refined(occupations_file: str) -> pd.DataFrame:
    """Load and filter ESCO occupations for IT/CS jobs with refined precision"""
    
    print("📊 Loading ESCO occupations data...")
    df = pd.read_csv(occupations_file)
    
    # Primary IT/CS ISCO groups (most reliable)
    primary_it_isco = ['25']  # ICT professionals - this is the main IT group
    
    # Secondary IT-related ISCO groups (need keyword validation)
    secondary_it_isco = ['35', '133', '214', '252']
    
    # Strict IT/CS keywords for primary filtering
    strict_it_keywords = [
        # Software Development
        'software developer', 'programmer', 'software engineer', 'application developer',
        'web developer', 'mobile developer', 'frontend developer', 'backend developer',
        'full stack developer', 'fullstack developer', 'game developer',
        
        # Data & Analytics
        'data scientist', 'data analyst', 'data engineer', 'business intelligence analyst',
        'machine learning engineer', 'ai engineer', 'artificial intelligence',
        
        # Infrastructure & Systems
        'system administrator', 'network administrator', 'database administrator',
        'cloud engineer', 'cloud architect', 'devops engineer', 'site reliability engineer',
        
        # Security
        'security analyst', 'cybersecurity', 'information security', 'security engineer',
        'penetration tester', 'ethical hacker',
        
        # Design & UX
        'ui designer', 'ux designer', 'user experience designer', 'user interface designer',
        'interaction designer', 'product designer',
        
        # Architecture & Leadership
        'software architect', 'solution architect', 'technical architect',
        'technical lead', 'engineering manager', 'cto', 'cio',
        
        # Quality & Testing
        'qa engineer', 'test engineer', 'automation tester', 'quality assurance',
        
        # Project Management (IT-specific)
        'technical project manager', 'product manager', 'scrum master', 'agile coach'
    ]
    
    # Exclusion keywords to filter out non-IT jobs
    exclusion_keywords = [
        'animal', 'agriculture', 'farming', 'medical', 'healthcare', 'hospital',
        'construction', 'building', 'manufacturing', 'factory', 'production',
        'retail', 'sales', 'marketing', 'finance', 'accounting', 'legal',
        'education', 'teaching', 'school', 'university', 'research',
        'transportation', 'logistics', 'shipping', 'warehouse',
        'food', 'restaurant', 'hotel', 'tourism', 'travel',
        'art', 'music', 'entertainment', 'media', 'journalism',
        'government', 'public', 'military', 'police', 'fire'
    ]
    
    # Filter by primary ISCO groups (high confidence)
    primary_filter = df['iscoGroup'].astype(str).str.startswith(tuple(primary_it_isco))
    primary_occupations = df[primary_filter].copy()
    
    # Filter by strict keywords for secondary groups
    secondary_filter = df['iscoGroup'].astype(str).str.startswith(tuple(secondary_it_isco))
    secondary_candidates = df[secondary_filter].copy()
    
    # Apply keyword filtering to secondary candidates
    keyword_matches = []
    for _, row in secondary_candidates.iterrows():
        title = str(row['preferredLabel']).lower()
        alt_labels = str(row.get('altLabels', '')).lower()
        all_text = f"{title} {alt_labels}"
        
        # Check for strict IT keywords
        has_it_keyword = any(keyword.lower() in all_text for keyword in strict_it_keywords)
        
        # Check for exclusion keywords
        has_exclusion = any(exclusion.lower() in all_text for exclusion in exclusion_keywords)
        
        if has_it_keyword and not has_exclusion:
            keyword_matches.append(row)
    
    secondary_occupations = pd.DataFrame(keyword_matches) if keyword_matches else pd.DataFrame()
    
    # Additional keyword-based filtering for any remaining occupations
    remaining_df = df[~df.index.isin(primary_occupations.index) & 
                     ~df.index.isin(secondary_occupations.index)]
    
    additional_matches = []
    for _, row in remaining_df.iterrows():
        title = str(row['preferredLabel']).lower()
        alt_labels = str(row.get('altLabels', '')).lower()
        all_text = f"{title} {alt_labels}"
        
        # Very strict matching for remaining occupations
        strict_matches = [
            'software', 'developer', 'programmer', 'data scientist', 'data analyst',
            'cybersecurity', 'information security', 'database administrator',
            'system administrator', 'network administrator', 'cloud engineer',
            'devops', 'ui designer', 'ux designer', 'qa engineer'
        ]
        
        has_strict_match = any(match in all_text for match in strict_matches)
        has_exclusion = any(exclusion.lower() in all_text for exclusion in exclusion_keywords)
        
        if has_strict_match and not has_exclusion:
            additional_matches.append(row)
    
    additional_occupations = pd.DataFrame(additional_matches) if additional_matches else pd.DataFrame()
    
    # Combine all filtered occupations
    it_occupations = pd.concat([primary_occupations, secondary_occupations, additional_occupations], 
                              ignore_index=True).drop_duplicates()
    
    print(f"✅ Found {len(it_occupations)} refined IT/CS occupations")
    print(f"  📊 Primary ISCO (25): {len(primary_occupations)}")
    print(f"  📊 Secondary ISCO + keywords: {len(secondary_occupations)}")
    print(f"  📊 Additional keyword matches: {len(additional_occupations)}")
    
    return it_occupations

def extract_refined_it_career_domains(it_occupations_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Extract IT/CS career domains with refined precision"""
    
    print("🔍 Extracting refined IT/CS career domains...")
    
    # More precise domain patterns
    domain_patterns = {
        'software_development': {
            'required': ['software', 'developer', 'programmer', 'engineer'],
            'keywords': [
                'software developer', 'programmer', 'software engineer', 'application developer',
                'backend developer', 'frontend developer', 'full stack developer', 'fullstack developer',
                'game developer', 'embedded developer', 'mobile app developer'
            ]
        },
        'data_science': {
            'required': ['data'],
            'keywords': [
                'data scientist', 'data analyst', 'business intelligence', 'data engineer',
                'machine learning', 'artificial intelligence', 'ai engineer', 'ml engineer',
                'research scientist', 'statistician', 'analytics specialist'
            ]
        },
        'cybersecurity': {
            'required': ['security', 'cyber'],
            'keywords': [
                'security analyst', 'cybersecurity', 'information security', 'security engineer',
                'penetration tester', 'ethical hacker', 'security consultant', 'security specialist',
                'incident response', 'forensic analyst'
            ]
        },
        'systems_administration': {
            'required': ['system', 'administrator', 'admin'],
            'keywords': [
                'system administrator', 'sysadmin', 'network administrator', 'it administrator',
                'server administrator', 'infrastructure engineer', 'systems engineer'
            ]
        },
        'database_administration': {
            'required': ['database'],
            'keywords': [
                'database administrator', 'dba', 'database engineer', 'database analyst',
                'database developer', 'sql developer'
            ]
        },
        'web_development': {
            'required': ['web'],
            'keywords': [
                'web developer', 'frontend developer', 'backend developer', 'fullstack developer',
                'web designer', 'web engineer'
            ]
        },
        'mobile_development': {
            'required': ['mobile', 'app'],
            'keywords': [
                'mobile developer', 'ios developer', 'android developer', 'mobile app developer',
                'app developer'
            ]
        },
        'cloud_engineering': {
            'required': ['cloud'],
            'keywords': [
                'cloud engineer', 'cloud architect', 'aws engineer', 'azure engineer',
                'cloud specialist', 'cloud consultant'
            ]
        },
        'devops': {
            'required': ['devops', 'sre', 'site reliability'],
            'keywords': [
                'devops engineer', 'site reliability engineer', 'sre', 'deployment engineer',
                'automation engineer', 'build engineer', 'release engineer'
            ]
        },
        'ui_ux_design': {
            'required': ['ui', 'ux', 'user interface', 'user experience', 'design'],
            'keywords': [
                'ui designer', 'ux designer', 'user interface designer', 'user experience designer',
                'interaction designer', 'product designer', 'visual designer'
            ]
        },
        'software_architecture': {
            'required': ['architect'],
            'keywords': [
                'software architect', 'solution architect', 'system architect',
                'technical architect', 'enterprise architect'
            ]
        },
        'quality_assurance': {
            'required': ['qa', 'quality', 'test'],
            'keywords': [
                'qa engineer', 'quality assurance', 'test engineer', 'automation tester',
                'software tester', 'quality analyst'
            ]
        },
        'project_management': {
            'required': ['project manager', 'product manager', 'scrum', 'agile'],
            'keywords': [
                'technical project manager', 'product manager', 'scrum master',
                'agile coach', 'delivery manager', 'program manager'
            ]
        }
    }
    
    career_domains = {}
    
    for domain, config in domain_patterns.items():
        domain_occupations = []
        required_terms = config['required']
        keywords = config['keywords']
        
        for _, row in it_occupations_df.iterrows():
            title = str(row['preferredLabel']).lower()
            alt_labels = str(row.get('altLabels', '')).lower()
            all_text = f"{title} {alt_labels}"
            
            # Check if occupation has required terms
            has_required = any(req.lower() in all_text for req in required_terms)
            
            # Check if occupation matches specific keywords
            has_keyword = any(keyword.lower() in all_text for keyword in keywords)
            
            if has_required or has_keyword:
                # Add main title
                clean_title = row['preferredLabel'].strip()
                if clean_title and clean_title not in domain_occupations:
                    domain_occupations.append(clean_title)
                
                # Add relevant alternative labels
                if pd.notna(row.get('altLabels')):
                    for alt_label in str(row['altLabels']).split('\n'):
                        alt_clean = alt_label.strip()
                        if alt_clean and alt_clean not in domain_occupations:
                            alt_lower = alt_clean.lower()
                            if any(req.lower() in alt_lower for req in required_terms) or \
                               any(keyword.lower() in alt_lower for keyword in keywords):
                                domain_occupations.append(alt_clean)
        
        # Clean and sort
        career_domains[domain] = sorted(list(set(domain_occupations)))
        print(f"  📋 {domain}: {len(career_domains[domain])} occupations")
    
    return career_domains

def validate_extracted_domains(career_domains: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Validate and clean extracted domains"""
    
    print("🔍 Validating extracted domains...")
    
    validated_domains = {}
    
    for domain, occupations in career_domains.items():
        validated_occupations = []
        
        for occupation in occupations:
            # Basic validation
            if len(occupation.strip()) > 3 and occupation.strip().lower() not in [
                'data', 'system', 'network', 'web', 'mobile', 'cloud', 'software'
            ]:
                validated_occupations.append(occupation.strip())
        
        if validated_occupations:
            validated_domains[domain] = validated_occupations[:100]  # Limit to 100 per domain
            print(f"  ✅ {domain}: {len(validated_domains[domain])} validated occupations")
    
    return validated_domains

def generate_refined_esco_config(occupations_file: str) -> Dict:
    """Generate refined ESCO-based IT/CS configuration"""
    
    print("🚀 Starting Refined ESCO IT/CS Career Domains Extraction")
    print("=" * 60)
    
    # Load refined IT/CS occupations
    it_occupations = load_esco_it_occupations_refined(occupations_file)
    
    # Extract career domains
    career_domains = extract_refined_it_career_domains(it_occupations)
    
    # Validate domains
    validated_domains = validate_extracted_domains(career_domains)
    
    # Create configuration
    esco_config = {
        'career_domains': validated_domains,
        'metadata': {
            'source': 'ESCO European Skills, Competences, Qualifications and Occupations',
            'focus': 'IT/CS occupations only (refined extraction)',
            'version': '1.2.0',
            'extraction_date': pd.Timestamp.now().isoformat(),
            'total_it_occupations': len(it_occupations),
            'domains_extracted': len(validated_domains),
            'extraction_method': 'refined_filtering_with_validation'
        }
    }
    
    return esco_config

if __name__ == "__main__":
    # Configuration
    occupations_file = "dataset/esco/occupations_en.csv"
    output_file = "esco_it_career_domains_refined.json"
    
    try:
        # Generate refined ESCO IT/CS configuration
        config = generate_refined_esco_config(occupations_file)
        
        # Save configuration
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Refined configuration saved to: {output_file}")
        
        # Print summary
        print("\n📋 Refined ESCO IT/CS Career Domains Summary")
        print("=" * 50)
        
        career_domains = config['career_domains']
        total_occupations = sum(len(occupations) for occupations in career_domains.values())
        
        print(f"📊 Total domains: {len(career_domains)}")
        print(f"📊 Total IT/CS occupations: {total_occupations}")
        
        print("\n🔍 Domain Breakdown:")
        for domain, occupations in career_domains.items():
            print(f"  {domain}: {len(occupations)} occupations")
            # Show first 3 examples
            examples = occupations[:3]
            if examples:
                print(f"    Examples: {', '.join(examples)}")
        
        print(f"\n✅ Ready to integrate with progression_constraints.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
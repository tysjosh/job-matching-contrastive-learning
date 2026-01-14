#!/usr/bin/env python3
"""
Analyze role mismatches in matched_datasets_pairs_full_with_uri.jsonl
"""
import json
import re
from collections import defaultdict

def normalize_role(role):
    """Normalize role names for comparison"""
    if not role:
        return ""
    
    # Convert to lowercase and remove common variations
    role = role.lower().strip()
    
    # Remove common prefixes/suffixes
    role = re.sub(r'\b(senior|junior|lead|principal|staff|associate|assistant)\b', '', role)
    role = re.sub(r'\b(i|ii|iii|iv|v|1|2|3|4|5)\b', '', role)
    
    # Normalize common role variations
    role_mappings = {
        'software engineer': ['software developer', 'programmer', 'developer', 'software dev'],
        'data scientist': ['data analyst', 'data engineer', 'analytics engineer'],
        'project manager': ['program manager', 'product manager', 'project lead'],
        'database administrator': ['dba', 'database admin', 'db administrator'],
        'web developer': ['frontend developer', 'backend developer', 'full stack developer', 'web dev'],
        'ai engineer': ['ai specialist', 'machine learning engineer', 'ml engineer'],
        'graphic designer': ['ui designer', 'ux designer', 'visual designer', 'designer'],
        'business analyst': ['systems analyst', 'data analyst', 'process analyst'],
        'devops engineer': ['cloud engineer', 'infrastructure engineer', 'site reliability engineer'],
        'qa engineer': ['test engineer', 'quality assurance', 'software tester'],
    }
    
    # Clean up whitespace and special characters
    role = re.sub(r'[^\w\s]', ' ', role)
    role = ' '.join(role.split())
    
    # Check for mappings
    for canonical, variations in role_mappings.items():
        if role in variations or any(var in role for var in variations):
            return canonical
        if canonical in role:
            return canonical
    
    return role

def analyze_role_mismatches(filename):
    """Analyze role mismatches in the dataset"""
    total_records = 0
    role_mismatches = 0
    mismatch_examples = []
    role_distribution = defaultdict(int)
    job_distribution = defaultdict(int)
    
    with open(filename, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                total_records += 1
                
                resume_role = data.get('resume', {}).get('role', '')
                job_title = data.get('job', {}).get('title', '')
                
                if not resume_role or not job_title:
                    continue
                
                # Track distributions
                role_distribution[resume_role] += 1
                job_distribution[job_title] += 1
                
                # Normalize for comparison
                norm_resume_role = normalize_role(resume_role)
                norm_job_title = normalize_role(job_title)
                
                # Check for mismatch
                if norm_resume_role != norm_job_title:
                    role_mismatches += 1
                    if len(mismatch_examples) < 20:  # Collect first 20 examples
                        mismatch_examples.append({
                            'resume_role': resume_role,
                            'job_title': job_title,
                            'normalized_resume': norm_resume_role,
                            'normalized_job': norm_job_title,
                            'label': data.get('label', 'unknown')
                        })
                        
            except json.JSONDecodeError:
                continue
    
    return {
        'total_records': total_records,
        'role_mismatches': role_mismatches,
        'mismatch_percentage': (role_mismatches / total_records * 100) if total_records > 0 else 0,
        'mismatch_examples': mismatch_examples,
        'top_resume_roles': dict(sorted(role_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
        'top_job_titles': dict(sorted(job_distribution.items(), key=lambda x: x[1], reverse=True)[:10])
    }

if __name__ == "__main__":
    results = analyze_role_mismatches('matched_datasets_pairs_full_with_uri.jsonl')
    
    print(f"=== ROLE MISMATCH ANALYSIS ===")
    print(f"Total records: {results['total_records']}")
    print(f"Role mismatches: {results['role_mismatches']}")
    print(f"Mismatch percentage: {results['mismatch_percentage']:.1f}%")
    
    print(f"\n=== MISMATCH EXAMPLES ===")
    for i, example in enumerate(results['mismatch_examples'], 1):
        print(f"{i}. {example['resume_role']} -> {example['job_title']} (Label: {example['label']})")
        print(f"   Normalized: {example['normalized_resume']} -> {example['normalized_job']}")
    
    print(f"\n=== TOP RESUME ROLES ===")
    for role, count in results['top_resume_roles'].items():
        print(f"{role}: {count}")
    
    print(f"\n=== TOP JOB TITLES ===")
    for title, count in results['top_job_titles'].items():
        print(f"{title}: {count}")
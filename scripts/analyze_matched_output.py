#!/usr/bin/env python3
"""
Comprehensive analysis of the matched dataset output from Sentence Transformer matching.
Analyzes data quality, matching patterns, role distributions, and skill alignments.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re

def load_matched_data(file_path: str) -> List[Dict]:
    """Load the matched dataset from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    return data

def analyze_basic_statistics(data: List[Dict]) -> Dict:
    """Analyze basic dataset statistics."""
    stats = {
        'total_records': len(data),
        'unique_applicants': len(set(record['job_applicant_id'] for record in data)),
        'label_distribution': Counter(record['label'] for record in data),
        'experience_levels': Counter(record['resume']['experience_level'] for record in data),
        'education_levels': Counter(record['resume']['education'] for record in data if record['resume'].get('education')),
    }
    
    # Text statistics
    text_lengths = [record['resume']['text_stats']['word_count'] for record in data if 'text_stats' in record['resume']]
    stats['text_length_stats'] = {
        'mean': np.mean(text_lengths),
        'median': np.median(text_lengths),
        'std': np.std(text_lengths),
        'min': np.min(text_lengths),
        'max': np.max(text_lengths)
    }
    
    # Skill statistics
    skill_counts = [record['resume']['skill_stats']['total_skills'] for record in data if 'skill_stats' in record['resume']]
    stats['skill_count_stats'] = {
        'mean': np.mean(skill_counts),
        'median': np.median(skill_counts),
        'std': np.std(skill_counts),
        'min': np.min(skill_counts),
        'max': np.max(skill_counts)
    }
    
    return stats

def analyze_role_matching(data: List[Dict]) -> Dict:
    """Analyze role-to-role matching patterns."""
    role_matches = defaultdict(Counter)
    role_distribution = Counter()
    job_title_distribution = Counter()
    
    for record in data:
        resume_role = record['resume']['role'].lower().strip()
        job_title = record['job']['title'].lower().strip()
        
        role_distribution[resume_role] += 1
        job_title_distribution[job_title] += 1
        role_matches[resume_role][job_title] += 1
    
    # Find most common role mappings
    common_mappings = []
    for resume_role, job_matches in role_matches.items():
        for job_title, count in job_matches.most_common(3):
            common_mappings.append({
                'resume_role': resume_role,
                'job_title': job_title,
                'count': count,
                'percentage': (count / role_distribution[resume_role]) * 100
            })
    
    # Sort by count
    common_mappings.sort(key=lambda x: x['count'], reverse=True)
    
    return {
        'role_distribution': dict(role_distribution.most_common(20)),
        'job_title_distribution': dict(job_title_distribution.most_common(20)),
        'top_role_mappings': common_mappings[:30],
        'unique_resume_roles': len(role_distribution),
        'unique_job_titles': len(job_title_distribution)
    }

def analyze_skill_alignment(data: List[Dict]) -> Dict:
    """Analyze skill alignment between resumes and job requirements."""
    skill_analysis = {
        'most_common_skills': Counter(),
        'skill_level_distribution': Counter(),
        'programming_languages': Counter(),
        'frameworks': Counter(),
        'databases': Counter(),
        'certifications': Counter()
    }
    
    for record in data:
        # Resume skills
        if 'skills' in record['resume'] and record['resume']['skills']:
            for skill in record['resume']['skills']:
                if isinstance(skill, dict) and 'name' in skill:
                    skill_name = skill['name'].lower().strip()
                    skill_analysis['most_common_skills'][skill_name] += 1
                    
                    if 'level' in skill:
                        skill_analysis['skill_level_distribution'][skill['level']] += 1
        
        # Programming languages, frameworks, databases from skill_stats
        if 'skill_stats' in record['resume']:
            stats = record['resume']['skill_stats']
            
            for lang in stats.get('programming_languages', []):
                skill_analysis['programming_languages'][lang.lower()] += 1
            
            for framework in stats.get('frameworks', []):
                skill_analysis['frameworks'][framework.lower()] += 1
            
            for db in stats.get('databases', []):
                skill_analysis['databases'][db.lower()] += 1
        
        # Certifications
        if 'certifications' in record['resume'] and record['resume']['certifications']:
            for cert in record['resume']['certifications']:
                if cert and cert.strip():
                    skill_analysis['certifications'][cert.lower().strip()] += 1
    
    # Convert to regular dicts with top items
    return {
        'most_common_skills': dict(skill_analysis['most_common_skills'].most_common(30)),
        'skill_level_distribution': dict(skill_analysis['skill_level_distribution']),
        'programming_languages': dict(skill_analysis['programming_languages'].most_common(15)),
        'frameworks': dict(skill_analysis['frameworks'].most_common(15)),
        'databases': dict(skill_analysis['databases'].most_common(15)),
        'certifications': dict(skill_analysis['certifications'].most_common(20))
    }

def analyze_data_quality(data: List[Dict]) -> Dict:
    """Analyze data quality metrics."""
    quality_metrics = {
        'missing_fields': defaultdict(int),
        'empty_fields': defaultdict(int),
        'field_completeness': {},
        'text_quality': {
            'very_short_resumes': 0,  # < 50 words
            'short_resumes': 0,       # 50-100 words
            'medium_resumes': 0,      # 100-300 words
            'long_resumes': 0,        # > 300 words
        },
        'skill_quality': {
            'no_skills': 0,
            'few_skills': 0,    # 1-3 skills
            'good_skills': 0,   # 4-10 skills
            'many_skills': 0    # > 10 skills
        }
    }
    
    total_records = len(data)
    
    for record in data:
        resume = record['resume']
        
        # Check for missing/empty fields
        key_fields = ['original_text', 'role', 'experience', 'skills', 'education']
        for field in key_fields:
            if field not in resume:
                quality_metrics['missing_fields'][field] += 1
            elif not resume[field] or (isinstance(resume[field], list) and len(resume[field]) == 0):
                quality_metrics['empty_fields'][field] += 1
        
        # Text quality analysis
        if 'text_stats' in resume and 'word_count' in resume['text_stats']:
            word_count = resume['text_stats']['word_count']
            if word_count < 50:
                quality_metrics['text_quality']['very_short_resumes'] += 1
            elif word_count < 100:
                quality_metrics['text_quality']['short_resumes'] += 1
            elif word_count < 300:
                quality_metrics['text_quality']['medium_resumes'] += 1
            else:
                quality_metrics['text_quality']['long_resumes'] += 1
        
        # Skill quality analysis
        if 'skill_stats' in resume and 'total_skills' in resume['skill_stats']:
            skill_count = resume['skill_stats']['total_skills']
            if skill_count == 0:
                quality_metrics['skill_quality']['no_skills'] += 1
            elif skill_count <= 3:
                quality_metrics['skill_quality']['few_skills'] += 1
            elif skill_count <= 10:
                quality_metrics['skill_quality']['good_skills'] += 1
            else:
                quality_metrics['skill_quality']['many_skills'] += 1
    
    # Calculate completeness percentages
    for field in ['original_text', 'role', 'experience', 'skills', 'education']:
        missing = quality_metrics['missing_fields'][field]
        empty = quality_metrics['empty_fields'][field]
        complete = total_records - missing - empty
        quality_metrics['field_completeness'][field] = (complete / total_records) * 100
    
    return quality_metrics

def analyze_matching_effectiveness(data: List[Dict]) -> Dict:
    """Analyze the effectiveness of the matching algorithm."""
    effectiveness = {
        'label_accuracy': {},
        'role_semantic_similarity': {},
        'experience_level_alignment': defaultdict(Counter),
        'education_alignment': defaultdict(Counter)
    }
    
    # Label distribution analysis
    label_counts = Counter(record['label'] for record in data)
    effectiveness['label_accuracy'] = {
        'positive_matches': label_counts[1],
        'negative_matches': label_counts[0],
        'positive_ratio': label_counts[1] / len(data),
        'balance_ratio': min(label_counts[0], label_counts[1]) / max(label_counts[0], label_counts[1])
    }
    
    # Experience level alignment
    for record in data:
        resume_exp = record['resume']['experience_level']
        job_exp = record['job']['experience_level']
        label = record['label']
        
        effectiveness['experience_level_alignment'][resume_exp][job_exp] += 1
    
    # Role semantic analysis (simplified)
    role_similarity_scores = []
    for record in data:
        resume_role = record['resume']['role'].lower()
        job_title = record['job']['title'].lower()
        
        # Simple semantic similarity based on word overlap
        resume_words = set(resume_role.split())
        job_words = set(job_title.split())
        
        if resume_words and job_words:
            overlap = len(resume_words.intersection(job_words))
            similarity = overlap / max(len(resume_words), len(job_words))
            role_similarity_scores.append(similarity)
    
    effectiveness['role_semantic_similarity'] = {
        'mean_similarity': np.mean(role_similarity_scores),
        'median_similarity': np.median(role_similarity_scores),
        'high_similarity_count': sum(1 for s in role_similarity_scores if s > 0.5),
        'exact_matches': sum(1 for s in role_similarity_scores if s == 1.0)
    }
    
    return effectiveness

def generate_report(data: List[Dict]) -> str:
    """Generate a comprehensive analysis report."""
    print("🔍 Analyzing matched dataset...")
    
    # Run all analyses
    basic_stats = analyze_basic_statistics(data)
    role_analysis = analyze_role_matching(data)
    skill_analysis = analyze_skill_alignment(data)
    quality_analysis = analyze_data_quality(data)
    effectiveness_analysis = analyze_matching_effectiveness(data)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("SENTENCE TRANSFORMER MATCHING - COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Basic Statistics
    report.append("\n📊 BASIC DATASET STATISTICS")
    report.append("-" * 40)
    report.append(f"Total Records: {basic_stats['total_records']:,}")
    report.append(f"Unique Applicants: {basic_stats['unique_applicants']:,}")
    report.append(f"Label Distribution: Positive={basic_stats['label_distribution'][1]:,}, Negative={basic_stats['label_distribution'][0]:,}")
    
    report.append(f"\nExperience Level Distribution:")
    for level, count in basic_stats['experience_levels'].most_common():
        report.append(f"  {level}: {count:,} ({count/basic_stats['total_records']*100:.1f}%)")
    
    report.append(f"\nText Length Statistics:")
    report.append(f"  Mean: {basic_stats['text_length_stats']['mean']:.1f} words")
    report.append(f"  Median: {basic_stats['text_length_stats']['median']:.1f} words")
    report.append(f"  Range: {basic_stats['text_length_stats']['min']:.0f} - {basic_stats['text_length_stats']['max']:.0f} words")
    
    report.append(f"\nSkill Count Statistics:")
    report.append(f"  Mean: {basic_stats['skill_count_stats']['mean']:.1f} skills per resume")
    report.append(f"  Median: {basic_stats['skill_count_stats']['median']:.1f} skills per resume")
    report.append(f"  Range: {basic_stats['skill_count_stats']['min']:.0f} - {basic_stats['skill_count_stats']['max']:.0f} skills")
    
    # Role Matching Analysis
    report.append("\n🎯 ROLE MATCHING ANALYSIS")
    report.append("-" * 40)
    report.append(f"Unique Resume Roles: {role_analysis['unique_resume_roles']}")
    report.append(f"Unique Job Titles: {role_analysis['unique_job_titles']}")
    
    report.append(f"\nTop Resume Roles:")
    for role, count in list(role_analysis['role_distribution'].items())[:10]:
        report.append(f"  {role}: {count:,}")
    
    report.append(f"\nTop Role-to-Job Mappings:")
    for mapping in role_analysis['top_role_mappings'][:15]:
        report.append(f"  {mapping['resume_role']} → {mapping['job_title']}: {mapping['count']} ({mapping['percentage']:.1f}%)")
    
    # Skill Analysis
    report.append("\n🛠️ SKILL ANALYSIS")
    report.append("-" * 40)
    report.append(f"Most Common Skills:")
    for skill, count in list(skill_analysis['most_common_skills'].items())[:15]:
        report.append(f"  {skill}: {count:,}")
    
    report.append(f"\nSkill Level Distribution:")
    for level, count in skill_analysis['skill_level_distribution'].items():
        report.append(f"  {level}: {count:,}")
    
    if skill_analysis['programming_languages']:
        report.append(f"\nTop Programming Languages:")
        for lang, count in list(skill_analysis['programming_languages'].items())[:10]:
            report.append(f"  {lang}: {count:,}")
    
    if skill_analysis['certifications']:
        report.append(f"\nTop Certifications:")
        for cert, count in list(skill_analysis['certifications'].items())[:10]:
            report.append(f"  {cert}: {count:,}")
    
    # Data Quality Analysis
    report.append("\n✅ DATA QUALITY ANALYSIS")
    report.append("-" * 40)
    report.append(f"Field Completeness:")
    for field, completeness in quality_analysis['field_completeness'].items():
        report.append(f"  {field}: {completeness:.1f}%")
    
    report.append(f"\nText Quality Distribution:")
    for category, count in quality_analysis['text_quality'].items():
        percentage = (count / basic_stats['total_records']) * 100
        report.append(f"  {category.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    report.append(f"\nSkill Quality Distribution:")
    for category, count in quality_analysis['skill_quality'].items():
        percentage = (count / basic_stats['total_records']) * 100
        report.append(f"  {category.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    # Matching Effectiveness
    report.append("\n🎯 MATCHING EFFECTIVENESS")
    report.append("-" * 40)
    eff = effectiveness_analysis['label_accuracy']
    report.append(f"Label Distribution:")
    report.append(f"  Positive Matches: {eff['positive_matches']:,} ({eff['positive_ratio']*100:.1f}%)")
    report.append(f"  Negative Matches: {eff['negative_matches']:,} ({(1-eff['positive_ratio'])*100:.1f}%)")
    report.append(f"  Balance Ratio: {eff['balance_ratio']:.2f}")
    
    role_sim = effectiveness_analysis['role_semantic_similarity']
    report.append(f"\nRole Semantic Similarity:")
    report.append(f"  Mean Similarity: {role_sim['mean_similarity']:.3f}")
    report.append(f"  High Similarity (>0.5): {role_sim['high_similarity_count']:,}")
    report.append(f"  Exact Role Matches: {role_sim['exact_matches']:,}")
    
    # Summary and Recommendations
    report.append("\n💡 KEY INSIGHTS & RECOMMENDATIONS")
    report.append("-" * 40)
    
    # Calculate key insights
    positive_ratio = eff['positive_ratio']
    avg_skills = basic_stats['skill_count_stats']['mean']
    role_diversity = role_analysis['unique_resume_roles']
    
    if positive_ratio > 0.7:
        report.append("✅ High positive match ratio indicates good semantic matching quality")
    elif positive_ratio < 0.3:
        report.append("⚠️ Low positive match ratio - consider adjusting similarity thresholds")
    else:
        report.append("✅ Balanced positive/negative ratio suitable for training")
    
    if avg_skills > 8:
        report.append("✅ Rich skill information available for matching")
    elif avg_skills < 4:
        report.append("⚠️ Limited skill information - consider skill extraction improvements")
    
    if role_diversity > 50:
        report.append("✅ Good role diversity for comprehensive model training")
    
    report.append(f"\n🚀 DATASET READY FOR:")
    report.append(f"  • Machine Learning Model Training")
    report.append(f"  • Resume-Job Matching Systems")
    report.append(f"  • Skill Gap Analysis")
    report.append(f"  • Career Progression Modeling")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

def main():
    """Main analysis function."""
    file_path = "matched_datasets_pairs_full.jsonl"
    
    print(f"Loading data from {file_path}...")
    data = load_matched_data(file_path)
    
    if not data:
        print("❌ No data loaded. Check file path and format.")
        return
    
    print(f"✅ Loaded {len(data):,} records")
    
    # Generate and display report
    report = generate_report(data)
    print(report)
    
    # Save report to file
    with open("matched_dataset_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: matched_dataset_analysis_report.txt")

if __name__ == "__main__":
    main()
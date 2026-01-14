#!/usr/bin/env python3
"""
Create visualizations for the matched dataset analysis.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def load_data(file_path: str):
    """Load the matched dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                continue
    return data

def create_visualizations(data):
    """Create comprehensive visualizations."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Label Distribution
    plt.subplot(4, 3, 1)
    labels = [record['label'] for record in data]
    label_counts = Counter(labels)
    plt.pie(label_counts.values(), labels=['Negative (0)', 'Positive (1)'], autopct='%1.1f%%', startangle=90)
    plt.title('Label Distribution\n(Positive vs Negative Matches)', fontsize=12, fontweight='bold')
    
    # 2. Experience Level Distribution
    plt.subplot(4, 3, 2)
    exp_levels = [record['resume']['experience_level'] for record in data]
    exp_counts = Counter(exp_levels)
    plt.bar(exp_counts.keys(), exp_counts.values(), color='skyblue')
    plt.title('Experience Level Distribution', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 3. Top Resume Roles
    plt.subplot(4, 3, 3)
    roles = [record['resume']['role'] for record in data]
    role_counts = Counter(roles).most_common(10)
    roles_list, counts_list = zip(*role_counts)
    plt.barh(range(len(roles_list)), counts_list, color='lightcoral')
    plt.yticks(range(len(roles_list)), [role[:25] + '...' if len(role) > 25 else role for role in roles_list])
    plt.title('Top 10 Resume Roles', fontsize=12, fontweight='bold')
    plt.xlabel('Count')
    
    # 4. Skill Count Distribution
    plt.subplot(4, 3, 4)
    skill_counts = [record['resume']['skill_stats']['total_skills'] for record in data if 'skill_stats' in record['resume']]
    plt.hist(skill_counts, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Skill Counts per Resume', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Skills')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(skill_counts), color='red', linestyle='--', label=f'Mean: {np.mean(skill_counts):.1f}')
    plt.legend()
    
    # 5. Text Length Distribution
    plt.subplot(4, 3, 5)
    text_lengths = [record['resume']['text_stats']['word_count'] for record in data if 'text_stats' in record['resume']]
    plt.hist(text_lengths, bins=30, color='gold', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Resume Text Lengths', fontsize=12, fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(text_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(text_lengths):.1f}')
    plt.legend()
    
    # 6. Top Skills
    plt.subplot(4, 3, 6)
    all_skills = []
    for record in data:
        if 'skills' in record['resume'] and record['resume']['skills']:
            for skill in record['resume']['skills']:
                if isinstance(skill, dict) and 'name' in skill:
                    all_skills.append(skill['name'].lower())
    
    skill_counts = Counter(all_skills).most_common(15)
    skills_list, counts_list = zip(*skill_counts)
    plt.barh(range(len(skills_list)), counts_list, color='mediumpurple')
    plt.yticks(range(len(skills_list)), [skill[:20] + '...' if len(skill) > 20 else skill for skill in skills_list])
    plt.title('Top 15 Most Common Skills', fontsize=12, fontweight='bold')
    plt.xlabel('Count')
    
    # 7. Education Level Distribution
    plt.subplot(4, 3, 7)
    education_levels = [record['resume']['education'] for record in data if record['resume'].get('education')]
    edu_counts = Counter(education_levels)
    plt.pie(edu_counts.values(), labels=edu_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Education Level Distribution', fontsize=12, fontweight='bold')
    
    # 8. Role Matching Heatmap (Top roles)
    plt.subplot(4, 3, 8)
    role_job_pairs = []
    for record in data:
        resume_role = record['resume']['role']
        job_title = record['job']['title']
        role_job_pairs.append((resume_role, job_title))
    
    # Get top 8 resume roles and job titles for heatmap
    top_resume_roles = [role for role, _ in Counter([pair[0] for pair in role_job_pairs]).most_common(8)]
    top_job_titles = [job for job, _ in Counter([pair[1] for pair in role_job_pairs]).most_common(8)]
    
    # Create matrix
    matrix = np.zeros((len(top_resume_roles), len(top_job_titles)))
    for resume_role, job_title in role_job_pairs:
        if resume_role in top_resume_roles and job_title in top_job_titles:
            i = top_resume_roles.index(resume_role)
            j = top_job_titles.index(job_title)
            matrix[i][j] += 1
    
    sns.heatmap(matrix, 
                xticklabels=[title[:15] + '...' if len(title) > 15 else title for title in top_job_titles],
                yticklabels=[role[:15] + '...' if len(role) > 15 else role for role in top_resume_roles],
                annot=True, fmt='g', cmap='Blues')
    plt.title('Role-to-Job Matching Heatmap', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 9. Skill Level Distribution
    plt.subplot(4, 3, 9)
    skill_levels = []
    for record in data:
        if 'skills' in record['resume'] and record['resume']['skills']:
            for skill in record['resume']['skills']:
                if isinstance(skill, dict) and 'level' in skill:
                    skill_levels.append(skill['level'])
    
    level_counts = Counter(skill_levels)
    # Filter out numeric levels and focus on main categories
    main_levels = {k: v for k, v in level_counts.items() if k in ['beginner', 'intermediate', 'advanced', 'expert', 'senior']}
    
    if main_levels:
        plt.bar(main_levels.keys(), main_levels.values(), color='orange')
        plt.title('Skill Level Distribution', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
    
    # 10. Experience vs Skills Scatter
    plt.subplot(4, 3, 10)
    exp_mapping = {'junior': 1, 'mid': 2, 'senior': 3}
    exp_numeric = []
    skill_counts_for_scatter = []
    
    for record in data:
        exp_level = record['resume']['experience_level']
        if exp_level in exp_mapping and 'skill_stats' in record['resume']:
            exp_numeric.append(exp_mapping[exp_level])
            skill_counts_for_scatter.append(record['resume']['skill_stats']['total_skills'])
    
    plt.scatter(exp_numeric, skill_counts_for_scatter, alpha=0.6, color='teal')
    plt.xlabel('Experience Level (1=Junior, 2=Mid, 3=Senior)')
    plt.ylabel('Number of Skills')
    plt.title('Experience Level vs Skill Count', fontsize=12, fontweight='bold')
    plt.xticks([1, 2, 3], ['Junior', 'Mid', 'Senior'])
    
    # 11. Top Certifications
    plt.subplot(4, 3, 11)
    all_certs = []
    for record in data:
        if 'certifications' in record['resume'] and record['resume']['certifications']:
            for cert in record['resume']['certifications']:
                if cert and cert.strip():
                    all_certs.append(cert.lower())
    
    cert_counts = Counter(all_certs).most_common(10)
    if cert_counts:
        certs_list, counts_list = zip(*cert_counts)
        plt.barh(range(len(certs_list)), counts_list, color='lightblue')
        plt.yticks(range(len(certs_list)), [cert[:30] + '...' if len(cert) > 30 else cert for cert in certs_list])
        plt.title('Top 10 Certifications', fontsize=12, fontweight='bold')
        plt.xlabel('Count')
    
    # 12. Label by Experience Level
    plt.subplot(4, 3, 12)
    exp_label_data = {}
    for record in data:
        exp_level = record['resume']['experience_level']
        label = record['label']
        if exp_level not in exp_label_data:
            exp_label_data[exp_level] = {'positive': 0, 'negative': 0}
        
        if label == 1:
            exp_label_data[exp_level]['positive'] += 1
        else:
            exp_label_data[exp_level]['negative'] += 1
    
    exp_levels = list(exp_label_data.keys())
    positive_counts = [exp_label_data[level]['positive'] for level in exp_levels]
    negative_counts = [exp_label_data[level]['negative'] for level in exp_levels]
    
    x = np.arange(len(exp_levels))
    width = 0.35
    
    plt.bar(x - width/2, positive_counts, width, label='Positive', color='lightgreen')
    plt.bar(x + width/2, negative_counts, width, label='Negative', color='lightcoral')
    
    plt.xlabel('Experience Level')
    plt.ylabel('Count')
    plt.title('Match Labels by Experience Level', fontsize=12, fontweight='bold')
    plt.xticks(x, exp_levels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('matched_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualizations saved as 'matched_dataset_analysis.png'")

def main():
    """Main function."""
    print("Loading data for visualization...")
    data = load_data("matched_datasets_pairs_full.jsonl")
    
    if not data:
        print("❌ No data loaded.")
        return
    
    print(f"✅ Loaded {len(data):,} records")
    print("Creating visualizations...")
    
    create_visualizations(data)
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
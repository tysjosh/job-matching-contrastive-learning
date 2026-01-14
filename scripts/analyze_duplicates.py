#!/usr/bin/env python3
"""
Script to analyze preprocessed_resumes.jsonl for duplicate resumes
"""

import json
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple


def load_resumes(filepath: str) -> List[Dict]:
    """Load resumes from JSONL file"""
    resumes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                resume = json.loads(line.strip())
                resume['_line_number'] = line_num
                resumes.append(resume)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    return resumes


def create_resume_hash(resume: Dict) -> str:
    """Create a hash of resume content for exact duplicate detection"""
    # Create a normalized string representation of the resume
    # Exclude the line number field we added
    resume_copy = {k: v for k, v in resume.items() if k != '_line_number'}
    resume_str = json.dumps(resume_copy, sort_keys=True)
    return hashlib.md5(resume_str.encode()).hexdigest()


def create_fuzzy_signature(resume: Dict) -> str:
    """Create a signature based on key fields for near-duplicate detection"""
    # Combine key identifying fields
    parts = []

    # Add summary (first 100 chars)
    summary = resume.get('summary', '').strip()
    if summary and summary != 'Not provided':
        parts.append(summary[:100].lower())

    # Add education degrees
    education = resume.get('education', [])
    if education:
        degrees = [edu.get('degree', '').lower()
                   for edu in education if edu.get('degree')]
        parts.extend(sorted(degrees))

    # Add skills
    skills = resume.get('skills', [])
    if skills:
        skill_names = [skill.get('name', '').lower() for skill in skills
                       if skill.get('name') and skill.get('name') != 'Unknown'
                       and skill.get('name') != 'Not Provided']
        parts.extend(sorted(set(skill_names)))

    # Add job titles
    experience = resume.get('experience', [])
    if experience:
        job_titles = [exp.get('job_title', '').lower() for exp in experience
                      if exp.get('job_title')]
        parts.extend(sorted(set(job_titles)))

    signature = '||'.join(parts)
    return hashlib.md5(signature.encode()).hexdigest()


def analyze_duplicates(resumes: List[Dict]) -> Tuple[Dict, Dict]:
    """Analyze for exact and near-duplicate resumes"""

    # Exact duplicates by full hash
    exact_duplicates = defaultdict(list)
    for resume in resumes:
        hash_val = create_resume_hash(resume)
        exact_duplicates[hash_val].append(resume['_line_number'])

    # Filter to only those with duplicates
    exact_duplicates = {k: v for k,
                        v in exact_duplicates.items() if len(v) > 1}

    # Near duplicates by fuzzy signature
    fuzzy_duplicates = defaultdict(list)
    for resume in resumes:
        signature = create_fuzzy_signature(resume)
        fuzzy_duplicates[signature].append({
            'line': resume['_line_number'],
            'summary': resume.get('summary', '')[:80],
            'job_title': resume.get('experience', [{}])[0].get('job_title', 'N/A') if resume.get('experience') else 'N/A'
        })

    # Filter to only those with duplicates
    fuzzy_duplicates = {k: v for k,
                        v in fuzzy_duplicates.items() if len(v) > 1}

    return exact_duplicates, fuzzy_duplicates


def print_analysis_report(resumes: List[Dict], exact_dupes: Dict, fuzzy_dupes: Dict):
    """Print comprehensive duplicate analysis report"""

    print("="*80)
    print("DUPLICATE RESUME ANALYSIS REPORT")
    print("="*80)
    print(f"\nTotal resumes analyzed: {len(resumes)}")
    print(
        f"Unique resume hashes: {len(set(create_resume_hash(r) for r in resumes))}")

    print("\n" + "="*80)
    print("EXACT DUPLICATES (identical JSON content)")
    print("="*80)

    if exact_dupes:
        print(f"\nFound {len(exact_dupes)} groups of exact duplicates")
        print(
            f"Total duplicate entries: {sum(len(v) - 1 for v in exact_dupes.values())}")

        for idx, (hash_val, line_numbers) in enumerate(exact_dupes.items(), 1):
            print(f"\n  Group {idx}: {len(line_numbers)} identical resumes")
            print(f"    Lines: {', '.join(map(str, sorted(line_numbers)))}")

            # Show a sample of the duplicate resume
            sample_line = line_numbers[0]
            sample_resume = next(
                r for r in resumes if r['_line_number'] == sample_line)
            print(
                f"    Summary: {sample_resume.get('summary', 'N/A')[:100]}...")
            print(
                f"    Job Title: {sample_resume.get('experience', [{}])[0].get('job_title', 'N/A')}")
    else:
        print("\n✓ No exact duplicates found")

    print("\n" + "="*80)
    print("NEAR DUPLICATES (similar content)")
    print("="*80)

    if fuzzy_dupes:
        print(
            f"\nFound {len(fuzzy_dupes)} groups of potentially similar resumes")
        print(f"(Based on summary, skills, education, and job titles)")

        for idx, (signature, resume_infos) in enumerate(fuzzy_dupes.items(), 1):
            if idx > 20:  # Limit output
                print(f"\n  ... and {len(fuzzy_dupes) - 20} more groups")
                break

            print(f"\n  Group {idx}: {len(resume_infos)} similar resumes")
            for info in resume_infos:
                print(f"    - Line {info['line']}: {info['job_title']}")
                print(f"      {info['summary'][:70]}...")
    else:
        print("\n✓ No near duplicates found")

    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)

    # Count by job title
    job_title_counts = defaultdict(int)
    for resume in resumes:
        experiences = resume.get('experience', [])
        if experiences:
            job_title = experiences[0].get('job_title', 'Unknown')
            job_title_counts[job_title] += 1

    print(f"\nTop 10 most common job titles:")
    for job_title, count in sorted(job_title_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {job_title}: {count}")

    # Summary analysis
    not_provided_count = sum(1 for r in resumes if r.get(
        'summary') in ['Not provided', '', None])
    print(
        f"\nResumes with 'Not provided' or empty summary: {not_provided_count}")

    # Experience level analysis
    exp_levels = defaultdict(int)
    for resume in resumes:
        experiences = resume.get('experience', [])
        if experiences:
            exp_level = experiences[0].get('experience_level', 'Unknown')
            exp_levels[exp_level] += 1

    print(f"\nExperience level distribution:")
    for level, count in sorted(exp_levels.items(), key=lambda x: x[1], reverse=True):
        print(f"  {level}: {count}")


def main():
    filepath = "dataset/preprocessed_resumes.jsonl"

    print(f"Loading resumes from {filepath}...")
    resumes = load_resumes(filepath)

    print(f"Analyzing {len(resumes)} resumes for duplicates...")
    exact_dupes, fuzzy_dupes = analyze_duplicates(resumes)

    print_analysis_report(resumes, exact_dupes, fuzzy_dupes)

    # Save detailed results to file
    output_file = "duplicate_analysis_results.json"
    results = {
        'total_resumes': len(resumes),
        'exact_duplicate_groups': len(exact_dupes),
        'exact_duplicate_lines': {
            hash_val: line_nums for hash_val, line_nums in exact_dupes.items()
        },
        'near_duplicate_groups': len(fuzzy_dupes),
        'near_duplicate_details': {
            idx: [info for info in infos]
            for idx, infos in fuzzy_dupes.items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

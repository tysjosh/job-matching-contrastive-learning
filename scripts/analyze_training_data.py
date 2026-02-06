"""
Comprehensive Training Data Analysis

This script analyzes the training data to identify potential issues that could
cause embedding collapse or training failures.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import hashlib


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_duplicates(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check for duplicate samples."""
    print("\n" + "="*80)
    print("1. DUPLICATE ANALYSIS")
    print("="*80)

    # Hash resume-job pairs
    pair_hashes = []
    resume_hashes = []
    job_hashes = []

    for sample in data:
        # Create hash of resume
        resume_str = json.dumps(sample.get('resume', {}), sort_keys=True)
        resume_hash = hashlib.md5(resume_str.encode()).hexdigest()
        resume_hashes.append(resume_hash)

        # Create hash of job
        job_str = json.dumps(sample.get('job', {}), sort_keys=True)
        job_hash = hashlib.md5(job_str.encode()).hexdigest()
        job_hashes.append(job_hash)

        # Create hash of pair
        pair_str = resume_str + job_str
        pair_hash = hashlib.md5(pair_str.encode()).hexdigest()
        pair_hashes.append(pair_hash)

    # Count unique items
    unique_pairs = len(set(pair_hashes))
    unique_resumes = len(set(resume_hashes))
    unique_jobs = len(set(job_hashes))

    # Find duplicates
    pair_counts = Counter(pair_hashes)
    duplicate_pairs = sum(1 for count in pair_counts.values() if count > 1)

    print(f"\n📊 Total samples: {len(data)}")
    print(f"   Unique resume-job pairs: {unique_pairs}")
    print(f"   Unique resumes: {unique_resumes}")
    print(f"   Unique jobs: {unique_jobs}")
    print(f"   Duplicate pairs: {duplicate_pairs}")

    if unique_pairs < len(data):
        print(
            f"\n⚠️  WARNING: {len(data) - unique_pairs} duplicate samples found!")
        print(
            f"   Duplication rate: {((len(data) - unique_pairs) / len(data)) * 100:.1f}%")
    else:
        print(f"\n✅ No duplicate samples found")

    # Check if all resumes/jobs are the same
    if unique_resumes == 1:
        print(f"\n❌ CRITICAL: All resumes are identical!")
    elif unique_resumes < 10:
        print(
            f"\n⚠️  WARNING: Only {unique_resumes} unique resumes (very low diversity)")

    if unique_jobs == 1:
        print(f"\n❌ CRITICAL: All jobs are identical!")
    elif unique_jobs < 10:
        print(
            f"\n⚠️  WARNING: Only {unique_jobs} unique jobs (very low diversity)")

    return {
        'total_samples': len(data),
        'unique_pairs': unique_pairs,
        'unique_resumes': unique_resumes,
        'unique_jobs': unique_jobs,
        'duplicate_pairs': duplicate_pairs
    }


def analyze_labels(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze label distribution."""
    print("\n" + "="*80)
    print("2. LABEL DISTRIBUTION ANALYSIS")
    print("="*80)

    labels = [sample.get('label', sample.get('match', None))
              for sample in data]

    # Count labels
    label_counts = Counter(labels)

    print(f"\n📊 Label distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")

    # Check for imbalance
    if None in label_counts:
        print(f"\n⚠️  WARNING: {label_counts[None]} samples have no label!")

    if len(label_counts) == 1:
        print(f"\n❌ CRITICAL: All samples have the same label!")
        print(f"   Model cannot learn with only one class")
    elif len(label_counts) == 2:
        values = list(label_counts.values())
        ratio = max(values) / min(values)
        if ratio > 3:
            print(
                f"\n⚠️  WARNING: Significant class imbalance (ratio: {ratio:.1f}:1)")
        else:
            print(f"\n✅ Reasonable class balance (ratio: {ratio:.1f}:1)")

    return {
        'label_distribution': dict(label_counts),
        'num_classes': len(label_counts)
    }


def analyze_text_content(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze text content quality."""
    print("\n" + "="*80)
    print("3. TEXT CONTENT ANALYSIS")
    print("="*80)

    # Sample first few items
    print(f"\n📝 Sample data (first 3 items):")

    for i, sample in enumerate(data[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Label: {sample.get('label', sample.get('match', 'N/A'))}")

        # Resume
        resume = sample.get('resume', {})
        resume_text = resume.get('experience', '')[
            :100] if resume.get('experience') else 'NO TEXT'
        print(f"Resume preview: {resume_text}...")
        print(f"Resume skills: {resume.get('skills', [])[:3]}...")

        # Job
        job = sample.get('job', {})
        job_desc = job.get('description', {})
        if isinstance(job_desc, dict):
            job_text = job_desc.get('original', '')[
                :100] if job_desc.get('original') else 'NO TEXT'
        else:
            job_text = str(job_desc)[:100] if job_desc else 'NO TEXT'
        print(f"Job preview: {job_text}...")
        print(f"Job skills: {job.get('skills', [])[:3]}...")

    # Analyze text presence
    resumes_with_text = 0
    jobs_with_text = 0
    resumes_with_skills = 0
    jobs_with_skills = 0

    for sample in data:
        resume = sample.get('resume', {})
        job = sample.get('job', {})

        # Handle resume experience (could be list or string with nested structure)
        experience = resume.get('experience', '')
        if isinstance(experience, list) and len(experience) > 0:
            if isinstance(experience[0], dict):
                # Handle nested description structure
                desc_field = experience[0].get('description', '')
                if isinstance(desc_field, list) and len(desc_field) > 0:
                    if isinstance(desc_field[0], dict):
                        experience_text = desc_field[0].get('description', '')
                    else:
                        experience_text = str(desc_field[0])
                elif isinstance(desc_field, str):
                    experience_text = desc_field
                else:
                    experience_text = str(desc_field) if desc_field else ''
            else:
                experience_text = str(experience[0])
        else:
            experience_text = str(experience)

        if experience_text and len(experience_text) > 10:
            resumes_with_text += 1

        # Handle job description (could be dict or string)
        description = job.get('description', {})
        if isinstance(description, dict):
            job_text = description.get('original', '')
        else:
            job_text = str(description)

        if job_text and len(job_text) > 10:
            jobs_with_text += 1

        if resume.get('skills') and len(resume['skills']) > 0:
            resumes_with_skills += 1

        if job.get('skills') and len(job['skills']) > 0:
            jobs_with_skills += 1

    print(f"\n📊 Content statistics:")
    print(
        f"   Resumes with text: {resumes_with_text}/{len(data)} ({resumes_with_text/len(data)*100:.1f}%)")
    print(
        f"   Jobs with text: {jobs_with_text}/{len(data)} ({jobs_with_text/len(data)*100:.1f}%)")
    print(
        f"   Resumes with skills: {resumes_with_skills}/{len(data)} ({resumes_with_skills/len(data)*100:.1f}%)")
    print(
        f"   Jobs with skills: {jobs_with_skills}/{len(data)} ({jobs_with_skills/len(data)*100:.1f}%)")

    if resumes_with_text < len(data) * 0.5:
        print(f"\n⚠️  WARNING: Less than 50% of resumes have meaningful text")

    if jobs_with_text < len(data) * 0.5:
        print(f"\n⚠️  WARNING: Less than 50% of jobs have meaningful text")

    return {
        'resumes_with_text': resumes_with_text,
        'jobs_with_text': jobs_with_text,
        'resumes_with_skills': resumes_with_skills,
        'jobs_with_skills': jobs_with_skills
    }


def analyze_career_paths(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze career path data."""
    print("\n" + "="*80)
    print("4. CAREER PATH ANALYSIS")
    print("="*80)

    has_career_data = 0
    career_distances = []

    for sample in data:
        if 'career_distance' in sample:
            has_career_data += 1
            career_distances.append(sample['career_distance'])

    print(f"\n📊 Career path statistics:")
    print(
        f"   Samples with career data: {has_career_data}/{len(data)} ({has_career_data/len(data)*100:.1f}%)")

    if career_distances:
        print(
            f"   Career distance range: {min(career_distances):.2f} - {max(career_distances):.2f}")
        print(
            f"   Average career distance: {sum(career_distances)/len(career_distances):.2f}")

        # Distribution
        dist_counts = Counter([int(d) for d in career_distances])
        print(f"\n   Distance distribution:")
        for dist in sorted(dist_counts.keys()):
            count = dist_counts[dist]
            print(
                f"     Distance {dist}: {count} samples ({count/len(career_distances)*100:.1f}%)")
    else:
        print(f"\n⚠️  WARNING: No career distance data found")

    return {
        'has_career_data': has_career_data,
        'career_distances': career_distances
    }


def analyze_augmentation(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze if data is augmented."""
    print("\n" + "="*80)
    print("5. AUGMENTATION ANALYSIS")
    print("="*80)

    augmented_count = 0
    augmentation_types = Counter()

    for sample in data:
        if 'augmentation_type' in sample:
            augmented_count += 1
            augmentation_types[sample['augmentation_type']] += 1

    print(f"\n📊 Augmentation statistics:")
    print(
        f"   Augmented samples: {augmented_count}/{len(data)} ({augmented_count/len(data)*100:.1f}%)")

    if augmentation_types:
        print(f"\n   Augmentation type distribution:")
        for aug_type, count in augmentation_types.most_common():
            print(
                f"     {aug_type}: {count} ({count/augmented_count*100:.1f}%)")

    return {
        'augmented_count': augmented_count,
        'augmentation_types': dict(augmentation_types)
    }


def check_text_encoder_potential_issues(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check for issues that might cause identical embeddings."""
    print("\n" + "="*80)
    print("6. EMBEDDING POTENTIAL ISSUES")
    print("="*80)

    # Check if all resumes have the exact same text
    resume_texts = []
    job_texts = []

    for sample in data[:100]:  # Sample first 100
        resume = sample.get('resume', {})
        job = sample.get('job', {})

        # Build text that would be encoded - handle nested list format
        resume_parts = []
        experience = resume.get('experience', [])
        if isinstance(experience, list) and len(experience) > 0:
            if isinstance(experience[0], dict):
                # Handle nested description structure
                desc_field = experience[0].get('description', '')
                if isinstance(desc_field, list) and len(desc_field) > 0:
                    if isinstance(desc_field[0], dict):
                        resume_parts.append(desc_field[0].get('description', ''))
                    else:
                        resume_parts.append(str(desc_field[0]))
                elif isinstance(desc_field, str):
                    resume_parts.append(desc_field)
                else:
                    resume_parts.append(str(desc_field) if desc_field else '')
            else:
                resume_parts.append(str(experience[0]))
        elif isinstance(experience, str):
            resume_parts.append(experience)

        if resume.get('skills'):
            skills_text = ' '.join([s.get('name', '') if isinstance(
                s, dict) else str(s) for s in resume['skills']])
            resume_parts.append(skills_text)
        resume_text = ' '.join(resume_parts)
        resume_texts.append(resume_text)

        job_parts = []
        description = job.get('description', {})
        if isinstance(description, dict):
            job_parts.append(description.get('original', ''))
        elif isinstance(description, str):
            job_parts.append(description)

        if job.get('skills'):
            skills_text = ' '.join([s.get('name', '') if isinstance(
                s, dict) else str(s) for s in job['skills']])
            job_parts.append(skills_text)
        job_text = ' '.join(job_parts)
        job_texts.append(job_text)

    # Check uniqueness
    unique_resume_texts = len(set(resume_texts))
    unique_job_texts = len(set(job_texts))

    print(f"\n📊 Text diversity (first 100 samples):")
    print(f"   Unique resume texts: {unique_resume_texts}/100")
    print(f"   Unique job texts: {unique_job_texts}/100")

    if unique_resume_texts == 1:
        print(f"\n❌ CRITICAL: All resume texts are IDENTICAL!")
        print(f"   This will cause all resume embeddings to be the same")
        print(f"   Sample text: {resume_texts[0][:200]}...")
    elif unique_resume_texts < 10:
        print(
            f"\n⚠️  WARNING: Very low resume text diversity ({unique_resume_texts} unique)")
    else:
        print(f"\n✅ Good resume text diversity")

    if unique_job_texts == 1:
        print(f"\n❌ CRITICAL: All job texts are IDENTICAL!")
        print(f"   This will cause all job embeddings to be the same")
        print(f"   Sample text: {job_texts[0][:200]}...")
    elif unique_job_texts < 10:
        print(
            f"\n⚠️  WARNING: Very low job text diversity ({unique_job_texts} unique)")
    else:
        print(f"\n✅ Good job text diversity")

    # Check for empty texts
    empty_resumes = sum(1 for t in resume_texts if len(t.strip()) < 10)
    empty_jobs = sum(1 for t in job_texts if len(t.strip()) < 10)

    if empty_resumes > 0:
        print(
            f"\n⚠️  WARNING: {empty_resumes} resumes have empty/minimal text")

    if empty_jobs > 0:
        print(f"\n⚠️  WARNING: {empty_jobs} jobs have empty/minimal text")

    return {
        'unique_resume_texts': unique_resume_texts,
        'unique_job_texts': unique_job_texts,
        'empty_resumes': empty_resumes,
        'empty_jobs': empty_jobs
    }


def main():
    """Run comprehensive data analysis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAINING DATA ANALYSIS")
    print("="*80)

    # Load data
    data_file = "augmented_combined_data_training_with_uri.jsonl"

    if not Path(data_file).exists():
        print(f"\n❌ ERROR: Data file not found: {data_file}")
        return

    print(f"\n📂 Loading data from: {data_file}")
    data = load_data(data_file)
    print(f"✅ Loaded {len(data)} samples")

    # Run analyses
    results = {}

    results['duplicates'] = analyze_duplicates(data)
    results['labels'] = analyze_labels(data)
    results['text_content'] = analyze_text_content(data)
    results['career_paths'] = analyze_career_paths(data)
    results['augmentation'] = analyze_augmentation(data)
    results['embedding_issues'] = check_text_encoder_potential_issues(data)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    critical_issues = []
    warnings = []

    # Check for critical issues
    if results['duplicates']['unique_resumes'] == 1:
        critical_issues.append("All resumes are identical")

    if results['duplicates']['unique_jobs'] == 1:
        critical_issues.append("All jobs are identical")

    if results['labels']['num_classes'] == 1:
        critical_issues.append("All samples have the same label")

    if results['embedding_issues']['unique_resume_texts'] == 1:
        critical_issues.append("All resume texts are identical")

    if results['embedding_issues']['unique_job_texts'] == 1:
        critical_issues.append("All job texts are identical")

    # Check for warnings
    if results['duplicates']['duplicate_pairs'] > 0:
        warnings.append(
            f"{results['duplicates']['duplicate_pairs']} duplicate samples found")

    if results['text_content']['resumes_with_text'] < len(data) * 0.5:
        warnings.append("Less than 50% of resumes have meaningful text")

    if results['text_content']['jobs_with_text'] < len(data) * 0.5:
        warnings.append("Less than 50% of jobs have meaningful text")

    # Display summary
    if critical_issues:
        print(f"\n❌ CRITICAL ISSUES FOUND ({len(critical_issues)}):")
        for issue in critical_issues:
            print(f"   • {issue}")
        print(f"\n   These issues WILL cause embedding collapse and training failure!")
    else:
        print(f"\n✅ No critical data issues found")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"   • {warning}")
    else:
        print(f"\n✅ No data quality warnings")

    if not critical_issues and not warnings:
        print(f"\n🎉 Data looks good! The embedding collapse is likely a model/training issue, not data.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

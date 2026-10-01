"""Read-only source audit for occupation-targeted recommendation experiments.

Outputs aggregate profiles and per-occupation support, never resume contents.
Exact normalized content is a proxy for identity where stable IDs are absent.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'career_v7': {'train': 'preprocess/learning_curve_v7/frac_100/train.jsonl',
                  'validation': 'preprocess/data_splits_v7/validation.jsonl',
                  'test': 'preprocess/data_splits_v7/test.jsonl'},
    'indian': {s: f'preprocess/indian_splits/{s}.jsonl' for s in ['train', 'validation', 'test']},
    'alitianchi_clean': {s: f'preprocess/alitianchi_splits_clean/{s}.jsonl' for s in ['train', 'validation', 'test']},
}


def normalize(value):
    if isinstance(value, str):
        return ' '.join(value.casefold().split())
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def key(record, side):
    fields = ('role', 'experience', 'experience_level', 'skills', 'keywords') if side == 'resume' else (
        'title', 'description', 'skills', 'experience_level')
    data = {k: record.get(side, {}).get(k) for k in fields}
    return hashlib.sha256(json.dumps(normalize(data), sort_keys=True).encode()).hexdigest()


def audit():
    with (ROOT / 'dataset/esco/occupations_en.csv').open() as f:
        occupation_names = {r['conceptUri']: r['preferredLabel'] for r in csv.DictReader(f)}
    results = {}
    support_rows = []
    for dataset, paths in SOURCES.items():
        splits = {}
        for split, relative in paths.items():
            with (ROOT / relative).open() as f:
                splits[split] = [json.loads(line) for line in f if line.strip()]
        resume_sets = {s: {key(r, 'resume') for r in rows} for s, rows in splits.items()}
        job_sets = {s: {key(r, 'job') for r in rows} for s, rows in splits.items()}
        prior_resumes = resume_sets['train'] | resume_sets['validation']
        stats = {}
        occupation_support = defaultdict(lambda: defaultdict(set))
        occupation_counts = defaultdict(Counter)
        for split, rows in splits.items():
            queries = defaultdict(list)
            pairs = Counter()
            per_pair_labels = defaultdict(set)
            per_resume_occupation_labels = defaultdict(set)
            labels = Counter()
            modes = Counter()
            scores = []
            for row in rows:
                resume, job = key(row, 'resume'), key(row, 'job')
                uri = row.get('job', {}).get('occupation_uri')
                label = row.get('metadata', {}).get('original_label', str(row.get('label')))
                labels[label] += 1
                modes[str(row.get('metadata', {}).get('occupation_match_mode'))] += 1
                score = row.get('metadata', {}).get('occupation_match_score')
                if isinstance(score, (int, float)): scores.append(score)
                queries[resume].append(row)
                pairs[(resume, job)] += 1
                per_pair_labels[(resume, job)].add(label)
                if uri:
                    occupation_support[uri][split + '_resumes'].add(resume)
                    occupation_counts[uri][split + '_rows'] += 1
                    per_resume_occupation_labels[(resume, uri)].add(label)
                    if row.get('label') == 1:
                        occupation_support[uri][split + '_positive_resumes'].add(resume)
                        occupation_counts[uri][split + '_positive_rows'] += 1
                    if split == 'test' and resume not in prior_resumes:
                        occupation_support[uri]['independent_test_resumes'].add(resume)
                        if row.get('label') == 1:
                            occupation_support[uri]['independent_test_positive_resumes'].add(resume)
            candidate_counts = Counter(len({key(r, 'job') for r in rs}) for rs in queries.values())
            stats[split] = {
                'rows': len(rows), 'labels': dict(labels), 'unique_resume_contents': len(queries),
                'unique_job_contents': len(job_sets[split]), 'distinct_pairs': len(pairs),
                'duplicate_pair_extra_rows': sum(n - 1 for n in pairs.values()),
                'conflicting_pair_labels': sum(len(v) > 1 for v in per_pair_labels.values()),
                'resume_occupation_groups_with_multiple_fit_labels': sum(len(v) > 1 for v in per_resume_occupation_labels.values()),
                'job_occupation_present_rows': sum(bool(r.get('job', {}).get('occupation_uri')) for r in rows),
                'invalid_occupation_rows': sum(bool(r.get('job', {}).get('occupation_uri')) and r['job']['occupation_uri'] not in occupation_names for r in rows),
                'distinct_job_occupations': len({r['job']['occupation_uri'] for r in rows if r.get('job', {}).get('occupation_uri')}),
                'resume_occupation_present_rows': sum(bool(r.get('resume', {}).get('occupation_uri')) for r in rows),
                'missing_applicant_id_rows': sum(not r.get('job_applicant_id') for r in rows),
                'mapping_modes': dict(modes),
                'mapping_score_below_80': sum(s < 80 for s in scores),
                'mapping_score_available': len(scores),
                'judged_jobs_per_resume_distribution': dict(sorted(candidate_counts.items())),
                'queries_with_positive_and_negative': sum(any(r.get('label') == 1 for r in rs) and any(r.get('label') == 0 for r in rs) for rs in queries.values()),
                'queries_with_at_least_10_judged_jobs': sum(n for k, n in candidate_counts.items() if k >= 10),
            }
        overlap = {}
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            overlap[a + '__' + b] = {'resume_contents': len(resume_sets[a] & resume_sets[b]),
                                     'job_contents': len(job_sets[a] & job_sets[b])}
        targets = []
        for uri, count in occupation_counts.items():
            row = {'dataset': dataset, 'occupation_uri': uri, 'occupation_name': occupation_names.get(uri, 'UNRESOLVED')}
            for split in ['train', 'validation', 'test']:
                for suffix in ['rows', 'positive_rows']: row[split + '_' + suffix] = count[split + '_' + suffix]
                for suffix in ['resumes', 'positive_resumes']: row[split + '_' + suffix] = len(occupation_support[uri][split + '_' + suffix])
            for field in ['independent_test_resumes', 'independent_test_positive_resumes']:
                row[field] = len(occupation_support[uri][field])
            targets.append(row)
        support_rows.extend(targets)
        feasibility = {}
        for min_train in [1, 5, 10, 25]:
            for min_test in [5, 10, 20]:
                for independent in [False, True]:
                    test_field = 'independent_test_positive_resumes' if independent else 'test_positive_resumes'
                    feasibility[f'train_positive_resumes>={min_train};{test_field}>={min_test}'] = sum(
                        t['train_positive_resumes'] >= min_train and t[test_field] >= min_test for t in targets)
        results[dataset] = {'sources': paths, 'source_sha256': {s: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for s, p in paths.items()},
                            'splits': stats, 'overlap': overlap, 'total_occupations': len(targets),
                            'test_resume_contents_absent_from_train_and_validation': len(resume_sets['test'] - prior_resumes),
                            'target_feasibility_screen': feasibility,
                            'top_targets_by_test_positive_support': sorted(targets, key=lambda t: -t['test_positive_resumes'])[:15]}
    return {'identity_definition': 'SHA256 of normalized resume content fields; exact-content proxy, not verified person identity',
            'positive_definition': 'top-level label == 1; original labels separately profiled',
            'feasibility_note': 'Counts are feasibility screens, not statistical power guarantees. Shared entities do not automatically invalidate warm-start tasks.',
            'datasets': results}, support_rows


if __name__ == '__main__':
    report, support = audit()
    out = ROOT / 'results/occupation_task_readiness'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'occupation_support.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(support[0]))
        writer.writeheader()
        writer.writerows(support)
    for dataset, detail in report['datasets'].items():
        print(dataset, json.dumps({k: detail[k] for k in ['splits', 'overlap', 'total_occupations', 'test_resume_contents_absent_from_train_and_validation', 'target_feasibility_screen']}, indent=2))

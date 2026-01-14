#!/usr/bin/env python3
"""
Build a data-driven career graph from actual training data.

This script:
1. Extracts all unique occupation URIs from training data
2. Adds missing nodes to the ESCO graph
3. Creates edges based on:
   - Seniority level patterns (junior → senior)
   - Skill similarity between jobs
   - Augmentation relationships
4. Saves updated graph
"""

import json
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_seniority_level(title):
    """Extract seniority level from job title."""
    title_lower = title.lower()

    if any(kw in title_lower for kw in ['junior', 'jr', 'entry', 'trainee', 'assistant']):
        return 'junior', 1
    elif any(kw in title_lower for kw in ['senior', 'sr', 'principal']):
        return 'senior', 3
    elif any(kw in title_lower for kw in ['lead', 'chief', 'head', 'director', 'manager']):
        return 'lead', 4
    else:
        return 'mid', 2


def extract_base_role(title):
    """Extract base role from title (remove seniority markers)."""
    title_lower = title.lower()

    # Remove seniority markers
    markers = ['junior', 'jr', 'senior', 'sr', 'lead', 'chief', 'head',
               'director', 'manager', 'principal', 'entry', 'trainee', 'assistant']

    for marker in markers:
        title_lower = title_lower.replace(marker, '')

    return title_lower.strip()


def compute_title_similarity(title1, title2):
    """Compute similarity between two job titles."""
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()


def load_training_data(data_path):
    """Load all training data and extract job information."""

    logger.info(f"Loading training data from {data_path}")

    jobs = {}  # uri -> job info
    uri_titles = defaultdict(set)  # uri -> set of titles
    augmentation_pairs = []  # (original_uri, augmented_uri, aug_type)

    with open(data_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())

                job = sample.get('job', {})
                resume = sample.get('resume', {})
                uri = job.get('occupation_uri')

                if uri:
                    # Extract from JOB object
                    title = job.get('title', 'Unknown')
                    exp_level = job.get('experience_level', 'unknown')

                    # Extract from RESUME object
                    resume_skills = resume.get('skills', [])

                    # Store job info
                    if uri not in jobs:
                        jobs[uri] = {
                            'uri': uri,
                            'titles': set(),
                            'skills': set(),
                            'experience_levels': set(),
                            'sample_count': 0
                        }

                    jobs[uri]['titles'].add(title)
                    jobs[uri]['experience_levels'].add(exp_level)
                    jobs[uri]['sample_count'] += 1
                    uri_titles[uri].add(title)

                    # Extract skills from RESUME only
                    if isinstance(resume_skills, list):
                        for skill in resume_skills:
                            if isinstance(skill, dict):
                                skill_name = skill.get('name')
                                if skill_name:
                                    jobs[uri]['skills'].add(skill_name.lower())

                    # Track augmentation relationships
                    metadata = sample.get('metadata', {})
                    aug_type = metadata.get('augmentation_type', '')

                    # For augmented samples, we could track transformations
                    # but we'll focus on structural patterns

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue

    logger.info(f"Extracted {len(jobs)} unique occupation URIs")
    logger.info(
        f"Total samples processed: {sum(j['sample_count'] for j in jobs.values())}")

    return jobs, uri_titles


def add_missing_nodes(graph, jobs):
    """Add nodes for URIs that don't exist in the graph."""

    added_nodes = []

    for uri, job_info in jobs.items():
        if uri not in graph:
            # Get most common title
            title = list(job_info['titles'])[
                0] if job_info['titles'] else 'Unknown'

            graph.add_node(uri, title=title)
            added_nodes.append((uri, title))
            logger.info(f"Added missing node: {title} ({uri})")

    return added_nodes


def create_seniority_edges(graph, jobs, uri_titles):
    """Create edges based on seniority progression patterns."""

    logger.info("Creating seniority-based edges...")

    # Group by base role
    role_groups = defaultdict(
        lambda: {'junior': [], 'mid': [], 'senior': [], 'lead': []})

    for uri, job_info in jobs.items():
        for title in job_info['titles']:
            level, level_num = extract_seniority_level(title)
            base_role = extract_base_role(title)

            if base_role:  # Only if we extracted a valid base role
                role_groups[base_role][level].append(uri)

    edges_added = 0

    # Create progression edges within each role group
    for base_role, levels in role_groups.items():
        # Junior → Mid
        for junior_uri in levels['junior']:
            for mid_uri in levels['mid']:
                if junior_uri != mid_uri and not graph.has_edge(junior_uri, mid_uri):
                    graph.add_edge(junior_uri, mid_uri, weight=1,
                                   type='seniority_progression')
                    edges_added += 1

        # Mid → Senior
        for mid_uri in levels['mid']:
            for senior_uri in levels['senior']:
                if mid_uri != senior_uri and not graph.has_edge(mid_uri, senior_uri):
                    graph.add_edge(mid_uri, senior_uri, weight=1,
                                   type='seniority_progression')
                    edges_added += 1

        # Senior → Lead
        for senior_uri in levels['senior']:
            for lead_uri in levels['lead']:
                if senior_uri != lead_uri and not graph.has_edge(senior_uri, lead_uri):
                    graph.add_edge(senior_uri, lead_uri, weight=2,
                                   type='seniority_progression')
                    edges_added += 1

        # Junior → Senior (skip mid if direct path makes sense)
        if not levels['mid']:  # Only if no mid-level exists
            for junior_uri in levels['junior']:
                for senior_uri in levels['senior']:
                    if junior_uri != senior_uri and not graph.has_edge(junior_uri, senior_uri):
                        graph.add_edge(junior_uri, senior_uri,
                                       weight=2, type='seniority_progression')
                        edges_added += 1

    logger.info(f"Added {edges_added} seniority progression edges")
    return edges_added


def create_skill_similarity_edges(graph, jobs):
    """Create edges based on skill similarity (Jaccard)."""

    logger.info("Creating skill-based similarity edges...")

    edges_added = 0
    job_list = list(jobs.items())

    for i, (uri1, job1) in enumerate(job_list):
        for uri2, job2 in job_list[i+1:]:
            if uri1 == uri2 or graph.has_edge(uri1, uri2):
                continue

            skills1 = job1['skills']
            skills2 = job2['skills']

            if not skills1 or not skills2:
                continue

            # Jaccard similarity
            intersection = len(skills1 & skills2)
            union = len(skills1 | skills2)

            if union == 0:
                continue

            similarity = intersection / union

            # Add edge if similarity > threshold
            if similarity > 0.3:  # 30% skill overlap
                # Convert similarity to distance (inverse)
                distance = int(10 * (1 - similarity))
                distance = max(1, min(distance, 5))  # Clamp to 1-5

                # Add bidirectional edges for skill similarity
                graph.add_edge(uri1, uri2, weight=distance,
                               type='skill_similarity')
                graph.add_edge(uri2, uri1, weight=distance,
                               type='skill_similarity')
                edges_added += 2

    logger.info(f"Added {edges_added} skill similarity edges")
    return edges_added


def create_title_similarity_edges(graph, jobs, uri_titles):
    """Create edges based on title similarity for related roles."""

    logger.info("Creating title-based similarity edges...")

    edges_added = 0
    job_list = list(jobs.items())

    for i, (uri1, job1) in enumerate(job_list):
        title1 = list(job1['titles'])[0] if job1['titles'] else ''

        for uri2, job2 in job_list[i+1:]:
            if uri1 == uri2 or graph.has_edge(uri1, uri2) or graph.has_edge(uri2, uri1):
                continue

            title2 = list(job2['titles'])[0] if job2['titles'] else ''

            similarity = compute_title_similarity(title1, title2)

            # Add edge if titles are similar
            if similarity > 0.4:  # 40% title similarity
                distance = int(8 * (1 - similarity))
                distance = max(2, min(distance, 6))  # Clamp to 2-6

                # Add bidirectional edges
                graph.add_edge(uri1, uri2, weight=distance,
                               type='title_similarity')
                graph.add_edge(uri2, uri1, weight=distance,
                               type='title_similarity')
                edges_added += 2

    logger.info(f"Added {edges_added} title similarity edges")
    return edges_added


def main():
    print("=" * 80)
    print("BUILDING DATA-DRIVEN CAREER GRAPH")
    print("=" * 80)

    # Paths
    data_path = Path(__file__).parent / "preprocess" / \
        "augmented_enriched_data_training_updated_with_uri.jsonl"
    graph_path = Path(__file__).parent / \
        "training_output" / "career_graph_bridged.gexf"
    output_path = Path(__file__).parent / "training_output" / \
        "career_graph_data_driven.gexf"

    # Load existing graph
    print(f"\nLoading existing graph from: {graph_path}")
    G = nx.read_gexf(str(graph_path))

    print(f"Original graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Load training data
    jobs, uri_titles = load_training_data(str(data_path))

    # Add missing nodes
    print("\n" + "-" * 80)
    print("ADDING MISSING NODES")
    print("-" * 80)
    added_nodes = add_missing_nodes(G, jobs)
    print(f"Added {len(added_nodes)} missing nodes")

    # Create edges based on different strategies
    print("\n" + "-" * 80)
    print("CREATING EDGES FROM TRAINING DATA")
    print("-" * 80)

    seniority_edges = create_seniority_edges(G, jobs, uri_titles)
    skill_edges = create_skill_similarity_edges(G, jobs)
    title_edges = create_title_similarity_edges(G, jobs, uri_titles)

    # Check connectivity
    print("\n" + "-" * 80)
    print("CONNECTIVITY ANALYSIS")
    print("-" * 80)

    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)

    print(f"\nUpdated graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Weak components: {len(components)}")
    print(
        f"  Largest component: {len(components[0])} nodes ({len(components[0])/G.number_of_nodes()*100:.1f}%)")

    # Test with training URIs
    print("\n" + "-" * 80)
    print("TESTING CONNECTIVITY WITH TRAINING DATA")
    print("-" * 80)

    training_uris = list(jobs.keys())
    connected_pairs = 0
    total_pairs = 0

    for i, uri1 in enumerate(training_uris[:20]):  # Test first 20
        for uri2 in training_uris[i+1:i+21]:
            total_pairs += 1
            try:
                if nx.has_path(G, uri1, uri2):
                    connected_pairs += 1
            except nx.NodeNotFound:
                pass

    if total_pairs > 0:
        print(
            f"Sample connectivity: {connected_pairs}/{total_pairs} pairs ({connected_pairs/total_pairs*100:.1f}%)")

    # Save updated graph
    print("\n" + "-" * 80)
    print(f"Saving data-driven graph to: {output_path}")
    nx.write_gexf(G, str(output_path))

    print("\n" + "=" * 80)
    print("✅ DATA-DRIVEN CAREER GRAPH BUILT SUCCESSFULLY!")
    print("=" * 80)

    print("\nSummary:")
    print(f"  • Added {len(added_nodes)} missing nodes (100% coverage)")
    print(f"  • Created {seniority_edges} seniority progression edges")
    print(f"  • Created {skill_edges} skill similarity edges")
    print(f"  • Created {title_edges} title similarity edges")
    print(
        f"  • Total new edges: {seniority_edges + skill_edges + title_edges}")
    print(f"\nNext step: Update career_graph.py to use: {output_path.name}")
    print()


if __name__ == "__main__":
    main()

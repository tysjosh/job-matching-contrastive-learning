"""
CareerGraph implementation for pathway-aware negative sampling in contrastive learning.

This module provides career progression analysis and pathway-aware negative sampling
for resume-job matching models using ESCO graph-based career distance calculations.
"""

import random
import os
from typing import Dict, List, Any
import logging

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    raise ImportError(
        "NetworkX is required for ESCO graph operations. Please install it via 'pip install networkx'.")

logger = logging.getLogger(__name__)


class CareerGraph:
    """
    Manages career pathway information for pathway-aware negative sampling.

    Provides methods to compute career distances between jobs and select
    pathway-aware negative samples based on career progression patterns.
    Uses ESCO NetworkX graph for accurate career distance calculations.
    """

    def __init__(self, esco_graph_path: str, hard_negative_max_distance: float = 2.0,
                 medium_negative_max_distance: float = 4.0, use_distance_cache: bool = True):
        """
        Initialize CareerGraph with ESCO graph support.

        Args:
            esco_graph_path: Path to pre-built ESCO NetworkX graph (.gexf file). Required.
            hard_negative_max_distance: Maximum distance for hard negatives.
            medium_negative_max_distance: Maximum distance for medium negatives.
            use_distance_cache: Whether to cache distance calculations for training performance.
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for ESCO graph operations. Please install it via 'pip install networkx'.")

        self.hard_negative_max_distance = hard_negative_max_distance
        self.medium_negative_max_distance = medium_negative_max_distance
        self.use_distance_cache = use_distance_cache

        # Initialize components
        self.esco_graph = None
        self.distance_cache = {} if use_distance_cache else None
        self.uri_to_title = {}  # For ESCO graph node lookup

        # Load ESCO graph
        self._load_esco_graph(esco_graph_path)

    def _load_esco_graph(self, esco_graph_path: str):
        """Load pre-built ESCO NetworkX graph for career distance calculations."""
        try:
            if not os.path.exists(esco_graph_path):
                raise FileNotFoundError(
                    f"ESCO graph file not found at: {esco_graph_path}")

            self.esco_graph = nx.read_gexf(esco_graph_path)

            # Build URI to title mapping for faster lookups
            for node in self.esco_graph.nodes():
                title = self.esco_graph.nodes[node].get('title', 'Unknown')
                self.uri_to_title[node] = title

            logger.info(
                f"Loaded ESCO graph with {self.esco_graph.number_of_nodes()} nodes and {self.esco_graph.number_of_edges()} edges")

        except Exception as e:
            logger.error(
                f"Error loading ESCO graph from {esco_graph_path}: {e}")
            raise

    def _get_esco_distance(self, job1_uri: str, job2_uri: str) -> float:
        """Get career distance using ESCO NetworkX graph."""
        if not self.esco_graph or job1_uri not in self.esco_graph or job2_uri not in self.esco_graph:
            return float('inf')

        try:
            # Check cache first for training performance
            if self.distance_cache is not None:
                cache_key = (job1_uri, job2_uri)
                if cache_key in self.distance_cache:
                    return self.distance_cache[cache_key]

            # Calculate shortest path distance
            distance = nx.shortest_path_length(
                self.esco_graph, source=job1_uri, target=job2_uri)

            # Cache the result
            if self.distance_cache is not None:
                self.distance_cache[(job1_uri, job2_uri)] = distance
                self.distance_cache[(job2_uri, job1_uri)
                                    ] = distance  # Symmetric

            return float(distance)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    def compute_career_distance(self, job1: Dict[str, Any], job2: Dict[str, Any]) -> float:
        """
        Compute career distance between two jobs using ESCO graph.

        Args:
            job1: First job dictionary with 'uri' field.
            job2: Second job dictionary with 'uri' field.

        Returns:
            Career distance (0.0 = same job, higher = more distant).
        """
        try:
            # Extract URIs from job data
            job1_uri = job1.get('uri') or job1.get(
                'job_uri') or job1.get('occupation_uri')
            job2_uri = job2.get('uri') or job2.get(
                'job_uri') or job2.get('occupation_uri')

            if not job1_uri or not job2_uri:
                logger.warning(
                    f"Missing URIs for career distance calculation: job1_uri={job1_uri}, job2_uri={job2_uri}")
                return float('inf')

            if job1_uri == job2_uri:
                return 0.0

            return self._get_esco_distance(job1_uri, job2_uri)

        except Exception as e:
            logger.warning(f"Error computing career distance: {e}")
            return float('inf')

    def select_pathway_negatives(self, positive_job: Dict[str, Any],
                                 candidate_jobs: List[Dict[str, Any]],
                                 num_negatives: int) -> List[Dict[str, Any]]:
        """
        Select pathway-aware negative samples based on career distances.

        Args:
            positive_job: The positive job for comparison.
            candidate_jobs: List of candidate negative jobs.
            num_negatives: Number of negatives to select.

        Returns:
            List of selected negative jobs.
        """
        if not candidate_jobs:
            return []

        if len(candidate_jobs) <= num_negatives:
            return candidate_jobs

        try:
            # Compute distances for all candidates
            job_distances = []
            for job in candidate_jobs:
                distance = self.compute_career_distance(positive_job, job)
                job_distances.append((job, distance))

            # Strategic selection based on distance thresholds
            selected = []

            # Categorize candidates by distance
            hard_candidates = [
                job for job, dist in job_distances if dist <= self.hard_negative_max_distance]
            medium_candidates = [
                job for job, dist in job_distances if self.hard_negative_max_distance < dist <= self.medium_negative_max_distance]
            easy_candidates = [
                job for job, dist in job_distances if dist > self.medium_negative_max_distance]

            # Calculate target counts (60% hard, 30% medium, 10% easy)
            hard_count = int(num_negatives * 0.6)
            medium_count = int(num_negatives * 0.3)
            easy_count = num_negatives - hard_count - medium_count

            # Select negatives with random sampling from each category to ensure diversity
            # This ensures we get medium and easy negatives, not just the hardest ones
            if hard_candidates and hard_count > 0:
                selected.extend(random.sample(hard_candidates,
                                min(hard_count, len(hard_candidates))))

            if medium_candidates and medium_count > 0:
                selected.extend(random.sample(medium_candidates, min(
                    medium_count, len(medium_candidates))))

            if easy_candidates and easy_count > 0:
                selected.extend(random.sample(easy_candidates,
                                min(easy_count, len(easy_candidates))))

            # Fill remaining slots if needed (prioritize harder negatives for remaining slots)
            if len(selected) < num_negatives:
                remaining_needed = num_negatives - len(selected)
                used_jobs = set(id(job) for job in selected)

                # Try to fill from each category in order of difficulty
                remaining_candidates = []
                for category in [hard_candidates, medium_candidates, easy_candidates]:
                    remaining_from_category = [
                        job for job in category if id(job) not in used_jobs]
                    remaining_candidates.extend(remaining_from_category)

                if remaining_candidates:
                    additional = random.sample(remaining_candidates, min(
                        remaining_needed, len(remaining_candidates)))
                    selected.extend(additional)

            return selected[:num_negatives]

        except Exception as e:
            logger.warning(f"Error in pathway negative selection: {e}")
            return random.sample(candidate_jobs, min(num_negatives, len(candidate_jobs)))

    def get_job_title(self, job_uri: str) -> str:
        """
        Get the job title for a given URI from the ESCO graph.

        Args:
            job_uri: The URI of the job/occupation.

        Returns:
            The job title or 'Unknown' if not found.
        """
        return self.uri_to_title.get(job_uri, 'Unknown')

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded ESCO graph.

        Returns:
            Dictionary with graph statistics.
        """
        if not self.esco_graph:
            return {'error': 'No ESCO graph loaded'}

        return {
            'nodes': self.esco_graph.number_of_nodes(),
            'edges': self.esco_graph.number_of_edges(),
            'is_directed': self.esco_graph.is_directed(),
            'cache_size': len(self.distance_cache) if self.distance_cache else 0
        }

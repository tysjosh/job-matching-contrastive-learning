#!/usr/bin/env python3
"""
Career Progression Graph Builder using ESCO Dataset
Implements both rule-based heuristics and skill-based analysis for career progression inference.
"""

import pandas as pd
import networkx as nx
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CareerGraphBuilder:
    """
    Builds a career progression graph using ESCO dataset with two methods:
    1. Rule-based heuristics (seniority patterns in job titles)
    2. Skill-based analysis (semantic skill progression)
    """

    def __init__(self, config_path: str = "config/career_graph_config.yaml"):
        """Initialize the career graph builder with configuration."""
        self.config = self._load_config(config_path)
        self.occupations_df = None
        self.skills_df = None
        self.relations_df = None
        self.occupation_skills = {}
        self.senior_skills_set = set()
        self.G_career = nx.DiGraph()

        # Configuration parameters
        self.OVERLAP_THRESHOLD = self.config.get('overlap_threshold', 0.80)
        self.MIN_SKILLS_THRESHOLD = self.config.get('min_skills_threshold', 5)

        # CS/IT specific configuration
        self.cs_config = self.config.get('cs_it_config', {})
        self.CS_ENABLED = self.cs_config.get('enabled', False)
        self.CS_OVERLAP_THRESHOLD = self.cs_config.get(
            'overlap_threshold', 0.50)
        self.CS_MIN_SKILLS_THRESHOLD = self.cs_config.get(
            'min_skills_threshold', 3)
        self.CS_INCLUDE_OPTIONAL = self.cs_config.get(
            'include_optional_skills', True)
        self.CS_ENABLE_LATERAL = self.cs_config.get(
            'enable_lateral_moves', True)

        # Cache for CS occupation URIs
        self.cs_occupation_uris = set()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(
                f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Return default configuration if config file is not found."""
        return {
            'overlap_threshold': 0.80,
            'min_skills_threshold': 5,
            'seniority_tiers': [
                "intern", "junior", "assistant", "trainee",
                "associate", "specialist", "",  # Mid-level (often no prefix)
                "senior", "lead", "principal", "staff",
                "manager", "director", "head", "chief"
            ],
            'seniority_keywords': [
                'manage', 'mentor', 'lead', 'supervise', 'governance', 'strategy',
                'architect', 'design', 'budget', 'planning', 'roadmap', 'business',
                'stakeholder', 'client relations', 'procurement', 'leadership',
                'team management', 'project management', 'strategic planning'
            ]
        }

    def load_esco_data(self, occupations_path: str, skills_path: str, relations_path: str):
        """Load ESCO dataset CSV files."""
        logger.info("Loading ESCO dataset...")

        try:
            self.occupations_df = pd.read_csv(occupations_path)
            self.skills_df = pd.read_csv(skills_path)
            self.relations_df = pd.read_csv(relations_path)

            logger.info(f"Loaded {len(self.occupations_df)} occupations")
            logger.info(f"Loaded {len(self.skills_df)} skills")
            logger.info(
                f"Loaded {len(self.relations_df)} occupation-skill relations")

            # Identify CS/IT occupations if CS config is enabled
            if self.CS_ENABLED:
                self._identify_cs_occupations()

        except FileNotFoundError as e:
            logger.error(f"ESCO data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading ESCO data: {e}")
            raise

    def _identify_cs_occupations(self):
        """Identify CS/IT occupations based on ISCO groups and keywords."""
        logger.info("Identifying CS/IT occupations...")

        cs_uris = set()

        # Method 1: ISCO group filtering
        isco_groups = self.cs_config.get('isco_groups', ['25'])
        for isco_group in isco_groups:
            mask = self.occupations_df['iscoGroup'].astype(
                str).str.startswith(isco_group)
            cs_uris.update(self.occupations_df[mask]['conceptUri'].tolist())

        # Method 2: Keyword matching
        keywords = self.cs_config.get('occupation_keywords', [])
        if keywords:
            pattern = '|'.join(keywords)
            mask = (
                self.occupations_df['preferredLabel'].str.contains(pattern, case=False, na=False) |
                self.occupations_df.get('altLabels', pd.Series()).str.contains(
                    pattern, case=False, na=False)
            )
            cs_uris.update(self.occupations_df[mask]['conceptUri'].tolist())

        self.cs_occupation_uris = cs_uris
        logger.info(
            f"Identified {len(self.cs_occupation_uris)} CS/IT occupations")

    def _is_cs_occupation(self, occupation_uri: str) -> bool:
        """Check if an occupation is a CS/IT occupation."""
        return self.CS_ENABLED and occupation_uri in self.cs_occupation_uris

    def _get_title(self, occupation_uri: str) -> str:
        """Get the title of an occupation by its URI."""
        matches = self.occupations_df[self.occupations_df['conceptUri']
                                      == occupation_uri]
        if len(matches) > 0:
            return matches['preferredLabel'].iloc[0]
        return ""

    def create_skill_profiles(self):
        """Create skill profiles for each occupation using essential skills only."""
        logger.info("Creating skill profiles for occupations...")

        # Filter for essential skills for all occupations
        essential_relations = self.relations_df[
            self.relations_df['relationType'] == 'essential'
        ].copy()

        # For CS jobs, also include optional skills if configured
        if self.CS_ENABLED and self.CS_INCLUDE_OPTIONAL and len(self.cs_occupation_uris) > 0:
            logger.info("Including optional skills for CS/IT occupations...")
            optional_relations = self.relations_df[
                (self.relations_df['relationType'] == 'optional') &
                (self.relations_df['occupationUri'].isin(
                    self.cs_occupation_uris))
            ].copy()

            # Combine essential + optional for CS
            all_relations = pd.concat(
                [essential_relations, optional_relations], ignore_index=True)
        else:
            all_relations = essential_relations

        # Group by occupation and aggregate essential skill URIs into sets
        self.occupation_skills = all_relations.groupby(
            'occupationUri')['skillUri'].apply(set).to_dict()

        logger.info(
            f"Created skill profiles for {len(self.occupation_skills)} occupations")

        # Log CS occupation statistics
        if self.CS_ENABLED:
            cs_with_skills = sum(
                1 for uri in self.cs_occupation_uris if uri in self.occupation_skills)
            logger.info(
                f"CS/IT occupations with skills: {cs_with_skills}/{len(self.cs_occupation_uris)}")

    def curate_seniority_skills(self):
        """Identify skills that indicate seniority based on keywords."""
        logger.info("Curating seniority skills lexicon...")

        seniority_keywords = self.config.get('seniority_keywords', [])

        # Add CS-specific seniority keywords if enabled
        if self.CS_ENABLED:
            cs_seniority_keywords = self.cs_config.get(
                'cs_seniority_keywords', [])
            seniority_keywords = list(
                set(seniority_keywords + cs_seniority_keywords))
            logger.info(
                f"Using {len(cs_seniority_keywords)} additional CS-specific seniority keywords")

        # Create regex pattern for keyword matching
        pattern = '|'.join(seniority_keywords)

        # Search in both skill title and description
        senior_skills_mask = (
            self.skills_df['preferredLabel'].str.contains(pattern, case=False, na=False) |
            self.skills_df['description'].str.contains(
                pattern, case=False, na=False)
        )

        # Get URIs of matching skills
        senior_skill_uris = self.skills_df[senior_skills_mask]['conceptUri'].tolist(
        )
        self.senior_skills_set = set(senior_skill_uris)

        logger.info(
            f"Identified {len(self.senior_skills_set)} potential seniority skills")

    def rule_based_progression(self) -> List[Tuple[str, str]]:
        """
        Infer progression using rule-based heuristics on job titles.
        Returns list of (junior_uri, senior_uri) tuples.
        """
        logger.info("Running Rule-based heuristics...")

        seniority_tiers = self.config.get('seniority_tiers', [])
        tier_map = {tier: i for i, tier in enumerate(seniority_tiers)}

        edges = []

        # Create a mapping of normalized titles to occupation URIs
        title_to_uris = {}
        for _, row in self.occupations_df.iterrows():
            title = row['preferredLabel'].lower().strip()
            uri = row['conceptUri']

            if title not in title_to_uris:
                title_to_uris[title] = []
            title_to_uris[title].append(uri)

        # Find progression patterns
        for title, uris in title_to_uris.items():
            if len(uris) < 2:
                continue

            # Extract base title and seniority modifiers
            title_variants = []
            for uri in uris:
                full_title = self.occupations_df[
                    self.occupations_df['conceptUri'] == uri
                ]['preferredLabel'].iloc[0]

                # Find seniority tier in title
                found_tier = None
                base_title = full_title.lower()

                for tier in seniority_tiers:
                    if tier and tier in base_title:
                        found_tier = tier
                        base_title = base_title.replace(tier, '').strip()
                        break

                if not found_tier:
                    found_tier = ""  # Mid-level default

                title_variants.append((uri, base_title, found_tier))

            # Create edges between different seniority levels of same base title
            for i, (uri1, base1, tier1) in enumerate(title_variants):
                for j, (uri2, base2, tier2) in enumerate(title_variants):
                    if i != j and base1 == base2:
                        tier1_level = tier_map.get(
                            tier1, len(seniority_tiers) // 2)
                        tier2_level = tier_map.get(
                            tier2, len(seniority_tiers) // 2)

                        if tier1_level < tier2_level:  # tier1 is junior to tier2
                            edges.append((uri1, uri2))

        logger.info(
            f"Rule-based heuristics found {len(edges)} progression edges")
        return edges

    def detect_lateral_cs_moves(self) -> List[Tuple[str, str]]:
        """
        Detect lateral career moves between CS specializations at same seniority level.
        Returns list of (uri1, uri2) tuples representing bidirectional lateral moves.
        """
        if not self.CS_ENABLED or not self.CS_ENABLE_LATERAL:
            return []

        logger.info("Detecting lateral CS moves...")

        edges = []
        specialization_groups = self.cs_config.get('specialization_groups', {})

        for spec_name, spec_keywords in specialization_groups.items():
            # Find roles matching this specialization
            spec_roles = []
            for uri in self.cs_occupation_uris:
                if uri not in self.occupation_skills:
                    continue

                title = self._get_title(uri).lower()
                if any(keyword.lower() in title for keyword in spec_keywords):
                    spec_roles.append(uri)

            # Create lateral edges between roles at similar seniority levels
            for uri1, uri2 in combinations(spec_roles, 2):
                if self._similar_seniority_level(uri1, uri2):
                    # Add bidirectional edges for lateral moves
                    edges.append((uri1, uri2))
                    edges.append((uri2, uri1))

        logger.info(f"Detected {len(edges)} lateral CS move edges")
        return edges

    def _similar_seniority_level(self, uri1: str, uri2: str) -> bool:
        """Check if two roles are at similar seniority levels."""
        title1 = self._get_title(uri1).lower()
        title2 = self._get_title(uri2).lower()

        # Get CS seniority tiers if available
        if self.CS_ENABLED:
            cs_tiers = self.cs_config.get('cs_seniority_tiers', [])
        else:
            cs_tiers = self.config.get('seniority_tiers', [])

        tier_map = {tier.lower(): i for i, tier in enumerate(cs_tiers) if tier}

        # Find tier levels
        tier1_level = None
        tier2_level = None

        for tier, level in tier_map.items():
            if tier in title1:
                tier1_level = level
            if tier in title2:
                tier2_level = level

        # If both have no tier markers, assume same level (mid-level)
        if tier1_level is None and tier2_level is None:
            return True

        # If both have tiers, check if within 1 level
        if tier1_level is not None and tier2_level is not None:
            return abs(tier1_level - tier2_level) <= 1

        return False

    def skill_based_progression(self) -> List[Tuple[str, str]]:
        """
        Infer progression using skill-based analysis.
        Returns list of (junior_uri, senior_uri) tuples.
        """
        logger.info("Running Skill-based analysis...")

        edges = []
        all_occupation_uris = list(self.occupation_skills.keys())

        # Compare all pairs of occupations
        for occ_A_uri, occ_B_uri in combinations(all_occupation_uris, 2):
            skills_A = self.occupation_skills.get(occ_A_uri, set())
            skills_B = self.occupation_skills.get(occ_B_uri, set())

            # Determine if either or both are CS occupations
            is_cs_A = self._is_cs_occupation(occ_A_uri)
            is_cs_B = self._is_cs_occupation(occ_B_uri)
            is_cs_pair = is_cs_A or is_cs_B

            # Use appropriate thresholds based on occupation type
            if is_cs_pair:
                min_skills_threshold = self.CS_MIN_SKILLS_THRESHOLD
                overlap_threshold = self.CS_OVERLAP_THRESHOLD
            else:
                min_skills_threshold = self.MIN_SKILLS_THRESHOLD
                overlap_threshold = self.OVERLAP_THRESHOLD

            # Skip if either role has too few skills
            if len(skills_A) < min_skills_threshold or len(skills_B) < min_skills_threshold:
                continue

            # Determine junior and senior roles
            junior_skills, senior_skills = None, None
            junior_uri, senior_uri = None, None

            # Check if one is a clear subset of the other
            if skills_A.issubset(skills_B):
                junior_skills, senior_skills = skills_A, skills_B
                junior_uri, senior_uri = occ_A_uri, occ_B_uri
            elif skills_B.issubset(skills_A):
                junior_skills, senior_skills = skills_B, skills_A
                junior_uri, senior_uri = occ_B_uri, occ_A_uri
            else:
                # Check for high overlap percentage
                overlap_A_to_B = len(
                    skills_A.intersection(skills_B)) / len(skills_A)
                overlap_B_to_A = len(
                    skills_B.intersection(skills_A)) / len(skills_B)

                if overlap_A_to_B >= overlap_threshold:
                    junior_skills, senior_skills = skills_A, skills_B
                    junior_uri, senior_uri = occ_A_uri, occ_B_uri
                elif overlap_B_to_A >= overlap_threshold:
                    junior_skills, senior_skills = skills_B, skills_A
                    junior_uri, senior_uri = occ_B_uri, occ_A_uri
                else:
                    continue  # No high overlap, likely unrelated roles

            # Apply three-filter logic (with relaxed rules for CS)

            # FILTER 1: Relevance (already passed by overlap check)

            # FILTER 2: Growth - Senior role must have new skills (or more skills for CS)
            new_skills = senior_skills - junior_skills
            if is_cs_pair:
                # For CS jobs, allow progression if new skills exist OR senior has more skills
                if not new_skills and len(senior_skills) <= len(junior_skills):
                    continue
            else:
                # For non-CS jobs, strictly require new skills
                if not new_skills:
                    continue

            # FILTER 3: Seniority - New skills must include seniority indicators (relaxed for CS)
            if is_cs_pair:
                # For CS jobs, check both skills and title patterns
                has_seniority_skills = bool(
                    new_skills.intersection(self.senior_skills_set))
                has_seniority_title = self._has_cs_seniority_pattern(
                    senior_uri, junior_uri)

                if not (has_seniority_skills or has_seniority_title):
                    continue
            else:
                # For non-CS jobs, strictly require seniority skills
                if not new_skills.intersection(self.senior_skills_set):
                    continue

            # Valid progression found
            edges.append((junior_uri, senior_uri))

        logger.info(
            f"Skill-based analysis found {len(edges)} progression edges")
        return edges

    def _has_cs_seniority_pattern(self, senior_uri: str, junior_uri: str) -> bool:
        """Check if the senior role has CS-specific seniority patterns in title."""
        if not self.CS_ENABLED:
            return False

        senior_title = self._get_title(senior_uri).lower()
        junior_title = self._get_title(junior_uri).lower()

        # Get CS seniority tiers
        cs_tiers = self.cs_config.get('cs_seniority_tiers', [])
        tier_map = {tier.lower(): i for i, tier in enumerate(cs_tiers) if tier}

        # Find tier levels
        senior_tier = None
        junior_tier = None

        for tier, level in tier_map.items():
            if tier in senior_title:
                senior_tier = level
            if tier in junior_title:
                junior_tier = level

        # Check if senior has higher tier
        if senior_tier is not None and junior_tier is not None:
            return senior_tier > junior_tier
        elif senior_tier is not None and junior_tier is None:
            return True  # Senior has tier, junior doesn't

        return False

    def build_career_graph(self):
        """Build the complete career progression graph using both methods."""
        logger.info("Building career progression graph...")

        # Ensure data is prepared
        if not self.occupation_skills:
            self.create_skill_profiles()
        if not self.senior_skills_set:
            self.curate_seniority_skills()

        # Get edges from both methods
        method_a_edges = self.rule_based_progression()
        method_b_edges = self.skill_based_progression()

        # Get lateral CS moves if enabled
        lateral_edges = []
        if self.CS_ENABLED and self.CS_ENABLE_LATERAL:
            lateral_edges = self.detect_lateral_cs_moves()
            logger.info(f"Added {len(lateral_edges)} lateral CS move edges")

        # Combine edges (remove duplicates)
        all_edges = list(set(method_a_edges + method_b_edges + lateral_edges))

        # Build the graph
        self.G_career = nx.DiGraph()
        self.G_career.add_edges_from(all_edges)

        logger.info(
            f"Career graph built with {self.G_career.number_of_nodes()} nodes and {self.G_career.number_of_edges()} edges")

        # Add node attributes (occupation titles)
        for node in self.G_career.nodes():
            title = self.occupations_df[
                self.occupations_df['conceptUri'] == node
            ]['preferredLabel'].iloc[0] if len(self.occupations_df[
                self.occupations_df['conceptUri'] == node
            ]) > 0 else "Unknown"
            self.G_career.nodes[node]['title'] = title

        # Log CS statistics
        if self.CS_ENABLED:
            cs_nodes = [n for n in self.G_career.nodes()
                        if self._is_cs_occupation(n)]
            logger.info(
                f"CS/IT occupations in graph: {len(cs_nodes)}/{len(self.cs_occupation_uris)}")

    def prune_graph(self, remove_transitive=False):
        """Prune and refine the career graph."""
        logger.info("Pruning career graph...")

        initial_edges = self.G_career.number_of_edges()

        # Remove bidirectional edges (much faster than finding all cycles)
        # Instead of finding all cycles, just check for bidirectional edges
        logger.info("Removing bidirectional edges...")
        edges_to_remove = []

        for u, v in list(self.G_career.edges()):
            # If edge (v, u) also exists, keep only one direction
            if self.G_career.has_edge(v, u):
                # Keep the edge with more "senior" target (heuristic)
                # Remove the reverse edge to make it unidirectional
                edges_to_remove.append((v, u))

        # Remove duplicate reverse edges
        edges_to_remove = list(set(edges_to_remove))
        self.G_career.remove_edges_from(edges_to_remove)
        logger.info(f"Removed {len(edges_to_remove)} bidirectional edges")

        # Remove transitive edges (OPTIONAL - disabled by default due to O(N^3) complexity)
        # For large graphs, this is too slow. Can be enabled for small graphs only.
        if remove_transitive and self.G_career.number_of_nodes() < 500:
            logger.info("Removing transitive edges (this may take a while)...")
            transitive_edges = []
            nodes = list(self.G_career.nodes())
            total_nodes = len(nodes)

            for idx, node in enumerate(nodes):
                if idx % 100 == 0:
                    logger.info(f"Processing node {idx}/{total_nodes}...")

                successors = list(self.G_career.successors(node))
                if len(successors) <= 1:
                    continue

                for succ1 in successors:
                    for succ2 in successors:
                        if succ1 != succ2:
                            # Check if there's a path from succ1 to succ2
                            try:
                                if nx.has_path(self.G_career, succ1, succ2):
                                    transitive_edges.append((node, succ2))
                            except:
                                pass

            self.G_career.remove_edges_from(transitive_edges)
            logger.info(f"Removed {len(transitive_edges)} transitive edges")
        else:
            logger.info(
                "Skipping transitive edge removal (disabled for performance)")

        final_edges = self.G_career.number_of_edges()
        logger.info(
            f"Pruning complete: {initial_edges} -> {final_edges} edges")

    def get_career_distance(self, role1_uri: str, role2_uri: str) -> float:
        """
        Calculate the shortest path distance between two roles in the career graph.

        Args:
            role1_uri: URI of the starting role
            role2_uri: URI of the target role

        Returns:
            Distance as number of career steps, or float('inf') if no path exists
        """
        try:
            distance = nx.shortest_path_length(
                self.G_career, source=role1_uri, target=role2_uri)
            return distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    def get_career_path(self, role1_uri: str, role2_uri: str) -> Optional[List[str]]:
        """
        Get the shortest career path between two roles.

        Args:
            role1_uri: URI of the starting role
            role2_uri: URI of the target role

        Returns:
            List of role URIs representing the career path, or None if no path exists
        """
        try:
            path = nx.shortest_path(
                self.G_career, source=role1_uri, target=role2_uri)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def save_graph(self, filepath: str):
        """Save the career graph to a file."""
        nx.write_gexf(self.G_career, filepath)
        logger.info(f"Career graph saved to {filepath}")

    def load_graph(self, filepath: str):
        """Load a career graph from a file."""
        self.G_career = nx.read_gexf(filepath)
        logger.info(f"Career graph loaded from {filepath}")

    def get_graph_statistics(self) -> Dict:
        """Get statistics about the career graph."""
        stats = {
            'nodes': self.G_career.number_of_nodes(),
            'edges': self.G_career.number_of_edges(),
            'density': nx.density(self.G_career),
            'is_dag': nx.is_directed_acyclic_graph(self.G_career),
            'weakly_connected_components': nx.number_weakly_connected_components(self.G_career),
            'strongly_connected_components': nx.number_strongly_connected_components(self.G_career)
        }
        return stats

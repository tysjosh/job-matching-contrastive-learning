"""
ESCO context builder for LLM augmentation prompts.

This module extracts lightweight ontology guidance from ESCO domain data and
the CS skills dataset to provide domain-aware hints to the LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class ESCOContext:
    """ESCO-derived context for prompt injection."""
    domain: Optional[str]
    adjacent_roles: List[str]
    skill_clusters: Dict[str, List[str]]

    def to_prompt_block(self) -> str:
        """Render context as a prompt block."""
        if not self.domain and not self.adjacent_roles and not self.skill_clusters:
            return ""

        lines = ["ESCO guidance:"]
        if self.domain:
            lines.append(f"- Domain: {self.domain}")
        if self.adjacent_roles:
            lines.append(f"- Adjacent roles: {', '.join(self.adjacent_roles)}")
        if self.skill_clusters:
            for cluster, skills in self.skill_clusters.items():
                if skills:
                    lines.append(f"- {cluster} skills: {', '.join(skills)}")
        return "\n".join(lines)


class ESCOContextBuilder:
    """Build ESCO context for prompt injection."""

    def __init__(
        self,
        esco_domains_path: Optional[str] = "esco_it_career_domains_refined.json",
        cs_skills_path: Optional[str] = "dataset/cs_skills.json"
    ):
        self.esco_domains_path = Path(esco_domains_path) if esco_domains_path else None
        self.cs_skills_path = Path(cs_skills_path) if cs_skills_path else None
        self._domain_map = self._load_esco_domains()
        self._skill_clusters = self._load_skill_clusters()

    def _load_esco_domains(self) -> Dict[str, List[str]]:
        if not self.esco_domains_path or not self.esco_domains_path.exists():
            return {}
        with open(self.esco_domains_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("career_domains", {})

    def _load_skill_clusters(self) -> Dict[str, List[str]]:
        if not self.cs_skills_path or not self.cs_skills_path.exists():
            return {}
        with open(self.cs_skills_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        clusters = {}
        for key, values in data.items():
            if isinstance(values, list):
                clusters[key] = values
        return clusters

    def build_context(
        self,
        role: str,
        job_title: str,
        skills: Optional[List[Dict[str, str]]] = None,
        max_adjacent_roles: int = 6,
        max_skills_per_cluster: int = 6
    ) -> ESCOContext:
        domain = self._match_domain(role, job_title)
        adjacent_roles = self._get_adjacent_roles(domain, max_adjacent_roles)
        skill_clusters = self._get_skill_clusters(skills or [], max_skills_per_cluster)

        return ESCOContext(
            domain=domain,
            adjacent_roles=adjacent_roles,
            skill_clusters=skill_clusters
        )

    def _match_domain(self, role: str, job_title: str) -> Optional[str]:
        if not self._domain_map:
            return None
        text = f"{role} {job_title}".lower()
        for domain, roles in self._domain_map.items():
            for role_name in roles:
                if role_name.lower() in text:
                    return domain
        return None

    def _get_adjacent_roles(self, domain: Optional[str], max_roles: int) -> List[str]:
        if not domain or domain not in self._domain_map:
            return []
        roles = self._domain_map.get(domain, [])
        return roles[:max_roles]

    def _get_skill_clusters(
        self,
        skills: List[Dict[str, str]],
        max_skills_per_cluster: int
    ) -> Dict[str, List[str]]:
        if not self._skill_clusters:
            return {}
        skill_names = {s.get("name", "").lower() for s in skills if isinstance(s, dict)}
        clusters: Dict[str, List[str]] = {}
        for cluster_name, cluster_skills in self._skill_clusters.items():
            matches: List[str] = []
            for skill in cluster_skills:
                if skill.lower() in skill_names:
                    matches.append(skill)
                if len(matches) >= max_skills_per_cluster:
                    break
            if matches:
                clusters[cluster_name] = matches
        return clusters

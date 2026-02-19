"""
Build an ESCO knowledge graph (skills hierarchy + occupation-skill relations
+ ISCO group links) from the ESCO CSVs.

Inputs (default paths relative to project root):
    dataset/esco/skills_en.csv
    dataset/esco/occupations_en.csv
    dataset/esco/occupationSkillRelations_en.csv
    dataset/esco/ISCOGroups_en.csv
    dataset/esco/skillsHierarchy_en.csv

Output:
    networkx.MultiDiGraph with:
    - Skill nodes          (URI)
    - Occupation nodes     (URI)
    - ISCO group nodes     (URI)
    - Edges:
        * (parent_skill) -[skill_parent_of]-> (child_skill)
        * (occupation)   -[has_skill {relationType, skillType}]-> (skill)
        * (occupation)   -[in_isco_group]-> (isco_group)

Usage:
    python scripts/build_esco_knowledge_graph.py
    python scripts/build_esco_knowledge_graph.py --data-dir dataset/esco --export esco_kg.graphml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def _split_labels(x) -> list[str]:
    """ESCO altLabels often come in newline-separated strings."""
    s = _safe_str(x)
    if not s:
        return []
    parts = [p.strip() for p in s.replace("\r", "\n").split("\n")]
    return [p for p in parts if p]


def _add_node_if_missing(G: nx.MultiDiGraph, uri: str, **attrs):
    if uri not in G:
        G.add_node(uri, **attrs)
    else:
        # merge attrs without clobbering existing non-empty fields
        for k, v in attrs.items():
            if v is None:
                continue
            if k not in G.nodes[uri] or G.nodes[uri].get(k) in (None, "", [], {}):
                G.nodes[uri][k] = v


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_esco_graph(
    data_dir: str = "dataset/esco",
) -> nx.MultiDiGraph:
    data = Path(data_dir)

    skills_df = pd.read_csv(data / "skills_en.csv")
    occ_df = pd.read_csv(data / "occupations_en.csv")
    occ_skill_df = pd.read_csv(data / "occupationSkillRelations_en.csv")
    isco_df = pd.read_csv(data / "ISCOGroups_en.csv")
    hier_df = pd.read_csv(data / "skillsHierarchy_en.csv")

    G = nx.MultiDiGraph(name="ESCO_KG")

    # ---- Skill nodes ----
    for _, row in skills_df.iterrows():
        uri = _safe_str(row.get("conceptUri"))
        if not uri:
            continue
        _add_node_if_missing(
            G, uri,
            node_type="skill",
            concept_type=_safe_str(row.get("conceptType")),
            skill_type=_safe_str(row.get("skillType")),
            reuse_level=_safe_str(row.get("reuseLevel")),
            label=_safe_str(row.get("preferredLabel")),
            alt_labels=_split_labels(row.get("altLabels")),
            description=_safe_str(row.get("description")),
            status=_safe_str(row.get("status")),
            modified_date=_safe_str(row.get("modifiedDate")),
        )

    # ---- Occupation nodes ----
    for _, row in occ_df.iterrows():
        uri = _safe_str(row.get("conceptUri"))
        if not uri:
            continue
        _add_node_if_missing(
            G, uri,
            node_type="occupation",
            concept_type=_safe_str(row.get("conceptType")),
            label=_safe_str(row.get("preferredLabel")),
            alt_labels=_split_labels(row.get("altLabels")),
            description=_safe_str(row.get("description")),
            status=_safe_str(row.get("status")),
            modified_date=_safe_str(row.get("modifiedDate")),
            esco_code=_safe_str(row.get("code")),
            isco_group_code=None if pd.isna(row.get("iscoGroup")) else int(row.get("iscoGroup")),
        )

    # ---- ISCO group nodes ----
    isco_code_to_uri: Dict[int, str] = {}
    for _, row in isco_df.iterrows():
        uri = _safe_str(row.get("conceptUri"))
        code = row.get("code")
        if not uri or pd.isna(code):
            continue
        code_int = int(code)
        isco_code_to_uri[code_int] = uri
        _add_node_if_missing(
            G, uri,
            node_type="isco_group",
            concept_type=_safe_str(row.get("conceptType")),
            isco_code=code_int,
            label=_safe_str(row.get("preferredLabel")),
            alt_labels=_split_labels(row.get("altLabels")),
            description=_safe_str(row.get("description")),
            status=_safe_str(row.get("status")),
        )

    # ---- Link Occupations -> ISCO groups ----
    for occ_uri, occ_data in list(G.nodes(data=True)):
        if occ_data.get("node_type") != "occupation":
            continue
        code = occ_data.get("isco_group_code")
        if isinstance(code, int) and code in isco_code_to_uri:
            isco_uri = isco_code_to_uri[code]
            G.add_edge(
                occ_uri, isco_uri,
                key=f"in_isco_group::{code}",
                edge_type="in_isco_group",
            )

    # ---- Skill hierarchy edges from skillsHierarchy_en.csv ----
    level_uri_cols = [
        "Level 0 URI", "Level 1 URI", "Level 2 URI", "Level 3 URI",
    ]
    level_term_cols = [
        "Level 0 preferred term", "Level 1 preferred term",
        "Level 2 preferred term", "Level 3 preferred term",
    ]

    for _, row in hier_df.iterrows():
        uris = [_safe_str(row.get(c)) for c in level_uri_cols]
        terms = [_safe_str(row.get(c)) for c in level_term_cols]

        for u, t in zip(uris, terms):
            if u:
                _add_node_if_missing(G, u, node_type="skill", label=t)

        for parent_u, child_u in zip(uris, uris[1:]):
            if parent_u and child_u and parent_u != child_u:
                G.add_edge(
                    parent_u, child_u,
                    key=f"skill_parent_of::{parent_u}->{child_u}",
                    edge_type="skill_parent_of",
                )

    # ---- Occupation -> Skill edges ----
    for _, row in occ_skill_df.iterrows():
        occ_uri = _safe_str(row.get("occupationUri"))
        skill_uri = _safe_str(row.get("skillUri"))
        if not occ_uri or not skill_uri:
            continue

        _add_node_if_missing(G, occ_uri, node_type="occupation")
        _add_node_if_missing(G, skill_uri, node_type="skill")

        rel_type = _safe_str(row.get("relationType"))
        skill_type = _safe_str(row.get("skillType"))

        G.add_edge(
            occ_uri, skill_uri,
            key=f"has_skill::{rel_type or 'unknown'}::{skill_type or 'unknown'}",
            edge_type="has_skill",
            relation_type=rel_type,
            skill_type=skill_type,
        )

    return G


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_graph(G: nx.MultiDiGraph, out_path: str) -> None:
    """Export graph to .graphml or .gexf for inspection (e.g. Gephi).

    GEXF/GraphML don't support list-valued or None attributes, so we
    normalize everything to strings before writing.
    """
    H = G.copy()
    for _, data in H.nodes(data=True):
        for k in list(data.keys()):
            v = data[k]
            if isinstance(v, list):
                data[k] = " | ".join(str(i) for i in v)
            elif v is None:
                data[k] = ""
    for _, _, data in H.edges(data=True):
        for k in list(data.keys()):
            v = data[k]
            if v is None:
                data[k] = ""

    ext = Path(out_path).suffix.lower()
    if ext == ".graphml":
        nx.write_graphml(H, out_path)
    elif ext == ".gexf":
        nx.write_gexf(H, out_path)
    else:
        raise ValueError(f"Unsupported export format '{ext}'. Use .graphml or .gexf")
    print(f"Graph exported to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ESCO knowledge graph")
    parser.add_argument(
        "--data-dir", default="dataset/esco",
        help="Directory containing the ESCO CSV files (default: dataset/esco)",
    )
    parser.add_argument(
        "--export", default=None,
        help="Optional output path (.graphml or .gexf) to export the graph",
    )
    args = parser.parse_args()

    G = build_esco_graph(data_dir=args.data_dir)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Quick breakdown by node type
    from collections import Counter
    type_counts = Counter(d.get("node_type", "unknown") for _, d in G.nodes(data=True))
    for ntype, count in type_counts.most_common():
        print(f"  {ntype}: {count}")

    if args.export:
        export_graph(G, args.export)

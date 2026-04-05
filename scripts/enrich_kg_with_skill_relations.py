#!/usr/bin/env python3
"""
Enrich the ESCO KG graph by adding direct skill-skill relations.

The existing KG only connects skills through occupation nodes (2+ hops).
skillSkillRelations.csv provides 5,899 direct skill-to-skill edges that
create 1-hop paths between functionally related skills.

Usage:
    python3 scripts/enrich_kg_with_skill_relations.py \
        --input dataset/esco/esco_kg.gexf \
        --skill-relations dataset/esco/skillSkillRelations.csv \
        --output dataset/esco/esco_kg_enriched.gexf
"""
import argparse, csv, logging
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/esco/esco_kg.gexf")
    parser.add_argument("--skill-relations", default="dataset/esco/skillSkillRelations.csv")
    parser.add_argument("--output", default="dataset/esco/esco_kg_enriched.gexf")
    args = parser.parse_args()

    logger.info(f"Loading KG from {args.input}")
    G = nx.read_gexf(args.input)
    orig_nodes, orig_edges = G.number_of_nodes(), G.number_of_edges()
    logger.info(f"Original: {orig_nodes} nodes, {orig_edges} edges")

    # Add skill-skill relations
    added, skipped_missing = 0, 0
    with open(args.skill_relations) as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["originalSkillUri"]
            v = row["relatedSkillUri"]
            rel = row.get("relationType", "related")
            if u in G and v in G:
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                    added += 1
            else:
                skipped_missing += 1

    logger.info(f"Added {added} skill-skill edges, skipped {skipped_missing} (nodes not in graph)")
    logger.info(f"Enriched: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (+{G.number_of_edges()-orig_edges})")

    logger.info(f"Saving to {args.output}")
    # Use gpickle to avoid GEXF attribute type issues
    if args.output.endswith('.gpickle'):
        import pickle
        with open(args.output, 'wb') as f:
            pickle.dump(G, f)
    else:
        nx.write_gexf(G, args.output)
    logger.info("Done")

if __name__ == "__main__":
    main()

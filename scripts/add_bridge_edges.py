#!/usr/bin/env python3
"""
Add bridge edges to connect graph components.
Uses skill-based and category-based approaches.
"""

import networkx as nx
import json
from collections import defaultdict
from difflib import SequenceMatcher

def load_esco_skills():
    """Load ESCO skills for skill-based matching."""
    # This would load from your ESCO skills file
    # For now, return empty dict (will use title-based matching)
    return {}

def get_job_category(title):
    """Categorize job by keywords."""
    title = title.lower()
    categories = []
    
    if any(kw in title for kw in ['software', 'developer', 'programmer']):
        categories.append('software_dev')
    if any(kw in title for kw in ['data', 'analyst', 'scientist']):
        categories.append('data')
    if any(kw in title for kw in ['engineer']):
        categories.append('engineering')
    if any(kw in title for kw in ['manager', 'director', 'supervisor']):
        categories.append('management')
    if any(kw in title for kw in ['designer', 'ux', 'ui']):
        categories.append('design')
    
    return categories if categories else ['other']

def title_similarity(title1, title2):
    """Compute similarity between two job titles."""
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()

def add_category_bridges(G, components, main_comp_id=0):
    """Add bridges between components based on job categories."""
    
    print("Adding category-based bridges...")
    
    # Group nodes by category and component
    category_nodes = defaultdict(lambda: defaultdict(list))
    
    for comp_id, comp in enumerate(components):
        for node in comp:
            title = G.nodes[node].get('title', '')
            categories = get_job_category(title)
            for cat in categories:
                category_nodes[cat][comp_id].append(node)
    
    bridges_added = 0
    
    # For each category, connect smaller components to main component
    for category, comp_nodes in category_nodes.items():
        if category == 'other':
            continue
        
        if main_comp_id not in comp_nodes:
            continue
        
        main_nodes = comp_nodes[main_comp_id]
        
        # Connect other components to main
        for comp_id, nodes in comp_nodes.items():
            if comp_id == main_comp_id:
                continue
            
            # For each node in small component
            for node in nodes:
                node_title = G.nodes[node].get('title', '')
                
                # Find best match in main component
                best_match = None
                best_similarity = 0
                
                for main_node in main_nodes:
                    main_title = G.nodes[main_node].get('title', '')
                    sim = title_similarity(node_title, main_title)
                    
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = main_node
                
                # Add edge if similarity > 0.3
                if best_match and best_similarity > 0.3:
                    # Calculate edge weight based on similarity
                    # Higher similarity = shorter distance
                    weight = int(5 * (1 - best_similarity))  # 0-3 range
                    weight = max(1, weight)  # At least 1
                    
                    G.add_edge(node, best_match, weight=weight, bridge_type='category')
                    G.add_edge(best_match, node, weight=weight, bridge_type='category')
                    bridges_added += 1
                    
                    if bridges_added <= 5:  # Show first 5
                        print(f"  Bridge: {node_title} ↔ {main_title} (weight={weight}, sim={best_similarity:.2f})")
    
    print(f"Added {bridges_added} category-based bridges")
    return bridges_added

def add_generic_bridges(G, components, main_comp_id=0):
    """Add generic bridges for remaining isolated components."""
    
    print("\nAdding generic bridges for isolated components...")
    
    main_comp = components[main_comp_id]
    bridges_added = 0
    
    # Get a representative node from main component (high degree)
    main_degrees = [(node, G.degree(node)) for node in main_comp]
    main_degrees.sort(key=lambda x: x[1], reverse=True)
    hub_node = main_degrees[0][0]
    hub_title = G.nodes[hub_node].get('title', '')
    
    print(f"  Using hub node: {hub_title} (degree={main_degrees[0][1]})")
    
    # Connect each small component to hub
    for comp_id, comp in enumerate(components):
        if comp_id == main_comp_id or len(comp) > 100:
            continue
        
        # Find highest degree node in this component
        comp_degrees = [(node, G.degree(node)) for node in comp]
        comp_degrees.sort(key=lambda x: x[1], reverse=True)
        comp_hub = comp_degrees[0][0]
        comp_title = G.nodes[comp_hub].get('title', '')
        
        # Check if already connected
        if nx.has_path(G.to_undirected(), hub_node, comp_hub):
            continue
        
        # Add bridge with high weight (distant connection)
        G.add_edge(hub_node, comp_hub, weight=5, bridge_type='generic')
        G.add_edge(comp_hub, hub_node, weight=5, bridge_type='generic')
        bridges_added += 1
        
        if bridges_added <= 5:
            print(f"  Bridge: {hub_title} ↔ {comp_title} (comp {comp_id}, size={len(comp)})")
    
    print(f"Added {bridges_added} generic bridges")
    return bridges_added

def main():
    print("=" * 80)
    print("ADDING BRIDGE EDGES TO CAREER GRAPH")
    print("=" * 80)
    
    # Load graph
    print("\nLoading graph...")
    G = nx.read_gexf('/Users/olukotunjosh/Downloads/CDCL/training_output/career_graph.gexf')
    
    print(f"Original graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Get components
    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)
    print(f"  Components: {len(components)}")
    print(f"  Largest component: {len(components[0])} nodes")
    
    # Add bridges
    print("\n" + "=" * 80)
    category_bridges = add_category_bridges(G, components, main_comp_id=0)
    generic_bridges = add_generic_bridges(G, components, main_comp_id=0)
    
    # Check new connectivity
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    new_components = list(nx.weakly_connected_components(G))
    new_components.sort(key=len, reverse=True)
    
    print(f"\nNew graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()} (+{G.number_of_edges() - 16510})")
    print(f"  Components: {len(new_components)} (was {len(components)})")
    print(f"  Largest component: {len(new_components[0])} nodes (was {len(components[0])})")
    
    # Calculate improvement
    original_coverage = len(components[0]) / G.number_of_nodes() * 100
    new_coverage = len(new_components[0]) / G.number_of_nodes() * 100
    
    print(f"\nConnectivity improvement:")
    print(f"  Original: {original_coverage:.1f}% in main component")
    print(f"  New: {new_coverage:.1f}% in main component")
    print(f"  Improvement: +{new_coverage - original_coverage:.1f}%")
    
    # Save new graph
    output_path = '/Users/olukotunjosh/Downloads/CDCL/training_output/career_graph_bridged.gexf'
    nx.write_gexf(G, output_path)
    print(f"\nSaved bridged graph to: {output_path}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

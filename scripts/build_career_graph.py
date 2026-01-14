#!/usr/bin/env python3
"""
Offline script to build career graph from ESCO data.
Run this BEFORE training to generate the career graph file.
"""

import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'contrastive_learning'))

from contrastive_learning.career_graph_builder import CareerGraphBuilder

def build_career_graph():
    """Build and save career graph from ESCO data."""
    print("Building career graph from ESCO data...")
    
    occupations_path = "dataset/esco/occupations_en.csv"
    skills_path = "dataset/esco/skills_en.csv"
    relations_path = "dataset/esco/occupationSkillRelations_en.csv"
    output_path = "training_output/career_graph.gexf"
    output_dir = "training_output"
    
    # Initialize builder
    builder = CareerGraphBuilder()
    
    # Load ESCO data
    builder.load_esco_data(
        occupations_path=occupations_path,
        skills_path=skills_path,
        relations_path=relations_path
    )
    
    # Build the career graph
    builder.build_career_graph()
    
    # Prune the graph
    builder.prune_graph()
    
    # Save the graph for training
    os.makedirs(output_dir, exist_ok=True)
    builder.save_graph(output_path)
    
    # Print statistics
    stats = builder.get_graph_statistics()
    print("\nCareer Graph Built Successfully!")
    print(f"Nodes: {stats['nodes']}")
    print(f"Edges: {stats['edges']}")
    print(f"Saved to: training_output/career_graph.gexf")
    
    return True

if __name__ == "__main__":
    success = build_career_graph()
    if success:
        print("\n✅ Career graph ready for training!")
    else:
        print("\n❌ Failed to build career graph!")
        sys.exit(1)

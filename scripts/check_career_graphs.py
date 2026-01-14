#!/usr/bin/env python3
"""Check which career graph files exist and their statistics"""

import xml.etree.ElementTree as ET
import os

files = [
    'training_output/career_graph_data_driven.gexf',
    'training_output/career_graph.gexf',
    'training_output/career_graph_bridged.gexf'
]

print("=" * 70)
print("Career Graph File Analysis")
print("=" * 70)

for filepath in files:
    if os.path.exists(filepath):
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            ns = {'gexf': 'http://www.gexf.net/1.2draft'}
            nodes = root.findall('.//gexf:node', ns)
            edges = root.findall('.//gexf:edge', ns)

            print(f'\n📊 {filepath}:')
            print(f'   ✓ Exists')
            print(f'   Nodes (occupations): {len(nodes):,}')
            print(f'   Edges (career transitions): {len(edges):,}')

            # Show a few sample occupations
            sample_titles = []
            for node in nodes[:5]:
                attvalues = node.findall('.//gexf:attvalue', ns)
                for attvalue in attvalues:
                    if attvalue.get('for') == '0':
                        sample_titles.append(attvalue.get('value'))
                        break

            if sample_titles:
                print(
                    f'   Sample occupations: {", ".join(sample_titles[:3])}...')

        except Exception as e:
            print(f'   ✗ Error reading file: {e}')
    else:
        print(f'\n❌ {filepath}: File not found')

print("\n" + "=" * 70)
print("Note: The augmentation script uses the FIRST file found in this order:")
print("  1. training_output/career_graph_data_driven.gexf")
print("  2. training_output/career_graph.gexf")
print("  3. colab_package/training_output/career_graph.gexf")
print("=" * 70)

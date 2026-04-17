#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Assignment Joern Graph Visualization Script

This script runs Joern on a simple assignment example and visualizes the graph.
"""

import os
import pandas as pd
import sastvd.helpers.joern as svdj
import sastvd as svd
from graphviz import Digraph

def get_simple_digraph(nodes, edges):
    """Create a simple digraph with clear visualization."""
    dot = Digraph(comment="Simple Assignment CPG")
    
    # Create node map with clear labels
    node_map = {}
    for _, node in nodes.iterrows():
        if node['lineNumber'] and pd.notna(node['lineNumber']):
            # Create clear label with line number and type
            label = f"Line {int(node['lineNumber'])}"
            if 'name' in node and pd.notna(node['name']):
                label += f": {node['name']}"
            elif 'node_label' in node and pd.notna(node['node_label']):
                # Extract just the type from node_label
                label_parts = str(node['node_label']).split(':')
                if len(label_parts) > 0:
                    label += f": {label_parts[0].strip()}"
            node_map[node['id']] = label
            dot.node(str(node['id']), label)
    
    # Add edges with clear labels
    for _, edge in edges.iterrows():
        if edge['innode'] in node_map and edge['outnode'] in node_map:
            # Show all edge types for this simple example
            dot.edge(str(edge['outnode']), str(edge['innode']), edge['etype'])
    
    return dot

def main():
    """Main function to run Joern and visualize the graph."""
    # Simple assignment example code
    example_code = """
void foo() {
    int a = 5;
    int b = 3;
    int c = a + b;
}
"""
    
    print("Running Joern on simple assignment example...")
    # Run Joern on the hardcoded code
    result = svdj.full_run_joern_from_string(example_code, "test", "test")
    
    if not result:
        print("Error: Joern analysis failed.")
        return
    
    print("Joern analysis completed successfully.")
    
    # Get node and edge data
    nodes, edges = result["nodes"], result["edges"]
    print(f"Original: {len(nodes)} nodes and {len(edges)} edges.")
    
    # Create simple graph
    print("Creating simple graph...")
    dot = get_simple_digraph(nodes, edges)
    
    # Save to file instead of opening automatically
    output_path = "./assignment_joern_graph"
    dot.render(output_path, format="pdf", cleanup=True)
    
    print(f"Assignment graph saved to: {output_path}.pdf")
    print("You can open this file manually to view the graph.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Joern Graph Visualization Script

This script runs Joern on a simple example and creates a highly simplified graph.
"""

import os
import pandas as pd
import sastvd.helpers.joern as svdj
import sastvd as svd
from graphviz import Digraph

def create_minimal_graph(code, output_file):
    """Create a minimal graph with only essential code elements."""
    # Run Joern
    result = svdj.full_run_joern_from_string(code, "test", "test")
    if not result:
        print("Error: Joern analysis failed.")
        return False
    
    nodes, edges = result["nodes"], result["edges"]
    print(f"Original: {len(nodes)} nodes, {len(edges)} edges")
    
    # Create a very minimal graph
    dot = Digraph(comment="Minimal CPG", graph_attr={'rankdir': 'TB'})
    
    # Only keep nodes with line numbers and meaningful content
    kept_nodes = {}
    line_node_map = {}
    
    for _, node in nodes.iterrows():
        if node['lineNumber'] and pd.notna(node['lineNumber']):
            line_num = int(node['lineNumber'])
            # Create simple label with line number
            label = f"Line {line_num}"
            
            # Add variable/function names if available
            if 'name' in node and pd.notna(node['name']) and node['name']:
                label += f": {node['name']}"
            
            # Add node to graph
            node_id = str(node['id'])
            dot.node(node_id, label)
            kept_nodes[node_id] = line_num
            
            # Map line numbers to node IDs for grouping
            if line_num not in line_node_map:
                line_node_map[line_num] = []
            line_node_map[line_num].append(node_id)
    
    # Only keep edges between kept nodes and limit edge types
    kept_edges = []
    edge_types = {'CFG', 'AST', 'DDG'}  # Only keep important edge types
    
    for _, edge in edges.iterrows():
        src_id = str(edge['outnode'])
        dst_id = str(edge['innode'])
        
        if src_id in kept_nodes and dst_id in kept_nodes:
            etype = edge['etype']
            if etype in edge_types:
                dot.edge(src_id, dst_id, etype)
                kept_edges.append(edge)
    
    print(f"Minimal: {len(kept_nodes)} nodes, {len(kept_edges)} edges")
    
    # Save the graph
    dot.render(output_file, format="pdf", cleanup=True)
    print(f"Minimal graph saved to: {output_file}.pdf")
    return True

def main():
    """Main function."""
    # Simple example code
    example_code = """
void foo() {
    int a = 5;
    int b = 3;
    int c = a + b;
}
"""
    
    print("Creating minimal Joern graph...")
    create_minimal_graph(example_code, "./minimal_assignment_graph")
    print("Done!")

if __name__ == "__main__":
    main()
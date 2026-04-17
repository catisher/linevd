#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joern Graph Visualization Script with Hardcoded Example

This script runs Joern on a hardcoded example and visualizes the resulting code property graph.
"""

import os
import sastvd.helpers.joern as svdj
import sastvd as svd

def main():
    """Main function to run Joern and visualize the graph."""
    # Hardcoded example code
    example_code = """
short add (short b) {
    short a = 32767;
    if (b > 0) {
        a = a + b;
    }
    return a;
}
"""
    
    print("Running Joern on hardcoded example...")
    # Run Joern on the hardcoded code
    result = svdj.full_run_joern_from_string(example_code, "test", "test")
    
    if not result:
        print("Error: Joern analysis failed.")
        return
    
    print("Joern analysis completed successfully.")
    
    # Get node and edge data
    nodes, edges = result["nodes"], result["edges"]
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges.")
    
    # Visualize the graph
    print("Visualizing the graph...")
    
    # Get the digraph object
    dot = svdj.get_digraph(
        nodes[["id", "node_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=True
    )
    
    # Save to file instead of opening automatically
    output_path = "./joern_graph"
    dot.render(output_path, format="pdf", cleanup=True)
    
    print(f"Graph saved to: {output_path}.pdf")
    print("You can open this file manually to view the graph.")

if __name__ == "__main__":
    main()
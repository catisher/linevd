#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joern Graph Visualization Script

This script runs Joern to parse code files and visualizes the resulting code property graph.
"""

import argparse
import os
import sastvd.helpers.joern as svdj
import sastvd as svd

def main():
    """Main function to run Joern and visualize the graph."""
    parser = argparse.ArgumentParser(description="Visualize Joern-parsed code property graphs")
    parser.add_argument("code_file", help="Path to the code file to analyze")
    parser.add_argument("--hop", type=int, default=1, help="Number of hops for subgraph visualization")
    parser.add_argument("--line", type=int, default=-1, help="Line number to focus on (default: entire graph)")
    parser.add_argument("--verbose", type=int, default=3, help="Verbosity level")
    args = parser.parse_args()
    
    # Check if code file exists
    if not os.path.exists(args.code_file):
        print(f"Error: File {args.code_file} does not exist.")
        return
    
    # Run Joern if not already run
    edges_file = args.code_file + ".edges.json"
    nodes_file = args.code_file + ".nodes.json"
    
    if not (os.path.exists(edges_file) and os.path.exists(nodes_file)):
        print(f"Running Joern on {args.code_file}...")
        result = svdj.full_run_joern(args.code_file, verbose=args.verbose)
        if not result:
            print("Error: Joern analysis failed.")
            return
        print("Joern analysis completed successfully.")
    else:
        print("Using existing Joern output files.")
    
    # Get node and edge data
    print("Loading node and edge data...")
    nodes, edges = svdj.get_node_edges(args.code_file)
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges.")
    
    # Visualize the graph
    print("Visualizing the graph...")
    if args.line > 0:
        # Visualize subgraph around specified line
        print(f"Visualizing subgraph around line {args.line} with {args.hop} hops.")
        svdj.plot_graph_node_edge_df(
            nodes, 
            edges, 
            nodeids=[args.line], 
            hop=args.hop, 
            edge_label=True
        )
    else:
        # Visualize entire graph
        print("Visualizing entire graph.")
        svdj.plot_graph_node_edge_df(nodes, edges, edge_label=True)
    
    print("Graph visualization completed. Check /tmp/tmp.gv for the output.")

if __name__ == "__main__":
    main()
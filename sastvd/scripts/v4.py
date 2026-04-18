#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Code Structure Visualization Script

This script creates a manually simplified graph of code structure.
"""

import os
from graphviz import Digraph

def create_manual_graph():
    """Create a manually simplified graph of code structure."""
    # Create a new digraph with top-to-bottom layout
    dot = Digraph(comment="Simplified Code Structure", graph_attr={'rankdir': 'TB'})
    
    # Create subgraph for main code structure
    with dot.subgraph(name='cluster_code') as c:
        c.attr(label='Code Structure', style='filled', fillcolor='lightyellow')
        # Add nodes for each line of code
        c.node('line1', 'void foo() {', shape='box', style='filled', fillcolor='lightblue')
        c.node('line2', 'int a = 5;', shape='box', style='filled', fillcolor='lightgreen')
        c.node('line3', 'int b = 3;', shape='box', style='filled', fillcolor='lightgreen')
        c.node('line4', 'int c = a + b;', shape='box', style='filled', fillcolor='lightgreen')
        c.node('line5', '}', shape='box', style='filled', fillcolor='lightblue')
        
        # Add edges to show control flow
        c.edge('line1', 'line2', 'CFG')
        c.edge('line2', 'line3', 'CFG')
        c.edge('line3', 'line4', 'CFG')
        c.edge('line4', 'line5', 'CFG')
        
        # Add data flow edges
        c.edge('line2', 'line4', 'DDG: a', style='dashed', color='green')
        c.edge('line3', 'line4', 'DDG: b', style='dashed', color='green')
    
    # Create subgraph for legend (on the right)
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='filled', fillcolor='lightgrey', rank='same')
        # Create legend nodes
        c.node('legend_title', 'Legend', shape='box', style='filled', fillcolor='lightgrey')
        c.node('legend_cfg', 'Control Flow', shape='box')
        c.node('legend_ddg', 'Data Flow', shape='box')
        c.node('legend_func', 'Function Def', shape='box', style='filled', fillcolor='lightblue')
        c.node('legend_var', 'Variable Def', shape='box', style='filled', fillcolor='lightgreen')
        
        # Connect legend nodes vertically
        c.edge('legend_title', 'legend_cfg', style='invis')
        c.edge('legend_cfg', 'legend_ddg', style='invis')
        c.edge('legend_ddg', 'legend_func', style='invis')
        c.edge('legend_func', 'legend_var', style='invis')
        
        # Add example edges for legend with clear labels
        c.edge('legend_cfg', 'legend_ddg', style='solid', color='black', label='Black line: CFG', constraint='false')
        c.edge('legend_ddg', 'legend_func', style='dashed', color='green', label='Green line: DDG', constraint='false')
    
    # Add invisible edge to align subgraphs horizontally
    dot.edge('line1', 'legend_title', style='invis')
    
    # Save the graph
    output_file = "./manual_code_structure"
    dot.render(output_file, format="pdf", cleanup=True)
    print(f"Manual graph saved to: {output_file}.pdf")
    
    return True

def main():
    """Main function."""
    print("Creating manual simplified code structure graph...")
    create_manual_graph()
    print("Done! This graph only shows the essential code structure.")

if __name__ == "__main__":
    main()
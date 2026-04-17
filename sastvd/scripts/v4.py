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
    
    # Add nodes for each line of code
    dot.node('line1', 'void foo() {', shape='box', style='filled', fillcolor='lightblue')
    dot.node('line2', 'int a = 5;', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('line3', 'int b = 3;', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('line4', 'int c = a + b;', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('line5', '}', shape='box', style='filled', fillcolor='lightblue')
    
    # Add edges to show control flow
    dot.edge('line1', 'line2', 'CFG')
    dot.edge('line2', 'line3', 'CFG')
    dot.edge('line3', 'line4', 'CFG')
    dot.edge('line4', 'line5', 'CFG')
    
    # Add data flow edges
    dot.edge('line2', 'line4', 'DDG: a', style='dashed', color='green')
    dot.edge('line3', 'line4', 'DDG: b', style='dashed', color='green')
    
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
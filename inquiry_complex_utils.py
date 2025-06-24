import json
import os
from typing import Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class InquiryComplexAnalyzer:
    """
    Utility class for analyzing and visualizing inquiry complexes.
    """
    
    def __init__(self, inquiry_complex: Dict):
        """
        Initialize with an inquiry complex dictionary.
        
        Args:
            inquiry_complex: Dictionary representation of the inquiry complex
        """
        self.graph_data = inquiry_complex
        self.nx_graph = self._build_networkx_graph()
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph from the inquiry complex"""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in self.graph_data.items():
            G.add_node(node_id, **node_data)
        
        # Add edges based on parent_id relationships
        for node_id, node_data in self.graph_data.items():
            parent_id = node_data.get("parent_id")
            if parent_id and parent_id in self.graph_data:
                G.add_edge(parent_id, node_id)
        
        return G
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed statistics about the inquiry complex"""
        stats = {
            "total_nodes": len(self.graph_data),
            "total_edges": self.nx_graph.number_of_edges(),
            "node_types": defaultdict(int),
            "depth_distribution": defaultdict(int),
            "questions": [],
            "terminal_nodes": 0,
            "nonsense_nodes": 0,
            "max_depth": 0,
            "avg_depth": 0
        }
        
        depths = []
        for node_data in self.graph_data.values():
            node_type = node_data["node_type"]
            depth = node_data["depth"]
            
            stats["node_types"][node_type] += 1
            if depth >= 0:  # Exclude reason nodes with depth -1
                stats["depth_distribution"][depth] += 1
                depths.append(depth)
            
            if node_data.get("terminal", False):
                stats["terminal_nodes"] += 1
            if node_data.get("nonsense", False):
                stats["nonsense_nodes"] += 1
            if node_type == "question":
                stats["questions"].append({
                    "id": node_data.get("id", "unknown"),
                    "summary": node_data["summary"],
                    "content": node_data["content"]
                })
        
        if depths:
            stats["max_depth"] = max(depths)
            stats["avg_depth"] = sum(depths) / len(depths)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats["node_types"] = dict(stats["node_types"])
        stats["depth_distribution"] = dict(stats["depth_distribution"])
        
        return stats
    
    def get_question_trees(self) -> Dict[str, Dict]:
        """Get separate trees for each central question"""
        question_trees = {}
        
        # Find all central questions
        for node_id, node_data in self.graph_data.items():
            if node_data.get("is_central_question", False):
                question_trees[node_id] = self._extract_subtree(node_id)
        
        return question_trees
    
    def _extract_subtree(self, root_id: str) -> Dict:
        """Extract a subtree rooted at the given node"""
        subtree = {}
        visited = set()
        
        def dfs(node_id):
            if node_id in visited or node_id not in self.graph_data:
                return
            
            visited.add(node_id)
            subtree[node_id] = self.graph_data[node_id].copy()
            
            # Find children
            for child_id in self.nx_graph.successors(node_id):
                dfs(child_id)
        
        dfs(root_id)
        return subtree
    
    def get_dialectical_paths(self, question_id: str) -> List[List[str]]:
        """Get all dialectical paths from a question to terminal nodes"""
        paths = []
        
        def dfs_paths(node_id, current_path):
            current_path = current_path + [node_id]
            
            # Check if this is a terminal node or has no children
            children = list(self.nx_graph.successors(node_id))
            if not children or self.graph_data[node_id].get("terminal", False):
                paths.append(current_path)
                return
            
            # Continue to children
            for child_id in children:
                dfs_paths(child_id, current_path)
        
        dfs_paths(question_id, [])
        return paths
    
    def visualize_graph(self, output_file: str = "inquiry_complex.png", 
                       figsize: tuple = (15, 10), node_size: int = 1000):
        """Create a visualization of the inquiry complex"""
        plt.figure(figsize=figsize)
        
        # Define colors for different node types
        color_map = {
            "question": "#FF6B6B",      # Red
            "thesis": "#4ECDC4",        # Teal
            "reason": "#45B7D1",        # Blue
            "antithesis": "#FFA07A",    # Orange
            "synthesis": "#98D8C8",     # Light Green
            "direct_reply": "#DDA0DD"   # Plum
        }
        
        # Set node colors based on type
        node_colors = []
        for node_id in self.nx_graph.nodes():
            node_type = self.graph_data[node_id]["node_type"]
            node_colors.append(color_map.get(node_type, "#CCCCCC"))
        
        # Use hierarchical layout
        pos = self._hierarchical_layout()
        
        # Draw the graph
        nx.draw(self.nx_graph, pos,
                node_color=node_colors,
                node_size=node_size,
                with_labels=False,
                arrows=True,
                arrowsize=20,
                edge_color="gray",
                alpha=0.7)
        
        # Add labels for question nodes only (to avoid clutter)
        question_nodes = {node_id: self.graph_data[node_id]["summary"][:30] + "..."
                         for node_id in self.nx_graph.nodes()
                         if self.graph_data[node_id]["node_type"] == "question"}
        
        if question_nodes:
            nx.draw_networkx_labels(self.nx_graph, pos, 
                                  labels=question_nodes, 
                                  font_size=8)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=node_type.title())
                          for node_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title("Inquiry Complex Visualization", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {output_file}")
    
    def _hierarchical_layout(self) -> Dict:
        """Create a hierarchical layout based on node depths"""
        pos = {}
        
        # Group nodes by depth
        depth_groups = defaultdict(list)
        for node_id, node_data in self.graph_data.items():
            depth = node_data["depth"]
            if depth >= 0:  # Exclude reason nodes
                depth_groups[depth].append(node_id)
        
        # Position nodes
        for depth, nodes in depth_groups.items():
            y = -depth * 2  # Vertical spacing
            x_spacing = 3 if len(nodes) > 1 else 0
            x_start = -(len(nodes) - 1) * x_spacing / 2
            
            for i, node_id in enumerate(nodes):
                x = x_start + i * x_spacing
                pos[node_id] = (x, y)
        
        # Position reason nodes near their parent thesis
        for node_id, node_data in self.graph_data.items():
            if node_data["depth"] == -1:  # Reason node
                parent_id = node_data["parent_id"]
                if parent_id in pos:
                    parent_x, parent_y = pos[parent_id]
                    # Place reason nodes to the right of their parent
                    pos[node_id] = (parent_x + 1, parent_y + 0.5)
        
        return pos
    
    def export_summary_report(self, output_file: str = "inquiry_complex_report.txt"):
        """Export a human-readable summary report"""
        stats = self.get_detailed_stats()
        
        with open(output_file, 'w') as f:
            f.write("=== INQUIRY COMPLEX ANALYSIS REPORT ===\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Total Nodes: {stats['total_nodes']}\n")
            f.write(f"Total Edges: {stats['total_edges']}\n")
            f.write(f"Max Depth: {stats['max_depth']}\n")
            f.write(f"Average Depth: {stats['avg_depth']:.2f}\n")
            f.write(f"Terminal Nodes: {stats['terminal_nodes']}\n")
            f.write(f"Nonsense Nodes: {stats['nonsense_nodes']}\n\n")
            
            # Node type distribution
            f.write("NODE TYPE DISTRIBUTION:\n")
            for node_type, count in stats['node_types'].items():
                f.write(f"  {node_type.title()}: {count}\n")
            f.write("\n")
            
            # Depth distribution
            f.write("DEPTH DISTRIBUTION:\n")
            for depth, count in sorted(stats['depth_distribution'].items()):
                f.write(f"  Depth {depth}: {count} nodes\n")
            f.write("\n")
            
            # Central questions
            f.write("CENTRAL QUESTIONS:\n")
            for i, question in enumerate(stats['questions'], 1):
                f.write(f"{i}. {question['summary']}\n")
                f.write(f"   Content: {question['content'][:200]}...\n\n")
            
            # Dialectical paths for each question
            question_trees = self.get_question_trees()
            for question_id, subtree in question_trees.items():
                question_data = self.graph_data[question_id]
                f.write(f"DIALECTICAL STRUCTURE FOR: {question_data['summary']}\n")
                f.write("-" * 50 + "\n")
                
                paths = self.get_dialectical_paths(question_id)
                f.write(f"Number of dialectical paths: {len(paths)}\n")
                
                for i, path in enumerate(paths[:5], 1):  # Show first 5 paths
                    f.write(f"\nPath {i}:\n")
                    for j, node_id in enumerate(path):
                        node_data = self.graph_data[node_id]
                        indent = "  " * j
                        f.write(f"{indent}- {node_data['node_type'].title()}: {node_data['summary'][:50]}...\n")
                
                if len(paths) > 5:
                    f.write(f"\n... and {len(paths) - 5} more paths\n")
                f.write("\n" + "=" * 70 + "\n\n")
        
        print(f"Summary report exported to {output_file}")


def load_inquiry_complex(filepath: str) -> Dict:
    """Load an inquiry complex from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_inquiry_complexes(complex1: Dict, complex2: Dict) -> Dict:
    """Compare two inquiry complexes and return comparison statistics"""
    def get_basic_stats(complex_data):
        stats = {
            "total_nodes": len(complex_data),
            "node_types": defaultdict(int)
        }
        for node_data in complex_data.values():
            stats["node_types"][node_data["node_type"]] += 1
        return stats
    
    stats1 = get_basic_stats(complex1)
    stats2 = get_basic_stats(complex2)
    
    comparison = {
        "complex1": dict(stats1),
        "complex2": dict(stats2),
        "differences": {
            "node_count_diff": stats1["total_nodes"] - stats2["total_nodes"],
            "type_differences": {}
        }
    }
    
    # Compare node type counts
    all_types = set(stats1["node_types"].keys()) | set(stats2["node_types"].keys())
    for node_type in all_types:
        count1 = stats1["node_types"].get(node_type, 0)
        count2 = stats2["node_types"].get(node_type, 0)
        comparison["differences"]["type_differences"][node_type] = count1 - count2
    
    return comparison


# Example usage and testing
if __name__ == "__main__":
    # Load inquiry complex from temp/inquiry_complex.json
    sample_complex = load_inquiry_complex("temp/inquiry_complex.json")
    
    # Analyze the inquiry complex
    analyzer = InquiryComplexAnalyzer(sample_complex)
    stats = analyzer.get_detailed_stats()
    analyzer.visualize_graph(output_file="temp/inquiry_complex_visualization.png")
    print("Analysis completed!")
    print(f"Statistics: {stats}")
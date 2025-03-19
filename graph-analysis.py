import json
from collections import defaultdict
from typing import Dict, Any

def analyze_graph_structure(filepath: str = 'graph.json') -> Dict[str, Any]:
    """
    Analyze the structure of the graph and return various metrics.
    
    Args:
        filepath: Path to the JSON file containing the graph data
        
    Returns:
        Dictionary containing various graph metrics
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Initialize tracking variables
        node_type_count = defaultdict(int)
        terminal_node_count = defaultdict(int)
        terminal_type_count = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for terminal types
        max_depth = 0
        total_depth = 0
        node_count = 0
        
        # Analyze the graph
        for node in graph_data.values():
            # Count node types
            node_type = node['node_type']
            node_type_count[node_type] += 1
            
            # Count terminal nodes by type and termination reason
            if node['terminal']:
                terminal_node_count[node_type] += 1
                # Extract termination type from the summary or content
                # Common termination indicators might be in the summary
                summary_lower = node['summary'].lower()
                if 'nonsense' in summary_lower or node['nonsense']:
                    term_type = 'nonsense'
                elif 'identical' in summary_lower or node['identical_to']:
                    term_type = 'identical'
                elif 'agreed' in summary_lower or 'agreement' in summary_lower:
                    term_type = 'agreement'
                elif 'disagree' in summary_lower or 'disagreement' in summary_lower:
                    term_type = 'disagreement'
                else:
                    term_type = 'other'
                
                terminal_type_count[node_type][term_type] += 1
            
            # Track max depth and calculate average
            depth = node['depth']
            max_depth = max(max_depth, depth)
            total_depth += depth
            node_count += 1
        
        # Calculate average depth
        avg_depth = total_depth / node_count if node_count > 0 else 0
        
        # Compile results
        results = {
            'node_type_distribution': dict(node_type_count),
            'terminal_nodes': dict(terminal_node_count),
            'terminal_types': {k: dict(v) for k, v in terminal_type_count.items()},
            'max_depth': max_depth,
            'average_depth': round(avg_depth, 2),
            'total_nodes': node_count
        }
        
        # Print results
        print("\nGraph Structure Analysis:")
        print("------------------------")
        print(f"Node type distribution: {results['node_type_distribution']}")
        print(f"\nTerminal nodes by type: {results['terminal_nodes']}")
        print("\nTerminal nodes by type and termination reason:")
        for node_type, term_types in results['terminal_types'].items():
            print(f"\n{node_type}:")
            for term_type, count in term_types.items():
                print(f"  - {term_type}: {count}")
        print(f"\nMaximum depth: {results['max_depth']}")
        print(f"Average depth: {results['average_depth']}")
        print(f"Total nodes: {results['total_nodes']}")
        
        return results
    
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        raise
    except Exception as e:
        print(f"Error analyzing graph: {str(e)}")
        raise

if __name__ == '__main__':
    analyze_graph_structure()
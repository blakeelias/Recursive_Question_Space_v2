import json
from typing import List, Dict, Any

def find_nonsense_nodes(filepath: str = 'graph.json') -> List[Dict[str, Any]]:
    """
    Find all nodes marked as nonsense in the graph and generate a report.
    
    Args:
        filepath: Path to the JSON file containing the graph data
        
    Returns:
        List of dictionaries containing nonsense nodes with their metadata
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        nonsense_nodes = []
        
        # Find all nonsense nodes
        for node_id, node in graph_data.items():
            # Check both the nonsense flag and text indicators in summary
            if node['nonsense'] or 'nonsense' in node['summary'].lower():
                # Get parent info if available
                parent_info = None
                if node['parent_id'] and node['parent_id'] in graph_data:
                    parent = graph_data[node['parent_id']]
                    parent_info = {
                        'id': node['parent_id'],
                        'summary': parent['summary'],
                        'content': parent['content']
                    }
                
                nonsense_node = {
                    'id': node_id,
                    'summary': node['summary'],
                    'content': node['content'],
                    'node_type': node['node_type'],
                    'depth': node['depth'],
                    'parent': parent_info,
                    'terminal': node['terminal']
                }
                nonsense_nodes.append(nonsense_node)
        
        # Sort nodes by depth for better organization
        nonsense_nodes.sort(key=lambda x: (x['depth'], x['node_type']))
        
        # Generate report
        report = ["Nonsense Nodes Analysis\n=====================\n"]
        report.append(f"Total nonsense nodes found: {len(nonsense_nodes)}\n")
        
        # Group by node type
        node_types = {}
        for node in nonsense_nodes:
            node_types[node['node_type']] = node_types.get(node['node_type'], 0) + 1
        
        report.append("\nNonsense nodes by type:")
        for ntype, count in node_types.items():
            report.append(f"{ntype}: {count}")
        
        report.append("\n\nDetailed Nonsense Nodes:\n")
        
        # Add detailed information for each nonsense node
        for idx, node in enumerate(nonsense_nodes, 1):
            report.append(f"\nNonsense Node {idx}:")
            report.append("-" * 50)
            report.append(f"ID: {node['id']}")
            report.append(f"Type: {node['node_type']}")
            report.append(f"Depth: {node['depth']}")
            report.append(f"Terminal: {node['terminal']}")
            report.append(f"\nSummary: {node['summary']}")
            report.append(f"Content: {node['content']}")
            
            if node['parent']:
                report.append(f"\nParent Node:")
                report.append(f"Parent ID: {node['parent']['id']}")
                report.append(f"Parent Summary: {node['parent']['summary']}")
                report.append(f"Parent Content: {node['parent']['content']}")
            
            report.append("\n")
        
        # Save report to file
        report_text = '\n'.join(report)
        output_file = 'nonsense_nodes_report.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Found {len(nonsense_nodes)} nonsense nodes")
        print(f"Report saved to {output_file}")
        
        return nonsense_nodes
    
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        raise
    except Exception as e:
        print(f"Error finding nonsense nodes: {str(e)}")
        raise

if __name__ == '__main__':
    find_nonsense_nodes()
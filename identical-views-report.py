import json
from typing import List, Dict, Any

def find_identical_views(filepath: str = 'graph_analysis.json') -> List[Dict[str, Any]]:
    """
    Find all pairs of identical views in the graph and generate a report.
    
    Args:
        filepath: Path to the JSON file containing the graph data
        
    Returns:
        List of dictionaries containing pairs of identical views
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        identical_pairs = []
        
        # Find all pairs of identical views
        for node_id, node in graph_data.items():
            if node['identical_to']:
                matching_node = graph_data.get(node['identical_to'])
                if matching_node:
                    pair = {
                        'view1': {
                            'id': node_id,
                            'summary': node['summary'],
                            'content': node['content']
                        },
                        'view2': {
                            'id': node['identical_to'],
                            'summary': matching_node['summary'],
                            'content': matching_node['content']
                        }
                    }
                    identical_pairs.append(pair)
        
        # Generate report
        report = []
        for idx, pair in enumerate(identical_pairs, 1):
            report.append(f"""
Identical View Pair {idx}:
--------------------------------
View 1 (ID: {pair['view1']['id']}):
Summary: {pair['view1']['summary']}
Content: {pair['view1']['content']}

View 2 (ID: {pair['view2']['id']}):
Summary: {pair['view2']['summary']}
Content: {pair['view2']['content']}
""")
        
        # Save report to file
        report_text = '\n'.join(report)
        with open('identical_views_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Found {len(identical_pairs)} pairs of identical views")
        print("Report saved to identical_views_report.txt")
        
        return identical_pairs
    
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        raise
    except Exception as e:
        print(f"Error finding identical views: {str(e)}")
        raise

if __name__ == '__main__':
    find_identical_views()
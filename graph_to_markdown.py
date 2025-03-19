import json
import os
import re

def slugify(text):
    """Convert text to a URL-friendly slug format."""
    # Remove special characters and replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug

def create_markdown_wiki(json_file_path, output_dir="wiki"):
    """
    Convert a JSON graph structure into a network of linked Markdown files.
    
    Args:
        json_file_path: Path to the JSON file containing the graph data
        output_dir: Directory where the Markdown files will be created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        graph_data = json.load(f)
    
    # First pass: Create a map of node IDs to their children
    children_map = {}
    for node_id, node_data in graph_data.items():
        parent_id = node_data.get('parent_id')
        if parent_id:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(node_id)
    
    # Second pass: Create Markdown files for each node
    for node_id, node_data in graph_data.items():
        # Extract node data
        summary = node_data.get('summary', 'No Summary')
        content = node_data.get('content', 'No Content')
        node_type = node_data.get('node_type', 'Unknown Type')
        parent_id = node_data.get('parent_id')
        depth = node_data.get('depth', 0)
        terminal = node_data.get('terminal', False)
        
        # Format content (replace curly braces with proper formatting)
        content = content.replace('{', '**').replace('}', '**')
        
        # Create filename based on summary or node ID if summary is missing
        filename = f"{slugify(summary)}.md" if summary else f"{node_id}.md"
        filepath = os.path.join(output_dir, filename)
        
        # Start building the Markdown content
        md_content = [
            f"# {summary}",
            "",
            f"**Node Type:** {node_type}",
            f"**Node ID:** {node_id}",
            f"**Depth:** {depth}",
            f"**Terminal:** {'Yes' if terminal else 'No'}",
            ""
        ]
        
        # Add parent link if it exists
        if parent_id and parent_id in graph_data:
            parent_summary = graph_data[parent_id].get('summary', 'Parent Node')
            parent_filename = f"{slugify(parent_summary)}.md" if parent_summary else f"{parent_id}.md"
            md_content.append(f"**Parent:** [{parent_summary}]({parent_filename})")
            md_content.append("")
        
        # Add content section
        md_content.append("## Content")
        md_content.append("")
        md_content.append(content)
        md_content.append("")
        
        # Add children links if they exist
        if node_id in children_map and children_map[node_id]:
            md_content.append("## Related Nodes")
            md_content.append("")
            for child_id in children_map[node_id]:
                if child_id in graph_data:
                    child_summary = graph_data[child_id].get('summary', 'Child Node')
                    child_filename = f"{slugify(child_summary)}.md" if child_summary else f"{child_id}.md"
                    child_type = graph_data[child_id].get('node_type', 'Unknown Type')
                    md_content.append(f"- [{child_summary}]({child_filename}) ({child_type})")
            md_content.append("")
        
        # Save the Markdown file
        with open(filepath, 'w') as f:
            f.write('\n'.join(md_content))
    
    # Create an index file that lists all nodes
    index_content = [
        "# Knowledge Graph Wiki",
        "",
        "## All Nodes",
        ""
    ]
    
    # Find the root nodes (nodes with no parents or central questions)
    root_nodes = []
    for node_id, node_data in graph_data.items():
        if node_data.get('parent_id') is None or node_data.get('is_central_question', False):
            root_nodes.append(node_id)
    
    # First add the root nodes
    if root_nodes:
        index_content.append("### Root Nodes")
        index_content.append("")
        for root_id in root_nodes:
            if root_id in graph_data:
                root_summary = graph_data[root_id].get('summary', 'Root Node')
                root_filename = f"{slugify(root_summary)}.md" if root_summary else f"{root_id}.md"
                root_type = graph_data[root_id].get('node_type', 'Unknown Type')
                index_content.append(f"- [{root_summary}]({root_filename}) ({root_type})")
        index_content.append("")
    
    # Add all other nodes by type
    node_types = {}
    for node_id, node_data in graph_data.items():
        node_type = node_data.get('node_type', 'Unknown Type')
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node_id)
    
    for node_type, type_nodes in node_types.items():
        if type_nodes:
            index_content.append(f"### {node_type.capitalize()} Nodes")
            index_content.append("")
            for node_id in type_nodes:
                if node_id in graph_data:
                    node_summary = graph_data[node_id].get('summary', 'Node')
                    node_filename = f"{slugify(node_summary)}.md" if node_summary else f"{node_id}.md"
                    index_content.append(f"- [{node_summary}]({node_filename})")
            index_content.append("")
    
    # Save the index file
    with open(os.path.join(output_dir, "index.md"), 'w') as f:
        f.write('\n'.join(index_content))
    
    print(f"Wiki created successfully in the '{output_dir}' directory!")
    print(f"Start browsing from 'index.md' or any of the root nodes.")

if __name__ == "__main__":
    # Replace with your actual JSON file path
    json_file_path = "graph_analysis.json"
    create_markdown_wiki(json_file_path)
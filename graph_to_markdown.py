import json
import os
import re

def slugify(text):
    """Convert text to a URL-friendly slug format."""
    # Remove special characters and replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug

def create_markdown_wiki(json_file_path, output_dir="Wiki/Storage/wiki"):
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
    
    # Helper function to create a unique filename for a node
    def get_unique_filename(node_id, node_data):
        summary = node_data.get('summary', '')
        node_type = node_data.get('node_type', 'unknown')
        
        # Use summary-nodetype-id format for filename to ensure uniqueness
        # but only use summary part for display
        if summary:
            return f"{slugify(summary)}-{slugify(node_type)}-{node_id}.md"
        else:
            return f"{node_id}.md"
    
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
        
        # Create unique filename for this node
        filename = get_unique_filename(node_id, node_data)
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
            parent_data = graph_data[parent_id]
            parent_summary = parent_data.get('summary', 'Parent Node')
            parent_filename = get_unique_filename(parent_id, parent_data)
            md_content.append(f"**Parent:** [{parent_summary}]({parent_filename})")
            md_content.append("")
        
        # Add content section
        md_content.append("## Content")
        md_content.append("")
        md_content.append(content)
        md_content.append("")
        
        # Add children links if they exist, grouped by node type
        if node_id in children_map and children_map[node_id]:
            md_content.append("## Related Nodes")
            md_content.append("")
            
            # Group child nodes by their type
            child_nodes_by_type = {}
            for child_id in children_map[node_id]:
                if child_id in graph_data:
                    child_data = graph_data[child_id]
                    child_type = child_data.get('node_type', 'Unknown Type')
                    
                    if child_type not in child_nodes_by_type:
                        child_nodes_by_type[child_type] = []
                    
                    child_nodes_by_type[child_type].append(child_id)
            
            # Display each group under its own heading
            for child_type, child_ids in sorted(child_nodes_by_type.items()):
                md_content.append(f"### {child_type.capitalize()} Nodes")
                md_content.append("")
                
                for child_id in child_ids:
                    child_data = graph_data[child_id]
                    child_summary = child_data.get('summary', 'Child Node')
                    child_filename = get_unique_filename(child_id, child_data)
                    md_content.append(f"- [{child_summary}]({child_filename})")
                
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
                root_data = graph_data[root_id]
                root_summary = root_data.get('summary', 'Root Node')
                root_type = root_data.get('node_type', 'Unknown Type')
                root_filename = get_unique_filename(root_id, root_data)
                index_content.append(f"- [{root_summary}]({root_filename}) ({root_type})")
        index_content.append("")
    
    # Create a list of all nodes with their metadata for sorting
    all_nodes = []
    for node_id, node_data in graph_data.items():
        all_nodes.append({
            'id': node_id,
            'summary': node_data.get('summary', 'Node'),
            'type': node_data.get('node_type', 'Unknown Type'),
            'depth': node_data.get('depth', 0),  # Use the depth value without absolute
            'filename': get_unique_filename(node_id, node_data)
        })
    
    # For index.md only: Sort nodes first by depth and then alphabetically by summary
    all_nodes.sort(key=lambda x: (x['depth'], x['summary'].lower()))
    
    # Group nodes by type for the "All Nodes" section
    node_types = {}
    for node in all_nodes:
        node_type = node['type']
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)
    
    # Add all other nodes by type
    for node_type, type_nodes in sorted(node_types.items()):
        if type_nodes:
            index_content.append(f"### {node_type.capitalize()} Nodes")
            index_content.append("")
            
            # For index.md: Sort nodes within each type already done by depth then alphabetically
            # (we're using the already sorted all_nodes list)
            
            for node in type_nodes:
                # Now show actual depth value (not absolute)
                index_content.append(f"- [{node['summary']}]({node['filename']}) (Depth: {node['depth']})")
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
import json
import os
import re

def slugify(text):
    """Convert text to a URL-friendly slug format."""
    # Remove special characters and replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_-]+', '-', slug)
    return slug

def escape_yaml_string(text):
    """
    Properly escape a string for YAML front matter to avoid quote issues.
    Uses single quotes for outer quotes if text contains double quotes,
    and escapes any existing single quotes in the text.
    """
    if not isinstance(text, str):
        return text
        
    # If text contains double quotes, use single quotes as outer quotes
    if '"' in text:
        # Escape any existing single quotes by doubling them
        text = text.replace("'", "''")
        return f"'{text}'"
    
    # Otherwise use double quotes
    # Escape any backslashes
    text = text.replace('\\', '\\\\')
    return f'"{text}"'

def create_markdown_wiki(json_file_path, output_dir="Wiki_Storage/Knowledge_Wiki__D5_Hugo_v2"):
    """
    Convert a JSON graph structure into a network of linked Markdown files.
    
    Args:
        json_file_path: Path to the JSON file containing the graph data
        output_dir: Directory where the Markdown files will be created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create content directory for Hugo
    content_dir = os.path.join(output_dir, "content")
    os.makedirs(content_dir, exist_ok=True)
    
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
    
    # Track filename mappings to handle duplicate slugs
    filename_map = {}
    filename_counter = {}
    
    # Function to get unique filename and handle duplicates
    def get_unique_filename_with_dedup(node_id, node_data):
        summary = node_data.get('summary', '')
        
        # Create slug from summary
        if summary:
            base_filename = f"{slugify(summary)}.md"
        else:
            base_filename = f"{node_id}.md"
        
        # If this would create a duplicate, add a number
        if base_filename in filename_counter:
            filename_counter[base_filename] += 1
            name, ext = os.path.splitext(base_filename)
            new_filename = f"{name}-{filename_counter[base_filename]}{ext}"
            filename_map[node_id] = new_filename
            return new_filename
        else:
            filename_counter[base_filename] = 0
            filename_map[node_id] = base_filename
            return base_filename
    
    # Second pass: Create Markdown files for each node
    for node_id, node_data in graph_data.items():
        # Extract node data
        summary = node_data.get('summary', 'No Summary')
        content = node_data.get('content', 'No Content')
        node_type = node_data.get('node_type', 'Unknown Type')
        parent_id = node_data.get('parent_id')
        depth = node_data.get('depth', 0)
        terminal = node_data.get('terminal', False)
        
        # Format content by splitting on curly braces and creating enumerated points
        if '{' in content and '}' in content:
            # Split the content by the pattern of closing brace possibly followed by comma and space
            points = re.split(r'\}\s*,?\s*', content)
            
            # Remove empty strings and process each point
            points = [point.strip() for point in points if point.strip()]
            
            # Format each point: remove opening braces and add enumeration
            formatted_points = []
            for i, point in enumerate(points, 1):
                # Remove opening brace if present
                if point.startswith('{'):
                    point = point[1:].strip()
                # Add the enumerated point
                formatted_points.append(f"{i}. **{point}**")
            
            # Join with newlines to create the formatted content
            content = '\n'.join(formatted_points)
        else:
            # If no curly braces are found, just keep the content as is
            content = content
        
        # Create unique filename for this node
        filename = get_unique_filename_with_dedup(node_id, node_data)
        filepath = os.path.join(content_dir, filename)
        
        # Use escape_yaml_string to properly handle quotes in all string values
        yaml_safe_summary = escape_yaml_string(summary)
        yaml_safe_node_type = escape_yaml_string(node_type)
        
        # Build Hugo front matter
        front_matter = [
            "---",
            f'title: {yaml_safe_summary}',
            f'nodetype: {yaml_safe_node_type}',
            f'nodeid: "{node_id}"',
            f'depth: {depth}',
            f'terminal: {str(terminal).lower()}',
        ]
        
        # Add parent info to front matter if it exists
        if parent_id and parent_id in graph_data:
            parent_data = graph_data[parent_id]
            parent_summary = parent_data.get('summary', 'Parent Node')
            # Fix quotes in parent summary
            yaml_safe_parent = escape_yaml_string(parent_summary)
            front_matter.append(f'parent: {yaml_safe_parent}')
            front_matter.append(f'parentid: "{parent_id}"')
        
        front_matter.append("---")
        
        # Start building the Markdown content
        md_content = front_matter + [
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
            # We'll set the exact filename in a third pass after we have all the mappings
            md_content.append(f"**Parent:** [{parent_summary}](PLACEHOLDER_PARENT_{parent_id})")
            md_content.append("")
        
        # Add content section
        md_content.append("## Content")
        md_content.append("")
        md_content.append(content)
        md_content.append("")
        
        # Add children links as placeholders (we'll fill in a third pass)
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
                    md_content.append(f"- [{child_summary}](PLACEHOLDER_CHILD_{child_id})")
                
                md_content.append("")
        
        # Save the Markdown file for now with placeholders
        with open(filepath, 'w') as f:
            f.write('\n'.join(md_content))
    
    # Third pass: Fill in the correct filenames for links
    for node_id, node_data in graph_data.items():
        filename = filename_map[node_id]
        filepath = os.path.join(content_dir, filename)
        
        # Read the file content
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace parent placeholder
        parent_id = node_data.get('parent_id')
        if parent_id and parent_id in filename_map:
            parent_filename = filename_map[parent_id]
            content = content.replace(f"PLACEHOLDER_PARENT_{parent_id}", parent_filename)
        
        # Replace child placeholders
        if node_id in children_map:
            for child_id in children_map[node_id]:
                if child_id in filename_map:
                    child_filename = filename_map[child_id]
                    content = content.replace(f"PLACEHOLDER_CHILD_{child_id}", child_filename)
        
        # Write updated content
        with open(filepath, 'w') as f:
            f.write(content)
    
    # Create Hugo config file
    config_toml = [
        'baseURL = "/"',
        'languageCode = "en-us"',
        'title = "Knowledge Graph Wiki"',
        '',
        '# URL handling',
        'uglyURLs = false',
        'canonifyURLs = true',
        '',
        '# Enable Git info',
        'enableGitInfo = true',
        '',
        '# Configure markdown renderer',
        '[markup]',
        '  [markup.goldmark]',
        '    [markup.goldmark.renderer]',
        '      unsafe = true'
    ]
    
    with open(os.path.join(output_dir, "config.toml"), 'w') as f:
        f.write('\n'.join(config_toml))
    
    # Create a minimal layout file for Hugo
    layouts_dir = os.path.join(output_dir, "layouts", "_default")
    os.makedirs(layouts_dir, exist_ok=True)
    
    # Create single.html template
    single_html = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '    <meta charset="utf-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1">',
        '    <title>{{ .Title }} | {{ .Site.Title }}</title>',
        '    <style>',
        '        body {',
        '            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;',
        '            line-height: 1.6;',
        '            margin: 0;',
        '            padding: 20px;',
        '            max-width: 800px;',
        '            margin: 0 auto;',
        '            color: #333;',
        '        }',
        '        a {',
        '            color: #0366d6;',
        '            text-decoration: none;',
        '        }',
        '        a:hover {',
        '            text-decoration: underline;',
        '        }',
        '        h1, h2, h3 {',
        '            margin-top: 1.5em;',
        '            margin-bottom: 0.5em;',
        '        }',
        '        .node-meta {',
        '            background: #f5f5f5;',
        '            padding: 1em;',
        '            border-radius: 4px;',
        '            margin-bottom: 1em;',
        '        }',
        '        .parent-link {',
        '            margin: 1em 0;',
        '        }',
        '        .content {',
        '            margin: 2em 0;',
        '        }',
        '        .site-header {',
        '            margin-bottom: 2em;',
        '            padding-bottom: 1em;',
        '            border-bottom: 1px solid #eee;',
        '        }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="site-header">',
        '        <h3><a href="/">{{ .Site.Title }}</a></h3>',
        '    </div>',
        '',
        '    <h1>{{ .Title }}</h1>',
        '',
        '    <div class="node-meta">',
        '        <p><strong>Node Type:</strong> {{ .Params.nodetype }}</p>',
        '        <p><strong>Depth:</strong> {{ .Params.depth }}</p>',
        '        <p><strong>Terminal:</strong> {{ if .Params.terminal }}Yes{{ else }}No{{ end }}</p>',
        '',
        '        {{ if .Params.parent }}',
        '        <p class="parent-link"><strong>Parent:</strong> <a href="{{ .Params.parentid | urlize }}.html">{{ .Params.parent }}</a></p>',
        '        {{ end }}',
        '    </div>',
        '',
        '    <div class="content">',
        '        {{ .Content }}',
        '    </div>',
        '</body>',
        '</html>'
    ]
    
    with open(os.path.join(layouts_dir, "single.html"), 'w') as f:
        f.write('\n'.join(single_html))
    
    # Create a custom index.html template
    index_html = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '    <meta charset="utf-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1">',
        '    <title>{{ .Site.Title }}</title>',
        '    <style>',
        '        body {',
        '            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;',
        '            line-height: 1.6;',
        '            margin: 0;',
        '            padding: 20px;',
        '            max-width: 800px;',
        '            margin: 0 auto;',
        '            color: #333;',
        '        }',
        '        a {',
        '            color: #0366d6;',
        '            text-decoration: none;',
        '        }',
        '        a:hover {',
        '            text-decoration: underline;',
        '        }',
        '        h1, h2, h3 {',
        '            margin-top: 1.5em;',
        '            margin-bottom: 0.5em;',
        '        }',
        '        #search-box {',
        '            width: 100%;',
        '            padding: 10px;',
        '            font-size: 16px;',
        '            margin-bottom: 20px;',
        '            border: 1px solid #ddd;',
        '            border-radius: 4px;',
        '        }',
        '    </style>',
        '</head>',
        '<body>',
        '    <h1>{{ .Site.Title }}</h1>',
        '',
        '    <input type="text" id="search-box" placeholder="Search pages...">',
        '',
        '    {{ .Content }}',
        '',
        '    <h2>All Pages</h2>',
        '    <div id="page-list">',
        '    <ul>',
        '        {{ range .Site.RegularPages }}',
        '        <li>',
        '            <a href="{{ .RelPermalink }}">{{ .Title }}</a>',
        '            {{ if .Params.nodetype }} ({{ .Params.nodetype }}){{ end }}',
        '            {{ if .Params.depth }} - Depth: {{ .Params.depth }}{{ end }}',
        '        </li>',
        '        {{ end }}',
        '    </ul>',
        '    </div>',
        '',
        '    <script>',
        '        // Simple search function',
        '        const searchBox = document.getElementById("search-box");',
        '        const pageList = document.getElementById("page-list");',
        '        const allItems = pageList.innerHTML;',
        '',
        '        searchBox.addEventListener("input", function() {',
        '            const query = this.value.toLowerCase();',
        '            if (query.length < 2) {',
        '                pageList.innerHTML = allItems;',
        '                return;',
        '            }',
        '',
        '            const links = pageList.querySelectorAll("li");',
        '            let html = "<ul>";',
        '',
        '            for (const link of links) {',
        '                const text = link.textContent.toLowerCase();',
        '                if (text.includes(query)) {',
        '                    html += `<li>${link.innerHTML}</li>`;',
        '                }',
        '            }',
        '',
        '            html += "</ul>";',
        '            pageList.innerHTML = html;',
        '        });',
        '    </script>',
        '</body>',
        '</html>'
    ]
    
    # Create layouts directory for the index page
    os.makedirs(os.path.join(output_dir, "layouts"), exist_ok=True)
    with open(os.path.join(output_dir, "layouts", "index.html"), 'w') as f:
        f.write('\n'.join(index_html))
    
    # Create a basic index file with proper quote handling
    index_title = escape_yaml_string("Knowledge Graph Wiki")
    index_content = [
        "---",
        f'title: {index_title}',
        "---",
        "",
        "Welcome to the Knowledge Graph Wiki. Browse through the nodes below or use the search to find specific topics.",
        "",
        "## Root Nodes",
        ""
    ]
    
    # Find the root nodes (nodes with no parents or central questions)
    root_nodes = []
    for node_id, node_data in graph_data.items():
        if node_data.get('parent_id') is None or node_data.get('is_central_question', False):
            root_nodes.append(node_id)
    
    # Add root nodes to the index
    if root_nodes:
        for root_id in root_nodes:
            if root_id in graph_data:
                root_data = graph_data[root_id]
                root_summary = root_data.get('summary', 'Root Node')
                root_type = root_data.get('node_type', 'Unknown Type')
                root_filename = filename_map[root_id]
                index_content.append(f"- [{root_summary}]({root_filename}) ({root_type})")
        
    # Create an _index.md file in the content directory
    with open(os.path.join(content_dir, "_index.md"), 'w') as f:
        f.write('\n'.join(index_content))
    
    # Create a .gitlab-ci.yml file for GitLab Pages, without the quote fixing script
    gitlab_ci = [
        'image: registry.gitlab.com/pages/hugo/hugo_extended:latest',
        '',
        'variables:',
        '  HUGO_ENV: production',
        '',
        'pages:',
        '  script:',
        '    - hugo',
        '  artifacts:',
        '    paths:',
        '      - public',
        '  only:',
        '    - main'
    ]
    
    with open(os.path.join(output_dir, ".gitlab-ci.yml"), 'w') as f:
        f.write('\n'.join(gitlab_ci))
    
    print(f"Hugo-compatible wiki created successfully in the '{output_dir}' directory!")
    print("To use with GitLab Pages:")
    print("1. Push the content to GitLab")
    print("2. The .gitlab-ci.yml file will automatically build and deploy your wiki")

if __name__ == "__main__":
    # Replace with JSON file path
    json_file_path = "graph_analysis.json"
    create_markdown_wiki(json_file_path)
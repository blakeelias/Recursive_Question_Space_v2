import json
import networkx as nx
import matplotlib.pyplot as plt

def create_knowledge_graph(data):
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node_id, node_data in data.items():
        # Create prefixed summary based on node type
        prefix = ''
        if node_data['node_type'] == 'thesis':
            prefix = '(T) '
        elif node_data['node_type'] == 'antithesis':
            prefix = '(A) '
        elif node_data['node_type'] == 'synthesis':
            prefix = '(S) '
            
        # Add node with its prefixed summary as label
        G.add_node(node_id, summary=prefix + node_data['summary'])
        
        # Add edge if there's a parent
        if node_data['parent_id'] is not None:
            G.add_edge(node_data['parent_id'], node_id)
    
    return G

def visualize_graph(G):
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create layout with stronger repulsion and more iterations
    pos = nx.spring_layout(G, 
                          k=2,  # Increase repulsion (default is 1)
                          iterations=100,  # More iterations for better convergence
                          seed=42,  # For consistency between runs
                          scale=2)  # Scale up the layout
    
    # Apply additional repulsion
    for _ in range(10):  # Extra iterations for fine-tuning
        for n1 in G.nodes():
            for n2 in G.nodes():
                if n1 < n2:  # Avoid processing pairs twice
                    # Get positions
                    x1, y1 = pos[n1]
                    x2, y2 = pos[n2]
                    
                    # Calculate distance
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = (dx * dx + dy * dy) ** 0.5
                    
                    if dist < 0.5:  # If nodes are too close
                        # Apply repulsive force
                        force = 0.1 * (0.5 - dist) / dist if dist > 0 else 0.1
                        pos[n1][0] -= dx * force
                        pos[n1][1] -= dy * force
                        pos[n2][0] += dx * force
                        pos[n2][1] += dy * force
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=2000,
                          alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20)
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'summary')
    # Wrap text at 30 characters
    wrapped_labels = {k: '\n'.join([v[i:i+30] for i in range(0, len(v), 30)]) 
                     for k, v in labels.items()}
    nx.draw_networkx_labels(G, pos, 
                           wrapped_labels,
                           font_size=8)
    
    plt.axis('off')
    return plt

# Load the data
with open('graph.json', 'r') as f:
    data = json.load(f)

# Create and visualize the graph
G = create_knowledge_graph(data)
plt = visualize_graph(G)

# Save the plot
plt.savefig('knowledge_graph.png', 
            bbox_inches='tight',
            dpi=300,
            format='png')
plt.close()
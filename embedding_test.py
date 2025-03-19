import faiss
import json
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

def load_graph_and_embeddings(export_dir: str):
    """Load graph, node mapping and FAISS index from an export directory"""
    with open(os.path.join(export_dir, "graph.json"), 'r') as f:
        graph = json.load(f)
    with open(os.path.join(export_dir, "node_mapping.json"), 'r') as f:
        node_mapping = json.load(f)
    index = faiss.read_index(os.path.join(export_dir, "faiss_index.index"))
    return graph, node_mapping, index

def get_embedding(client: OpenAI, text: str) -> np.ndarray:
    """Get embedding for text using same model as graph"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

def find_similar_nodes(query: str, graph: dict, index: faiss.Index, node_mapping: dict, client: OpenAI, k: int = 3):
    """Find k most similar nodes to query"""
    # Get query embedding
    query_embedding = get_embedding(client, query)
    
    # Search FAISS index
    distances, faiss_indices = index.search(query_embedding, k)
    
    # Match FAISS indices to node IDs
    results = []
    for i, faiss_idx in enumerate(faiss_indices[0]):
        # Find node ID for this FAISS index
        node_id = None
        for nid, idx in node_mapping.items():
            if idx == faiss_idx:
                node_id = nid
                break
        
        if node_id:
            node = graph[node_id]
            results.append({
                'node_id': node_id,
                'similarity_score': float(distances[0][i]),
                'node_type': node['node_type'],
                'summary': node['summary'],
                'content': node['content']
            })
    
    return results

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Load most recent export
    export_dir = "./graph_export_test"  # Update this path
    graph, node_mapping, index = load_graph_and_embeddings(export_dir)
    
    while True:
        # Get user input
        query = input("\nEnter your philosophical question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        # Find similar nodes
        similar_nodes = find_similar_nodes(query, graph, index, node_mapping, client)
        
        # Display results
        print("\nMost similar positions in our philosophical graph:")
        for i, node in enumerate(similar_nodes, 1):
            print(f"\n{i}. Node Type: {node['node_type']}")
            print(f"Similarity Score: {node['similarity_score']}")
            print(f"Summary: {node['summary']}")
            print(f"Full Content: {node['content']}")
            print("-" * 80)
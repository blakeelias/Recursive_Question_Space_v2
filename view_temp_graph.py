import pickle
import os
import json
import pprint
from pathlib import Path

def load_and_display_graph_state(save_dir="./temp"):
    """Load pickled graph state and display its contents"""
    # Find the pickle file
    pickle_path = os.path.join(save_dir, 'graph_state.pickle')
    
    if not os.path.exists(pickle_path):
        print(f"Error: Could not find pickle file at {pickle_path}")
        return
    
    print(f"Loading pickle file from {pickle_path}")
    
    # Load the pickled object
    try:
        with open(pickle_path, 'rb') as f:
            state = pickle.load(f)
            
        # Print the main structure
        print("\n== STATE STRUCTURE ==")
        for key, value in state.items():
            if key == 'graph':
                print(f"graph: [Dictionary with {len(value)} nodes]")
            elif key == 'edges':
                print(f"edges: [List with {len(value)} connections]")
            elif key == 'config':
                print(f"config: {value}")
            elif key == 'state':
                print("state:")
                for state_key, state_value in value.items():
                    print(f"  {state_key}: {len(state_value)} items")
            else:
                print(f"{key}: {value}")
                
        # Display details about the graph
        print("\n== GRAPH SUMMARY ==")
        node_types = {}
        for node_id, node_data in state['graph'].items():
            node_type = node_data.get('node_type')
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
            
            # Check if this is the central question
            if node_data.get('is_central_question', False):
                print(f"Central Question: {node_data['content']}")
                
        print("\nNode Types:")
        for node_type, count in node_types.items():
            print(f"  {node_type}: {count} nodes")
            
        # Display some example nodes of each type
        print("\n== SAMPLE NODES ==")
        shown_types = set()
        for node_id, node_data in state['graph'].items():
            node_type = node_data.get('node_type')
            if node_type not in shown_types and node_type != 'question':
                print(f"\nSample {node_type} node (ID: {node_id}):")
                print(f"  Summary: {node_data['summary']}")
                print(f"  Content (first 100 chars): {node_data['content'][:100]}...")
                shown_types.add(node_type)
                
        # Show processing status
        print("\n== PROCESSING STATUS ==")
        for status_name, status_values in state['state'].items():
            print(f"{status_name}: {len(status_values)} items processed")
            
        # Option to export as JSON
        export = input("\nExport full state as JSON? (y/n): ")
        if export.lower() == 'y':
            output_path = os.path.join(save_dir, 'graph_state_export.json')
            # Convert sets to lists for JSON serialization
            json_state = state.copy()
            for key, value in json_state.get('state', {}).items():
                if isinstance(value, list):
                    json_state['state'][key] = value
                    
            with open(output_path, 'w') as f:
                json.dump(json_state, f, indent=2)
            print(f"Exported state to {output_path}")
            
    except Exception as e:
        print(f"Error unpickling the state: {e}")

if __name__ == "__main__":
    # You can customize the save directory here
    save_dir = input("Enter the save directory path (default: ./temp): ") or "./temp"
    load_and_display_graph_state(save_dir)